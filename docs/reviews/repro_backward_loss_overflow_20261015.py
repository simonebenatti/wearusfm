"""Healthy numerical-stop regressions; both intentionally FAIL on reviewed 59b8c5b.

Use the real trainer, model, calibrated Step 1 smooth-L1 objective and synthetic
CPU data. A finite keep-head bias of 1e37 and the accepted keep_weight=20 make
the weighted local numerator overflow before division, while the numerators,
global mean and gradients remain finite. No fabricated loss terms are injected.

Run from the checkout with PYTHONPATH=src:tests/cpu, PYTHONDONTWRITEBYTECODE=1,
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 and the wearusfm-tok
Python: python -m pytest -q -p no:cacheprovider <this file>.

For a controlled verification of the proposed minimal fix, prefix the same
command with WEARUSFM_REVIEW_IN_MEMORY_FIX=1. The autouse fixture then replaces
only _accumulate_terms in memory; no source file is modified. Expected: 2 PASS.
The normal run keeps the reviewed implementation and is expected to report
2 FAIL until Claude implements the trainer correction. No gloo, GPU or SSH.
"""

import copy
import inspect
import json
import os
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.jepa import LossTerms, make_teacher
from test_spectral_step1 import _calibration, _setup


@pytest.fixture(autouse=True)
def _optional_in_memory_fix(monkeypatch):
    """Apply the exact proposed guard only when explicitly requested by env."""
    if os.environ.get("WEARUSFM_REVIEW_IN_MEMORY_FIX") != "1":
        return
    source = inspect.getsource(R._accumulate_terms)
    before = "            finite = mismatch or all(bool(torch.isfinite(v).all()) for v in terms.num.values())\n"
    after = (
        "            loss = None if mismatch else terms.backward_loss(global_den, world_size=ctx.world_size)\n"
        "            finite = mismatch or (all(bool(torch.isfinite(v).all()) for v in terms.num.values())\n"
        "                                  and bool(torch.isfinite(loss).all()))\n"
    )
    later = "            loss = terms.backward_loss(global_den, world_size=ctx.world_size)\n            loss.backward()\n"
    assert source.count(before) == source.count(later) == 1, "reviewed implementation changed; inspect the patch before adapting"
    source = source.replace(before, after).replace(later, "            loss.backward()\n")
    namespace = {}
    exec(compile(source, "<review-only finite-backward-loss guard>", "exec"), R.__dict__, namespace)
    monkeypatch.setattr(R, "_accumulate_terms", namespace["_accumulate_terms"])


def _case(tmp_path):
    root, manifest, cfg, _, y, valid = _setup(tmp_path)
    calibration = tmp_path / "calibration.npz"
    _calibration(calibration, cfg, manifest, y, valid)
    cfg = replace(R.with_spectral_keep(cfg, calibration, 20.0), batch_size=4, accum_steps=2)
    return root, manifest, cfg, y, valid


def _seed_finite_head_bias(student):
    with torch.no_grad():
        student.spectral_keep.linear.bias.fill_(1e37)
    assert torch.isfinite(student.spectral_keep.linear.bias).all()


def _assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, want in zip(actual, expected):
            _assert_same(item, want)
    else:
        assert actual == expected


def test_finite_numerators_and_gradients_do_not_authorize_an_infinite_backward_loss(tmp_path, monkeypatch):
    root, manifest, cfg, y, valid = _case(tmp_path)
    student = WearUsFM(cfg.model)
    student.spectral_keep.fit_target_stats(y, valid)
    _seed_finite_head_bias(student)
    teacher = make_teacher(student)
    index = R.load_index(manifest, [root], cfg.datasets)
    batch = L.PretrainLoader(index, cfg.loader).batch(cfg.batch_size, np.random.default_rng(7))
    micros = [R._prepare(part, "cpu") for part in L.split_batch(batch, cfg.accum_steps)]
    counts = [R.loss_counts(m.inp, m.visible, cfg.jepa, m.anchor_targets, model_cfg=student.cfg,
                            keep_target_active=student.spectral_keep.target_active) for m in micros]
    global_den = {key: sum(part[key] for part in counts) for key in counts[0]}

    # Establish the fault with REAL objective terms before exercising the guard.
    for micro in micros:
        terms = R.jepa_loss_terms(student, teacher, micro.inp, micro.visible, cfg.jepa,
                                  anchor_targets=micro.anchor_targets, kinds=micro.kinds)
        assert all(torch.isfinite(value).all() for value in terms.num.values())
        mathematical_keep = cfg.jepa.keep_weight * terms.num["keep"].double() / global_den["keep"]
        assert torch.isfinite(mathematical_keep) and mathematical_keep < torch.finfo(torch.float32).max
        loss = terms.backward_loss(global_den)
        assert torch.isinf(loss), "the fault must be overflow in the actual backward loss"
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in student.parameters())
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip))
        student.zero_grad(set_to_none=True)

    calls = []
    real_backward = torch.Tensor.backward

    def record_backward(tensor, *args, **kwargs):
        calls.append(float(tensor.detach()))
        return real_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    failure, metrics = R._accumulate_terms(None, student, teacher, micros, cfg, R.DistContext(), "cpu", nullcontext())
    assert failure == "perdita non finita", (failure, calls)
    assert metrics == {} and calls == []
    assert all(parameter.grad is None for parameter in student.parameters())


def test_infinite_backward_loss_stops_real_train_before_optimizer_ema_and_checkpoint(tmp_path, monkeypatch):
    root, manifest, cfg, _, _ = _case(tmp_path)
    out = tmp_path / "run"
    observed, losses = {}, []
    calls = {"backward": 0, "optimizer": 0, "ema": 0}
    real_teacher, real_init = R.make_teacher, torch.optim.AdamW.__init__
    real_step, real_ema = torch.optim.AdamW.step, R.ema_update
    real_loss, real_backward = LossTerms.backward_loss, torch.Tensor.backward

    def capture_teacher(student):
        _seed_finite_head_bias(student)
        teacher = real_teacher(student)
        observed.update(student=student, teacher=teacher,
                        student_before=copy.deepcopy(student.state_dict()), teacher_before=copy.deepcopy(teacher.state_dict()))
        return teacher

    def capture_optimizer(optimizer, *args, **kwargs):
        real_init(optimizer, *args, **kwargs)
        observed["optimizer"] = optimizer

    def record_loss(terms, *args, **kwargs):
        if "optimizer_before" not in observed:
            # The trainer has already selected this step's warmup LR. Compare
            # optimizer state from here, before any backward or optimizer step.
            observed["optimizer_before"] = copy.deepcopy(observed["optimizer"].state_dict())
        loss = real_loss(terms, *args, **kwargs)
        losses.append({"finite_numerators": all(bool(torch.isfinite(value).all()) for value in terms.num.values()),
                       "loss": float(loss.detach())})
        return loss

    def record_backward(tensor, *args, **kwargs):
        calls["backward"] += 1
        return real_backward(tensor, *args, **kwargs)

    def record_step(optimizer, *args, **kwargs):
        calls["optimizer"] += 1
        return real_step(optimizer, *args, **kwargs)

    def record_ema(*args, **kwargs):
        calls["ema"] += 1
        return real_ema(*args, **kwargs)

    monkeypatch.setattr(R, "make_teacher", capture_teacher)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", capture_optimizer)
    monkeypatch.setattr(LossTerms, "backward_loss", record_loss)
    monkeypatch.setattr(torch.Tensor, "backward", record_backward)
    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    monkeypatch.setattr(R, "ema_update", record_ema)
    summary = R.train(cfg, manifest, [root], out, log=lambda _: None)

    assert losses and all(item["finite_numerators"] and np.isinf(item["loss"]) for item in losses)
    assert summary["stopped"] == "perdita non finita al passo 0", (summary, calls, losses)
    assert summary["steps"] == 0 and calls == {"backward": 0, "optimizer": 0, "ema": 0}
    _assert_same(observed["student"].state_dict(), observed["student_before"])
    _assert_same(observed["teacher"].state_dict(), observed["teacher_before"])
    _assert_same(observed["optimizer"].state_dict(), observed["optimizer_before"])
    assert all(parameter.grad is None for parameter in observed["student"].parameters())
    assert not (out / "checkpoint.pt").exists() and not (out / "checkpoint.tmp").exists()
    assert (out / "metrics.jsonl").read_bytes() == b""
    assert (out / "STOP").read_text().strip() == summary["stopped"]
    assert json.loads((out / "summary.json").read_text()) == summary

    before_calls, before_losses = dict(calls), len(losses)
    blocked = R.train(cfg, manifest, [root], out, log=lambda _: None)
    assert blocked["steps"] is None and blocked["stopped"].startswith("fermato in precedenza: perdita non finita")
    assert calls == before_calls and len(losses) == before_losses
