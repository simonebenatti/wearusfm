"""Small real-trainer checks of the matched contrastive arms and global DDP loss.

Synthetic signals pass through the actual loader, tokenizer, encoder, loss
contract, AdamW, EMA and checkpoint code. This is not a CUDA timing benchmark.
"""

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from test_pose_training import _pose_setup, _r3_config
from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime, physical_pair_mask
from wearusfm.training.contrastive_memory import WindowIdentity
from wearusfm.training.jepa import make_teacher, ema_update
from wearusfm.model.rank_contrast import rank_contrast_pairs


def _config(pose_root, arm="C0", *, calibration=False):
    base = _r3_config(pose_root, total=4, keep=())
    contrast = ContrastiveConfig(arm=arm, capacity_per_dataset=8, min_negatives=1, gap_seconds=.001,
                                fill_steps=0, ramp_steps=1, rnc_stride=2, calibration_only=calibration)
    return replace(base, max_steps=4, batch_size=4, fixed_eval_every=0,
                   model=replace(base.model, contrastive_projection=True, muscle_dropout=0.),
                   loader=replace(base.loader, min_window_s=1., max_window_s=1.),
                   contrastive=asdict(contrast), datasets=("emg2pose",))


def _load(path):
    return torch.load(Path(path) / "checkpoint.pt", weights_only=False)


def _same_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _same_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same_nested(a, b)
    else:
        assert left == right


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("contrastive_runtime")
    root, manifest, pose_root = _pose_setup(tmp)
    configs, summaries = {}, {}
    for arm in ("C0", "CN", "CR"):
        cfg = _config(pose_root, arm)
        configs[arm] = cfg
        summaries[arm] = R.train(cfg, manifest, [root], tmp / arm, log=lambda _: None)
    return tmp, root, manifest, pose_root, configs, summaries


def test_all_three_arms_complete_real_training_and_advance_memory(trained):
    tmp, _, _, _, configs, summaries = trained
    for arm, summary in summaries.items():
        assert summary["stopped"] == "max_steps" and summary["steps"] == 4
        ckpt = _load(tmp / arm)
        state = ckpt["contrastive_state"]
        assert state["queue"]["last_step"] == 3 and state["queue"]["entries"]
        assert state["expected_datasets"] == ["emg2pose"]
        assert state["coverage"]["updates"] == 4
        assert state["coverage"]["datasets"]["emg2pose"][0] == 16
        recs = [json.loads(line) for line in (tmp / arm / "metrics.jsonl").read_text().splitlines()]
        assert len(recs) == 4
        assert all(np.isfinite(r["total"]) and np.isfinite(r["contrast_rnc"]) and np.isfinite(r["contrast_nce"]) for r in recs)
        assert any(r["contrast_rnc_windows"] > 0 for r in recs)
        assert any(r["contrast_nce_eligible"] > 0 for r in recs)
        assert all(r["contrast_nce_weight"] == pytest.approx(configs[arm].contrastive["nce_weight"] if arm == "CN" else 0.) for r in recs)
        assert all(r["contrast_rnc_weight"] == pytest.approx(configs[arm].contrastive["rnc_weight"] if arm == "CR" else 0.) for r in recs)
    control = _load(tmp / "C0")["student"]
    for arm in ("CN", "CR"):
        candidate = _load(tmp / arm)["student"]
        assert any(not torch.equal(control[k], candidate[k]) for k in control if k.startswith("backbone."))


def test_zero_weight_control_preserves_base_training_updates(trained):
    tmp, root, manifest, _, configs, _ = trained
    cfg = configs["C0"]
    baseline = replace(cfg, contrastive=None, model=replace(cfg.model, contrastive_projection=False))
    R.train(baseline, manifest, [root], tmp / "base", log=lambda _: None)
    old, control = _load(tmp / "base"), _load(tmp / "C0")
    for part in ("student", "teacher"):
        for name, value in old[part].items():
            # Base historical single-process path and additive-contract path may
            # differ by roundoff in the summation order; no stochastic mismatch.
            torch.testing.assert_close(control[part][name], value, rtol=2e-5, atol=2e-6, msg=f"{part}: {name}")
    assert torch.equal(old["torch_rng"], control["torch_rng"])


def test_resume_preserves_model_optimizer_rng_and_queue_bit_for_bit(trained):
    tmp, root, manifest, _, configs, _ = trained
    cfg = configs["CN"]
    R.train(replace(cfg, max_steps=2), manifest, [root], tmp / "resume", log=lambda _: None)
    R.train(cfg, manifest, [root], tmp / "resume", log=lambda _: None)
    whole, resumed = _load(tmp / "CN"), _load(tmp / "resume")
    for part in ("student", "teacher", "optimizer", "torch_rng", "contrastive_state"):
        _same_nested(whole[part], resumed[part])
    changed = replace(cfg, contrastive={**cfg.contrastive, "rnc_stride": 4})
    with pytest.raises(ValueError, match="ricetta diversa"):
        R.train(changed, manifest, [root], tmp / "resume", log=lambda _: None)


def test_calibration_advances_only_queue_and_measures_global_gradients(trained):
    tmp, root, manifest, pose_root, _, _ = trained
    cfg = _config(pose_root, "CN", calibration=True)
    torch.manual_seed(cfg.seed)
    init = WearUsFM(cfg.model).state_dict()
    summary = R.train(cfg, manifest, [root], tmp / "calibration", log=lambda _: None)
    ckpt = _load(tmp / "calibration")
    assert summary["contrastive"]["optimizer_updates"] == 0
    assert summary["contrastive"]["initial_sha256"] == summary["contrastive"]["final_student_sha256"]
    assert summary["contrastive"]["initial_sha256"] == summary["contrastive"]["final_teacher_sha256"]
    assert not ckpt["optimizer"]["state"]
    assert ckpt["contrastive_state"]["queue"]["last_step"] == 3
    for part in ("student", "teacher"):
        _same_nested(init, ckpt[part])
    recs = [json.loads(line) for line in (tmp / "calibration" / "metrics.jsonl").read_text().splitlines()]
    assert all(r["contrast_global_grad_base"] > 0 and r["contrast_global_grad_rnc"] > 0 for r in recs)
    assert any(r["contrast_global_grad_nce"] > 0 for r in recs)
    assert all(r["contrastive_calibration_only"] for r in recs)
    with pytest.raises(ValueError, match="start fresh"):
        R.train(cfg, manifest, [root], tmp / "calibration", log=lambda _: None)


@pytest.mark.parametrize("change,match", [
    ({"accum_steps": 2}, "accum_steps=1"),
    ({"contrastive": None}, "explicit matched"),
])
def test_invalid_runtime_contract_is_rejected_before_training(trained, change, match):
    tmp, root, manifest, _, configs, _ = trained
    with pytest.raises(ValueError, match=match):
        R.train(replace(configs["C0"], **change), manifest, [root], tmp / "invalid", log=lambda _: None)


def _global_update(cfg, init, batch, ctx):
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    wrapped = R._ddp_wrap(student, cfg, ctx, "cpu") if ctx.active else None
    engine = ContrastiveRuntime(cfg.contrastive)
    engine.prepare(batch, 0)
    micro = R._prepare(batch, "cpu", torch.float64)
    student.train()
    failure, metrics = R._accumulate_terms(wrapped, student, teacher, [micro], cfg, ctx, "cpu", nullcontext(), contrastive=engine)
    assert failure is None, failure
    grads = {name: p.grad.clone() for name, p in student.named_parameters() if p.grad is not None}
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
    opt.step()
    ema_update(teacher, student, cfg.jepa.ema_momentum)
    engine.commit()
    return {"metrics": metrics, "grads": grads, "student": student.state_dict(), "teacher": teacher.state_dict(),
            "queue": engine.state_dict()}


def _worker(rank, world, store, cfg, init, batch, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=45))
    try:
        local = L.split_batch(batch, world)[rank]
        result = _global_update(cfg, init, local, R.DistContext(rank, world, rank))
        torch.save(result, Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _spawn_bounded(world, store, cfg, init, batch, output):
    """Own child processes only; a collective failure must not hang pytest."""
    context = mp.spawn(_worker, args=(world, store, cfg, init, batch, output), nprocs=world, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("contrastive CPU DDP test exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.parametrize("arm,world", [("CN", 2), ("CR", 2), ("CR", 4)])
def test_global_full_model_gradients_match_ddp_with_rank_without_pose(tmp_path, arm, world):
    root, manifest, pose_root = _pose_setup(tmp_path)
    cfg = _config(pose_root, arm)
    cfg = replace(cfg, contrastive={**cfg.contrastive, "grad_every": 1})
    # Full population of physical windows, with some valid pose labels removed
    # to force a whole rank to contribute only as an unlabelled/zero-pose rank.
    loader = L.PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader, {}, None)
    batch = loader.batch(4, np.random.default_rng(3))
    for i in range(2):
        batch.pose[i] = None
    assert any(item is not None for item in batch.pose[2:])
    # Distinct synthetic sessions avoid relying on a lucky temporal gap for NCE.
    batch.rows = [{**row, "session": f"synthetic_session_{i}"} for i, row in enumerate(batch.rows)]
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    reference = _global_update(cfg, init, batch, R.DistContext())
    output = tmp_path / "ddp"
    output.mkdir()
    _spawn_bounded(world, str(tmp_path / "store"), cfg, init, batch, str(output))
    values = [torch.load(output / f"rank{rank}.pt", weights_only=False) for rank in range(world)]
    for got in values:
        assert set(got["grads"]) == set(reference["grads"])
        for key in ("total", "contrast_nce", "contrast_rnc"):
            assert got["metrics"][key] == pytest.approx(reference["metrics"][key], rel=1e-9, abs=1e-10)
        for key in ("contrast_global_grad_base", "contrast_global_grad_nce", "contrast_global_grad_rnc"):
            assert got["metrics"][key] == pytest.approx(reference["metrics"][key], rel=2e-6, abs=1e-8)
        for part in ("grads", "student", "teacher"):
            for key, value in reference[part].items():
                torch.testing.assert_close(got[part][key], value, rtol=2e-8, atol=2e-9, msg=f"{arm}: {part} {key}")
        # Queue values are projection outputs; tolerate only the tiny grouping
        # roundoff of one complete-batch encoder versus split encoders.
        assert got["queue"]["queue"]["last_step"] == 0
        assert [e["identity"] for e in got["queue"]["queue"]["entries"]] == [e["identity"] for e in reference["queue"]["queue"]["entries"]]


def test_physical_pair_mask_excludes_duplicate_instants_in_both_roles():
    # The same instant appears at patch 2 of a window starting at raw sample 0,
    # patch 0 of another starting at 100, and in a duplicate montage of window 0.
    # Montage deliberately is not a WindowIdentity field.
    a = WindowIdentity("emg2pose", "u1", "recording", None, 0, 4000, 2000.)
    b = WindowIdentity("emg2pose", "u1", "recording", None, 100, 4100, 2000.)
    s, p = torch.tensor([0, 1, 2, 0]), torch.tensor([2, 0, 2, 4])
    allowed, points = physical_pair_mask([a, b, a], s, p, 25., "cpu")
    assert points[0] == points[1] == points[2] and points[3] != points[0]
    assert not allowed[:3, :3].any()
    assert allowed[:3, 3].all() and allowed[3, :3].all()
    assert torch.equal(allowed, allowed.T)
    # The numerical objective never puts excluded pairs into a denominator or
    # numerator: their logits get exactly zero gradient.
    logits = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    num, chance, _ = rank_contrast_pairs(logits, torch.zeros_like(logits), allowed)
    grad, = torch.autograd.grad((num / chance.clamp(min=1)).sum(), logits)
    assert (grad[~allowed] == 0).all()
    assert torch.isfinite(grad).all()
