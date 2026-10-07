"""Fault injection nel trainer reale: loss finita, gradienti NaN/Inf, nessun update o checkpoint corrotto."""

import copy
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from wearusfm.training import run as R  # noqa: E402
from test_pretraining_loader import _tree  # noqa: E402


def _assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_same(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("bad_grad", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative_inf"])
@pytest.mark.parametrize("scenario", ["first_step", "resume", "resume_unsaved_step"])
def test_nonfinite_gradient_stops_before_optimizer_ema_and_save(tmp_path, monkeypatch, bad_grad, scenario):
    root, manifest = _tree(tmp_path)
    out = tmp_path / "run"
    cfg = replace(R.small_config(datasets=None, max_steps=4), eval_every=100, ckpt_every=100)
    checkpoint = out / "checkpoint.pt"
    if scenario != "first_step":
        R.train(replace(cfg, max_steps=1), manifest, [root], out, log=lambda _: None)
        original_checkpoint = checkpoint.read_bytes()
    else:
        original_checkpoint = None

    calls = {"loss": 0, "optimizer": 0, "ema": 0}
    observed = {}
    real_losses, real_step, real_ema = R.jepa_losses, torch.optim.AdamW.step, R.ema_update
    fail_on = 2 if scenario == "resume_unsaved_step" else 1

    # L'optimizer si costruisce normalmente; la cattura serve a verificare anche i suoi momenti e gruppi.
    real_init = torch.optim.AdamW.__init__

    def capture_optimizer(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        observed["optimizer"] = self

    def count_step(self, *args, **kwargs):
        calls["optimizer"] += 1
        return real_step(self, *args, **kwargs)

    def count_ema(*args, **kwargs):
        calls["ema"] += 1
        return real_ema(*args, **kwargs)

    def inject_gradient(student, teacher, *args, **kwargs):
        losses = real_losses(student, teacher, *args, **kwargs)
        calls["loss"] += 1
        if calls["loss"] == fail_on:
            assert torch.isfinite(losses["total"]), "il guasto deve essere solo nel backward"
            next(p for p in student.parameters() if p.requires_grad).register_hook(lambda grad: torch.full_like(grad, bad_grad))
            observed["student"], observed["teacher"] = student, teacher
            observed["student_before"] = copy.deepcopy(student.state_dict())
            observed["teacher_before"] = copy.deepcopy(teacher.state_dict())
            observed["optimizer_before"] = copy.deepcopy(observed["optimizer"].state_dict())
        return losses

    monkeypatch.setattr(torch.optim.AdamW, "__init__", capture_optimizer)
    monkeypatch.setattr(torch.optim.AdamW, "step", count_step)
    monkeypatch.setattr(R, "ema_update", count_ema)
    monkeypatch.setattr(R, "jepa_losses", inject_gradient)
    logs = []
    summary = R.train(cfg, manifest, [root], out, log=logs.append)
    completed = 0 if scenario == "first_step" else (2 if scenario == "resume_unsaved_step" else 1)
    assert summary["stopped"].startswith(f"gradienti non finiti al passo {completed}")
    assert summary["steps"] == completed
    assert calls == {"loss": fail_on, "optimizer": fail_on - 1, "ema": fail_on - 1}
    _assert_same(observed["student"].state_dict(), observed["student_before"])
    _assert_same(observed["teacher"].state_dict(), observed["teacher_before"])
    _assert_same(observed["optimizer"].state_dict(), observed["optimizer_before"])
    assert all(p.grad is None for p in observed["student"].parameters())
    assert all(torch.isfinite(p).all() for model in (observed["student"], observed["teacher"]) for p in model.parameters())
    if original_checkpoint is None:
        assert not checkpoint.exists()  # nessun checkpoint se il primissimo backward fallisce
    else:
        assert checkpoint.read_bytes() == original_checkpoint
        assert torch.load(checkpoint, weights_only=False)["step"] == 1
    assert not checkpoint.with_suffix(".tmp").exists()
    assert (out / "STOP").read_text().strip() == summary["stopped"]
    assert json.loads((out / "summary.json").read_text()) == summary
    assert any(summary["stopped"] in entry for entry in logs)
    metrics = (out / "metrics.jsonl").read_bytes()
    records = [json.loads(line) for line in metrics.splitlines()]
    assert [rec["step"] for rec in records] == list(range(1, completed + 1))
    assert all(torch.isfinite(torch.tensor(rec["grad_norm"])) for rec in records)

    # La catena non riparte dal checkpoint precedente dopo uno stop numerico.
    before_calls = dict(calls)
    blocked = R.train(cfg, manifest, [root], out, log=lambda _: None)
    assert blocked["stopped"].startswith("fermato in precedenza: gradienti non finiti")
    assert blocked["steps"] is None and calls == before_calls
    assert (out / "metrics.jsonl").read_bytes() == metrics
    if original_checkpoint is not None:
        assert checkpoint.read_bytes() == original_checkpoint


def test_nonfinite_loss_also_preserves_last_checkpoint(tmp_path, monkeypatch):
    root, manifest = _tree(tmp_path)
    out = tmp_path / "run"
    cfg = R.small_config(datasets=None, max_steps=1)
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    checkpoint = out / "checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    original_metrics = (out / "metrics.jsonl").read_bytes()

    def bad_loss(*args, **kwargs):
        return {"total": torch.tensor(float("nan"), requires_grad=True)}

    def forbidden_update(*args, **kwargs):
        pytest.fail("loss non finita: optimizer ed EMA non devono essere chiamati")

    monkeypatch.setattr(R, "jepa_losses", bad_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    monkeypatch.setattr(R, "ema_update", forbidden_update)
    summary = R.train(replace(cfg, max_steps=3), manifest, [root], out, log=lambda _: None)
    assert summary["stopped"] == "perdita non finita al passo 1" and summary["steps"] == 1
    assert checkpoint.read_bytes() == original_checkpoint
    assert not checkpoint.with_suffix(".tmp").exists()
    assert (out / "metrics.jsonl").read_bytes() == original_metrics
    assert (out / "STOP").read_text().strip() == summary["stopped"]
