"""Read-only review reproducers, pinned to Claude run.py at 5f62444/366ab11.

These assert the observed BUGS; PASS confirms the review finding, not correctness.
Not part of tests/cpu CI. Run with PYTHONPATH pointing to the reviewed src and
tests/cpu directories, OMP/MKL/OPENBLAS=1 and pytest -p no:cacheprovider.
Only synthetic fixtures and temporary output directories are created.
"""
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wearusfm.training import run as R
from wearusfm.training.jepa import LossTerms
from test_pretraining_loader import _tree


def test_plain_resume_accepts_changed_seed(tmp_path):
    root, manifest = _tree(tmp_path)
    cfg = R.small_config(datasets=("ninapro_db2",), max_steps=1)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    changed = replace(cfg, seed=cfg.seed + 1, max_steps=2)
    summary = R.train(changed, manifest, [root], out, log=lambda _: None)
    assert summary["steps"] == 2


def test_fixed_panel_identity_does_not_protect_session_scales_on_resume(tmp_path):
    root, manifest = _tree(tmp_path)
    cfg = replace(R.small_config(datasets=("ninapro_db2",), max_steps=1), fixed_eval_every=1, fixed_eval_batches=1)
    index = R.load_index(manifest, [root], cfg.datasets)
    scales1 = {R._session_key(row): 1.0 for row in index.rows}
    scales2 = {key: 2.0 for key in scales1}
    a, ida = R._fixed_eval_set(index, cfg, scales1, None)
    b, idb = R._fixed_eval_set(index, cfg, scales2, None)
    assert ida == idb
    assert any(not np.array_equal(x, y) for ba, bb in zip(a, b) for x, y in zip(ba.signals, bb.signals))
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, scales=scales1, log=lambda _: None)
    saved = json.loads((out / "fixed_eval_set.json").read_text())
    summary = R.train(replace(cfg, max_steps=2), manifest, [root], out, scales=scales2, log=lambda _: None)
    assert summary["steps"] == 2
    assert json.loads((out / "fixed_eval_set.json").read_text()) == saved


def test_optional_metric_implicitly_counts_missing_batches_as_zero(monkeypatch):
    cfg = R.small_config()
    results = iter([{"total": torch.tensor(1.), "jepa_topo_grid_2d": torch.tensor(2.)}, {"total": torch.tensor(1.)}])
    monkeypatch.setattr(R.L, "to_model_inputs", lambda b: (None, None, None))
    monkeypatch.setattr(R, "_to_device", lambda *a: (None, None, None))
    monkeypatch.setattr(R, "jepa_losses", lambda *a, **k: next(results))

    class Batch:
        rvq_codes = None
        anchor_targets = None
        kind = np.empty(0)

    values = R._fixed_eval_losses(torch.nn.Module(), torch.nn.Module(), [Batch(), Batch()], cfg, "cpu", nullcontext())
    assert values["jepa_topo_grid_2d"] == 1.0  # sole observed value2.0 wrongly divided by both batches


def test_missing_forward_component_escapes_the_coordinated_guard(monkeypatch):
    student = torch.nn.Module()
    student.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    student.cfg = None
    student.spectral_keep = None
    cfg = SimpleNamespace(jepa=SimpleNamespace(keep_weight=0.0, branch_weight=0.0))
    monkeypatch.setattr(R, "loss_counts", lambda *a, **kw: {"jepa": torch.tensor(1), "rvq": torch.tensor(1)})

    def missing(*args, **kwargs):
        terms = LossTerms()
        terms.num = {"jepa": student.weight ** 2}
        terms.den = {"jepa": torch.tensor(1)}
        terms.weight = {"jepa": 1.0}
        return terms

    monkeypatch.setattr(R, "jepa_loss_terms", missing)
    micro = R._Micro(True, None, None, None, None, None)
    with pytest.raises(ValueError, match="inconsistent component keys"):
        R._accumulate_terms(None, student, None, [micro], cfg, R.DistContext(), "cpu", nullcontext())
