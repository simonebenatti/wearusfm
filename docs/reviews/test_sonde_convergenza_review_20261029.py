"""Regression expectations for Claude's 22bf2cd/b98ece0, intentionally red there.

Run from the reviewed checkout with PYTHONPATH=src and pytest on this file.
No real caches, training campaign, or remote jobs are used.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("sonde_review_pc", Path.cwd() / "scripts/probe_convergence.py")
PC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PC)


def test_parity_rejects_changed_validation_score(tmp_path):
    oof = tmp_path / "oof.npz"
    rows = np.arange(4)
    np.savez(oof, datasets=np.array(["d"]), d_y=[0, 1, 0, 1], d_pred=[0, 1, 1, 1],
             d_s=["u", "u", "v", "v"], d_row=rows)
    fold = dict(train=["t"], val=["w"], held=["u", "v"], c=1.,
                held_bacc=.75, val_bacc=.75, train_bacc=.8)
    rep = {**{k: 0 for k in PC.SC.PROBE_KEYS}, "kind": "aggregate", "phase": "A",
           "weights": {"d": 1.}, "splits_sha256": "split", "extraction_recipe": {},
           "feature_set": "pool", "checkpoint_sha256": "checkpoint", "S": .75, "S_se": .1,
           "oof": str(oof), "oof_sha256": PC.sha256(oof),
           "datasets": {"d": {"input_sha256": "input", "oof_rows_sha256": PC.DA.fingerprint(rows),
                              "subjects": ["u", "v"], "s": .75, "folds": [fold]}}}
    original, new = tmp_path / "original.json", tmp_path / "new.json"
    original.write_text(json.dumps(rep))
    changed = copy.deepcopy(rep)
    changed["datasets"]["d"]["folds"][0]["val_bacc"] = .5
    new.write_text(json.dumps(changed))
    parity = PC.parity_check(original, PC.sha256(original), PC.sha256(oof), new)
    assert parity["datasets"]["d"]["val_bacc_abs_diff"] == [.25]
    assert not parity["identical"], "A changed validation score must close the exact-parity gate"


def _stub_probe(tmp_path, monkeypatch, phase, seed=0):
    """Stub input loading only; retain run_probe's real gate and execution ordering."""
    out = tmp_path / "output"
    recipe_a = PC.solver_recipe(3000, seed=0)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"protocol": PC.PROTOCOL, "phase": "A", "passed": True,
                               "solver_recipe_sha256": PC.DA.fingerprint(recipe_a)}))
    pin = {"code_sha": "cachecode", "checkpoint_sha256": "checkpoint", "input_sha256": "input",
           "splits_sha256": PC.CP.P2_SPLITS_SHA256, "sha256": "cache"}
    monkeypatch.setattr(PC, "load_pins", lambda p: {})
    monkeypatch.setattr(PC, "verify_cache_inputs", lambda *a: {"epn612": pin})
    monkeypatch.setattr(PC.SC, "load_cache", lambda *a: {**pin, "x": np.zeros((4, 2), np.float32),
                        "y": np.array([0, 1, 0, 1]), "s": np.array(["u", "u", "v", "v"]),
                        "subjects": ["t", "w", "u", "v"]})
    monkeypatch.setattr(PC.CP, "fold_roles", lambda *a: [dict(train=["t"], val=["w"], held=["u", "v"])])
    return SimpleNamespace(phase=phase, allow_unfrozen=True, out=out, gate=gate, datasets=["epn612"],
                           task="p2", encoder="B", folds=[0], pins=tmp_path / "pins.json", seed=seed,
                           features=tmp_path / "features", splits=None)


def test_fit_exception_preserves_partial_audit_on_disk(tmp_path, monkeypatch):
    args = _stub_probe(tmp_path, monkeypatch, "A")

    def failed_fit(*a, audit, **kw):
        audit.append(dict(dataset="epn612", fold=0, phase="candidate", C=.001,
                          exception="RuntimeError: injected fit failure"))
        raise RuntimeError("injected fit failure")

    monkeypatch.setattr(PC.SC, "oof_predictions", failed_fit)
    with pytest.raises(RuntimeError, match="injected fit failure"):
        PC.run_probe(args)
    assert not (args.out / "shard.json").exists(), "Failure must never publish a complete shard"
    audit_files = list(args.out.glob("*audit*.jsonl"))
    assert audit_files, "The recorded failure must survive the process, outside the success report"
    assert "injected fit failure" in audit_files[0].read_text()


def test_phase_b_rejects_recipe_change_before_entering_fits(tmp_path, monkeypatch):
    # A was seed=0; B is seed=1. The normalized solver recipe can be checked
    # against the gate's stored SHA before reading any reports or doing any fit.
    args = _stub_probe(tmp_path, monkeypatch, "B", seed=1)
    reached = []

    def fit_spy(*a, **kw):
        reached.append(True)
        raise RuntimeError("FIT_REACHED_WITH_DIFFERENT_RECIPE")

    monkeypatch.setattr(PC.SC, "oof_predictions", fit_spy)
    try:
        PC.run_probe(args)
    except (PC.Problem, RuntimeError):
        pass
    assert not reached, "The B job must reject a different recipe before spending its fit budget"
