"""Scettico 1, R2-N1: passed per coppia contro errori globali di confine/supporto.

Riproduzione indipendente con il trainer CPU reale (1 rank, fork 2, fine 4):
- reload_b reale con exp_avg di Adam azzerato subito dopo opt.load_state_dict;
- copie dei percorsi genuini con un solo campo di confine alterato (reload_b, poi continuous);
- supporto mancante in una sola fase.
Ogni esito viene scritto in JSON accanto a questo file.
"""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
import torch

import contrastive_target_resume as C
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from test_contrastive_target_resume import _live_train, _fresh_reload
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

HERE = Path(__file__).resolve().parent


def _fresh_reload_adam_zero(rank, folder, cfg, manifest, data, phase, initial_sha, provenance):
    torch.set_num_threads(1)
    original = torch.optim.Optimizer.load_state_dict

    def zeroing(self, state_dict):
        out = original(self, state_dict)
        for values in self.state.values():
            if "exp_avg" in values:
                values["exp_avg"].zero_()
        return out

    with patch.object(torch.optim.AdamW, "load_state_dict", zeroing):
        _live_train(folder, cfg, manifest, data, phase, initial_sha, provenance)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sk1_r2n1")
    data, manifest, pose = _pose_setup(tmp)
    base = _config(pose, "CR")
    cfg = replace(base, max_steps=4, model=replace(base.model, muscle_dropout=.4), log_batch_origin=True,
                  schedule=replace(base.schedule, total_steps=20, keep_checkpoints=(2,)))
    calibration = replace(cfg, contrastive={**cfg.contrastive, "calibration_only": True})
    provenance = {"source_sha256": "test", "input_sha256": {}}
    R.train(calibration, manifest, [data], tmp / "calibration", log=lambda _: None,
            contrastive_provenance=provenance)
    initial = C.checkpoint(tmp / "calibration/checkpoint.pt")["student"]
    initial_sha = state_sha256(initial)
    paths = {phase: tmp / phase for phase in C.PHASES}
    _live_train(paths["continuous"], cfg, manifest, data, "continuous", initial_sha, provenance)
    fork = paths["continuous"] / "checkpoint_step0000002.pt"
    fork_sha = C.file_sha(fork)
    for phase in C.PHASES[1:]:
        C.prepare_reload(fork, fork_sha, paths[phase], cfg, provenance, initial_sha,
                         phase=phase, fork_step=2, end_step=4, ranks=1)
        torch.multiprocessing.spawn(_fresh_reload,
            args=(paths[phase], cfg, manifest, data, phase, initial_sha, provenance), nprocs=1)
    mutated = tmp / "reload_b_adam_zero"
    C.prepare_reload(fork, fork_sha, mutated, cfg, provenance, initial_sha,
                     phase="reload_b", fork_step=2, end_step=4, ranks=1)
    torch.multiprocessing.spawn(_fresh_reload_adam_zero,
        args=(mutated, cfg, manifest, data, "reload_b", initial_sha, provenance), nprocs=1)
    return tmp, paths, mutated, cfg, initial_sha, initial, fork_sha


def _summary(result):
    return {
        "status": result["status"], "passed": result["passed"],
        "boundary_mismatches": result["boundary_mismatches"],
        "missing_active_support": result["missing_active_support"],
        "pairs": {name: {"passed": p["passed"], "failure_reasons_count": len(p.get("failure_reasons", [])),
                         "failure_reasons_head": p.get("failure_reasons", [])[:3],
                         "differences_count": len(p["input_rng_counter_order_differences"]),
                         "relative": p.get("relative_to_left_displacement_from_initial")}
                  for name, p in result["pairs"].items()},
    }


def _per_pair_view(result):
    """Verdetto che userebbe solo le voci di confine/supporto delle due fasi della coppia."""
    view = {}
    for name, p in result["pairs"].items():
        left, right = name.split("__")
        own_boundary = [e for e in result["boundary_mismatches"] if e["phase"] in (left, right)]
        own_support = [e for e in result["missing_active_support"] if e["phase"] in (left, right)]
        numeric_ok = not p.get("failure_reasons")
        view[name] = numeric_ok and not p["input_rng_counter_order_differences"] and not own_boundary and not own_support
    return view


def _dump(name, result):
    payload = {"summary": _summary(result), "per_pair_view": _per_pair_view(result)}
    (HERE / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def _edit_first_row(folder, mutate):
    path = folder / "trace-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_0_baseline(runs):
    tmp, paths, mutated, cfg, initial_sha, initial, fork_sha = runs
    result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    out = _dump("case0_baseline", result)
    assert result["status"] == "PASSED" and all(p["passed"] for p in result["pairs"].values())
    assert all(out["per_pair_view"].values())


def test_1_real_adam_zero_in_reload_b(runs):
    tmp, paths, mutated, cfg, initial_sha, initial, fork_sha = runs
    result = C.compare_three_paths({**paths, "reload_b": mutated}, cfg, initial_sha, initial, fork_sha,
                                   ranks=1, fork_step=2, end_step=4)
    out = _dump("case1_real_adam_zero_reload_b", result)
    assert result["status"] == "FAILED_REQUIRES_REVIEW" and not result["passed"]
    assert {e["phase"] for e in result["boundary_mismatches"]} == {"reload_b"}
    ca = result["pairs"]["continuous__reload_a"]
    assert ca["passed"] is False and ca["failure_reasons"] == [] and ca["input_rng_counter_order_differences"] == []
    assert out["per_pair_view"]["continuous__reload_a"] is True


def test_2_synthetic_boundary_only_reload_b(runs):
    tmp, paths, mutated, cfg, initial_sha, initial, fork_sha = runs
    copy = tmp / "reload_b_boundary_teacher"
    shutil.copytree(paths["reload_b"], copy)

    def mutate(rows):
        rows[0]["boundary"]["teacher"] = "changed"
    _edit_first_row(copy, mutate)
    result = C.compare_three_paths({**paths, "reload_b": copy}, cfg, initial_sha, initial, fork_sha,
                                   ranks=1, fork_step=2, end_step=4)
    out = _dump("case2_synthetic_boundary_reload_b", result)
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert result["boundary_mismatches"] == [{"phase": "reload_b", "rank": 0, "field": "teacher"}]
    for name, p in result["pairs"].items():
        assert p["passed"] is False and p["failure_reasons"] == [] and p["input_rng_counter_order_differences"] == []
    assert out["per_pair_view"] == {"continuous__reload_a": True, "continuous__reload_b": False,
                                    "reload_a__reload_b": False}


def test_3_synthetic_boundary_only_continuous(runs):
    tmp, paths, mutated, cfg, initial_sha, initial, fork_sha = runs
    copy = tmp / "continuous_boundary_moments"
    shutil.copytree(paths["continuous"], copy)

    def mutate(rows):
        rows[0]["boundary"]["optimizer_moments"] = "changed"
    _edit_first_row(copy, mutate)
    result = C.compare_three_paths({**paths, "continuous": copy}, cfg, initial_sha, initial, fork_sha,
                                   ranks=1, fork_step=2, end_step=4)
    out = _dump("case3_synthetic_boundary_continuous", result)
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert result["boundary_mismatches"] == [{"phase": "continuous", "rank": 0, "field": "optimizer_moments"}]
    ab = result["pairs"]["reload_a__reload_b"]
    assert ab["passed"] is False and ab["failure_reasons"] == [] and ab["input_rng_counter_order_differences"] == []
    assert out["per_pair_view"]["reload_a__reload_b"] is True


def test_4_support_missing_only_reload_b(runs):
    tmp, paths, mutated, cfg, initial_sha, initial, fork_sha = runs
    copy = tmp / "reload_b_support_zero"
    shutil.copytree(paths["reload_b"], copy)
    for name in ("trace-0.jsonl", "metrics.jsonl"):
        path = copy / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row.get("update", row.get("step")) == 4:
                row["contrast_rnc_pairs"] = 0.
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = C.compare_three_paths({**paths, "reload_b": copy}, cfg, initial_sha, initial, fork_sha,
                                   ranks=1, fork_step=2, end_step=4)
    out = _dump("case4_support_missing_reload_b", result)
    ca = result["pairs"]["continuous__reload_a"]
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert ca["passed"] is False and ca["failure_reasons"] == [] and ca["input_rng_counter_order_differences"] == []
    assert out["per_pair_view"]["continuous__reload_a"] is True
