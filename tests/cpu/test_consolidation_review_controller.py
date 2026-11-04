"""Persisted candidate outcomes and the prespecified timing/reading policies."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_finetune as C
from wearusfm.harness import finetuning as F


@pytest.mark.parametrize("two,one,expected", [(14100,8000,"joint"), (14101,14100,"split"), (29000,14101,"blocked")])
def test_packaging_never_changes_dose_or_exceeds_single_job_gate(two, one, expected):
    assert C.fit_packaging(two, one) == expected


@pytest.mark.parametrize("two,one", [(float("nan"),2), (1,2), (0,0)])
def test_invalid_costs_cannot_pass_gate(two, one):
    with pytest.raises(ValueError):
        C.fit_packaging(two, one)


@pytest.mark.parametrize("two,one,expected", [
    (24900, 15000, "joint"), (24901, 24900, "split"), (50000, 24901, "blocked"),
])
def test_seven_hour_amendment_has_its_own_fixed_gate(two, one, expected):
    assert C.fit_packaging(two, one, 24900) == expected
    # These costs must still fail the unamended four-hour allocation.
    assert C.fit_packaging(two, one) == "blocked"


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_job_limit_never_authorizes_a_fit(limit):
    with pytest.raises(ValueError, match="time limit"):
        C.fit_packaging(2, 1, limit)


def walltime_plan():
    plan = json.loads((ROOT / "results/infra/consolidation_20261104/finetune_plan.json").read_text())
    plan.pop("fit_allocation_each_fold_regime", None)
    plan["fit_allocation_by_regime"] = copy.deepcopy(C.FIT_ALLOCATION_BY_REGIME)
    plan["fit_packaging_policy"] = copy.deepcopy(C.FIT_PACKAGING_POLICY)
    return plan


def test_resource_amendment_is_explicit_and_does_not_change_scientific_recipe():
    original = json.loads((ROOT / "results/infra/consolidation_20261104/finetune_plan.json").read_text())
    assert C.fit_resource_policy(original) == {"frozen": 14100, "finetune": 24900, "scratch": 24900}
    plan = walltime_plan()
    assert C.recipe_identity(plan) == C.recipe_identity(original) == plan["recipe_sha256"]
    assert C.fit_resource_policy(plan) == {"frozen": 14100, "finetune": 24900, "scratch": 24900}
    plan["fit_allocation_by_regime"]["finetune"]["seconds"] = 28800
    with pytest.raises(ValueError, match="allocation policy"):
        C.fit_resource_policy(plan)
    plan = walltime_plan()
    plan["fit_packaging_policy"]["joint_if_two_candidates_seconds_lte"]["frozen"] = 24900
    with pytest.raises(ValueError, match="packaging policy"):
        C.fit_resource_policy(plan)
    plan = walltime_plan()
    plan["fit_allocation_each_fold_regime"] = {"seconds": 14400}
    with pytest.raises(ValueError, match="legacy"):
        C.fit_resource_policy(plan)


def test_benchmark_gate_rejects_old_source_and_wrong_operational_limits():
    plan = walltime_plan()
    gate = {
        "complete": True, "protocol": C.PROTOCOL, "mode": "benchmark",
        "source_sha256": "current-controller", "recipe_sha256": plan["recipe_sha256"],
        "fit_allocation_by_regime": copy.deepcopy(plan["fit_allocation_by_regime"]),
        "fit_packaging_policy": copy.deepcopy(plan["fit_packaging_policy"]),
        "fit_limit_seconds_by_regime": C.fit_resource_policy(plan),
        "projected_two_candidates_seconds": {"frozen": 20000, "finetune": 40000, "scratch": 45000},
        "projected_one_candidate_seconds": {"frozen": 10000, "finetune": 21000, "scratch": 23000},
        "fit_packaging": {regime: "split" for regime in F.REGIMES},
        "resource_gate_passed": True,
    }
    assert C.benchmark_packaging(gate, plan, "current-controller") == gate["fit_packaging"]
    with pytest.raises(ValueError, match="provenance"):
        C.benchmark_packaging(gate, plan, "new-controller")
    wrong_limits = copy.deepcopy(gate)
    wrong_limits["fit_limit_seconds_by_regime"]["finetune"] = 14100
    with pytest.raises(ValueError, match="resource policy"):
        C.benchmark_packaging(wrong_limits, plan, "current-controller")
    wrong_packaging = copy.deepcopy(gate)
    wrong_packaging["fit_packaging"]["finetune"] = "joint"
    with pytest.raises(ValueError, match="measured projections"):
        C.benchmark_packaging(wrong_packaging, plan, "current-controller")
    wrong_memory = copy.deepcopy(gate)
    wrong_memory["resource_gate_passed"] = False
    with pytest.raises(ValueError, match="memory"):
        C.benchmark_packaging(wrong_memory, plan, "current-controller")


def test_launcher_rejects_unknown_regime_before_python_or_cluster_setup(tmp_path):
    launcher = ROOT / "scripts/slurm/consolidation_finetune.sbatch"
    result = subprocess.run(["bash", str(launcher), "fit", str(tmp_path / "out"), "0", "unknown"],
                            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 2
    assert "unknown fit regime" in result.stderr
    assert "WORK" not in result.stderr


def candidate(number, score=None, best=3, delta=.0):
    return {"candidate": number, "val_bacc": score, "best_epoch": best,
            "stopped": "fixed_budget" if score is not None else "nonfinite",
            "history": [{"epoch": i, "val_bacc": .8 + (delta if i == 5 else 0)} for i in range(1, 6)]}


def test_selection_and_scratch_rule_use_validation_only():
    assert C.select_candidate([candidate(0), candidate(1,.8)])["candidate"] == 1
    assert C.select_candidate([candidate(1,.8), candidate(0,.8)])["candidate"] == 0
    assert C.select_candidate([candidate(0), candidate(1)]) is None
    reports = [candidate(0,.8,best=5) for _ in range(3)] + [candidate(0,.8)]
    assert C.scratch_still_improving(reports)["flag"]
    assert C.scratch_still_improving([candidate(0,.8,delta=.006) for _ in range(4)])["flag"]
    assert not C.scratch_still_improving([candidate(0,.8,delta=.001) for _ in range(4)])["flag"]
    with pytest.raises(ValueError, match="incomplete"):
        C.scratch_still_improving([candidate(0) for _ in range(4)])


def test_head_multipliers_are_part_of_recipe_hash():
    plan = json.loads((ROOT / "results/infra/consolidation_20261104/finetune_plan.json").read_text())
    assert plan["head_lr_multiplier"] == F.HEAD_LR_MULTIPLIER
    assert plan["scratch_learning_rule"] == C.SCRATCH_RULE
    assert C.recipe_identity(plan) == plan["recipe_sha256"]
    plan["head_lr_multiplier"]["finetune"] = 10
    assert C.recipe_identity(plan) != plan["recipe_sha256"]


def test_divergence_persists_without_losing_first_candidate_or_masking_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "build_classifier", lambda *a, **k: torch.nn.Linear(2,2))
    calls = []
    def fit(*a, **k):
        number = len(calls)
        calls.append(number)
        if number == 0:
            return {"val_bacc": .8, "best_epoch": 1, "stopped": "fixed_budget", "history": []}
        raise F.NumericalDivergence("loss", step=2, epoch=1)
    monkeypatch.setattr(F, "fit_candidate", fit)
    args = ({}, None, None, {}, None, np.array(["move", "rest"]), "finetune")
    plan = {"seed":0, "batch":8, "epochs":5}
    first = C.run_candidate(*args, 0, plan, "cpu", tmp_path, {"test": True})
    first_bytes = (tmp_path / "candidate_0.json").read_bytes()
    second = C.run_candidate(*args, 1, plan, "cpu", tmp_path, {"test": True})
    assert (tmp_path / "candidate_0.json").read_bytes() == first_bytes
    assert second["val_bacc"] is None and second["stopped"] == "nonfinite"
    json.dumps(second, allow_nan=False)
    assert C.select_candidate([first, second])["candidate"] == 0
    def oom(*a, **k):
        raise torch.OutOfMemoryError("not a numerical candidate result")
    monkeypatch.setattr(F, "fit_candidate", oom)
    with pytest.raises(torch.OutOfMemoryError):
        C.run_candidate(*args, 1, plan, "cpu", tmp_path, {})


def test_split_second_reads_own_first_receipt_and_rejects_changed_identity(tmp_path):
    first = tmp_path / "candidate0"
    second = tmp_path / "candidate1"
    first.mkdir()
    provenance = {"fold":0,"regime":"scratch","recipe_sha256":"fixed"}
    item = {**candidate(0), "complete":True,"provenance":provenance}
    path = first / "candidate_0.json"
    path.write_text(json.dumps(item))
    receipt = {"complete":True,"candidate_only":0,"candidate_provenance":provenance,
               "candidate_file_sha256":{"0":C.T.R.sha256_file(path)}}
    (first / "report.json").write_text(json.dumps(receipt))
    assert C.prior_candidate(second, provenance)["stopped"] == "nonfinite"
    with pytest.raises(ValueError, match="wrong first"):
        C.prior_candidate(second, {**provenance,"fold":1})
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="report changed"):
        C.prior_candidate(second, provenance)
