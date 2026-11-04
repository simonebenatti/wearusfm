"""Persisted candidate outcomes and the prespecified timing/reading policies."""
import json
from pathlib import Path
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
