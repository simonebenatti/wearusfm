"""Adversarial release and completed-predecessor guards; no submit or cluster access."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import execution as E, provenance as P
from test_pose_temponet_provenance import prepared, release, numeric_fixture


def test_release_booleans_cannot_replace_actual_operational_review(prepared):
    repo, plan = prepared
    forged = copy.deepcopy(plan)
    forged["submission_released"] = True
    forged["status"] = "claude_review_converged"
    forged["reader"]["code_sha"] = "a" * 40
    with pytest.raises(ValueError, match="actual Claude operational review required"):
        P.validate_plan(forged, source_root=repo, require_review=True)
    forged["review"]["operational"] = {
        "status": "claude_review_converged", "commit": "f" * 40,
        "proposal_sha256": forged["proposal"]["sha256"],
        "source_sha256": forged["reader"]["source_sha256"],
        "artifact": {"path": "docs/reviews/nonexistent.md", "sha256": "f" * 64,
                     "local_copy": "docs/reviews/nonexistent.md"},
    }
    with pytest.raises((ValueError, FileNotFoundError)):
        P.validate_plan(forged, source_root=repo, require_review=True)


def completed_receipt(repo, plan, monkeypatch):
    plan = release(repo, plan, monkeypatch)
    receipt = P.make_receipt(plan, "preflight", numeric=numeric_fixture(),
                              gates={"integrity": True, "historical_R1": True},
                              details={"allocation": {"slurm_job_id": "999123", "cpus": 2, "gpus": 0, "nodes": 1}},
                              source_root=repo)
    pin = P.write_receipt(plan, receipt)
    return plan, pin


@pytest.mark.parametrize("accounting", [
    "999123|RUNNING|0:0\n", "999123|FAILED|1:0\n", "999123|COMPLETED|0:9\n",
    "999123|COMPLETED|1:0\n", "999124|COMPLETED|0:0\n", "",
    "999123.batch|COMPLETED|0:0\n",
    "999123|COMPLETED|0:0\n999123|COMPLETED|0:0\n",
])
def test_only_unambiguous_exact_allocation_completed_zero_zero_removes_dependency(prepared, monkeypatch, tmp_path, accounting):
    repo, plan = prepared
    plan, receipt = completed_receipt(repo, plan, monkeypatch)
    sacct = tmp_path / "accounting.psv"
    sacct.write_text(accounting)
    proof = {"preflight": {"receipt": receipt, "sacct": P.file_pin(sacct)}}
    with pytest.raises(ValueError, match="unambiguous sacct COMPLETED 0:0"):
        E.verify_completed_predecessors(plan, proof)


def test_receipt_plus_exact_sacct_required_and_read_change_detected(prepared, monkeypatch, tmp_path):
    repo, plan = prepared
    plan, receipt = completed_receipt(repo, plan, monkeypatch)
    sacct = tmp_path / "accounting.psv"
    sacct.write_text("999123|COMPLETED|0:0\n999123.batch|COMPLETED|0:0\n")
    proof = {"preflight": {"receipt": receipt, "sacct": P.file_pin(sacct)}}
    assert E.verify_completed_predecessors(plan, proof) == {"preflight"}
    wrong_id = {"bench": proof["preflight"]}
    with pytest.raises(ValueError, match="wrong predecessor stage"):
        E.verify_completed_predecessors(plan, wrong_id)
    sacct.write_text("999123|FAILED|1:0\n")
    with pytest.raises(ValueError, match="SHA256 mismatch|byte count"):
        E.verify_completed_predecessors(plan, proof)
