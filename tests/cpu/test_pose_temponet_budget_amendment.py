"""Real pinned receipts: budget-only bridge, no CUDA, files or jobs on cluster."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import budget_amendment as A1, execution as E, provenance as P, runner as R

ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT / "results/infra/pose_temponet_execution_20261120/launcher"


@pytest.fixture
def plan():
    return A1.prepare(A1.base_plan(ROOT), source_root=ROOT)


def receipts(plan):
    return {identifier: P.verified_receipt(
        P.file_pin(MIRROR / (identifier + ".json")), plan, verify_artifacts=False,
        receipt_mirror_dir=MIRROR) for identifier in A1.LEGACY}


def test_authorized_caps_are_exactly_150_percent_of_frozen_measured_need():
    from decimal import Decimal
    for key, need in zip(A1.CAPS, ("35.9", "37.4", "77", "376.2")):
        assert Decimal(str(A1.CAPS[key])) == Decimal(need) * Decimal("1.5")


def test_preparation_preserves_science_and_never_claims_review_or_release(plan):
    base = A1.base_plan(ROOT)
    assert plan["status"] == "REVIEW_REQUIRED"
    assert plan["submission_released"] is False
    assert plan["reader"]["code_sha"] is None
    assert plan["execution_sha256"] != base["execution_sha256"]
    for key in ("inputs", "proposal", "recipe_sha256", "producers", "jobs", "resource_policy"):
        assert plan[key] == base[key]
    with pytest.raises(ValueError):
        P.validate_plan(plan, require_review=True)


def test_exact_three_real_receipts_survive_source_identity_change(plan):
    rows = receipts(plan)
    assert set(rows) == set(A1.LEGACY)
    assert all(r["source_sha256"] == A1.OLD_SOURCE for r in rows.values())
    assert plan["reader"]["source_sha256"] != A1.OLD_SOURCE
    assert rows["TCNH-preflight-20261120"]["gates"]["historical_R1"] is True


def test_old_operational_review_cannot_release_the_new_sources(plan):
    plan["submission_released"] = True
    plan["status"] = "claude_review_converged"
    plan["reader"]["code_sha"] = A1.base_plan(ROOT)["reader"]["code_sha"]
    plan["review"]["operational"] = A1.base_plan(ROOT)["review"]["operational"]
    with pytest.raises(ValueError, match="operational review targets other sources"):
        P.validate_plan(plan, require_review=True)


def test_real_resource_gate_passes_only_with_explicit_A1_and_same_walltimes(plan):
    rows = receipts(plan)
    amended = R._resource_projection(plan, rows)
    old = R._resource_projection(A1.base_plan(ROOT), rows)
    assert amended["all_passed"] is True
    assert not old["all_passed"]
    assert {k for k, v in old["gates"].items() if not v} == {
        "neural_gpu_hours", "gpu_total", "core_total"}
    for key in ("jobs", "matrix", "projection", "policy"):
        assert amended[key] == old[key]
    packed = E.apply_training_pack(E.job_graph(), R._training_pack(amended), budget_caps=A1.CAPS)
    assert len(packed) == 24
    preview = copy.deepcopy(plan)
    preview["jobs"] = packed
    P.validate_plan(preview)
    with pytest.raises(ValueError, match="campaign caps"):
        E.apply_training_pack(E.job_graph(), R._training_pack(amended))


@pytest.mark.parametrize("field", ["neural_gpu_hours", "gpu_total_hours", "cpu_only_core_hours", "total_core_hours"])
def test_budget_cannot_be_changed_by_editing_plan(plan, field):
    plan["budget_amendment"]["authorized_caps"][field] += 1
    plan["execution_sha256"] = P.execution_identity(plan)
    with pytest.raises(ValueError, match="unauthorized A1"):
        P.validate_plan(plan)


@pytest.mark.parametrize("identifier", list(A1.LEGACY))
def test_rehashed_modified_legacy_receipt_is_rejected(plan, tmp_path, identifier):
    receipt = json.loads((MIRROR / (identifier + ".json")).read_text())
    receipt["details"]["allocation"]["slurm_job_id"] = "99999999"
    receipt["receipt_sha256"] = P.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    path = tmp_path / (identifier + ".json")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact pinned completed receipt"):
        P.verified_receipt(P.file_pin(path), plan, verify_artifacts=False)


def test_unknown_old_source_receipt_cannot_use_legacy_bridge(plan, tmp_path):
    receipt = json.loads((MIRROR / "TCNH-preflight-20261120.json").read_text())
    job = next(j for j in plan["jobs"] if j["stage"] == "wls_calibration")
    receipt.update(job_id=job["id"], stage=job["stage"])
    receipt["receipt_sha256"] = P.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    path = tmp_path / "untrusted.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identity differs"):
        P.verified_receipt(P.file_pin(path), plan, verify_artifacts=False)


def test_old_receipts_rejected_without_A1_even_if_current_sources_match(plan):
    del plan["budget_amendment"]
    plan["execution_sha256"] = P.execution_identity(plan)
    with pytest.raises(ValueError, match="identity differs"):
        receipts(plan)


def test_scientific_fields_and_unrelated_source_diffs_cannot_enter_amendment(plan):
    changed = copy.deepcopy(plan)
    changed["jobs"][5]["head"] = "L13"
    with pytest.raises(ValueError, match="job graph"):
        A1.validate(changed, source_root=ROOT)
    changed = copy.deepcopy(plan)
    changed["reader"]["source_manifest"]["scripts/pose_temponet/train.py"] = "0" * 64
    with pytest.raises(ValueError, match="outside resource/provenance"):
        A1.validate(changed, source_root=ROOT)
    changed = copy.deepcopy(plan)
    changed["inputs"]["cache_train"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scientific/base"):
        A1.validate(changed, source_root=ROOT)


def test_benchmark_input_identity_is_checked_against_immutable_base(plan):
    rows = receipts(plan)
    rows["TCNH-gpu-benchmark-20261120"]["details"]["benchmark"]["input_identity"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="benchmark input identity"):
        R._resource_projection(plan, rows)


def test_original_budget_proposal_and_base_plan_cannot_be_overwritten(plan, tmp_path):
    with pytest.raises(ValueError, match="exact pinned released base"):
        A1.prepare(plan, source_root=ROOT)
    path = tmp_path / A1.BASE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        A1.base_plan(tmp_path)


def test_completed_predecessors_never_emit_new_submission_commands(plan):
    # Command admission below is a unit simulation, not an operational review.
    plan["submission_released"] = True
    for job in plan["jobs"][:3]:
        with pytest.raises(ValueError, match="must never be resubmitted"):
            E.submit_command(plan, job, checkout_sha="a" * 40,
                             plan_path="/tmp/plan.json", plan_sha256="b" * 64)


def test_remaining_commands_use_new_checkout_and_retain_completed_dependencies(plan):
    rows = receipts(plan)
    projection = R._resource_projection(plan, rows)
    plan["jobs"] = E.apply_training_pack(plan["jobs"], R._training_pack(projection), budget_caps=A1.CAPS)
    plan["submission_released"] = True  # isolated command unit test, never a release
    job = next(j for j in plan["jobs"] if j["stage"] == "wls_calibration")
    command = E.submit_command(plan, job, checkout_sha="a" * 40,
                               plan_path="/tmp/plan.json", plan_sha256="b" * 64)
    assert A1.CAMPAIGN in command
    assert any("CODE_DIR=" + A1.CODE_DIR in arg for arg in command)
    assert job["after"] == list(A1.LEGACY)[1:]
