"""Read the real readiness gate with temporary evidence; never grant runtime readiness."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scaling_readiness_test", ROOT / "scripts/scaling_train.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
CONTRACT_SHA = "c" * 64
SOURCES = {"scripts/scaling_evaluation.py", "scripts/scaling_projection.py", "scripts/scaling_pose_readout.py",
           "scripts/scaling_probe.py", "scripts/scaling_compare.py", "scripts/scaling_reading.py"}


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": M.R.sha256_file(path)}


@pytest.fixture
def readiness(tmp_path):
    """All certificates here are explicit synthetic unit-test fixtures, not run artifacts."""
    sources = {path: M.R.sha256_file(ROOT / path) for path in sorted(SOURCES)}
    verification = {"fixture": True, "complete": True, "passed": 1, "failed": 0, "source_files": sources}
    resources = {"fixture": True, "complete": True, "checkpoint_step": 5000,
                 "includes_30m_classification_extraction": True, "includes_dimension_matched_secondary": True,
                 "gpu_hours_cap": 1., "cpu_core_hours_cap": 1.}
    review = tmp_path / "review.txt"
    review.write_text("Synthetic unit-test APPROVATO fixture; not an operational review or launch authorization.\n")
    return {"protocol": "scaling-evaluation-readiness-20261104-v1", "complete": True,
            "checkpoint_step": 5000, "training_source_contract_sha256": CONTRACT_SHA,
            "dimensions": {"T100-G-30M": 384, M.ARM: 896},
            "pool_widths": {"T100-G-30M": 768, M.ARM: 1792}, "source_files": sources,
            "capabilities": {"dimension_aware_extraction": True, "pose_ridge": True,
                             "classification_probe": True, "dimension_matched_secondary": True,
                             "convergence_audit": True, "paired_comparison": True},
            "verification_report": write(tmp_path / "tests.json", verification),
            "resource_plan": write(tmp_path / "resources.json", resources),
            "review": {"path": str(review), "sha256": M.R.sha256_file(review)},
            "review_verdict": "APPROVATO", "control_required_until_utc": "2026-11-11T00:00:00Z"}


def check(report):
    states = {}
    M.check_evaluation_readiness(report, CONTRACT_SHA, states)
    return states


def mutate_pinned_json(report, key, mutation):
    path = Path(report[key]["path"])
    value = json.loads(path.read_text())
    mutation(value)
    report[key] = write(path, value)


def test_complete_readiness_reads_all_real_source_and_evidence_pins(readiness):
    states = check(readiness)
    assert SOURCES <= M.REQUIRED_EVALUATORS
    assert set(states) == {"evaluation_tests", "evaluation_resources", "evaluation_review"} | {
        "evaluation_source:" + source for source in SOURCES}
    for source in SOURCES:
        assert states["evaluation_source:" + source]["sha256"] == readiness["source_files"][source]
    assert states["evaluation_tests"]["path"] == readiness["verification_report"]["path"]
    assert states["evaluation_resources"]["sha256"] == readiness["resource_plan"]["sha256"]
    assert states["evaluation_review"]["sha256"] == readiness["review"]["sha256"]


@pytest.mark.parametrize("bad", ["protocol", "complete", "dose", "contract", "dimension", "pool", "capability",
                                "capability_flag", "source_hash", "review_verdict", "review_hash", "horizon"])
def test_missing_or_inconsistent_readiness_is_rejected(readiness, bad):
    if bad == "protocol": readiness["protocol"] = "other"
    if bad == "complete": readiness["complete"] = False
    if bad == "dose": readiness["checkpoint_step"] = 19618
    if bad == "contract": readiness["training_source_contract_sha256"] = "b" * 64
    if bad == "dimension": readiness["dimensions"][M.ARM] = 384
    if bad == "pool": readiness["pool_widths"][M.ARM] = 768
    if bad == "capability": readiness["capabilities"].pop("paired_comparison")
    if bad == "capability_flag": readiness["capabilities"]["dimension_matched_secondary"] = False
    if bad == "source_hash": readiness["source_files"]["scripts/scaling_probe.py"] = "b" * 64
    if bad == "review_verdict": readiness["review_verdict"] = "DA RIVEDERE"
    if bad == "review_hash": readiness["review"]["sha256"] = "b" * 64
    if bad == "horizon": readiness["control_required_until_utc"] = "after_training"
    with pytest.raises(ValueError):
        check(readiness)


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_each_evaluator_including_projection_and_reading_is_required(readiness, source):
    readiness["source_files"].pop(source)
    # Also alter the test report, so the required-source check must reject this
    # even when the remaining implementation and its tests agree with each other.
    mutate_pinned_json(readiness, "verification_report", lambda value: value["source_files"].pop(source))
    with pytest.raises(ValueError, match="evaluators missing"):
        check(readiness)


@pytest.mark.parametrize("bad", ["incomplete", "failed", "zero_passed", "boolean_passed", "unbound_sources", "pin"])
def test_tests_must_have_passed_and_be_bound_to_these_exact_sources(readiness, bad):
    if bad == "pin":
        readiness["verification_report"]["sha256"] = "b" * 64
    else:
        def mutation(value):
            if bad == "incomplete": value["complete"] = False
            if bad == "failed": value["failed"] = 1
            if bad == "zero_passed": value["passed"] = 0
            if bad == "boolean_passed": value["passed"] = True
            if bad == "unbound_sources": value["source_files"]["scripts/scaling_pose_readout.py"] = "b" * 64
        mutate_pinned_json(readiness, "verification_report", mutation)
    with pytest.raises(ValueError):
        check(readiness)


@pytest.mark.parametrize("bad", ["incomplete", "dose", "control_extraction", "matched", "gpu_zero", "cpu_zero",
                                "gpu_negative", "cpu_boolean", "cpu_nan", "pin"])
def test_resource_certificate_covers_both_arms_and_secondary_with_positive_caps(readiness, bad):
    if bad == "pin":
        readiness["resource_plan"]["sha256"] = "b" * 64
    else:
        def mutation(value):
            if bad == "incomplete": value["complete"] = False
            if bad == "dose": value["checkpoint_step"] = 19618
            if bad == "control_extraction": value["includes_30m_classification_extraction"] = False
            if bad == "matched": value["includes_dimension_matched_secondary"] = False
            if bad == "gpu_zero": value["gpu_hours_cap"] = 0.
            if bad == "cpu_zero": value["cpu_core_hours_cap"] = 0.
            if bad == "gpu_negative": value["gpu_hours_cap"] = -1.
            if bad == "cpu_boolean": value["cpu_core_hours_cap"] = True
            if bad == "cpu_nan": value["cpu_core_hours_cap"] = float("nan")
        mutate_pinned_json(readiness, "resource_plan", mutation)
    with pytest.raises(ValueError):
        check(readiness)


def test_unpinned_mutation_of_review_file_is_rejected(readiness):
    Path(readiness["review"]["path"]).write_text("changed after review pin")
    with pytest.raises(ValueError, match="SHA"):
        check(readiness)


def test_offline_check_does_not_invent_a_readiness_artifact_or_change_report(readiness, tmp_path):
    before = copy.deepcopy(readiness)
    paths = sorted(str(path) for path in tmp_path.iterdir())
    check(readiness)
    assert readiness == before
    assert sorted(str(path) for path in tmp_path.iterdir()) == paths
