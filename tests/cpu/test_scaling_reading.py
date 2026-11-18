import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scaling_reading", ROOT / "scripts/scaling_reading.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def example():
    return {"protocol": "scaling-comparison-5000-20261104-v1", "checkpoint_step": 5000,
            "candidate": "T100-G-300M", "control": "T100-G-30M",
            "bootstrap": copy.deepcopy(S.POLICY["bootstrap"]),
            "quality": {"artifact_integrity": True, "paired_support": True, "solver_converged": True},
            "epn": {"delta_pp": 0.5, "ci95": [-0.2, 1.2]},
            "pose": {"control_E_dyn": 1., "candidate_E_dyn": 0.97, "delta_E_dyn": -0.03, "ci95": [-0.05, 0.01]},
            "p1": {"delta_pp": 0., "ci95": [-0.8, 0.8]},
            "p2": {"delta_pp": -0.5, "ci95": [-1.2, 0.2]}}


def test_unresolved_supports_a_dose_question_not_a_claim_or_submission():
    result = S.read_comparison(example())
    assert result["recommendation"] == "propose_19618_dose_question"
    assert result["primary_reading"] == {"epn": "unresolved", "pose": "unresolved"}
    assert not result["pose_practical_signal"] and not result["continuation_authorized"]
    assert result["classification_noninferiority"]["p2"] == "unresolved"


def test_positive_epn_and_negative_pose_cannot_be_cherry_picked():
    row = example()
    row["epn"]["ci95"] = [.1, .9]
    row["pose"] = {"control_E_dyn": 1., "candidate_E_dyn": 1.01, "delta_E_dyn": .01, "ci95": [.001, .02]}
    result = S.read_comparison(row)
    assert result["primary_reading"] == {"epn": "better", "pose": "worse"}
    assert result["recommendation"] == "pause_this_recipe_review_before_19618"


def test_two_percent_is_point_relevance_not_a_new_confidence_margin():
    row = example()
    row["pose"]["ci95"] = [-.055, -.001]
    result = S.read_comparison(row)
    assert result["pose_practical_signal"]
    assert result["recommendation"] == "propose_19618_confirmation"
    row["pose"].update(candidate_E_dyn=.99, delta_E_dyn=-.01, ci95=[-.02, -.001])
    result = S.read_comparison(row)
    assert result["primary_reading"]["pose"] == "better"
    assert not result["pose_practical_signal"]


@pytest.mark.parametrize("interval,expected", [([-1., 0.], "unresolved"), ([-2., -1.], "unresolved"),
                                                ([-2., -1.01], "loss_beyond_margin"), ([-.99, 0.], "supported")])
def test_classification_margin_is_strict_and_uses_interval(interval, expected):
    row = example()
    row["p1"]["ci95"] = interval
    result = S.read_comparison(row)
    assert result["classification_noninferiority"]["p1"] == expected
    assert (result["recommendation"] == "pause_this_recipe_review_before_19618") == (expected == "loss_beyond_margin")


@pytest.mark.parametrize("interval", [[0., 1.], [-1., 0.]])
def test_touching_zero_is_unresolved_for_epn(interval):
    row = example()
    row["epn"]["ci95"] = interval
    assert S.read_comparison(row)["primary_reading"]["epn"] == "unresolved"


def test_fit_or_artifact_failure_prevents_a_scientific_reading():
    row = example()
    row["quality"]["solver_converged"] = False
    result = S.read_comparison(row)
    assert not result["reading_valid"] and not result["continuation_authorized"]
    assert "primary_reading" not in result


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_numbers_do_not_become_unresolved_evidence(bad):
    row = example()
    row["epn"]["delta_pp"] = bad
    with pytest.raises(ValueError):
        S.read_comparison(row)


def test_pose_ratio_and_difference_must_reconcile():
    row = example()
    row["pose"]["delta_E_dyn"] = .3
    with pytest.raises(ValueError, match="disagree"):
        S.read_comparison(row)


def test_cost_uses_slower_measurement_margin_reserve_and_15min_rounding():
    result = S.continuation_training_cap(15000, 4.)
    # 14618 * 4 * 1.2 + 1200 = 71366.4 seconds, rounded up to 72000.
    assert result["walltime_seconds"] == 72000 and result["gpu_hours_cap"] == 80.
    assert not result["extractions_and_cpu_included"] and not result["submission_authorized"]


def test_policy_file_matches_the_implemented_prespecification():
    import json
    policy = json.loads((ROOT / "results/infra/scaling_300m_20261104/scientific_reading_policy.json").read_text())
    assert policy == S.POLICY
