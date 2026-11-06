import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scaling_reading19618", ROOT / "scripts/scaling_reading19618.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def example():
    return {"protocol": "scaling-comparison-19618-20261106-v1", "checkpoint_step": 19618,
            "candidate": "T100-G-300M", "control": "T100-G-30M",
            "bootstrap": copy.deepcopy(S.POLICY["bootstrap"]),
            "quality": {"artifact_integrity": True, "paired_support": True, "solver_converged": True},
            "epn": {"delta_pp": 0.5, "ci95": [-0.2, 1.2]},
            "pose": {"control_E_dyn": 1., "candidate_E_dyn": 0.97, "delta_E_dyn": -0.03, "ci95": [-0.05, 0.01]},
            "p1": {"delta_pp": 0., "ci95": [-0.8, 0.8]},
            "p2": {"delta_pp": -0.5, "ci95": [-1.2, 0.2]}}


def test_unresolved_is_described_without_a_claim_recommendation_or_submission():
    result = S.read_comparison(example())
    assert result["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result
    assert result["primary_reading"] == {"epn": "unresolved", "pose": "unresolved"}
    assert not result["pose_practical_signal"] and not result["further_training_authorized"]
    assert result["classification_noninferiority"]["p2"] == "unresolved"


def test_positive_epn_and_negative_pose_cannot_be_cherry_picked():
    row = example()
    row["epn"]["ci95"] = [.1, .9]
    row["pose"] = {"control_E_dyn": 1., "candidate_E_dyn": 1.01, "delta_E_dyn": .01, "ci95": [.001, .02]}
    result = S.read_comparison(row)
    assert result["primary_reading"] == {"epn": "better", "pose": "worse"}
    assert result["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result


def test_two_percent_is_point_relevance_not_a_new_confidence_margin():
    row = example()
    row["pose"]["ci95"] = [-.055, -.001]
    result = S.read_comparison(row)
    assert result["pose_practical_signal"]
    assert "recommendation" not in result
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
    assert "recommendation" not in result and result["reading_status"] == "descriptive_endpoint_reading"


@pytest.mark.parametrize("interval", [[0., 1.], [-1., 0.]])
def test_touching_zero_is_unresolved_for_epn(interval):
    row = example()
    row["epn"]["ci95"] = interval
    assert S.read_comparison(row)["primary_reading"]["epn"] == "unresolved"


def test_fit_or_artifact_failure_prevents_a_scientific_reading():
    row = example()
    row["quality"]["solver_converged"] = False
    result = S.read_comparison(row)
    assert not result["reading_valid"] and not result["further_training_authorized"]
    assert "primary_reading" not in result and result["reading_status"] == "repair_evaluation_no_scientific_reading"
    assert result["failed_checks"] == ["solver_converged"] and "recommendation" not in result


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


def test_reading_has_no_budget_formula_dose_proposal_or_checkpoint_selection():
    source = (ROOT / "scripts/scaling_reading19618.py").read_text()
    assert not hasattr(S, "continuation_training_cap")
    for forbidden in ("propose_", "pause_this_recipe", "49046", "best_checkpoint", "continuation_is_authorized"):
        assert forbidden not in source
    assert S.POLICY["recommendation_emitted"] is False and S.POLICY["s8_operational_table_reapplied"] is False
    assert S.POLICY["checkpoint_selection_between_doses"] is False and S.POLICY["further_training_authorized"] is False
    assert S.POLICY["pretraining_seeds"] == 1 and S.POLICY["checkpoint_step"] == 19618


def test_policy_file_matches_the_implemented_prespecification():
    import json
    policy = json.loads((ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261106/"
                         "scientific_reading_policy_19618.json").read_text())
    assert policy == S.POLICY


def frozen_reading():
    spec5 = importlib.util.spec_from_file_location("scaling_reading", ROOT / "scripts/scaling_reading.py")
    module = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(module)
    return module


def test_definitions_are_those_frozen_for_the_5000_reading():
    import json
    frozen = json.loads((ROOT / "results/infra/scaling_300m_20261104/scientific_reading_policy.json").read_text())
    shared = ("candidate", "control", "bootstrap", "epn_margin_pp", "pose_practical_point_reduction",
              "p1_p2_noninferiority_margin_pp", "historical_pair_is_noise_floor")
    assert {key: S.POLICY[key] for key in shared} == {key: frozen[key] for key in shared}
    assert S.POLICY["definitions_inherited_from"] == frozen["protocol"] == frozen_reading().PROTOCOL
    assert frozen["checkpoint_step"] == 5000 and S.POLICY["bootstrap"] == {"unit": "subject", "replicates": 2000, "seed": 20261102}


def test_labels_equal_the_frozen_5000_definitions_on_a_grid_of_intervals_and_points():
    frozen = frozen_reading()
    intervals = [[-2.5, -1.5], [-2., -1.], [-1.5, -.5], [-1., 0.], [-.5, 0.], [-.5, .5], [0., .5], [0., 1.], [.5, 1.5]]
    pose_intervals = [[-.06, -.01], [-.03, 0.], [-.02, .01], [0., .02], [.001, .02]]
    compared = 0
    for epn in intervals:
        for p1 in intervals:
            for pose_ci in pose_intervals:
                for candidate in (.97, .98, .99, 1.01):
                    row = example()
                    row["epn"]["ci95"], row["p1"]["ci95"], row["p2"]["ci95"] = epn, p1, [-p1[1], -p1[0]]
                    row["pose"] = {"control_E_dyn": 1., "candidate_E_dyn": candidate, "delta_E_dyn": candidate - 1.,
                                   "ci95": pose_ci}
                    new = S.read_comparison(row)
                    old = frozen.read_comparison({**row, "protocol": "scaling-comparison-5000-20261104-v1",
                                                  "checkpoint_step": 5000})
                    for key in ("reading_valid", "source_quality", "primary_reading", "pose_relative_point_reduction",
                                "pose_practical_signal", "classification_noninferiority"):
                        assert new[key] == old[key]
                    assert "recommendation" in old and "recommendation" not in new
                    compared += 1
    assert compared == 9 * 9 * 5 * 4
