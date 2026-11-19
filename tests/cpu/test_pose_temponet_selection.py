from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import selection as S


def curve(lr, mse, best=20, horizon=300, edyn_best=None, selected_edyn=1.0):
    edyn_best = best if edyn_best is None else edyn_best
    epochs = sorted({0, best, edyn_best, horizon})
    points = tuple(S.CurvePoint(float(e), mse if e == best else mse + .1,
                               .8 if e == edyn_best else selected_edyn) for e in epochs)
    return S.CandidateCurve(lr, horizon, points)


def test_pooling_uses_window_weights_and_dynamic_sums_not_fold_ratios():
    scores = [S.FoldScore(2, 1, 1, 1), S.FoldScore(18, 9, 9, 99)]
    point = S.pool_scores(scores, 1, [7, 9])
    assert point.mse == 2
    assert point.edyn == .1
    assert point.edyn != pytest.approx((1 + 9 / 99) / 2)
    assert point.fold_steps == (7, 9)
    with pytest.raises(S.NonDiagnosticError):
        S.pool_scores([S.FoldScore(1, 1, 0, 0)], 1)


def test_zero_is_diagnostic_and_ties_choose_exposure_then_lr():
    a = S.CandidateCurve(.001, 3, (S.CurvePoint(0, 0, 0), S.CurvePoint(.5, .4, .7), S.CurvePoint(3, .4, .3)))
    b = S.CandidateCurve(.0001, 3, (S.CurvePoint(0, 0, 0), S.CurvePoint(.5, .4, .9), S.CurvePoint(3, .5, .1)))
    got = S.choose([a, b])
    assert (got.lr, got.exposure, got.mse, got.edyn) == (.0001, .5, .4, .9)


def test_dynamic_tail_triggers_extension_even_when_mse_stopped_early():
    candidate = curve(.001, .7, best=43, horizon=150, edyn_best=150)
    assert S.needs_epoch_extension(candidate)
    assert not S.needs_epoch_extension(curve(.001, .7, best=43, horizon=150, edyn_best=140))


def test_claude_lower_edge_counterexample_does_not_remove_the_comparator():
    candidates = [curve(.003, .7330), curve(.001, .7290), curve(.0003, .7272),
                  curve(.0001, .7269, best=17), curve(.00003, .7268, best=57)]
    result = S.assessment(candidates)
    assert result["selected"]["lr"] == .00003
    assert result["lower_edge_relative_mse_gain"] == pytest.approx(.0001 / .7269)
    assert not result["optimization_unresolved"]
    candidates[-1] = curve(.00003, .72, best=57)
    assert "lower_lr_edge_material_gain" in S.assessment(candidates)["optimizer_alarms"]


def test_upper_edge_and_final_dynamic_minimum_are_optimizer_alarms():
    candidates = [curve(lr, 1 + i / 100) for i, lr in enumerate(S.INITIAL_LRS)]
    candidates.append(curve(S.UPPER_LR, .8, best=20, edyn_best=298))
    result = S.assessment(candidates)
    assert set(result["optimizer_alarms"]) == {"upper_lr_edge_after_extension", "minimum_in_last_five_maximum_epochs"}


def test_s2_is_only_a_tradeoff_and_does_not_select_dynamic_optimum():
    candidates = [curve(.001, .7, best=93, horizon=300, edyn_best=150, selected_edyn=.9)]
    result = S.assessment(candidates)
    assert result["selected"]["exposure"] == 93
    assert result["objective_tradeoff"]["flag"] is True
    assert result["objective_tradeoff"]["dynamic_relative_regret"] == pytest.approx(.125)
    assert not result["optimization_unresolved"]


def test_first_quarter_alarm_and_zero_dynamic_optimum_are_explicit():
    candidate = S.CandidateCurve(.001, 300, (S.CurvePoint(0, 5, 5), S.CurvePoint(.25, 1, .1),
                                           S.CurvePoint(10, 2, 0), S.CurvePoint(300, 3, .2)))
    result = S.assessment([candidate])
    assert result["optimizer_alarms"] == ["selected_first_quarter_epoch"]
    assert result["objective_tradeoff"]["regret_infinite"]
    assert result["objective_tradeoff"]["dynamic_relative_regret"] is None


def test_first_epoch_schedule_deduplicates_joint_steps_without_inventing_quarter_alarm():
    assert S.checkpoint_schedule([1, 1], 2) == [(0., (0, 0)), (1., (1, 1)), (2., (2, 2))]
    assert S.checkpoint_schedule([1, 2], 1) == [(0., (0, 0)), (.5, (1, 1)), (1., (1, 2))]
    assert S.checkpoint_schedule([7, 9], 1)[1] == (.25, (2, 3))
    assert S.checkpoint_schedule([7, 9], 2, after_exposure=1) == [(2., (14, 18))]


def test_lr_search_adds_only_the_selected_edge_and_stops_after_it():
    candidates = [curve(lr, .7 + i / 10) for i, lr in enumerate(S.INITIAL_LRS)]
    assert S.next_learning_rate(candidates) == S.LOWER_LR
    candidates.append(curve(S.LOWER_LR, .69))
    assert S.next_learning_rate(candidates) is None
    with pytest.raises(ValueError, match="four initial"):
        S.next_learning_rate(candidates[1:])


@pytest.mark.parametrize("points", [
    (S.CurvePoint(0, 1, 1), S.CurvePoint(1, 1, 1)),
    (S.CurvePoint(0, 1, 1), S.CurvePoint(3, float("nan"), 1)),
    (S.CurvePoint(0, 1, 1), S.CurvePoint(3, 1, 1), S.CurvePoint(3, 1, 1)),
])
def test_incomplete_nonfinite_or_duplicate_curve_is_not_selectable(points):
    with pytest.raises(ValueError):
        S.choose([S.CandidateCurve(.001, 3, points)])
