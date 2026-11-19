from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet.controls import (window_noise, validate_outer_partition,
    inner_partitions, normalize_component, generate_components, synthetic_target,
    calibrate_grid, linear_reference_gate, linear_head_gate,
    temporal_reference_gate, temporal_head_gate)
from pose_temponet.controls import (_quadratic_panel, _pose_variance_gram, _quad,
    score_quadratic_panel, statistic_values, build_positive_control_fold,
    evaluate_outer_references, combine_reference_folds, aggregate_quadratic_panel,
    score_aggregate)
from pose_temponet.metrics import dynamic_variance, selection_statistics
from pose_temponet.data import central_support


def test_window_noise_keyed_by_metadata_is_reorder_invariant():
    meta = [{"user": "u0", "window": 0}, {"user": "u0", "window": 1}]
    a = window_noise(19, meta, 160, 2)
    b = window_noise(19, meta[::-1], 160, 2)
    np.testing.assert_array_equal(a, b[::-1])
    assert not np.array_equal(a[0, :, 0], a[1, :, 0])
    assert not np.array_equal(a[0, :, 0], a[0, :, 1])


def test_partition_never_mixes_outer_held_into_inner_or_development():
    meta = [{"user": str(i//2)} for i in range(8)]
    result = validate_outer_partition(meta, [0, 1, 4, 5], [2, 3, 6, 7], ["0", "1", "2", "3"])
    assert result["fitting_users"] == ["0", "2"]
    inner = inner_partitions(meta, [0, 1, 4, 5])
    for fit, held in inner:
        assert set(fit) | set(held) == {0, 1, 4, 5}
    with pytest.raises(ValueError, match="users overlap"):
        validate_outer_partition(meta, [0], [1], ["0", "1", "2", "3"])
    with pytest.raises(ValueError, match="non-training"):
        validate_outer_partition(meta, [0, 1], [2, 3], ["0", "1"])


def test_normalization_uses_only_outer_fitting_statistics():
    rng = np.random.default_rng(7)
    comp = rng.normal(size=(4, 16, 2))
    support = np.ones((4, 4, 2), dtype=bool)
    a, stats_a = normalize_component(comp, support, [0, 1])
    changed = comp.copy(); changed[2:] += 100
    b, stats_b = normalize_component(changed, support, [0, 1])
    assert stats_a == stats_b
    np.testing.assert_array_equal(a[:2], b[:2])
    np.testing.assert_allclose(a[:2, 6:-6].mean((0, 1)), 0, atol=1e-15)


def test_generator_seed_and_saved_teacher_states_are_reproducible():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 20, 12)).astype(np.float16)
    meta = [{"user": str(i), "window": i} for i in range(4)]
    support = np.ones((4, 8, 3), dtype=bool)
    args = (x, meta, support, [0, 2], [1, 3], ["0", "1", "2", "3"])
    a = generate_components(*args, outer_fold=0, n_joints=3, block_windows=2)
    b = generate_components(*args, outer_fold=0, n_joints=3, block_windows=2)
    assert a["generator_hashes"] == b["generator_hashes"]
    for key in ("L", "T", "epsilon"):
        np.testing.assert_array_equal(a[key], b[key])
    clean, observed = synthetic_target(a, .4, .2)
    np.testing.assert_allclose(observed-clean, .2*a["epsilon"])


def test_grid_has_1936_pairs_and_ridge_boundary_skips_only_that_pair():
    calls = []
    def score(a, sigma):
        calls.append((a, sigma))
        return {"E_cal": .8+abs(sigma-2)*.01, "E_cleanLS": .8,
                "E_oracle": .8*(1-(.08 if a == .4 else .075)),
                "boundary_unresolved": a == .4 and sigma == 2}
    result = calibrate_grid({"E_dyn": .8}, score)
    assert result["status"] == "calibrated"
    assert len(calls) == len(set(calls)) == result["evaluations"] == 1936
    assert result["selected"]["linear"]["sigma"] == 2
    assert result["selected"]["temporal"]["a"] == .4
    assert result["selected"]["temporal"]["sigma"] != 2
    blocked = calibrate_grid({"E_dyn": .8, "boundary_unresolved": True}, lambda *_: pytest.fail("must not evaluate"))
    assert blocked["status"] == "control_non_calibrated"


def test_linearity_gate_uses_recoverable_ridge_margin_not_oracle():
    # Claude's counterexample: ridge90.8%, WLS-like reader89.8% of oracle gain.
    reference = {split: {metric: {"constant": 1., "oracle": .9, "ridge": .9092}
                         for metric in ("MSE", "E_dyn")} for split in ("fold0", "fold1", "pooled")}
    heads = {split: {metric: .9102 for metric in ("MSE", "E_dyn")} for split in reference}
    assert linear_reference_gate(reference)["status"] == "diagnostic"
    result = linear_head_gate(reference, heads, optimization_alarms=["low_lr_boundary"])
    assert result["status"] == "passed"  # alarm alone never vetoes a passing control
    heads["fold0"]["MSE"] = .99
    assert linear_head_gate(reference, heads)["status"] == "recipe_not_admitted"
    assert linear_head_gate(reference, heads, optimization_alarms=["tail"])["status"] == "optimization_unresolved"
    reference["fold1"]["MSE"]["ridge"] = .99
    assert linear_head_gate(reference, heads)["status"] == "control_non_diagnostic"


def test_temporal_clean_baseline_gate_and_mlp_are_distinct():
    reference = {split: {"clean_ls": 1., "oracle": .92} for split in ("fold0", "fold1", "pooled")}
    heads = {split: .95 for split in reference}
    mlp = {split: .94 for split in reference}
    result = temporal_head_gate(reference, heads, mlp_records=mlp)
    assert result["status"] == "passed" and result["mlp_is_descriptive_only"]
    assert all(x > .5 for x in result["recovery"].values())
    reference["fold0"]["oracle"] = .8
    assert temporal_head_gate(reference, heads)["status"] == "control_non_diagnostic"


def test_nondiagnostic_denominators_precede_neural_alarms():
    ref = {s: {m: {"constant": 1., "ridge": 1., "oracle": 1.} for m in ("MSE", "E_dyn")}
           for s in ("fold0", "fold1", "pooled")}
    assert linear_reference_gate(ref)["status"] == "control_non_diagnostic"
    with pytest.raises(ValueError, match="panel incomplete"):
        temporal_reference_gate({"pooled": {"clean_ls": 1., "oracle": .92}})


def test_quadratic_scores_equal_direct_historical_metrics_with_masks():
    rng = np.random.default_rng(88)
    target_basis = [rng.normal(size=(5, 20, 3)) for _ in range(3)]
    pred_basis = [rng.normal(size=(5, 8, 3)) for _ in range(3)]
    mask = np.ones((5, 20, 3), dtype=bool); mask[0, :12, 0] = False; mask[2, 7:, 1] = False
    support = central_support(mask, np.full(5, 20, dtype=np.int64))
    coefficients = np.array([1., .4, 1.75])
    target = sum(c*y for c, y in zip(coefficients, target_basis))
    prediction = sum(c*y for c, y in zip(coefficients, pred_basis))
    clean = target_basis[0]+.4*target_basis[1]
    pv = dynamic_variance(clean, mask); threshold = float(np.median(pv))
    data = {"target": target, "mask": mask, "pose_var": pv, "std": np.ones((5, 3)),
            "meta": [{"user": str(i), "n_patches": 20} for i in range(5)]}
    panel = _quadratic_panel([p-y[:, 6:-6] for p, y in zip(pred_basis, target_basis)],
                            [y[:, 6:-6] for y in target_basis], support)
    q = score_quadratic_panel(panel, coefficients, pv >= threshold)
    direct = selection_statistics(prediction, data, threshold)
    aggregate = score_aggregate(aggregate_quadratic_panel(panel, pv >= threshold), coefficients)
    for key in q:
        assert q[key] == pytest.approx(direct[key], rel=1e-13, abs=1e-12)
        assert aggregate[key] == pytest.approx(direct[key], rel=1e-13, abs=1e-12)
    pg = _pose_variance_gram(target_basis[:2], mask)
    np.testing.assert_allclose(_quad(pg, [1., .4]), pv, rtol=1e-13)


def _builder_fixture():
    rng = np.random.default_rng(51)
    x = rng.normal(size=(8, 20, 4)).astype(np.float16)
    z = (x.astype(float)-x.astype(float).mean(-1, keepdims=True))
    z /= np.sqrt(np.mean(z*z, axis=-1, keepdims=True)+1e-5)
    l = z @ rng.normal(size=(4, 2))
    temporal = np.roll(z, 1, axis=1) @ rng.normal(size=(4, 2))
    epsilon = rng.normal(size=l.shape)
    observed = l + .7*rng.normal(size=l.shape)
    mask = np.ones(observed.shape, dtype=bool)
    data = {"lat": x, "target": observed, "mask": mask, "std": np.ones((8, 2)),
            "pose_var": dynamic_variance(observed, mask),
            "meta": [{"user": str(i), "n_patches": 20} for i in range(8)]}
    components = {"L": l, "T": temporal, "epsilon": epsilon}
    return data, components


def test_full_inner_builder_never_uses_outer_held_targets_for_calibration():
    data, components = _builder_fixture()
    fit, held = [0, 2, 4, 6], [1, 3, 5, 7]
    result = build_positive_control_fold(data, fit, held, list(map(str, range(8))),
                                         outer_fold=0, components=components, block_windows=2)
    changed = dict(data); changed["target"] = data["target"].copy(); changed["target"][held] += 10000
    again = build_positive_control_fold(changed, fit, held, list(map(str, range(8))),
                                        outer_fold=0, components=components, block_windows=2)
    assert result["real_reference"] == again["real_reference"]
    assert result["calibration"] == again["calibration"]
    assert result["calibration"]["evaluations"] == 1936
    assert result["outer_scored_after_inner_selection"] == (result["calibration"]["status"] == "calibrated")


def test_outer_references_and_pooled_gate_sum_moments_not_ratios():
    data, components = _builder_fixture()
    selected = {"linear": {"a": 0., "sigma": 1., "alpha": .1},
                "temporal": {"a": .4, "sigma": 1., "alpha": .1}}
    results = []
    for fold in (0, 1):
        fit, held = list(range(fold, 8, 2)), list(range(1-fold, 8, 2))
        stats = evaluate_outer_references(data, components, fit, held, selected, block_windows=2)
        results.append({"outer_fold": fold, "calibration": {"status": "calibrated"},
                        "outer_scored_after_inner_selection": True, "outer_reference_statistics": stats})
    gate = combine_reference_folds(results)
    a, b = [r["outer_reference_statistics"]["linear"]["ridge"] for r in results]
    expected = (a["dynamic_sse"]+b["dynamic_sse"])/(a["dynamic_target_ss"]+b["dynamic_target_ss"])
    assert gate["linear_records"]["pooled"]["E_dyn"]["ridge"] == pytest.approx(expected)


def test_missing_dynamic_support_reports_nondiagnostic_without_nan_json():
    import json
    values = statistic_values({"mse_sum": 1., "window_count": 1, "dynamic_sse": 0., "dynamic_target_ss": 0.})
    assert values["E_dyn"] is None
    report = calibrate_grid(values, lambda *_: pytest.fail("no scan with undefined real anchor"))
    assert report["status"] == "control_non_diagnostic"
    json.dumps(report, allow_nan=False)
