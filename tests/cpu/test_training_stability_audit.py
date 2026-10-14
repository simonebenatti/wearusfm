import numpy as np
import pytest

from wearusfm.harness.training_stability import audit_curve, moving_average


def _rows(loss):
    return [{"step": i + 1, "keep_branches": float(x), "grad_norm": .2} for i, x in enumerate(loss)]


def test_v1_reproduced_and_single_peak_mass_is_not_confused_with_drift():
    x = np.full(700, .05)
    x[350] = 1.5
    rows = _rows(x)
    rows[350]["grad_norm"] = 3.0
    r = audit_curve(rows)
    ma = moving_average(x, 50)
    expected = (ma[100:] / np.minimum.accumulate(ma)[100:]).max()
    assert r["v1"]["max_rise"] == pytest.approx(expected)
    assert not r["v1"]["stable"]
    assert r["critical_window"]["largest_batch_update"] == 351
    assert r["critical_window"]["ratio_without_largest_batch"] == pytest.approx(1.)
    assert r["critical_window"]["median"] == pytest.approx(.05)
    assert r["gradient"]["clipped_fraction"] == pytest.approx(1 / 700)
    assert r["blocks"][-1]["mean"] == pytest.approx(.05)


def test_sustained_rise_survives_removal_of_largest_batch():
    x = np.r_[np.full(400, .05), np.full(400, .2)]
    r = audit_curve(_rows(x))
    assert not r["v1"]["stable"]
    assert r["critical_window"]["ratio_without_largest_batch"] == pytest.approx(4.)
    assert r["critical_window"]["median"] == pytest.approx(.2)
    assert r["blocks"][-1]["mean"] == pytest.approx(.2)


def test_zeros_and_invalid_values_remain_explicit():
    r = audit_curve(_rows(np.zeros(300)))
    assert not r["v1"]["stable"] and r["v1"]["max_rise"] is None
    assert not r["v1"]["reference_positive"]
    x = np.zeros(300)
    x[-1] = .1
    assert not audit_curve(_rows(x))["v1"]["stable"]
    r = audit_curve(_rows(np.r_[np.zeros(200), np.ones(100)]))
    assert r["critical_window"]["max"] == 1.
    assert r["critical_window"]["selection"] == "first infinite rise from a zero reference"
    rows = _rows(np.ones(300))
    rows[-1]["grad_norm"] = float("inf")
    r = audit_curve(rows)
    assert not r["finite_nonnegative"] and r["invalid_updates"] == [300]
    rows = _rows(np.ones(300))
    rows[5]["step"] = 5
    with pytest.raises(ValueError, match="consecutive"):
        audit_curve(rows)
