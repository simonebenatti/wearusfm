"""Sonda e verdetto del gate D8, senza torch (scikit-learn e torch insieme vanno in segfault su macOS)."""

import numpy as np

from wearusfm.gate import consistency as G


def _xy(n_subj=10, per=10, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, us = [], [], []
    for s in range(n_subj):
        for _ in range(per):
            base = rng.normal(size=20)
            for label in (0, 1):
                xs.append(base + rng.normal(scale=0.01, size=20) + label * shift)
                ys.append(label)
                us.append(f"s{s}")
    return np.array(xs), np.array(ys), np.array(us)


def test_probe_at_chance_when_versions_are_identical_and_high_when_shifted():
    x, y, u = _xy()
    p = G.probe_case(x, y, u)
    assert p["chance"] == 0.5 and p["balanced_accuracy"] <= 0.65
    x2, y2, u2 = _xy(shift=1.0)
    assert G.probe_case(x2, y2, u2)["balanced_accuracy"] > 0.9


def test_verdict_rules():
    ok_err = {"total": 0.01, "fourier": 0.02, "spline": 0.0, "mlp": 0.0}
    ok_probe = {"balanced_accuracy": 0.52, "ci95": [0.45, 0.58]}
    assert G.verdict(ok_err, ok_probe)["passes"]
    assert not G.verdict({**ok_err, "spline": 0.051}, ok_probe)["passes"]  # una famiglia oltre il 5%: non passa (4a firmata)
    assert not G.verdict(ok_err, {"balanced_accuracy": 0.56, "ci95": [0.50, 0.62]})["passes"]
    assert not G.verdict(ok_err, {"balanced_accuracy": 0.54, "ci95": [0.51, 0.57]})["passes"]  # l'intervallo non contiene 0,5
