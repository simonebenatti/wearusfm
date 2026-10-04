"""Sonda lineare del passo 5 (regime «encoder congelato + probe», metriche di D12) su dati sintetici."""

import numpy as np
import pytest

from wearusfm.harness.probe import ProbeResult, bootstrap_bacc_se, combine_seeds, linear_probe


def _data(rng, n_sub, per_sub, n_cls, sep):
    x, y, s = [], [], []
    centers = rng.normal(size=(n_cls, 16)) * sep
    for k in range(n_sub):
        shift = rng.normal(size=16) * 0.3  # ogni soggetto un po' diverso
        for _ in range(per_sub):
            c = int(rng.integers(n_cls))
            x.append(centers[c] + shift + rng.normal(size=16))
            y.append(c)
            s.append(f"u{k}")
    return np.array(x), np.array(y), np.array(s)


def test_separable_classes_give_high_accuracy_and_noise_gives_chance():
    rng = np.random.default_rng(0)
    xa, ya, sa = _data(rng, 20, 30, 4, sep=0.6)
    k = np.array([int(v[1:]) for v in sa])
    tr, va, te = k < 14, (k >= 14) & (k < 16), k >= 16
    r = linear_probe(xa[tr], ya[tr], xa[va], ya[va], xa[te], ya[te], sa[te], n_boot=200)
    assert 0.6 < r.test_bacc < 1.0 and 0 < r.test_bacc_se < 0.15 and r.n_test_subjects == 4
    noise = rng.normal(size=xa.shape)
    r0 = linear_probe(noise[tr], ya[tr], noise[va], ya[va], noise[te], ya[te], sa[te], n_boot=200)
    assert abs(r0.test_bacc - 0.25) < 0.12  # caso con 4 classi
    with pytest.raises(ValueError):
        linear_probe(xa[:0], ya[:0], xa[va], ya[va], xa[te], ya[te], sa[te])


def test_bootstrap_resamples_subjects_and_seeds_combine_conservatively():
    y = np.array([0, 1] * 10)
    pred = y.copy()
    pred[:4] = 1 - pred[:4]  # errori concentrati nei soggetti 0 e 1
    s = np.repeat([f"u{i}" for i in range(10)], 2)
    se = bootstrap_bacc_se(y, pred, s, n_boot=500)
    assert se > 0
    a = ProbeResult(0.80, 0.03, 0, 0, 1.0, 1, 1, 1, 1)
    b = ProbeResult(0.84, 0.03, 0, 0, 1.0, 1, 1, 1, 1)
    m, se2 = combine_seeds([a, b])
    assert m == pytest.approx(0.82) and se2 >= 0.03  # stessi soggetti di test: la parte dei soggetti non si riduce
    assert combine_seeds([a]) == (0.80, 0.03)
