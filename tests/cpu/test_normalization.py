import numpy as np
import pytest

from wearusfm.harness.normalization import fit_per_session_first_n_seconds, fit_train_only


def test_fit_train_only_zero_median_after_apply():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(1000, 4))
    stats = fit_train_only(data)
    normalized = stats.apply(data)
    assert np.allclose(np.median(normalized, axis=0), 0.0, atol=0.1)


def test_fit_train_only_handles_3d_windows():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(50, 200, 3))  # (N campioni, T, C)
    stats = fit_train_only(data)
    assert stats.median.shape == (3,)
    assert stats.mad.shape == (3,)


def test_apply_does_not_divide_by_zero_for_constant_channel():
    data = np.zeros((100, 2))
    data[:, 1] = np.random.default_rng(2).normal(size=100)
    stats = fit_train_only(data)
    out = stats.apply(data)
    assert np.isfinite(out).all()


def test_normalization_fit_on_train_does_not_use_test_stats():
    # se le stats venissero (erroneamente) stimate su tutto il dataset, la media di
    # holdout con un forte shift non si allineerebbe piu' a zero dopo la normalizzazione
    rng = np.random.default_rng(3)
    train = rng.normal(loc=0.0, scale=1.0, size=(500, 2))
    holdout = rng.normal(loc=50.0, scale=1.0, size=(100, 2))  # shift grande, mai visto in train

    stats = fit_train_only(train)  # SOLO train
    normalized_holdout = stats.apply(holdout)
    # lo shift resta visibile: la normalizzazione train-only non lo nasconde
    assert np.median(normalized_holdout) > 10


def test_fit_per_session_first_n_seconds_uses_only_prefix():
    fs_hz = 100.0
    rng = np.random.default_rng(4)
    prefix = rng.normal(loc=0.0, scale=1.0, size=(200, 2))  # primi 2s
    rest = rng.normal(loc=100.0, scale=1.0, size=(800, 2))  # shift grande nel resto
    session = np.concatenate([prefix, rest], axis=0)

    stats = fit_per_session_first_n_seconds(session, fs_hz=fs_hz, n_seconds=2.0)
    assert abs(stats.median[0]) < 5  # deve riflettere il prefisso, non il resto shiftato


def test_fit_per_session_first_n_seconds_rejects_zero_samples():
    with pytest.raises(ValueError):
        fit_per_session_first_n_seconds(np.zeros((10, 2)), fs_hz=100.0, n_seconds=0.0)
