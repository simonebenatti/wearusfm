import numpy as np

from wearusfm.harness.features_hudgins import (
    extract_hudgins_features,
    extract_hudgins_features_batch,
    mean_absolute_value,
    slope_sign_changes,
    waveform_length,
    zero_crossings,
)


def test_mean_absolute_value_constant_signal():
    window = np.full((100, 3), -2.0)
    mav = mean_absolute_value(window)
    assert np.allclose(mav, 2.0)


def test_zero_crossings_counts_sign_flips():
    # alternanza perfetta +1/-1: ogni coppia consecutiva attraversa lo zero
    signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0])[:, None]
    zc = zero_crossings(signal)
    assert zc[0] == 4


def test_zero_crossings_no_flips_on_constant_positive_signal():
    signal = np.full((50, 1), 3.0)
    zc = zero_crossings(signal)
    assert zc[0] == 0


def test_zero_crossings_threshold_ignores_small_flips():
    # attraversamenti veri ma di ampiezza minuscola: la soglia li deve scartare
    signal = np.array([0.001, -0.001, 0.001, -0.001])[:, None]
    zc_no_threshold = zero_crossings(signal, threshold=0.0)
    zc_with_threshold = zero_crossings(signal, threshold=0.5)
    assert zc_no_threshold[0] == 3
    assert zc_with_threshold[0] == 0


def test_waveform_length_zero_for_constant_signal():
    window = np.full((100, 2), 5.0)
    wl = waveform_length(window)
    assert np.allclose(wl, 0.0)


def test_waveform_length_accumulates_absolute_differences():
    signal = np.array([0.0, 1.0, 0.0, 1.0])[:, None]
    wl = waveform_length(signal)
    assert wl[0] == 3.0  # |1-0| + |0-1| + |1-0|


def test_slope_sign_changes_detects_local_extrema():
    # 0,1,0,1,0: zigzag puro, tutti e 3 i punti interni (indici 1,2,3) sono un
    # massimo o un minimo locale
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])[:, None]
    ssc = slope_sign_changes(signal)
    assert ssc[0] == 3


def test_slope_sign_changes_short_window_returns_zero():
    signal = np.array([1.0, 2.0])[:, None]
    ssc = slope_sign_changes(signal)
    assert ssc[0] == 0


def test_extract_hudgins_features_shape():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(200, 5))
    feats = extract_hudgins_features(window)
    assert feats.shape == (4 * 5,)


def test_extract_hudgins_features_batch_shape():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(10, 200, 5))
    feats = extract_hudgins_features_batch(windows)
    assert feats.shape == (10, 20)
