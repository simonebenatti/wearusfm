import numpy as np
import pytest

from wearusfm.preprocessing.canonical_view import (
    apply_scale,
    bandpass_resample,
    patchify,
    resample_ratio,
    session_scale,
)


def _tone(fs, freq, seconds=4.0, amp=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return (amp * np.sin(2 * np.pi * freq * t))[:, None]


def _steady_rms(x):
    core = x[len(x) // 4 : -len(x) // 4]  # esclude i bordi del filtro
    return float(np.sqrt(np.mean(core**2)))


@pytest.mark.parametrize(
    "fs, expected",
    [(1000.0, (1, 1)), (2000.0, (1, 2)), (2048.0, (125, 256)), (5120.0, (25, 128)), (1926.0, (500, 963))],
)
def test_resample_ratio_native_rates(fs, expected):
    assert resample_ratio(fs) == expected


def test_resample_ratio_approximates_non_integer_fs():
    up, down = resample_ratio(1111.11)  # NinaPro DB8 "~1111 Hz", non verificata
    assert (up, down) == (9, 10)


def test_resample_ratio_rejects_below_1khz():
    with pytest.raises(ValueError, match="non e' eleggibile"):
        resample_ratio(200.0)  # DB5


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    fs = 2000.0
    in_band = bandpass_resample(_tone(fs, 100.0), fs)
    low = bandpass_resample(_tone(fs, 5.0), fs)
    dc = bandpass_resample(np.full((int(4 * fs), 1), 3.0), fs)
    assert _steady_rms(in_band) == pytest.approx(1 / np.sqrt(2), rel=0.05)
    assert _steady_rms(low) < 0.1 * (1 / np.sqrt(2))
    assert _steady_rms(dc) < 1e-3


@pytest.mark.parametrize("fs", [1000.0, 2000.0, 2048.0, 5120.0])
def test_same_tone_gives_same_output_on_1khz_grid(fs):
    out = bandpass_resample(_tone(fs, 120.0, seconds=4.0), fs)
    assert out.shape == (4000, 1)
    assert _steady_rms(out) == pytest.approx(1 / np.sqrt(2), rel=0.05)


def test_bandpass_resample_rejects_low_fs():
    with pytest.raises(ValueError):
        bandpass_resample(np.zeros((400, 1)), 200.0)


def test_session_scale_streaming_equals_concatenated():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(scale=3.0, size=(500 + 10 * i, 4)) for i in range(5)]
    whole = np.concatenate(chunks, axis=0)
    assert session_scale(chunks) == pytest.approx(whole.std(), rel=1e-9)


def test_session_scale_excludes_qc_discarded_channels():
    rng = np.random.default_rng(1)
    good = rng.normal(scale=2.0, size=(2000, 3))
    bad = np.full((2000, 1), 1000.0)  # canale saturo: non deve pesare
    chunk = np.concatenate([good, bad], axis=1)
    mask = np.array([True, True, True, False])
    assert session_scale([chunk], mask) == pytest.approx(good.std(), rel=1e-9)


def test_session_scale_is_one_number_for_all_channels():
    rng = np.random.default_rng(2)
    chunk = np.stack([rng.normal(scale=1.0, size=5000), rng.normal(scale=4.0, size=5000)], axis=1)
    scaled = apply_scale(chunk, session_scale([chunk]))
    assert scaled.std() == pytest.approx(1.0, rel=1e-9)
    # i rapporti di ampiezza fra canali sopravvivono (niente z-score per canale)
    assert scaled[:, 1].std() / scaled[:, 0].std() == pytest.approx(4.0, rel=0.05)


def test_session_scale_rejects_empty_and_zero():
    with pytest.raises(ValueError):
        session_scale([])
    with pytest.raises(ValueError):
        session_scale([np.zeros((100, 2))])
    with pytest.raises(ValueError):
        session_scale([np.ones((100, 2))], np.array([False, False]))


def test_apply_scale_target_scale():
    x = np.array([[2.0, -2.0]])
    assert np.allclose(apply_scale(x, 2.0, target_scale=5.0), [[5.0, -5.0]])


def test_patchify_shape_and_anchoring():
    x = np.arange(1050 * 2, dtype=float).reshape(1050, 2)
    p = patchify(x)
    assert p.shape == (2, 5, 200)  # 1050 // 200 = 5, resto scartato
    assert p[0, 0, 0] == x[0, 0] and p[1, 4, 199] == x[999, 1]
    with pytest.raises(ValueError):
        patchify(np.zeros((100, 2)))
