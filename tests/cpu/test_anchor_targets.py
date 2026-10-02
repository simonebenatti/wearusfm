"""Target delle ancore (v10 §6.3) su segnali sintetici a contenuto noto."""

import math

import numpy as np
import pytest

from wearusfm.model import anchor_targets as A


def _sine(f_hz, fs, seconds=2.0, amp=1.0, n_ch=1):
    t = np.arange(int(seconds * fs)) / fs
    return np.tile(amp * np.sin(2 * math.pi * f_hz * t), (n_ch, 1))


def test_band_shape_puts_the_power_in_the_right_band():
    tg = A.anchor_targets(_sine(50.0, 2000.0), 2000.0, 450.0)  # 50 Hz cade nella seconda banda (37-70 Hz)
    p = tg.band_shape.shape[1] // 2
    assert tg.spec_valid[0, p] and int(np.argmax(tg.band_shape[0, p])) == 1
    assert tg.band_shape[0, p, 1] == pytest.approx(0.0, abs=0.05)  # quasi tutta la potenza: log(frazione) ~ 0
    assert (tg.band_shape[0, p, [0, 2, 3, 4]] < -3).all()


def test_rms_envelope_and_edges():
    tg = A.anchor_targets(_sine(100.0, 2000.0, amp=2.0), 2000.0, 450.0)
    assert np.allclose(tg.log_rms[0, 1:-1], math.log(2.0 / math.sqrt(2)), atol=1e-3) and tg.rms_valid.all()
    assert np.allclose(tg.log_env[0][tg.env_valid[0]], math.log(2.0 / math.sqrt(2)), atol=1e-3)
    p = tg.log_rms.shape[1]
    assert p == 80 and not tg.spec_valid[0, :3].any() and tg.spec_valid[0, 4:p - 4].all()  # 200 ms centrati: le prime 3 patch escono
    assert not tg.env_valid[0, :9].any() and tg.env_valid[0, 10]  # 500 ms centrati


def test_bands_beyond_the_available_band_are_masked():
    tg = A.anchor_targets(_sine(30.0, 200.0), 200.0, 100.0)  # Myo (DB5): Nyquist 100 Hz
    assert tg.band_available[0].tolist() == [True, True, False, False, False]  # 69,6-129,9 Hz supera i 100 Hz: fuori anche la terza
    p = tg.band_shape.shape[1] // 2
    avail = np.exp(tg.band_shape[0, p, :2])
    assert avail.sum() == pytest.approx(1.0, abs=1e-3)  # frazioni normalizzate solo sulle bande disponibili
    two = A.anchor_targets(np.vstack([_sine(30.0, 2000.0)[0], _sine(30.0, 2000.0)[0]]), 2000.0, np.array([450.0, 100.0]))
    assert two.band_available.tolist() == [[True] * 5, [True, True, False, False, False]]  # limite per canale (es. banda dichiarata)


def test_rvq_windows_only_where_a_whole_grid_window_is_hidden():
    vis = np.ones((3, 32), dtype=bool)
    vis[:, 8:16] = False  # slab allineato: finestra 1
    vis[:, 19:27] = False  # slab non allineato: nessuna finestra intera
    valid = np.ones((3, 32), dtype=bool)
    on = np.array([True, False, True])
    w = A.rvq_target_windows(vis, valid, on)
    assert w.shape == (3, 4) and w[:, 1].tolist() == [True, False, True] and not w[:, [0, 2, 3]].any()
    valid[:, 12:] = False  # padding: una finestra che finisce nel padding non ha target
    assert not A.rvq_target_windows(vis, valid, on).any()
    with pytest.raises(ValueError, match="non divide"):
        A.rvq_target_windows(vis, valid, on, patch_ms=30.0)


def test_rvq_never_on_channel_masking():
    vis = np.ones((3, 16), dtype=bool)
    vis[1] = False  # un canale nascosto per intero, gli altri visibili: masking di canale
    assert not A.rvq_target_windows(vis, np.ones((3, 16), dtype=bool), np.ones(3, dtype=bool)).any()
