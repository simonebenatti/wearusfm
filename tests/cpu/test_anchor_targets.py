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


def _reference(x, fs, band_limit_hz, patch_ms=25.0):
    """L'implementazione prima del 03/10/2026 (finestre materializzate, float64): la versione veloce deve dare gli stessi target."""
    x = np.asarray(x, dtype=np.float64)
    c, t = x.shape
    n_patch = int(np.floor(t / fs / (patch_ms / 1000.0) + 1e-9))
    centers = (np.arange(n_patch) + 0.5) * (patch_ms / 1000.0)  # come nel modulo: lo stesso arrotondamento dei centri
    limit = np.minimum(np.broadcast_to(np.asarray(band_limit_hz, dtype=np.float64), (c,)), fs / 2.0)
    edges = np.asarray(A.BAND_EDGES_HZ)
    avail = edges[None, 1:] <= limit[:, None] + 1e-9
    seg, _ = A._windows(x, fs, centers, A.RMS_WINDOW_MS)
    log_rms = np.log(np.sqrt(np.mean(seg ** 2, axis=-1)) + A.EPS)
    seg, _ = A._windows(x, fs, centers, A.SPECTRAL_WINDOW_MS)
    n = seg.shape[-1]
    power = np.abs(np.fft.rfft(seg * np.hanning(n), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    bp = np.stack([power[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) for lo, hi in zip(edges[:-1], edges[1:])], axis=-1) * avail[:, None, :]
    band_shape = np.log(bp / (bp.sum(axis=-1, keepdims=True) + A.EPS) + A.EPS)
    seg, _ = A._windows(x, fs, centers, A.ENVELOPE_WINDOW_MS)
    return log_rms, band_shape, np.log(np.sqrt(np.mean(seg ** 2, axis=-1)) + A.EPS)


def test_fast_targets_equal_the_reference_also_at_the_edges():
    rng = np.random.default_rng(3)
    for c, fs, t, lim in [(4, 2000.0, 8000, 450.0), (3, 1000.0, 1000, 450.0), (5, 200.0, 800, 100.0), (2, 2048.0, 3000, [450.0, 100.0])]:
        x = rng.normal(size=(c, t)) * np.linspace(0.1, 3.0, t)[None, :]  # ampiezza che cambia: i bordi contano
        got = A.anchor_targets(x.astype(np.float32), fs, lim)
        ref_rms, ref_shape, ref_env = _reference(x.astype(np.float32), fs, lim)
        assert np.allclose(got.log_rms, ref_rms, atol=1e-6) and np.allclose(got.log_env, ref_env, atol=1e-6)  # tutte le patch, anche fuori
        ok = got.spec_valid[:, :, None] & got.band_available[:, None, :]
        assert np.allclose(got.band_shape[ok], ref_shape[ok], atol=1e-3)  # float32 nello spettro


def test_rvq_windows_ignore_channels_discarded_by_qc():
    vis = np.ones((3, 16), dtype=bool)
    vis[:2, 0:8] = False  # canali 0 e 1 nascosti sulla prima finestra; il canale 2 e' scartato dal QC (le maschere lo lasciano visibile)
    valid, on = np.ones((3, 16), dtype=bool), np.ones(3, dtype=bool)
    assert not A.rvq_target_windows(vis, valid, on).any()  # senza QC il canale 2 «visibile» impedisce lo slab
    w = A.rvq_target_windows(vis, valid, on, qc_valid=np.array([True, True, False]))
    assert w[:2, 0].all() and not w[2].any() and not w[:, 1].any()


def test_multiscale_anchor_targets_scales_bands_and_masks():
    """Ancora multi-scala (Simone, 04/10/2026): 6 bande a 50 ms e 24 bande a 500 ms, centrate sulla patch; bande oltre il limite non disponibili."""
    fs = 2000.0
    t = np.arange(8000) / fs
    x = np.sin(2 * np.pi * 300 * t)[None, :] * np.ones((2, 1))
    tg = A.anchor_targets(x, fs, 450.0, multiscale=True)
    assert tg.ms_fast.shape == (2, 160, 6) and tg.ms_slow.shape == (2, 160, 24)
    e24 = np.geomspace(20.0, 450.0, 25)
    k = int(np.searchsorted(e24, 300.0) - 1)
    assert int(np.argmax(tg.ms_slow[0, 80])) == k and int(np.argmax(tg.ms_fast[0, 80])) == 5  # 300 Hz: banda 265-450 della scala rapida
    assert tg.ms_slow_valid[:, 10:150].all() and not tg.ms_slow_valid[:, :5].any()  # 500 ms: niente target vicino ai bordi della finestra
    assert tg.ms_fast_available.all() and tg.ms_slow_available.all()
    low = A.anchor_targets(np.random.default_rng(0).normal(size=(2, 800)), 200.0, 100.0, multiscale=True)
    assert not low.ms_fast_available[:, -1].any() and low.ms_fast_available[:, 0].all()  # a 200 Hz le bande alte non ci sono
    g = A.anchor_targets(x, fs, 450.0, multiscale=True, guard=(200, 0))
    assert not g.ms_fast_valid[:, :4].any()  # la zona di bordo vale anche per la multi-scala
    assert A.anchor_targets(x, fs, 450.0).ms_fast is None  # spenta: nessun target
