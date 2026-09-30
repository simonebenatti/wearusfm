"""Buchi di un canale (valore identico per >= 1 s): rilevamento, sidecar, esclusione nel campionamento (docs/decisioni.md, 01/10/2026)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_processed import _write  # noqa: E402

from wearusfm.data.processed import constant_runs, load_session, runs_in_segments, validate_session  # noqa: E402
from wearusfm.tokenizer_checks.sessions import (  # noqa: E402
    CanonicalSession,
    draw_from_session,
    to_canonical,
)


def _noise(n, c, seed=0, amp=500):
    return np.round(np.random.default_rng(seed).normal(scale=amp, size=(n, c))).astype(np.int16)


def test_constant_runs_2d_across_chunks_and_3d():
    x = _noise(5000, 3)
    x[1000:2500, 1] = 0  # 1500 campioni identici
    x[4200:, 2] = 7  # tratto finale fino alla fine dell'array
    x[300:340, 0] = 3  # corto: sotto soglia
    for chunk in (97, 1000, 10**6):  # il risultato non dipende dai blocchi di lettura
        from wearusfm.data import processed as P

        # 800 < 1000: il tratto finale del canale 2 e' sotto soglia; quello da 40 campioni del canale 0 pure
        assert P._constant_runs_2d(x, 1000, chunk_rows=chunk) == [{"channel": 1, "start": 1000, "n_samples": 1500}]
    assert constant_runs(x, 800)[-1] == {"channel": 2, "start": 4200, "n_samples": 800}
    x3 = np.stack([x[:2500], x[2500:]])  # (2, 2500, 3)
    r3 = constant_runs(x3, 1000)
    assert {"trial": 0, "channel": 1, "start": 1000, "n_samples": 1500} in r3


def test_natural_noise_has_no_long_runs():
    assert constant_runs(_noise(20000, 4, amp=3), 1000) == []  # anche rumore molto basso cambia valore di continuo


def test_runs_in_segments_split_at_trial_boundaries():
    arr = np.zeros((1000, 2), dtype=np.int16)
    meta = {"trials": [{"offset": 0, "n_samples": 600}, {"offset": 600, "n_samples": 400}]}
    segs = runs_in_segments([{"channel": 1, "start": 500, "n_samples": 300}], arr, meta)
    assert segs == [(0, 1, 500, 100), (1, 1, 0, 200)]
    assert runs_in_segments([{"channel": 0, "start": 10, "n_samples": 5}], arr, {}) == [(0, 0, 10, 5)]
    arr3 = np.zeros((2, 50, 2), dtype=np.int16)
    assert runs_in_segments([{"trial": 1, "channel": 0, "start": 3, "n_samples": 4}], arr3, {}) == [(1, 0, 3, 4)]


def test_validate_and_load_session_with_constant_runs(tmp_path):
    d = _write(tmp_path / "s", _noise(3000, 4), extra={"constant_runs": [{"channel": 2, "start": 100, "n_samples": 1200}]})
    assert validate_session(d) == []
    s = load_session(d, "ds", "s1", "a")
    assert s.constant_runs == [(0, 2, 100, 1200)]
    bad = _write(tmp_path / "b", _noise(3000, 4), extra={"constant_runs": [{"channel": 9, "start": 100, "n_samples": 10}]})
    assert any("constant_runs" in p for p in validate_session(bad))


def test_bad_patches_cover_the_run_plus_margin(tmp_path):
    fs = 2000
    x = _noise(fs * 10, 4)
    x[fs * 4 : fs * 5, 1] = 0  # 1 s di buco sul canale 1, da 4 s a 5 s
    d = _write(tmp_path / "s", x, fs=float(fs), extra={"constant_runs": [{"channel": 1, "start": fs * 4, "n_samples": fs}]})
    cs = to_canonical(load_session(d, "ds", "s1", "a"))
    bad = cs.bad_patches
    assert bad.shape == cs.stream.shape[:2] and not bad[[0, 2, 3]].any()
    # a 1 kHz il buco va da 4000 a 5000 campioni = patch 20-24; piu' UNA patch per lato (decisioni.md, 01/10/2026): 19-25.
    # Numeri scritti a mano, non derivati da MARGIN_PATCHES: altrimenti il test non si accorgerebbe se il margine cambiasse.
    assert np.flatnonzero(bad[1]).tolist() == [19, 20, 21, 22, 23, 24, 25]


def _draw_old(cs, n, rng):
    """L'algoritmo di estrazione PRIMA del 01/10/2026, copiato: serve a verificare che senza buchi le estrazioni restino identiche."""
    out, prov = [], []
    for _ in range(n):
        ch = int(rng.choice(np.flatnonzero(cs.qc_valid)))
        a0 = int(rng.integers(0, cs.stream.shape[1] - 16 + 1))
        out.append(cs.stream[ch, a0 : a0 + 16].reshape(-1))
        prov.append((cs.subject, cs.session, ch, a0))
    return np.stack(out), prov


def _cs(bad=None, n_patch=60):
    stream = np.random.default_rng(0).normal(size=(4, n_patch, 200)).astype(np.float32)
    return CanonicalSession("ds", "s1", "a", stream, np.array([True, True, False, True]), 1.0, bad)


def test_without_runs_draws_are_identical_to_before():
    for bad in (None, np.zeros((4, 60), dtype=bool)):
        new, pnew = draw_from_session(_cs(bad), 50, np.random.default_rng(7))
        old, pold = _draw_old(_cs(), 50, np.random.default_rng(7))
        assert np.array_equal(new, old) and pnew == pold


def test_draws_never_touch_a_run_and_fail_loudly_if_impossible():
    bad = np.zeros((4, 60), dtype=bool)
    bad[1, 10:40] = True
    _, prov = draw_from_session(_cs(bad), 300, np.random.default_rng(1))
    assert all(not bad[ch, a0 : a0 + 16].any() for _, _, ch, a0 in prov)
    assert any(ch == 1 for _, _, ch, _ in prov)  # il canale 1 si usa ancora, fuori dal buco
    with pytest.raises(ValueError, match="buco"):
        draw_from_session(_cs(np.ones((4, 60), dtype=bool)), 1, np.random.default_rng(0))
