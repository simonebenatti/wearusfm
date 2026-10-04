"""Target candidati per le ancore in frequenza (protocollo congelato il 04/10/2026): spettro a 16 bande, momenti, sonde."""

import numpy as np
import pytest

from wearusfm.harness import anchor_candidates as A


def _tone(f, fs=2000.0, n=400, c=2, seed=0):
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * f * t)[None, :] * np.ones((c, 1)) + rng.normal(scale=0.01, size=(c, n))


def test_bands_and_moments_follow_the_tone():
    edges = np.geomspace(20.0, 450.0, 17)
    for f in (40.0, 120.0, 300.0):
        lp = A.band_logpower(_tone(f), 2000.0)
        k = int(np.searchsorted(edges, f) - 1)
        assert lp.shape == (2, 16) and int(np.nanargmax(lp[0])) == k
        mom = A.spectral_moments(_tone(f), 2000.0)
        assert mom[0, 0] == pytest.approx(f, rel=0.15) and mom[0, 1] == pytest.approx(f, rel=0.15)
    shape = A.band_shape(_tone(120.0), 2000.0)
    assert shape.shape == (2, 5) and np.allclose(np.exp(shape).sum(axis=-1), 1.0, atol=1e-6)
    assert np.isnan(A.band_logpower(_tone(40.0, fs=200.0, n=40), 200.0)).any()  # a 200 Hz bande strette senza frequenze: NaN, non zero


def test_probes_and_rule():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 8))
    y = x[:, :2] @ rng.normal(size=(2, 3)) + rng.normal(scale=0.1, size=(2000, 3))
    assert A.ridge_r2(x[:1500], y[:1500], x[1500:], y[1500:]) > 0.9
    assert A.ridge_r2(x[:1500], rng.normal(size=(1500, 3)), x[1500:], rng.normal(size=(500, 3))) < 0.05
    lab = (x[:, 0] > 0).astype(int) + 2 * (x[:, 1] > 0)
    g = A.logistic_gain(x[:1500], lab[:1500], x[1500:], lab[1500:])
    assert g["gain"] > 0.5 and 0.15 < g["majority"] < 0.4
    grp = A.kmeans_labels(x[:1500], [x[1500:]], k=8)[0]
    assert grp.shape == (500,) and len(np.unique(grp)) > 4
    assert A.verdict(r2_h=0.31) and not A.verdict(r2_h=0.29) and A.verdict(gain_h=0.12) and not A.verdict(gain_h=0.05)


def test_multiscale_has_fast_and_slow_parts():
    fs = 2000.0
    t = np.arange(8000) / fs
    x = np.sin(2 * np.pi * 100 * t)[None, :] * np.ones((3, 1))
    x[:, 4400:4500] += np.sin(2 * np.pi * 300 * t[4400:4500])  # un evento rapido nel terzo sottoblocco da 50 ms del blocco [4200, 4600)
    ms = A.multiscale_logpower(x, fs, 4200, 4600)
    assert ms.shape == (3, 48)
    fast = ms[0, :24].reshape(4, 6)
    assert fast[2, 5] > fast[0, 5] + 3  # la banda alta (268-450 Hz) si accende solo nel sottoblocco dell'evento
    assert np.isfinite(A.multiscale_logpower(x, fs, 0, 400)).sum() > 0  # al bordo della finestra il tratto da 500 ms si taglia


def test_probe_stage_on_synthetic_units(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("anchor_candidates_script", Path(__file__).resolve().parents[2] / "scripts" / "anchor_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    m = 600
    dims = {"V": 16, "R": 16, "H": 16, "c16": 16, "mom": 2, "ms": 48, "cur": 6, "nrvq": 128}
    syn = {k: rng.normal(size=(m, d)).astype(np.float16) for k, d in dims.items()}
    syn["H"] = (syn["c16"].astype(np.float32) @ rng.normal(size=(16, 16))).astype(np.float16)  # le 16 bande si leggono da H
    syn.update(code=rng.integers(0, 8192, size=m), h_mask=np.ones(m, bool), train_user=np.arange(m) < 450, user=np.array(["u"] * m),
               codebook=rng.normal(size=(8192, 128)).astype(np.float32))
    np.savez(tmp_path / "syn.npz", **syn)
    assert mod.main(["probe", "--units", str(tmp_path / "syn.npz"), "--out", str(tmp_path / "rep.json")]) == 0
    res = json.loads((tmp_path / "rep.json").read_text())["results"]
    assert set(res) == {"a_rvq_group", "b_rvq_vector", "c_16_bands", "d_moments", "g_multiscale", "e_neurorvq_features", "f_spectral_family"}
    assert res["c_16_bands"]["predictable"] and not res["b_rvq_vector"]["predictable"] and not res["a_rvq_group"]["predictable"]
