"""Regressione DB8 per la replica di TEMPONet: matrice DoF->DoA, finestre, EMA, scelta di alpha, metriche e cache."""

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.harness import db8_regression as db8


def test_dof_to_doa_matches_krasoulis_s2():
    a = db8.DOF_TO_DOA
    assert a.shape == (18, 5)
    assert np.count_nonzero(a) == 13
    assert a[0, 0] == 0.639 and a[1, 0] == 0.383 and a[3, 0] == -0.639 and a[16, 0] == -0.19
    assert a[2, 1] == 1.0
    assert (a[4, 2], a[5, 2]) == (0.4, 0.6) and (a[6, 3], a[7, 3]) == (0.4, 0.6)
    assert (a[9, 4], a[10, 4], a[12, 4], a[13, 4]) == (0.1667, 0.3333, 0.1667, 0.3333)
    # le DoA di flessione sono medie pesate dei loro DoF (pesi a somma 1)
    np.testing.assert_allclose(a[:, 1:].sum(axis=0), 1.0)
    # nessun peso sui DoF 9, 12, 15, 16, 18 (indici 8, 11, 14, 15, 17)
    assert not a[[8, 11, 14, 15, 17]].any()


def test_glove_to_doa():
    g = np.zeros((2, 18))
    g[0, 2] = 30.0
    g[1, 4], g[1, 5] = 10.0, 20.0
    np.testing.assert_allclose(db8.glove_to_doa(g), [[0, 30, 0, 0, 0], [0, 0, 16, 0, 0]])
    with pytest.raises(ValueError):
        db8.glove_to_doa(np.zeros((3, 17)))


def test_window_ends():
    np.testing.assert_array_equal(db8.window_ends(300), [255, 287])
    assert db8.window_ends(255).size == 0
    np.testing.assert_array_equal(db8.window_ends(256), [255])


def test_ema_recursion_and_identity():
    y = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 0.0]])
    np.testing.assert_array_equal(db8.ema(y, 0.0), y)
    out = db8.ema(y, 0.5)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.5])  # y'_0 = y_0, poi 0,5 y'_{t-1} + 0,5 y_t
    np.testing.assert_allclose(out[:, 1], [4.0, 4.0, 2.0])
    per_col = db8.ema(y, np.array([0.0, 0.5]))
    np.testing.assert_allclose(per_col[:, 0], y[:, 0])
    np.testing.assert_allclose(per_col[:, 1], out[:, 1])
    with pytest.raises(ValueError):
        db8.ema(y, 1.0)


def test_tune_alpha_prefers_smoothing_for_noisy_predictions_and_zero_for_exact():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 4000)
    target = np.stack([np.sin(t), np.cos(0.5 * t)], axis=1) * 30
    noisy = target + rng.normal(0, 5, size=target.shape)
    alpha, mae = db8.tune_alpha([(noisy, target)])
    assert mae.shape == (len(db8.ALPHA_GRID), 2)
    assert (alpha > 0.3).all()
    alpha0, _ = db8.tune_alpha([(target.copy(), target)])
    np.testing.assert_array_equal(alpha0, [0.0, 0.0])  # pareggi: vince l'alpha piu' piccolo


def test_tune_alpha_restarts_ema_per_segment():
    seg1 = (np.full((50, 1), 10.0), np.full((50, 1), 10.0))
    seg2 = (np.full((50, 1), -10.0), np.full((50, 1), -10.0))
    alpha, mae = db8.tune_alpha([seg1, seg2])
    # ripartendo a ogni segmento l'EMA di una costante e' esatta per ogni alpha
    np.testing.assert_allclose(mae, 0.0)
    assert alpha[0] == 0.0


def test_metrics():
    target = np.zeros((4, 5))
    pred = np.zeros((4, 5))
    pred[0] += 9.0  # MAE istantaneo 9
    pred[1] += 10.0  # 10: non sotto la tolleranza
    pred[2] += 14.0
    pred[3] += 20.0
    assert db8.tolerance_accuracy(pred, target, 10.0) == 0.25
    assert db8.tolerance_accuracy(pred, target, 15.0) == 0.75
    ev = db8.evaluate(pred, target)
    assert ev["mae_deg"] == pytest.approx(13.25)
    assert ev["acc_10deg"] == 0.25 and ev["acc_15deg"] == 0.75 and ev["n_instants"] == 4
    assert len(ev["mae_per_doa_deg"]) == 5


def _fake_mat(path, n=1000, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    emg = (rng.normal(0, 5e-5, size=(n, 16))).astype(np.float32)
    if nan:
        emg[3, 2] = np.nan
    glove = rng.uniform(0, 90, size=(n, 18)).astype(np.float32)
    lab = np.zeros((n, 1), dtype=np.int8)
    lab[n // 2 :] = 3
    rep = np.zeros((n, 1), dtype=np.int8)
    rep[n // 2 :] = 1
    sio.savemat(path, {"emg": emg, "glove": glove, "restimulus": lab, "rerepetition": rep, "acc": np.zeros((n, 48))})
    return emg, glove


def test_read_raw_cache_roundtrip(tmp_path):
    (tmp_path / "DB8").mkdir()
    src = db8.acquisition_path(tmp_path, 2, 3)
    assert src.name == "S2_E1_A3.mat"
    emg, glove = _fake_mat(src)
    raw = db8.read_raw(src)
    assert raw["restimulus"].shape == (1000,) and raw["emg"].shape == (1000, 16)
    arrays = db8.to_cache_arrays(raw)
    for name, arr in arrays.items():
        np.save(db8.cache_file(tmp_path, 2, 3, name), arr)
    rec = db8.load_cached(tmp_path, 2, 3)
    np.testing.assert_array_equal(rec.emg, emg)
    np.testing.assert_allclose(rec.doa, db8.glove_to_doa(glove), rtol=1e-6)
    assert rec.n_samples == 1000
    with pytest.raises(ValueError):
        db8.cache_file(tmp_path, 13, 1, "emg")


def test_non_finite_values_are_rejected(tmp_path):
    (tmp_path / "DB8").mkdir()
    src = db8.acquisition_path(tmp_path, 1, 1)
    _fake_mat(src, nan=True)
    with pytest.raises(ValueError, match="non finiti"):
        db8.to_cache_arrays(db8.read_raw(src))
    with pytest.raises(ValueError, match="coda"):
        db8.trim_raw(db8.read_raw(src))  # NaN interno: non e' una coda


def _raw(n=100):
    return {"emg": np.zeros((n, 16)), "glove": np.zeros((n, 18)), "restimulus": np.zeros(n), "rerepetition": np.zeros(n)}


def test_trailing_non_finite_tail_is_trimmed():
    raw = _raw()
    raw["glove"][-1, :] = np.nan  # come S5_E1_A1: guanto NaN nell'ultimo campione
    out, trimmed = db8.trim_raw(raw)
    assert trimmed == 1 and all(v.shape[0] == 99 for v in out.values())
    db8.to_cache_arrays(out)  # nessun errore dopo il taglio
    clean, zero = db8.trim_raw(_raw())
    assert zero == 0 and clean["emg"].shape[0] == 100
    raw = _raw()
    raw["emg"][97:, 3] = np.inf
    raw["glove"][98, 0] = np.nan
    assert db8.trim_raw(raw)[1] == 3


def test_long_or_interrupted_tails_are_errors():
    raw = _raw(5000)
    raw["glove"][-2500:, 0] = np.nan  # piu' di un secondo
    with pytest.raises(ValueError, match="oltre"):
        db8.finite_length(raw)
    raw = _raw()
    raw["glove"][[95, 97, 98, 99], 0] = np.nan  # buco a 96: non e' una coda contigua
    with pytest.raises(ValueError, match="coda"):
        db8.finite_length(raw)
