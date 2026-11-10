"""TEMPONet su emg2pose (`docs/temponet_emg2pose_plan.md`): protocollo ufficiale contro reimplementazioni letterali del codice di emg2pose,
istanti, scala dell'ingresso, blocchi di training, EMA e alpha contro quelle di DB8, sintesi per utente, cancello, passo 0 su un tar sintetico."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wearusfm.harness import db8_regression as db8
from wearusfm.harness import emg2pose_regression as R
from wearusfm.ingest import emg2pose_pose as PP

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import emg2pose_eval_targets  # noqa: E402
from test_ingest_emg2pose_pose import _csv, _hdf5_bytes, _tar  # noqa: E402


# --- riferimenti letterali del codice di emg2pose (letto il 10/11/2026) ----------------------------------------------------------------

def _ref_get_contiguous_ones(binary_vector):
    if (binary_vector == 0).all():
        return []
    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [(ones[i], ones[j]) for i, j in zip(np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1))]


def _ref_windows(no_ik_failure, window_length, stride):
    """WindowedEmgDataset(window_length, stride, skip_ik_failures=True): blocchi, finestre per blocco, offset (precompute_windows)."""
    blocks = [(t0, t1) for t0, t1 in _ref_get_contiguous_ones(no_ik_failure) if (t1 - t0) >= window_length]
    lens = [(b[1] - b[0] - window_length) // stride + 1 for b in blocks]
    cumsum = np.cumsum([0] + lens)
    out = []
    for idx in range(int(cumsum[-1])):
        block_idx = np.searchsorted(cumsum, idx, "right") - 1
        start_idx, _ = blocks[block_idx]
        out.append(start_idx + (idx - cumsum[block_idx]) * stride)
    return out


def test_contiguous_ones_matches_emg2pose_and_keeps_the_end_inclusive():
    assert R.contiguous_ones(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert R.contiguous_ones(np.zeros(5)) == [] and R.contiguous_ones(np.ones(4)) == [(0, 3)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = rng.random(rng.integers(1, 80)) < rng.random()
        assert R.contiguous_ones(b) == [(int(a), int(c)) for a, c in _ref_get_contiguous_ones(b)]


def test_no_ik_failure_is_the_all_zero_row_with_isclose():
    a = np.ones((5, 20), np.float32)
    a[1] = 0.0
    a[2] = 0.0
    a[2, 7] = 1e-9  # dentro la tolleranza di np.isclose: ancora un fallimento
    a[3] = 0.0
    a[3, 0] = 1e-7  # fuori tolleranza: valido
    assert R.no_ik_failure(a).tolist() == [True, False, False, True, True]


def test_official_windows_match_the_windowed_dataset_of_emg2pose():
    rng = np.random.default_rng(1)
    for _ in range(300):
        mask = rng.random(rng.integers(1, 400)) < 0.97
        mask[rng.integers(0, mask.size, size=rng.integers(0, 4))] = False
        got = R.official_window_starts(mask, predicted=40, context=7, stride=40)
        assert got.tolist() == [int(o) for o in _ref_windows(mask, 47, 40)]
    run = np.ones(10_255, bool)  # t1 - t0 = 10.254 < 10.255: nessuna finestra
    assert R.official_window_starts(run).size == 0
    assert R.official_window_starts(np.ones(10_256, bool)).tolist() == [0]
    assert R.official_window_starts(np.ones(30_256, bool)).tolist() == [0, 10_000, 20_000]


def test_eval_instants_have_the_context_inside_the_window():
    inst = R.eval_instants(np.array([0, 10_000]))
    assert inst.shape == (2, 1000) and inst[0, 0] == 255 and inst[0, -1] == 255 + 9_990 and inst[1, 0] == 10_255
    assert (inst[:, -1] < np.array([0, 10_000]) + 10_255).all()  # dentro la finestra effettiva


def test_scale_exponent_rule():
    assert R.scale_exponent(0.0123) == 3 and R.scale_exponent(12.5) == 0 and R.scale_exponent(31.0) == -1
    assert R.scale_exponent(3.0) == 1  # 3 e 30 entrambi ammessi: vince 30, piu' vicino a 10 in scala logaritmica
    with pytest.raises(ValueError):
        R.scale_exponent(0.0)


def test_train_blocks_need_history_validity_and_the_stride():
    valid = np.ones(100, bool)
    valid[40] = False
    j = R.train_blocks(valid, n_samples=1000, stride_blocks=8)
    assert j.tolist() == [32, 48, 56, 64, 72, 80, 88, 96]  # 0-24 senza storia (10 j + 9 < 255), 40 non valido
    assert (R.block_end(j) >= 255).all() and R.train_blocks(valid, n_samples=900, stride_blocks=8).max() == 88


def test_ema_and_alpha_match_the_db8_versions():
    import torch

    rng = np.random.default_rng(2)
    pred = rng.normal(size=(6, 30, 4))
    target = pred + rng.normal(scale=0.5, size=pred.shape)
    alpha = np.array([0.0, 0.3, 0.9, 0.99])
    got = R.ema_windows(torch.tensor(pred), torch.tensor(alpha)).numpy()
    for w in range(6):
        np.testing.assert_allclose(got[w], db8.ema(pred[w], alpha), rtol=1e-12, atol=1e-12)
    grid = np.round(np.arange(0.0, 1.0, 0.05), 2)
    a_got, mae_got = R.tune_alpha_windows(torch.tensor(pred), torch.tensor(target), grid)
    a_ref, mae_ref = db8.tune_alpha([(pred[w], target[w]) for w in range(6)], grid)
    np.testing.assert_allclose(mae_got, mae_ref, rtol=1e-10)
    assert a_got.tolist() == a_ref.tolist()


def test_user_summary_gate_and_reading():
    s = R.summarize_users({"a": (np.deg2rad(10.0) * 4, 4), "b": (np.deg2rad(14.0) * 1, 1)})
    assert s["mean_over_users_deg"] == pytest.approx(12.0) and s["pooled_deg"] == pytest.approx(10.8)
    assert s["sd_over_users_deg"] == pytest.approx(np.std([10.0, 14.0], ddof=1))
    g = R.gate({"a": (np.deg2rad(8.9) * 2, 2)}, {"a": (np.deg2rad(10.0) * 2, 2)})
    assert g["passed"] and g["ratio"] == pytest.approx(0.89)
    assert not R.gate({"a": (np.deg2rad(9.0), 1)}, {"a": (np.deg2rad(10.0), 1)})["passed"]  # 0,9 non e' < 0,9
    assert R.reading(12.0, "user") == "sotto vemg2pose" and R.reading(12.2, "user") == "fra vemg2pose e NeuroPose"
    assert R.reading(18.0, "stage") == "fra NeuroPose e SensingDynamics" and R.reading(19.0, "user_stage") == "sopra SensingDynamics"


# --- passo 0 su dati sintetici ----------------------------------------------------------------------------------------------------------

def _angles(n, zero_runs=(), seed=0):
    t = np.arange(n) / 2000.0
    rng = np.random.default_rng(seed)
    a = (0.5 * np.sin(2 * np.pi * (0.3 + 0.05 * np.arange(20))[None, :] * t[:, None] + rng.normal(size=20)) + 0.7).astype(np.float32)
    for s, e in zero_runs:
        a[s:e] = 0.0
    return a


def _synthetic(tmp_path, specs):
    """specs: [(stem, user, split, generalization, n, zero_runs)] -> tar, CSV, processato e pose a 200 Hz."""
    recs, rows = {}, []
    proc, pose = tmp_path / "proc", tmp_path / "pose"
    rng = np.random.default_rng(5)
    for stem, user, split, gen, n, zr in specs:
        ang = _angles(n, zr, seed=len(recs))
        recs[stem] = _hdf5_bytes(ang)
        rows.append({"user": user, "filename": stem, "split": split, "generalization": gen, "side": "left", "moving_hand": "left",
                     "stage": "s1", "session": stem})
        d = proc / f"u{user}" / stem
        d.mkdir(parents=True)
        emg = rng.normal(scale=0.02, size=(n, 16))
        scale = 32767 / np.abs(emg).max()
        np.save(d / "data_int16.npy", np.round(emg * scale).astype(np.int16))
        (d / "metadata.json").write_text(json.dumps({"shape": [n, 16], "int16_scale": scale}))
        p = pose / f"u{user}" / stem
        p.mkdir(parents=True)
        blocks, valid = PP.block_reduce(ang, ~PP.invalid_rows(ang, 0))
        np.save(p / "pose_200hz_f16.npy", blocks)
        np.save(p / "pose_valid.npy", valid)
    tar = _tar(tmp_path / "e.tar", recs)
    csv = _csv(tmp_path / "meta.csv", rows)
    return tar, csv, proc, pose


def test_eval_targets_end_to_end(tmp_path):
    specs = [("v1", "uval", "val", "user", 25_000, [(12_000, 12_050)]),  # tratti 0-11.999 (1 finestra) e 12.050-24.999 (1 finestra)
             ("t1", "utest", "test", "user", 21_000, []),  # 0-20.999: finestre a 0 e 10.000
             ("t2", "utrain", "test", "stage", 9_000, []),  # troppo corta: nessuna finestra
             ("tr1", "utrain", "train", "none", 6_000, []), ("tr2", "utrain", "train", "none", 6_000, [])]
    tar, csv, proc, pose = _synthetic(tmp_path, specs)
    out = tmp_path / "out"
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out), "--n-scale", "2"]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["complete"] and not m["problems"] and set(m["recordings"]) == {"v1", "t1", "t2"}
    assert m["totals"]["val/user"]["windows"] == 2 and m["totals"]["test/user"]["windows"] == 2
    assert m["totals"]["test/stage"]["recordings_without_windows"] == 1
    z = np.load(out / "test" / "uutest" / "t1.npz")
    assert z["starts"].tolist() == [0, 10_000] and z["targets"].shape == (2, 1000, 20)
    ang = _angles(21_000, (), seed=1)
    np.testing.assert_array_equal(z["targets"], ang[R.eval_instants(z["starts"])])
    v = np.load(out / "val" / "uuval" / "v1.npz")
    assert v["starts"].tolist() == [0, 12_050]
    assert m["block_check"]["max_rad"] < 0.01  # medie di 10 campioni di un segnale lento, allineate per indice
    assert 3 <= m["scale"]["median_channel_sd_scaled"] <= 30 and m["scale"]["exponent_k"] == 3  # deviazione 0,02 -> 20
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out)]) == 2  # cartella non vuota


def test_eval_targets_flags_misalignment(tmp_path):
    tar, csv, proc, pose = _synthetic(tmp_path, [("t1", "u1", "test", "user", 12_000, []), ("tr", "u2", "train", "none", 3_000, [])])
    meta = proc / "uu1" / "t1" / "metadata.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "shape": [11_999, 16]}))
    out = tmp_path / "out"
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out), "--n-scale", "1"]) == 1
    assert "processato" in json.loads((out / "manifest.json").read_text())["problems"][0]
