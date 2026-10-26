"""Ingest della posa di emg2pose (seconda ondata del pilot, `docs/pilot_posa_v1.md` §2 e §6): lettura dei soli angoli, righe a zero e margine,
blocchi a 200 Hz allineati per indice, profilo dei bordi e minimi simultanei, selezione del collaudo, statistiche per lato e insieme, driver."""

import csv
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from wearusfm.ingest import emg2pose_pose as PP

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ingest_emg2pose_pose.py"


def _angles(n=3000, seed=0, zero_runs=(), nan_rows=(), v_dip_at=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 2000.0
    a = 0.8 * np.sin(2 * np.pi * (0.5 + np.arange(20) * 0.05)[None, :] * t[:, None] + rng.normal(size=20)[None, :]) + 1.0
    a = a.astype(np.float32)
    for s, e in zero_runs:
        a[s:e] = 0.0
    for r in nan_rows:
        a[r, 3] = np.nan
    if v_dip_at is not None:  # una «V» verso lo zero larga un frame, senza righe esattamente nulle
        half = 17
        for k in range(-half, half + 1):
            a[v_dip_at + k] *= abs(k) / half if k else 1e-3
    return a


def _hdf5_bytes(angles, *, rate=2000.0, attrs=None):
    n = angles.shape[0]
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    arr = np.zeros(n, dtype=dt)
    arr["time"] = np.arange(n) / rate
    arr["joint_angles"] = angles
    arr["emg"] = np.random.default_rng(1).normal(size=(n, 16)).astype(np.float32)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=arr)
        g.attrs["sample_rate"] = rate
        for k, v in (attrs or {}).items():
            g.attrs[k] = v
    return buf.getvalue()


def _tar(path, recs):
    """recs: {stem: bytes}; i nomi dei membri come nel tar vero."""
    with tarfile.open(path, "w") as tf:
        for stem, data in recs.items():
            info = tarfile.TarInfo(f"emg2pose_data/{stem}.hdf5")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _csv(path, rows):
    cols = ["session", "user", "stage", "start", "end", "side", "filename", "moving_hand", "held_out_user", "held_out_stage", "split", "generalization"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


# --- validita' e blocchi ----------------------------------------------------------------------------------------------------------------

def test_read_pose_reads_only_the_angles_and_checks_the_layout():
    a = _angles(500)
    rec = PP.read_pose(io.BytesIO(_hdf5_bytes(a, attrs={"side": b"left"})))
    assert rec["angles"].shape == (500, 20) and rec["angles"].dtype == np.float32 and rec["attrs"]["side"] == "left"
    assert np.allclose(rec["angles"], a)
    with pytest.raises(ValueError, match="sample_rate"):
        PP.read_pose(io.BytesIO(_hdf5_bytes(a, rate=1000.0)))


def test_zero_rows_margin_and_nonfinite_rows():
    a = _angles(1000, zero_runs=[(100, 140), (600, 601)], nan_rows=[800])
    z = PP.zero_rows(a)
    assert z.sum() == 41 and PP.runs_of(z) == [(100, 140), (600, 601)]
    assert PP.nonfinite_rows(a).sum() == 1
    bad0 = PP.invalid_rows(a, 0)
    assert bad0.sum() == 42 and bad0[800]
    bad = PP.invalid_rows(a, 33)
    # il margine copre 33 campioni per lato di ogni tratto a zero (e la riga non finita)
    assert bad[100 - 33] and not bad[100 - 34] and bad[139 + 33] and not bad[139 + 34] and bad[600 + 33] and bad.sum() == 41 + 66 + 66 + 1
    d = PP.distance_to(z)
    assert d[100] == 0 and d[99] == 1 and d[140] == 1 and d[370] == 230 and d[0] == 100


def test_blocks_at_200hz_are_aligned_by_sample_index_and_valid_only_if_whole():
    a = _angles(1005, zero_runs=[(23, 25)])  # 100 blocchi interi, 5 campioni di resto scartati
    blocks, valid = PP.block_reduce(a, ~PP.invalid_rows(a, 0))
    assert blocks.shape == (100, 20) and blocks.dtype == np.float16 and valid.shape == (100,)
    assert not valid[2] and valid[1] and valid[3] and valid.sum() == 99
    # blocco j = campioni [10 j, 10 j + 10): media esatta sui campioni validi
    assert np.allclose(blocks[5].astype(np.float64), a[50:60].astype(np.float64).mean(axis=0), atol=2e-3)
    # il blocco non valido tiene la media dei suoi 8 campioni validi (mai usata: la maschera lo dice)
    assert np.allclose(blocks[2].astype(np.float64), a[np.r_[20:23, 25:30]].astype(np.float64).mean(axis=0), atol=2e-3)
    # nessun campione valido nel blocco -> 0
    a2 = _angles(100, zero_runs=[(10, 20)])
    b2, v2 = PP.block_reduce(a2, ~PP.invalid_rows(a2, 0))
    assert not v2[1] and float(np.abs(b2[1].astype(np.float64)).max()) == 0.0


# --- misure del collaudo ----------------------------------------------------------------------------------------------------------------

def test_edge_profile_sees_a_ramp_next_to_a_zero_block_and_dips_count_the_v_shapes():
    a = _angles(4000, zero_runs=[(1000, 1100)])
    # rampa lineare di 33 campioni verso lo zero ai due bordi (l'ipotesi dell'interpolazione dopo gli zeri)
    for k in range(1, 34):
        a[1000 - k] *= k / 33
        a[1099 + k] *= k / 33
    prof = PP.edge_profile(a, PP.zero_rows(a))
    assert prof["n_zero_rows"] == 100 and prof["n_zero_runs"] == 1 and len(prof["bins"]) == 40 and prof["bins"][0]["lag_from"] == 1
    near = [b["mean_abs_diff"] for b in prof["bins"][:3]]
    far = prof["far"]["mean"]
    assert all(v is not None and v > 3 * far for v in near)  # dentro il frame la differenza prima e' molto piu' grande che lontano
    assert prof["bins"][10]["mean_abs_diff"] < 1.5 * far  # a 100 campioni il profilo e' tornato al valore lontano
    flat = PP.edge_profile(_angles(1000), np.zeros(1000, dtype=bool))
    assert flat["bins"] == [] and flat["far"]["n"] > 0
    v = _angles(4000, zero_runs=[(1000, 1100)], v_dip_at=3000)
    dips = PP.simultaneous_dips(v, PP.zero_rows(v))
    assert dips["n_dip_runs"] == 1 and 1 <= dips["n_dip_rows"] <= 5
    none = PP.simultaneous_dips(a, PP.zero_rows(a))
    assert none["n_dip_runs"] == 0  # la rampa sta entro due frame dagli zeri: non conta come «V»


def test_collaudo_picker_balances_sides_and_moving_hand():
    pk = PP.CollaudoPicker(6, both_sides=True)
    rows = [{"side": s, "moving_hand": m} for s in ("left", "right") for m in ("left", "right", "both")]
    assert all(pk.accept(r) for r in rows) and pk.full
    assert not pk.accept({"side": "left", "moving_hand": "left"})
    pk2 = PP.CollaudoPicker(4, both_sides=True)
    assert [pk2.accept({"side": "left", "moving_hand": "both"}) for _ in range(3)] == [True, False, False]  # un solo «both» per lato
    assert pk2.accept({"side": "left", "moving_hand": "left"}) and not pk2.accept({"side": "left", "moving_hand": "right"})  # lato pieno (2)
    pk3 = PP.CollaudoPicker(2, both_sides=False)
    assert pk3.accept(None) and pk3.accept(None) and not pk3.accept(None)


# --- una registrazione e le statistiche -----------------------------------------------------------------------------------------------------

def test_ingest_recording_writes_blocks_mask_and_meta(tmp_path):
    a = _angles(2000, zero_runs=[(200, 240)], nan_rows=[900])
    ref = PP.RecordingRef("emg2pose_data/2021-01-01-1-abc-cv-emg-pose-x@2-recording-0_left.hdf5",
                          "2021-01-01-1-abc-cv-emg-pose-x@2-recording-0_left")
    row = {"user": "u1", "side": "left", "moving_hand": "left", "split": "train"}
    s = PP.ingest_pose_recording(io.BytesIO(_hdf5_bytes(a)), ref, tmp_path, row, zero_margin_samples=10)
    d = PP.out_dir_for(tmp_path, "u1", ref.stem)
    blocks, valid, meta = np.load(d / PP.POSE_FILE), np.load(d / PP.VALID_FILE), json.loads((d / PP.META_FILE).read_text())
    assert blocks.shape == (200, 20) and valid.shape == (200,) and s["n_blocks"] == 200
    # zeri 200-239 piu' 10 di margine per lato: blocchi 19..24 non validi; la riga non finita 900 invalida il blocco 90
    assert not valid[19:25].any() and valid[18] and valid[25] and not valid[90] and valid.sum() == 200 - 6 - 1 == s["n_valid_blocks"]
    assert meta["n_zero_rows"] == 40 and meta["n_nonfinite_rows"] == 1 and meta["zero_margin_samples"] == 10 and meta["split"] == "train"
    assert meta["joint_stats_valid_blocks"]["n"] == 193 and len(meta["max_abs_per_joint"]) == 20 and "edge_profile" in meta
    assert meta["joints"][0] == "THUMB_CMC_FE" and meta["joints"][-1] == "PINKY_DIP_FE" and len(meta["joints"]) == 20


def test_aggregate_stats_uses_only_the_pretraining_split_and_applies_the_side_criterion(tmp_path):
    def rec(stem, user, side, split, shift=0.0, scale=1.0, seed=0):
        a = (_angles(1500, seed=seed) - 1.0) * scale + 1.0 + shift
        PP.ingest_pose_recording(io.BytesIO(_hdf5_bytes(a)), PP.RecordingRef(f"x/{stem}.hdf5", stem), tmp_path,
                                 {"user": user, "side": side, "moving_hand": "both", "split": split}, measure=False)
    # stesso contenuto (seme 0) a destra e a sinistra: il criterio per lato non deve scattare
    rec("r1_left", "a", "left", "train")
    rec("r2_left", "b", "left", "train")
    rec("r3_right", "a", "right", "train")
    rec("r4_right", "c", "right", "test", shift=5.0)  # fuori dal pretraining: non entra
    st = PP.aggregate_stats(tmp_path)
    assert st["n_recordings"] == 3 and st["n_recordings_per_side"] == {"left": 2, "right": 1} and st["n_recordings_other_splits_skipped"] == 1
    expected = _angles(1500)[:1500].astype(np.float64).reshape(150, 10, 20).mean(axis=1).mean(axis=0)
    assert st["all"]["n"] == 3 * 150 and len(st["all"]["mean"]) == 20 and np.allclose(st["all"]["mean"], expected, atol=2e-3)
    assert st["side_criterion"]["per_side"] is False and st["side_criterion"]["max_mean_gap_sd"] < 1e-6 and st["side_criterion"]["max_sd_ratio"] < 1.01
    assert (tmp_path / PP.STATS_FILE).exists() and json.loads((tmp_path / PP.STATS_FILE).read_text())["side_criterion"]["per_side"] is False
    # un lato spostato di una deviazione intera fa scattare il criterio
    rec("r5_right", "d", "right", "train", shift=1.0, seed=5)
    rec("r6_right", "e", "right", "train", shift=1.0, seed=6)
    st2 = PP.aggregate_stats(tmp_path)
    assert st2["side_criterion"]["per_side"] is True and st2["side_criterion"]["max_mean_gap_sd"] > 0.5


# --- driver ------------------------------------------------------------------------------------------------------------------------------

def test_driver_collaudo_then_full_ingest_with_skip_and_stats(tmp_path):
    stems = [f"2021-01-0{i}-1-h{i}-cv-emg-pose-s@2-recording-0_{side}" for i, side in enumerate(("left", "right", "left", "right", "left", "right"), 1)]
    recs = {s: _hdf5_bytes(_angles(1200, seed=i, zero_runs=[(300, 340)] if i % 2 else [])) for i, s in enumerate(stems)}
    recs["._junk"] = b"not hdf5"
    tarp = _tar(tmp_path / "pose.tar", recs)
    rows = [{"user": f"u{i % 3}", "side": s.rsplit("_", 1)[1], "moving_hand": ("left", "right", "both")[i % 3], "split": "train" if i < 5 else "test",
             "filename": s + ".hdf5", "start": 0, "end": 0.6} for i, s in enumerate(stems)]
    csvp = _csv(tmp_path / "meta.csv", rows)
    out, rep = tmp_path / "pose_root", tmp_path / "rep.json"
    env = {"PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, str(SCRIPT), "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep),
                        "--limit", "4", "--both-sides"], capture_output=True, text=True, env={**env, "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(rep.read_text())
    assert report["n_recordings"] == 4 and report["stopped"] == "limit" and report["collaudo"]["per_side"] == {"left": 2, "right": 2}
    assert report["collaudo"]["fields_in_first_file"]["timeseries_fields"] == ["time", "joint_angles", "emg"]
    assert report["collaudo"]["edge_profile"]["bins"][0]["n"] > 0 and report["collaudo"]["frac_zero_rows"] > 0
    assert len(list(out.rglob(PP.META_FILE))) == 4
    # ingest intero con margine, ripresa (--skip-existing) e statistiche: le 4 gia' fatte si saltano, le altre 2 si fanno
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep),
                         "--skip-existing", "--zero-margin-samples", "33", "--no-measure", "--stats"], capture_output=True, text=True,
                        env={**env, "PATH": "/usr/bin:/bin"})
    assert r2.returncode == 0, r2.stdout + r2.stderr
    report2 = json.loads(rep.read_text())
    assert report2["n_recordings"] == 2 and report2["n_skipped_existing"] == 4 and report2["params"]["zero_margin_samples"] == 33
    assert report2["stats"]["n_recordings"] == 5 and (out / PP.STATS_FILE).exists()  # 5 di train su 6
    r3 = subprocess.run([sys.executable, str(SCRIPT), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep), "--stats-only"],
                        capture_output=True, text=True, env={**env, "PATH": "/usr/bin:/bin"})
    assert r3.returncode == 0 and json.loads(r3.stdout)["n_recordings"] == 5


def test_launcher_uses_the_frozen_checkout_and_writes_outside_the_repo():
    text = (REPO / "scripts" / "slurm" / "ingest_emg2pose_pose.sbatch").read_text()
    assert "lib_code_dir.sh" in text and "code_dir_setup" in text and 'cd "$WORK/wearusfm"' not in text
    assert "emg2pose_pose_ingest_report_${SLURM_JOB_ID}.json" in text and "$SCRATCH/data/processed/emg2pose_pose" in text
    assert "--time-budget-s 13200" in text and "lrd_all_serial" in text and "--cpus-per-task=2" in text
