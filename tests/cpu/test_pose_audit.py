"""Audit della catena dei bersagli di posa su dati costruiti con geometria reale (`scripts/pose_audit.py`, controllo 1 della diagnosi GPT del
27/10/2026): identita' e ricostruzione dal tar, oracolo delle patch, ridge da log-RMS che batte il costante quando l'EMG porta la posa, scansione
del ritardo a zero quando EMG e posa sono allineati e NON a zero quando la posa e' spostata; split temporale fit / separazione / test."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ingest_emg2pose_pose import _hdf5_bytes, _tar  # noqa: E402
from test_pose_chain_oracle import FS, M, _ingest  # noqa: E402
from test_pose_targets import _stats  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.ingest import emg2pose  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "pose_audit.py"
N = 60000  # 30 s: fino a 7 finestre da 4 s per registrazione (fit 60% / separazione / test con almeno 2 finestre)


def _pose_and_emg(n, seed, shift_blocks=0):
    """Posa lenta per giunto; EMG con l'ampiezza del canale c modulata dal giunto c (log-RMS lineare nella posa): la ridge deve batterla.
    `shift_blocks` sposta la POSA scritta nell'albero rispetto all'EMG (posa disallineata di proposito)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    pose = (0.6 * np.sin(2 * np.pi * (0.25 + 0.05 * np.arange(20))[None, :] * t[:, None] + rng.normal(size=20)[None, :]) + 1.0).astype(np.float32)
    amp = 0.3 + 0.7 * (pose[:, :16] - 0.4) / 1.2  # (n, 16) in [0,3; 1,0]
    emg = rng.normal(size=(n, 16)) * amp * 3.0
    pose_written = np.roll(pose, shift_blocks * PT.BLOCK, axis=0) if shift_blocks else pose
    pose_written[5000:5300] = 0.0  # un blocco di zeri (fallimento della cinematica inversa)
    return pose_written, emg


def _setup(tmp_path, shift_blocks=0):
    root, pose_root = tmp_path / "processed" / "emg2pose", tmp_path / "pose"
    rows, recs = [], {}
    specs = [("1", "sA", "A", "left", 1), ("1", "sB", "A", "left", 2), ("2", "sC", "A", "right", 3), ("2", "sD", "B", "right", 4)]
    for user, sess, stage, side, seed in specs:
        stem = f"2021-01-0{seed}-1-h{seed}-cv-emg-pose-{stage}@2-recording-0_{side}"
        pose, emg = _pose_and_emg(N, seed, shift_blocks)
        m = emg2pose.build_montage_metadata(user, stem, side)
        d = root / f"u{user}" / stem
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "data_int16.npy", np.round(emg * 1000).astype(np.int16))
        (d / "metadata.json").write_text(json.dumps({"montage": montage_to_dict(m, [True] * 16), "native_fs_hz": FS, "shape": [N, 16],
                                                     "int16_scale": 1000.0, "csv_row": {"side": side, "moving_hand": "both", "user": user}}))
        _ingest(pose_root, user, stem, pose, side)
        recs[stem] = _hdf5_bytes(pose)
        rows.append({"session": sess, "user": user, "stage": stage, "start": 0, "end": 20, "side": side, "filename": stem + ".hdf5",
                     "moving_hand": "both", "held_out_user": False, "held_out_stage": False, "split": "train", "generalization": "user"})
    _stats(pose_root, mean=[1.0] * 20, std=[0.42] * 20)
    csvp = tmp_path / "meta.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tar = _tar(tmp_path / "emg2pose_dataset.tar", recs)
    return root, pose_root, csvp, tar


def _run(tmp_path, root, pose_root, csvp, tar, out):
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, str(SCRIPT), "--csv", str(csvp), "--processed-root", str(root), "--pose-root", str(pose_root), "--tar", str(tar),
           "--n-users", "2", "--n-recordings", "4", "--n-windows", "5", "--context", "1", "--lags", "6", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text())


def test_audit_confirms_an_aligned_chain_and_the_ridge_beats_the_constant(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path)
    rep = _run(tmp_path, root, pose_root, csvp, tar, tmp_path / "audit.json")
    s = rep["sintesi"]
    assert rep["n_recordings"] == 4 and rep["tar"]["found"] == 4 and rep["tar"]["missing"] == 0
    assert s["identita_ok_tutte"] and s["ricostruzione_ok_tutte"] and s["n_ricostruite"] == 4 and s["oracolo_ok_tutte"] and s["n_finestre_oracolo"] >= 12
    assert sum(s["blocchi_validi_per_patch_istogramma_0_5"]) == s["n_finestre_oracolo"] * 160 and s["blocchi_validi_per_patch_istogramma_0_5"][5] > 0
    for u, v in rep["ridge"].items():
        assert v["n_fit"] >= 2 and v["n_gap"] >= 1 and v["n_test"] >= 1 and v["n_fit"] + v["n_gap"] + v["n_test"] <= v["n_windows"]
        assert v["ratio_all"] < 0.8 and v["lag_scan"]["best_lag"] == 0 and v["misaligned"]["ratio_vs_const"] > v["ratio_all"]
    assert s["lag_zero_per_tutti"] and s["ridge_ratio_all_mediana_utenti"] < 0.8
    per = {r["stem"]: r for r in rep["per_recording"]}
    assert all(r["identita"]["side_emg"] and r["identita"]["margin"] for r in per.values())
    assert all(r["frac_valid_blocks"] < 1.0 for r in per.values())  # il blocco di zeri col margine M toglie qualcosa ovunque


def test_audit_detects_a_pose_shifted_against_the_emg(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path, shift_blocks=40)  # posa spostata di 40 blocchi = 8 patch = 400 ms
    rep = _run(tmp_path, root, pose_root, csvp, tar, tmp_path / "audit_shift.json")
    s = rep["sintesi"]
    assert s["identita_ok_tutte"] and s["ricostruzione_ok_tutte"] and s["oracolo_ok_tutte"]  # la catena e' coerente con l'albero: il difetto e' nei dati
    assert not s["lag_zero_per_tutti"] and all(abs(k) >= 4 for k in s["lag_migliore_per_utente"])  # ma lo spostamento si vede nel ritardo
