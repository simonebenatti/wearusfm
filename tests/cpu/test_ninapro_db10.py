"""Ingest di NinaPro DB10 (MeganePro MDS1) su .mat sintetici nel formato verificato: emg (T, C) single, ts con pause, etichette int8 (con le `re*`)."""

import json

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.data.processed import load_session, validate_session
from wearusfm.ingest import ninapro_db10 as D

FS = 1926.0


def _mat(path, n_cols=12, n=6000, gaps=((2000, 3.0), (4000, 0.5)), seed=0, fs=FS):
    rng = np.random.default_rng(seed)
    ts = 173.0 + np.arange(n) / fs
    for at, extra in gaps:  # pausa: da `at` in poi i tempi saltano di `extra` secondi
        ts[at:] += extra
    lab = np.zeros((n, 1), np.int8)
    lab[n // 3: 2 * n // 3] = 4
    mat = {"ts": ts[:, None], "emg": rng.normal(scale=2e-5, size=(n, n_cols)).astype(np.float32), "acc": np.zeros((n, 3 * n_cols), np.float32)}
    for k in D.LABEL_FIELDS:
        mat[k] = lab
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, mat)
    return path


def test_scan_skips_orig_files(tmp_path):
    d = tmp_path / "DB10" / "MDS1"
    _mat(d / "S010_ex1.mat")
    _mat(d / "S101_ex1.mat")
    (d / "S010_ex1_orig.mat").write_bytes(b"x")
    found = D.scan_db10(tmp_path)
    assert sorted(found) == [10, 101] and found[10].name == "S010_ex1.mat"  # mai il _orig, che non ha EMG


def test_segments_split_at_pauses_and_rate_from_ts():
    ts = np.arange(1000) / FS
    ts[400:] += 2.0
    segs, fs = D.segments_from_ts(ts)
    assert [(s, n) for s, n, _ in segs] == [(0, 400), (400, 600)] and abs(fs - FS) < 1e-6
    with pytest.raises(ValueError, match="crescente"):
        D.segments_from_ts(np.array([0.0, 1.0, 1.0, 2.0]))


def test_montage_able_bodied_rings_with_distal_between_gaps():
    m = D.build_montage_metadata(10, 12)
    prox, dist = m.groups
    assert (prox.topology.value, prox.symmetry, dist.topology.value, dist.symmetry) == ("ring", "D_8", "ring", "D_4")
    assert [c.sensor_coords.ring_angle_deg for c in prox.channels] == [45.0 * k for k in range(8)]
    assert [c.sensor_coords.ring_angle_deg for c in dist.channels] == [22.5, 112.5, 202.5, 292.5]  # fra gli elettrodi 1-2, 3-4, ...
    assert [c.sensor_coords.channel_index for g in m.groups for c in g.channels] == list(range(12))
    assert prox.channels[0].chirality.value == "right" and not prox.channels[0].anatomical_identity.nominal


def test_montage_exceptions_from_the_paper():
    s024 = D.build_montage_metadata(24, 11)  # manca l'elettrodo 8: anello prossimale con le posizioni 1-7
    assert [c.sensor_coords.ring_angle_deg for c in s024.groups[0].channels] == [45.0 * k for k in range(7)]
    assert [c.sensor_coords.channel_index for c in s024.groups[1].channels] == [7, 8, 9, 10]
    s039 = D.build_montage_metadata(39, 11)  # 7 prossimali a spaziatura non dichiarata: sparso
    assert all(g.topology.value == "sparse" for g in s039.groups) and s039.groups[0].channels[0].chirality.value == "right"
    s108 = D.build_montage_metadata(108, 8)  # amputato col solo anello prossimale
    assert [g.group_id for g in s108.groups] == ["proximal"] and s108.groups[0].channels[0].anatomical_identity.nominal
    assert s108.groups[0].channels[0].chirality.value == "unknown"
    with pytest.raises(ValueError, match="colonne"):
        D.build_montage_metadata(10, 11)


def test_ingest_writes_segments_labels_and_reads_back(tmp_path):
    p = _mat(tmp_path / "raw" / "DB10" / "MDS1" / "S115_ex1.mat", n_cols=11)
    r = D.ingest_subject(p, tmp_path / "out", 115)
    assert r["n_segments"] == 3 and r["n_channels"] == 11 and r["amputee"] and r["groups"] == ["proximal", "distal"]
    assert abs(r["longest_gap_s"] - 3.0) < 0.01 and r["n_grasps"] == 1
    out = tmp_path / "out" / "s115" / "ex1"
    meta = json.loads((out / "metadata.json").read_text())
    assert [(t["offset"], t["n_samples"]) for t in meta["trials"]] == [(0, 2000), (2000, 2000), (4000, 2000)]
    assert meta["native_fs_hz"] == 1926.0 and meta["laterality"] == "" and meta["nominal_anatomy"] is True
    s = load_session(out, "ninapro_db10", "s115", "ex1")
    assert [seg.shape for seg in s.segments] == [(2000, 11)] * 3 and s.fs == 1926.0
    assert not validate_session(out)
    assert set(np.load(out / "labels.npz").files) == set(D.LABEL_FIELDS)


def test_ingest_refuses_wrong_rate(tmp_path):
    p = _mat(tmp_path / "S010_ex1.mat", fs=2000.0)
    with pytest.raises(ValueError, match="frequenza"):
        D.ingest_subject(p, tmp_path / "out", 10)


def test_script_collaudo_resume_and_suspect_channel(tmp_path):
    import importlib.util
    from pathlib import Path

    d = tmp_path / "raw" / "DB10" / "MDS1"
    p = _mat(d / "S010_ex1.mat")
    m = sio.loadmat(p)
    m = {k: v for k, v in m.items() if not k.startswith("__")}
    m["emg"][:, 7] *= 0.01  # canale molto piu' debole (come il canale 7 di S010 reale): sospetto, non scartato
    sio.savemat(p, m)
    _mat(d / "S011_ex1.mat", seed=1)
    spec = importlib.util.spec_from_file_location("ingest_ninapro_db10", Path(__file__).resolve().parents[2] / "scripts" / "ingest_ninapro_db10.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--raw-root", str(tmp_path / "raw"), "--out-root", str(tmp_path / "out"), "--report", str(tmp_path / "r.json")]
    assert mod.main(args + ["--subjects", "10"]) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["suspect_low_channels_by_subject"] == {"10": [7]} and r["per_subject"][0]["discarded_channels"] == []
    assert mod.main(args + ["--skip-existing"]) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["subjects_processed"] == [11] and r["n_skipped_existing"] == 1
