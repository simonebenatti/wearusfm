import io
import json
import zipfile

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.ninapro_db5 import (
    N_CHANNELS,
    build_montage_metadata,
    ingest_subject,
    load_subject,
    scan_db5,
)


def _mat_bytes(subject, exercise, n, *, freq=200.0, sensor="Double Myo", lat="r", rng=None):
    rng = rng or np.random.default_rng(exercise)
    emg = np.round(rng.normal(scale=8.0, size=(n, N_CHANNELS))).clip(-128, 127).astype(np.float32)
    rest = np.zeros((n, 1), dtype=np.int8)
    rest[n // 2 :] = exercise  # meta' riposo, meta' movimento
    d = {
        "emg": emg, "acc": np.zeros((n, 3), np.float32), "glove": np.zeros((n, 22), np.float32),
        "stimulus": rest.copy(), "restimulus": rest.copy(),
        "repetition": np.ones((n, 1), np.int8), "rerepetition": np.ones((n, 1), np.int8),
        "subject": np.array([[subject]], float), "exercise": np.array([[exercise]], float),
        "frequency": np.array([[freq]], float), "laterality": np.array([lat]), "sensor": np.array([sensor]),
        "age": np.array([[23.0]]), "gender": np.array(["m"]), "height": np.array([[187.0]]),
    }
    buf = io.BytesIO()
    sio.savemat(buf, d)
    return buf.getvalue()


def _make_zip(root, subject=1, exercises=(1, 2, 3), **kw):
    (root / "DB5").mkdir(parents=True, exist_ok=True)
    path = root / "DB5" / f"s{subject}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for e in exercises:
            zf.writestr(f"s{subject}/S{subject}_E{e}_A1.mat", _mat_bytes(subject, e, 400 + 50 * e, **kw))
    return path


def test_scan_and_load_subject_orders_exercises(tmp_path):
    _make_zip(tmp_path, 1)
    _make_zip(tmp_path, 2)
    assert set(scan_db5(tmp_path)) == {1, 2}
    ex = load_subject(scan_db5(tmp_path)[1], 1)
    assert [e.exercise for e in ex] == [1, 2, 3]
    assert ex[0].emg.shape == (450, 16) and ex[0].labels["restimulus"].shape == (450,)
    assert ex[0].laterality == "r"


def test_load_subject_rejects_inconsistencies(tmp_path):
    with pytest.raises(ValueError, match="esercizi"):
        load_subject(_make_zip(tmp_path, 1, exercises=(1, 2)), 1)
    with pytest.raises(ValueError, match="frequenza"):
        load_subject(_make_zip(tmp_path, 2, freq=2000.0), 2)
    with pytest.raises(ValueError, match="sensore"):
        load_subject(_make_zip(tmp_path, 3, sensor="Single Myo"), 3)
    p = _make_zip(tmp_path, 4)
    with pytest.raises(ValueError, match="soggetto"):
        load_subject(p, 5)


def test_montage_has_two_rings_of_eight_and_chirality():
    m = build_montage_metadata(3, "l")
    assert [g.group_id for g in m.groups] == ["myo_1", "myo_2"]
    assert all(len(g.channels) == 8 and g.topology.value == "ring" for g in m.groups)
    assert {c.chirality.value for g in m.groups for c in g.channels} == {"left"}
    assert m.groups[1].channels[0].sensor_coords.channel_index == 8


def test_ingest_subject_writes_data_labels_and_sidecar(tmp_path):
    zip_path = _make_zip(tmp_path / "raw", 7)
    out = tmp_path / "out"
    res = ingest_subject(zip_path, out, 7)
    d = out / "s07" / "session1"
    data = np.load(d / "data_int16.npy")
    assert data.shape == (450 + 500 + 550, 16) and data.dtype == np.int16
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["native_fs_hz"] == 200.0 and [t["exercise"] for t in meta["trials"]] == [1, 2, 3]
    assert [t["offset"] for t in meta["trials"]] == [0, 450, 950]
    flags = [c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]]
    assert len(flags) == 16
    lab = np.load(d / "labels.npz")
    assert set(lab.files) == {"stimulus", "restimulus", "repetition", "rerepetition"} and len(lab["restimulus"]) == len(data)
    # ricostruzione fedele: i valori 8-bit interi sopravvivono alla quantizzazione
    rec = data.astype(np.float64) / meta["int16_scale"]
    assert np.allclose(rec, np.round(rec)) and abs(res["rest_fraction_restimulus"] - 0.5) < 0.01
