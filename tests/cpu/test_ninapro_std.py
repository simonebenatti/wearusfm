import dataclasses
import io
import json
import zipfile

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.ninapro_std import DB4, build_montage_metadata, ingest_subject, load_subject, scan_zips


def _mat_bytes(subject, exercise, n, *, freq=2000, sensor="Cometa", lat="r", subject_field=None,
               drop=(), n_ch=12):
    rng = np.random.default_rng(exercise)
    k = {1: 3, 2: 5, 3: 7}[exercise]
    rest = np.zeros((n, 1), dtype=np.uint8)
    rest[n // 2 :, 0] = 1 + np.arange(n - n // 2) % k
    d = {
        "emg": rng.normal(scale=300.0, size=(n, n_ch)).astype(np.float32),
        "stimulus": rest.astype(np.int8), "restimulus": rest, "repetition": np.ones((n, 1), np.int8),
        "rerepetition": np.ones((n, 1), np.uint8),
        "subject": np.array([[subject if subject_field is None else subject_field]], np.uint8),
        "exercise": np.array([[{1: 2, 2: 1, 3: 3}[exercise]]], np.uint8),  # come nei dati reali di DB4
        "frequency": np.array([[freq]], np.uint16), "laterality": np.array([lat]), "sensor": np.array([sensor]),
        "age": np.array([[30]], np.uint8), "gender": np.array(["f"]),
    }
    for k_ in drop:
        d.pop(k_)
    buf = io.BytesIO()
    sio.savemat(buf, d)
    return buf.getvalue()


def _make_zip(root, subject=1, exercises=(1, 2, 3), **kw):
    (root / "DB4").mkdir(parents=True, exist_ok=True)
    path = root / "DB4" / f"s{subject}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for e in exercises:
            zf.writestr(f"s{subject}/S{subject}_E{e}_A1.mat", _mat_bytes(subject, e, 600 + 100 * e, **kw))
    return path


def test_scan_and_load_subject(tmp_path):
    _make_zip(tmp_path, 1)
    _make_zip(tmp_path, 3)
    assert set(scan_zips(tmp_path, DB4)) == {1, 3}
    ex = load_subject(scan_zips(tmp_path, DB4)[1], 1, DB4)
    assert [e.exercise for e in ex] == [1, 2, 3] and ex[0].emg.shape == (700, 12)
    assert [e.exercise_field_in_file for e in ex] == [2, 1, 3] and [e.n_movements for e in ex] == [3, 5, 7]
    assert set(ex[0].labels) == {"stimulus", "restimulus", "repetition", "rerepetition"}


def test_load_rejects_inconsistencies(tmp_path):
    with pytest.raises(ValueError, match="esercizi"):
        load_subject(_make_zip(tmp_path, 1, exercises=(1, 2)), 1, DB4)
    with pytest.raises(ValueError, match="frequenza"):
        load_subject(_make_zip(tmp_path, 2, freq=200), 2, DB4)
    with pytest.raises(ValueError, match="sensore"):
        load_subject(_make_zip(tmp_path, 3, sensor="Delsys"), 3, DB4)
    with pytest.raises(ValueError, match="colonne"):
        load_subject(_make_zip(tmp_path, 4, n_ch=16), 4, DB4)
    with pytest.raises(ValueError, match="soggetto"):
        load_subject(_make_zip(tmp_path, 5), 6, DB4)


def test_optional_fields_may_be_absent(tmp_path):
    """DB2/DB3 non hanno sensor/frequency/laterality nei metadati: il parser non deve pretenderli."""
    p = _make_zip(tmp_path, 1, drop=("sensor", "frequency", "laterality", "exercise", "subject"))
    ex = load_subject(p, 1, DB4)
    assert ex[0].laterality == "" and ex[0].exercise_field_in_file is None and ex[0].subject_field_in_file is None


def test_montage_ring8_plus_four_targeted_muscles():
    m = build_montage_metadata(DB4, 1, "r")
    ring, targ = m.groups
    assert ring.topology.value == "ring" and ring.symmetry == "D_8" and len(ring.channels) == 8
    assert targ.topology.value == "sparse" and targ.symmetry == "none"
    assert [c.anatomical_identity.muscle for c in targ.channels] == ["FDS", "EDC", "BB", "TB"]
    assert [c.sensor_coords.channel_index for c in targ.channels] == [8, 9, 10, 11]
    assert ring.channels[2].sensor_coords.ring_angle_deg == 90.0
    assert ring.channels[0].anatomical_identity.precision.value == "unknown"
    assert not any(c.anatomical_identity.nominal for g in m.groups for c in g.channels)


def test_amputee_subjects_get_nominal_anatomy():
    cfg = dataclasses.replace(DB4, amputee_subjects=frozenset({7}))
    m = build_montage_metadata(cfg, 7, "l")
    assert all(c.anatomical_identity.nominal for g in m.groups for c in g.channels)
    assert {c.chirality.value for g in m.groups for c in g.channels} == {"left"}


def test_ingest_subject_writes_everything_and_records_anomalies(tmp_path):
    zp = _make_zip(tmp_path / "raw", 2, subject_field=11)
    res = ingest_subject(zp, tmp_path / "out", 2, DB4)
    d = tmp_path / "out" / "s02" / "session1"
    data = np.load(d / "data_int16.npy")
    assert data.shape == (700 + 800 + 900, 12) and data.dtype == np.int16
    meta = json.loads((d / "metadata.json").read_text())
    assert [t["offset"] for t in meta["trials"]] == [0, 700, 1500] and meta["native_fs_hz"] == 2000.0
    assert len([c for g in meta["montage"]["groups"] for c in g["channels"]]) == 12
    assert meta["label_fields"] == ["repetition", "rerepetition", "restimulus", "stimulus"]
    assert set(np.load(d / "labels.npz").files) == set(meta["label_fields"])
    assert res["subject_field_matches_filename"] is False and res["exercise_field_matches_filename"] is False
    assert res["movements_per_exercise"] == [3, 5, 7] and res["n_channels_discarded"] == 0
    rec = data.astype(np.float64) / meta["int16_scale"]
    assert abs(rec.std() - 300.0) < 5.0
