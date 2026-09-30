import dataclasses
import io
import json
import zipfile

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.ninapro_std import (
    DB3, DB4, build_montage_metadata, ingest_subject, load_subject, parse_exercise, scan_zips,
)


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
    with pytest.raises(ValueError, match="lunghezze incompatibili"):  # oltre la tolleranza
        parse_exercise(_mat_dict(n=300, label_n=280), 1, DB4)


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


def test_db3_config_volts_amputees_and_wrapper_folder(tmp_path):
    """DB3: zip s<N>_0.zip con cartella DB3_s<N>/ e .DS_Store, emg in volt, solo `subject`/`exercise`."""
    (tmp_path / "DB3").mkdir()
    path = tmp_path / "DB3" / "s3_0.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("DB3_s3/.DS_Store", b"junk")
        for e in (1, 2, 3):
            buf = io.BytesIO()
            emg = rng.normal(scale=2e-5, size=(500, 12)).astype(np.float32)  # volt
            emg[:, 5] = 0.0  # canale morto
            rest = np.zeros((500, 1), np.int8)
            rest[250:, 0] = 1
            sio.savemat(buf, {"emg": emg, "stimulus": rest, "restimulus": rest, "repetition": np.ones((500, 1), np.int8),
                              "rerepetition": np.ones((500, 1), np.int8), "subject": np.array([[3]], np.uint8),
                              "exercise": np.array([[e]], np.uint8)})
            zf.writestr(f"DB3_s3/S3_E{e}_A1.mat", buf.getvalue())
    assert set(scan_zips(tmp_path, DB3)) == {3}
    res = ingest_subject(path, tmp_path / "out", 3, DB3)
    assert res["discarded_channels"] == [5]  # il canale a zero e' scartato; i canali da 2e-5 V no
    assert res["exercise_field_matches_filename"] is True and res["subject_field_matches_filename"] is True
    meta = json.loads((tmp_path / "out" / "s03" / "session1" / "metadata.json").read_text())
    assert meta["nominal_anatomy"] is True
    assert all(c["anatomical_identity"]["nominal"] for g in meta["montage"]["groups"] for c in g["channels"])
    assert meta["montage"]["groups"][0]["channels"][0]["chirality"] == "unknown"  # nessuna `laterality` nei dati
    flags = [c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]]
    assert flags.count(False) == 1 and flags[5] is False


def test_low_amplitude_volt_channels_are_not_flagged_flat(tmp_path):
    """Il vecchio limite assoluto (std < 1e-6) avrebbe scartato un canale valido da 0,5 uV in volt."""
    zp = _make_zip(tmp_path / "raw", 1)
    with zipfile.ZipFile(zp, "w") as zf:
        rng = np.random.default_rng(1)
        for e in (1, 2, 3):
            buf = io.BytesIO()
            emg = rng.normal(scale=2e-5, size=(500, 12)).astype(np.float32)
            emg[:, 3] = rng.normal(scale=5e-7, size=500)  # canale molto quieto ma vivo
            rest = np.zeros((500, 1), np.int8)
            sio.savemat(buf, {"emg": emg, "restimulus": rest, "subject": np.array([[1]], np.uint8),
                              "exercise": np.array([[e]], np.uint8), "frequency": np.array([[2000]], np.uint16),
                              "sensor": np.array(["Cometa"]), "laterality": np.array(["r"])})
            zf.writestr(f"s1/S1_E{e}_A1.mat", buf.getvalue())
    res = ingest_subject(zp, tmp_path / "out", 1, DB4)
    assert res["discarded_channels"] == []


def test_db2_zip_naming_and_intact_subjects(tmp_path):
    (tmp_path / "DB2").mkdir()
    path = tmp_path / "DB2" / "DB2_s7.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for e in (1, 2, 3):
            buf = io.BytesIO()
            rest = np.zeros((400, 1), np.int8)
            rest[200:, 0] = 1
            sio.savemat(buf, {"emg": rng.normal(scale=1e-5, size=(400, 12)).astype(np.float32), "restimulus": rest,
                              "stimulus": rest, "repetition": rest, "rerepetition": rest,
                              "subject": np.array([[7]], np.uint8), "exercise": np.array([[e]], np.uint8)})
            zf.writestr(f"DB2_s7/S7_E{e}_A1.mat", buf.getvalue())
    from wearusfm.ingest.ninapro_std import DB2

    assert set(scan_zips(tmp_path, DB2)) == {7}
    res = ingest_subject(path, tmp_path / "out", 7, DB2)
    meta = json.loads((tmp_path / "out" / "s07" / "session1" / "metadata.json").read_text())
    assert meta["nominal_anatomy"] is False and res["n_channels_discarded"] == 0
    assert meta["montage"]["subject_id"] == "ninapro_db2_s07"


def _mat_dict(n=300, label_n=None):
    label_n = n if label_n is None else label_n
    return {
        "emg": np.random.default_rng(0).normal(size=(n, 12)).astype(np.float32),
        "restimulus": np.zeros((label_n, 1), np.uint8), "stimulus": np.zeros((label_n, 1), np.int8),
        "repetition": np.zeros((label_n, 1), np.int8), "rerepetition": np.zeros((label_n, 1), np.uint8),
    }


def test_label_length_mismatches_pad_with_minus_one_and_never_touch_emg():
    """Anomalie dei dati reali (DB2): etichette piu' corte di emg (1 campione in s1, 272 in s12)."""
    mat = _mat_dict(n=100_000, label_n=100_000)
    mat["restimulus"] = np.ones((99_728, 1), np.uint8)  # 272 in meno, come s12 (0,27%: dentro l'1%)
    mat["rerepetition"] = np.full((99_728, 1), 3, np.uint8)
    ex = parse_exercise(mat, 3, DB4)
    assert ex.emg.shape[0] == 100_000 and all(v.shape[0] == 100_000 for v in ex.labels.values())
    assert ex.label_length_adjustments == {"restimulus": -272, "rerepetition": -272}
    assert (ex.labels["restimulus"][:99_728] == 1).all() and (ex.labels["restimulus"][99_728:] == -1).all()
    assert ex.n_movements == 1  # il -1 di riempimento non e' un movimento
    ok = parse_exercise(_mat_dict(n=300), 3, DB4)
    assert ok.label_length_adjustments == {} and ok.emg.shape[0] == 300


def test_longer_label_is_truncated_and_gross_mismatch_raises():
    mat = _mat_dict(n=1000)
    mat["stimulus"] = np.zeros((1003, 1), np.int8)
    ex = parse_exercise(mat, 1, DB4)
    assert ex.label_length_adjustments == {"stimulus": 3} and ex.labels["stimulus"].shape[0] == 1000
    mat2 = _mat_dict(n=1000)
    mat2["restimulus"] = np.zeros((900, 1), np.uint8)  # 10% in meno: oltre l'1%
    with pytest.raises(ValueError, match="lunghezze incompatibili"):
        parse_exercise(mat2, 1, DB4)


def test_db7_two_exercises_amputees_21_22_and_big_unused_variables_are_skipped(tmp_path):
    (tmp_path / "DB7").mkdir()
    path = tmp_path / "DB7" / "Subject_21.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as zf:
        for e in (1, 2):
            buf = io.BytesIO()
            rest = np.zeros((400, 1), np.int8)
            rest[200:, 0] = 1
            sio.savemat(buf, {"emg": rng.normal(scale=2e-5, size=(400, 12)).astype(np.float32),
                              "acc": np.zeros((400, 36), np.float32), "gyro": np.zeros((400, 36), np.float32),
                              "mag": np.zeros((400, 36), np.float32), "glove": np.zeros((400, 18), np.float32),
                              "stimulus": rest, "restimulus": rest, "repetition": rest, "rerepetition": rest,
                              "subject": np.array([[1]], np.uint8), "exercise": np.array([[e]], np.uint8)})
            zf.writestr(f"S21_E{e}_A1.mat", buf.getvalue())  # alla radice dello zip, senza cartella
    from wearusfm.ingest.ninapro_std import DB7

    assert set(scan_zips(tmp_path, DB7)) == {21}
    res = ingest_subject(path, tmp_path / "out", 21, DB7)
    meta = json.loads((tmp_path / "out" / "s21" / "session1" / "metadata.json").read_text())
    assert meta["nominal_anatomy"] is True and [t["exercise"] for t in meta["trials"]] == [1, 2]
    assert res["subject_field_matches_filename"] is False  # lo scalare interno non coincide: registrato
    assert all(c["anatomical_identity"]["nominal"] for g in meta["montage"]["groups"] for c in g["channels"])
