import io
import json
import zipfile

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.ninapro_db6 import (
    build_montage_metadata,
    ingest_subject,
    load_subject_sessions,
    parse_session,
    scan_db6,
)


def _mat(day, time, n=600, subj=1, empty=(8, 9), label_n=None):
    rng = np.random.default_rng(day * 10 + time)
    emg = rng.normal(scale=2e-5, size=(n, 16)).astype(np.float32)
    emg[:, list(empty)] = 0.0
    rest = np.zeros((n if label_n is None else label_n, 1), np.int8)
    rest[len(rest) // 2 :, 0] = 3
    return {
        "emg": emg, "acc": np.zeros((n, 48), np.float32), "stimulus": rest, "restimulus": rest,
        "repetition": rest, "rerepetition": rest, "repetition_object": rest, "object": rest, "reobject": rest,
        "subj": np.array([[subj]], np.uint8), "daytesting": np.array([[day]], np.uint8), "time": np.array([[time]], np.uint8),
    }


def _make_zips(root, subject=1, missing=None, wrong_subject=None):
    (root / "DB6").mkdir(parents=True, exist_ok=True)
    paths = []
    for part, days in (("a", (1, 2, 3)), ("b", (4, 5))):
        p = root / "DB6" / f"DB6_s{subject}_{part}.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(f"DB6_s{subject}_{part}/.DS_Store", b"junk")
            for d in days:
                for t in (1, 2):
                    if (d, t) == missing:
                        continue
                    buf = io.BytesIO()
                    sio.savemat(buf, _mat(d, t, subj=subject))
                    sn = subject if wrong_subject is None else wrong_subject
                    zf.writestr(f"DB6_s{subject}_{part}/S{sn}_D{d}_T{t}.mat", buf.getvalue())
        paths.append(p)
    return paths


def test_scan_pairs_a_and_b_zips(tmp_path):
    _make_zips(tmp_path, 1)
    _make_zips(tmp_path, 4)
    got = scan_db6(tmp_path)
    assert set(got) == {1, 4} and [p.name for p in got[1]] == ["DB6_s1_a.zip", "DB6_s1_b.zip"]


def test_load_subject_sessions_ten_sessions_in_day_time_order(tmp_path):
    paths = _make_zips(tmp_path, 1)
    ses = load_subject_sessions(paths, 1)
    assert [(s.day, s.time) for s in ses] == [(d, t) for d in range(1, 6) for t in (1, 2)]
    assert ses[0].emg.shape == (600, 16) and ses[3].day_field_in_file == 2 and ses[3].time_field_in_file == 2


def test_missing_session_and_wrong_subject_raise(tmp_path):
    with pytest.raises(ValueError, match="sessioni"):
        load_subject_sessions(_make_zips(tmp_path / "a", 1, missing=(3, 2)), 1)
    with pytest.raises(ValueError, match="soggetto"):
        load_subject_sessions(_make_zips(tmp_path / "b", 2, wrong_subject=9), 2)


def test_parse_session_label_length_rule():
    mat = _mat(1, 1, n=100_000)
    mat["restimulus"] = np.ones((99_900, 1), np.int8)
    s = parse_session(mat, 1, 1)
    assert s.label_length_adjustments == {"restimulus": -100} and (s.labels["restimulus"][-100:] == -1).all()
    bad = _mat(1, 1, n=100_000)
    bad["restimulus"] = np.ones((90_000, 1), np.int8)
    with pytest.raises(ValueError, match="lunghezze incompatibili"):
        parse_session(bad, 1, 1)


def test_montage_three_groups_in_column_order_and_day_index():
    m = build_montage_metadata(3, 4, 2)
    assert [g.group_id for g in m.groups] == ["ring8_radiohumeral", "empty_columns", "distal6"]
    assert m.day_index == 4 and m.session_id == "D4_T2" and m.subject_id == "ninapro_db6_s03"
    idx = [c.sensor_coords.channel_index for g in m.groups for c in g.channels]
    assert idx == list(range(16))  # l'ordine dei gruppi coincide con l'ordine delle colonne


def test_ingest_subject_writes_ten_sessions_and_discards_empty_columns(tmp_path):
    paths = _make_zips(tmp_path / "raw", 1)
    res = ingest_subject(paths, tmp_path / "out", 1)
    assert res["n_sessions"] == 10 and res["discarded_channels_union"] == [8, 9] and res["all_fields_match_filename"]
    d = tmp_path / "out" / "s01" / "D3_T2"
    data = np.load(d / "data_int16.npy")
    assert data.shape == (600, 16) and data.dtype == np.int16
    meta = json.loads((d / "metadata.json").read_text())
    flags = [c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]]
    assert flags[8] is False and flags[9] is False and sum(flags) == 14
    assert meta["montage"]["day_index"] == 3 and meta["day"] == 3 and meta["time_of_day"] == 2
    assert set(np.load(d / "labels.npz").files) == set(meta["label_fields"])
