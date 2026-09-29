import io
import json
import tarfile

import h5py
import numpy as np
import pytest

from wearusfm.ingest.kaifosh import (
    N_CHANNELS,
    RecordingRef,
    build_montage_metadata,
    ingest_recording,
    load_split_table,
    read_recording,
    scan_tar,
    time_axis_report,
)


def _hdf5_bytes(n=4000, *, task="discrete_gestures", fields=("emg", "time"), gap_at=None, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype([("emg", "<f4", (N_CHANNELS,)), ("time", "<f8")])
    arr = np.zeros(n, dtype=dt)
    arr["emg"] = rng.normal(scale=5.0, size=(n, N_CHANNELS))
    t = 1632777632.0 + np.arange(n) / 2000.0
    if gap_at is not None:
        t[gap_at:] += 0.01  # un buco di 10 ms
    arr["time"] = t
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        if fields == ("emg", "time"):
            d = f.create_dataset("data", data=arr)
        else:
            bad = np.zeros(n, dtype=np.dtype([(fields[0], "<f4"), (fields[1], "<f8")]))
            d = f.create_dataset("data", data=bad)
        d.attrs["task"] = task
        g = f.create_group("prompts")  # gruppo pandas: non viene letto
        g.create_dataset("axis0", data=np.array([b"name", b"time"]))
    return buf.getvalue()


def _make_tar(tmp_path, members):
    path = tmp_path / "full_data.tar"
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_scan_tar_sorted_and_ignores_other_files(tmp_path):
    tar = _make_tar(tmp_path, {
        "discrete_gestures_user_002_dataset_000.hdf5": _hdf5_bytes(),
        "discrete_gestures_user_001_dataset_001.hdf5": _hdf5_bytes(),
        "discrete_gestures_user_001_dataset_000.hdf5": _hdf5_bytes(),
        "README.txt": b"x",
    })
    with tarfile.open(tar) as tf:
        refs = scan_tar(tf)
    assert [(r.user, r.dataset) for r in refs] == [(1, 0), (1, 1), (2, 0)]


def test_read_recording_streams_from_tar_without_temp_file(tmp_path):
    tar = _make_tar(tmp_path, {"discrete_gestures_user_003_dataset_000.hdf5": _hdf5_bytes(n=3000)})
    with tarfile.open(tar) as tf:
        ref = scan_tar(tf)[0]
        emg, t = read_recording(tf.extractfile(ref.member))
    assert emg.shape == (3000, 16) and emg.dtype == np.float32 and t.shape == (3000,)


def test_read_recording_rejects_wrong_task_and_fields():
    with pytest.raises(ValueError, match="task"):
        read_recording(io.BytesIO(_hdf5_bytes(task="typing")))
    with pytest.raises(ValueError, match="campi"):
        read_recording(io.BytesIO(_hdf5_bytes(fields=("a", "b"))))


def test_time_axis_report_regular_and_gap():
    ok = time_axis_report(1000.0 + np.arange(4000) / 2000.0)
    assert ok["regular"] and ok["n_gaps"] == 0 and abs(ok["estimated_rate_hz"] - 2000.0) < 1.0
    t = 1000.0 + np.arange(4000) / 2000.0
    t[2000:] += 0.01
    gap = time_axis_report(t)
    assert gap["n_gaps"] == 1 and gap["regular"] is True  # la mediana resta regolare; il buco e' contato
    assert gap["gaps"] == [{"index": 2000, "dt_s": pytest.approx(0.0105)}] and gap["gaps_truncated"] is False
    assert ok["gaps"] == []
    assert time_axis_report(np.array([1.0]))["regular"] is False


def test_load_split_table(tmp_path):
    p = tmp_path / "corpus.csv"
    p.write_text("start,end,split,dataset_number,user_number,dataset\n"
                 "1.0,2.0,train,0,17,a.hdf5\n3.0,4.0,train,0,17,a.hdf5\n5.0,6.0,val,0,18,b.hdf5\n7.0,8.0,test,1,18,c.hdf5\n")
    assert load_split_table(p) == {(17, 0): ["train"], (18, 0): ["val"], (18, 1): ["test"]}


def test_montage_is_one_ring_of_16_with_hardware_band():
    m = build_montage_metadata(5, 2)
    assert len(m.groups) == 1 and m.groups[0].topology.value == "ring" and len(m.groups[0].channels) == 16
    c = m.groups[0].channels[4]
    assert c.native_fs_hz == 2000.0 and c.effective_band_hz == (40.0, 850.0) and c.sensor_coords.ring_angle_deg == 90.0
    assert m.subject_id == "kaifosh_u005" and m.session_id == "dataset002"


def test_ingest_recording_writes_data_and_sidecar(tmp_path):
    tar = _make_tar(tmp_path, {"discrete_gestures_user_017_dataset_000.hdf5": _hdf5_bytes(n=6000)})
    out = tmp_path / "out"
    with tarfile.open(tar) as tf:
        ref = scan_tar(tf)[0]
        res = ingest_recording(tf.extractfile(ref.member), ref, out, {(17, 0): ["train"]})
    d = out / "u017" / "dataset000"
    data = np.load(d / "data_int16.npy")
    assert data.shape == (6000, 16) and data.dtype == np.int16
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["native_fs_hz"] == 2000.0 and meta["official_split"] == ["train"]
    assert meta["time_axis"]["regular"] and len([c for g in meta["montage"]["groups"] for c in g["channels"]]) == 16
    rec = data.astype(np.float64) / meta["int16_scale"]
    assert abs(rec.std() - 5.0) < 0.3  # ricostruzione fedele
    assert res["n_channels_discarded"] == 0 and res["user"] == 17
