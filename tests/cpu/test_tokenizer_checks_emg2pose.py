import numpy as np
import pytest

from wearusfm.tokenizer_checks.emg2pose import load_emg2pose_session, load_user_map


def _write(path, n=500, ch=16, fs=2000.0):
    import h5py

    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (ch,))])
    arr = np.zeros(n, dtype=dt)
    arr["emg"] = np.arange(n * ch, dtype=np.float32).reshape(n, ch)
    with h5py.File(path, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=arr)
        g.attrs["sample_rate"] = fs
        g.attrs["num_channels"] = ch


def test_load_user_map_uses_stem(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("session,user,filename\ns1,u7,rec-a_left.hdf5\ns2,u9,rec-b_right\n")
    assert load_user_map(p) == {"rec-a_left": "u7", "rec-b_right": "u9"}


def test_load_emg2pose_session(tmp_path):
    f = tmp_path / "rec-a_left.hdf5"
    _write(f)
    s = load_emg2pose_session(f, {"rec-a_left": "u7"})
    assert s.dataset == "emg2pose" and s.subject == "u7" and s.session == "rec-a_left"
    assert s.fs == 2000.0 and len(s.segments) == 1 and s.segments[0].shape == (500, 16)
    assert s.qc_valid.all() and s.segments[0][1, 0] == 16.0
    assert load_emg2pose_session(f, {}).subject == "sconosciuto"


def test_load_emg2pose_rejects_wrong_channel_count(tmp_path):
    f = tmp_path / "x.hdf5"
    _write(f, ch=8)
    with pytest.raises(ValueError, match="forma emg inattesa"):
        load_emg2pose_session(f, {})
