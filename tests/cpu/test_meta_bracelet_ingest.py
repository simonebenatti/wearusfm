import gzip
import io
import json
import tarfile

import h5py
import numpy as np
import pytest

from wearusfm.ingest import emg2pose as E2P
from wearusfm.ingest import emg2qwerty as E2Q


def _q_bytes(n=3000, *, user="09456349", rate=2000.0, ch=16, fields=None, seed=0):
    rng = np.random.default_rng(seed)
    names = fields or ("emg_right", "time", "emg_left")
    dt = np.dtype([(names[0], "<f4", (ch,)), (names[1], "<f8"), (names[2], "<f4", (ch,))])
    arr = np.zeros(n, dtype=dt)
    arr[names[2]] = rng.normal(scale=5.0, size=(n, ch))  # ruolo "left"
    arr[names[0]] = rng.normal(scale=8.0, size=(n, ch))  # ruolo "right"
    arr[names[1]] = 1597354281.0 + np.arange(n) / 2000.0
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("emg2qwerty")
        d = g.create_dataset("timeseries", data=arr)
        g.attrs.update({"user": user, "condition": "on_keyboard", "daq_channels": 16, "daq_sample_rate": rate,
                        "session_name": "2020-08-13-1597354281-keystrokes", "quality_check_tags": np.array([], dtype="S1")})
        g.attrs["keystrokes"] = json.dumps([{"key": "a", "start": 1.0}])
        d.attrs["prompts"] = json.dumps([{"name": "p1"}])
    return buf.getvalue()


def _targz(tmp_path, members):
    p = tmp_path / "q.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return p


def test_qwerty_streams_targz_in_order_and_ignores_other_files(tmp_path):
    p = _targz(tmp_path, {"2020-08-13-1597354281-keystrokes.hdf5": _q_bytes(seed=1), "notes.txt": b"x",
                          "2020-09-01-1598900000-keystrokes-abc.hdf5": _q_bytes(seed=2)})
    got = [(r.session_name, len(b) > 100) for r, b in E2Q.iter_tar_recordings(p)]
    assert got == [("2020-08-13-1597354281-keystrokes", True), ("2020-09-01-1598900000-keystrokes-abc", True)]


def test_qwerty_read_and_rejects_wrong_layout():
    rec = E2Q.read_recording(_q_bytes(n=500))
    assert rec["emg"].shape == (500, 32) and rec["attrs"]["user"] == "09456349" and rec["attrs"]["daq_sample_rate"] == 2000.0
    left_std, right_std = rec["emg"][:, :16].std(), rec["emg"][:, 16:].std()
    assert left_std < right_std  # colonne [left | right]: left ha scala 5, right 8
    with pytest.raises(ValueError, match="daq_sample_rate"):
        E2Q.read_recording(_q_bytes(rate=1000.0))
    with pytest.raises(ValueError, match="campi"):
        E2Q.read_recording(_q_bytes(fields=("a", "time", "b")))


def test_qwerty_ingest_two_wrist_rings_with_real_chirality_and_labels(tmp_path):
    ref = E2Q.RecordingRef("x/2020-08-13-1597354281-keystrokes.hdf5", "2020-08-13-1597354281-keystrokes")
    res = E2Q.ingest_recording(ref, _q_bytes(n=4000), tmp_path / "out")
    d = tmp_path / "out" / "u09456349" / "2020-08-13-1597354281-keystrokes"
    data = np.load(d / "data_int16.npy")
    meta = json.loads((d / "metadata.json").read_text())
    assert data.shape == (4000, 32) and data.dtype == np.int16 and res["discarded_channels"] == []
    groups = meta["montage"]["groups"]
    assert [g["group_id"] for g in groups] == ["left_wrist", "right_wrist"]
    assert {c["chirality"] for c in groups[0]["channels"]} == {"left"} and {c["chirality"] for c in groups[1]["channels"]} == {"right"}
    assert groups[1]["channels"][0]["sensor_coords"]["channel_index"] == 16
    assert json.loads((d / "keystrokes.json").read_text())[0]["key"] == "a" and json.loads((d / "prompts.json").read_text())[0]["name"] == "p1"
    assert meta["time_axis"]["regular"] and meta["attrs"]["condition"] == "on_keyboard" and "keystrokes" not in meta["attrs"]
    rec = data.astype(np.float64) / meta["int16_scale"]
    assert abs(rec[:, :16].std() - 5.0) < 0.3 and abs(rec[:, 16:].std() - 8.0) < 0.3


def _p_bytes(n=3000, *, rate=2000.0, ch=16, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (ch,))])
    arr = np.zeros(n, dtype=dt)
    arr["emg"] = rng.normal(scale=4.0, size=(n, ch))
    arr["time"] = 1670313600.0 + np.arange(n) / 2000.0
    arr["joint_angles"] = 99.0  # non deve entrare nell'output
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=arr)
        g.attrs.update({"sample_rate": rate, "num_channels": ch, "stage": "s", "split": "val", "held_out_user": True})
    return buf.getvalue()


def test_pose_scan_metadata_and_ingest(tmp_path):
    tarp = tmp_path / "e.tar"
    with tarfile.open(tarp, "w") as tf:
        for nm in ("emg2pose_data/rec-a_left.hdf5", "emg2pose_data/rec-b_right.hdf5"):
            data = _p_bytes()
            info = tarfile.TarInfo(nm)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    csvp = tmp_path / "meta.csv"
    csvp.write_text("session,user,stage,start,end,side,filename,moving_hand,held_out_user,held_out_stage,split,generalization\n"
                    "s1,u7,st,0,1,left,rec-a_left.hdf5,both,True,False,val,user\n")
    meta = E2P.load_metadata_csv(csvp)
    assert meta["rec-a_left"]["user"] == "u7"
    with tarfile.open(tarp) as tf:
        refs = E2P.scan_tar(tf)
        assert [r.stem for r in refs] == ["rec-a_left", "rec-b_right"]
        r1 = E2P.ingest_recording(tf.extractfile(refs[0].member), refs[0], tmp_path / "out", meta.get("rec-a_left"))
        r2 = E2P.ingest_recording(tf.extractfile(refs[1].member), refs[1], tmp_path / "out", meta.get("rec-b_right"))
    assert r1["user"] == "u7" and r1["in_csv"] and r1["split"] == "val"
    assert r2["user"] == "sconosciuto" and not r2["in_csv"]  # riga mancante: segnalata, non inventata
    d = tmp_path / "out" / "uu7" / "rec-a_left"
    data = np.load(d / "data_int16.npy")
    m = json.loads((d / "metadata.json").read_text())
    assert data.shape == (3000, 16) and m["montage"]["groups"][0]["channels"][0]["chirality"] == "left"
    assert m["attrs"]["split"] == "val" and "non ingeriti" in m["joint_angles"]
    d2 = tmp_path / "out" / "usconosciuto" / "rec-b_right"
    assert json.loads((d2 / "metadata.json").read_text())["montage"]["groups"][0]["channels"][0]["chirality"] == "right"  # dal suffisso


def test_pose_rejects_wrong_rate_and_channels():
    with pytest.raises(ValueError, match="sample_rate"):
        E2P.read_recording(io.BytesIO(_p_bytes(rate=1000.0)))
    with pytest.raises(ValueError, match="forma emg"):
        E2P.read_recording(io.BytesIO(_p_bytes(ch=8)))


def test_emg2pose_scan_tar_skips_macos_metadata_files(tmp_path):
    """Un tar creato su Mac contiene `._<nome>.hdf5` (metadati, non HDF5): non sono registrazioni."""
    import tarfile

    from wearusfm.ingest.emg2pose import scan_tar

    tarp = tmp_path / "t.tar"
    with tarfile.open(tarp, "w") as tf:
        for nm in ("emg2pose_data/rec-a_left.hdf5", "emg2pose_data/._rec-a_left.hdf5", "__MACOSX/emg2pose_data/rec-b_right.hdf5",
                   "emg2pose_data/rec-b_right.hdf5", "emg2pose_data/note.txt"):
            data = b"x"
            info = tarfile.TarInfo(nm)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with tarfile.open(tarp) as tf:
        assert [r.stem for r in scan_tar(tf)] == ["rec-a_left", "rec-b_right"]



def _pose_tar(path, n=3):
    with tarfile.open(path, "w") as tf:
        for k in range(n):
            data = _p_bytes()
            info = tarfile.TarInfo(f"emg2pose_data/rec-{k}_left.hdf5")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with tarfile.open(path) as tf:
        return [m.offset for m in tf.getmembers()]


class _ReadSpy(io.FileIO):
    """File che ricorda fin dove e' stato letto."""

    max_pos = 0

    def read(self, n=-1):
        out = super().read(n)
        self.max_pos = max(self.max_pos, self.tell())
        return out

    def readinto(self, b):
        k = super().readinto(b)
        self.max_pos = max(self.max_pos, self.tell())
        return k


def test_iter_recordings_reads_only_as_far_as_needed(tmp_path):
    tarp = tmp_path / "e.tar"
    offsets = _pose_tar(tarp)
    with _ReadSpy(tarp) as f, tarfile.open(fileobj=f, mode="r:") as tf:
        assert next(E2P.iter_recordings(tf)).stem == "rec-0_left"
        assert f.max_pos <= offsets[1]  # l'intestazione della seconda registrazione non e' ancora stata letta
    with _ReadSpy(tarp) as f, tarfile.open(fileobj=f, mode="r:") as tf:
        E2P.scan_tar(tf)
        assert f.max_pos > offsets[2]  # scan_tar legge l'indice intero


def test_pose_ingest_script_streams_the_tar(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    tarp = tmp_path / "e.tar"
    _pose_tar(tarp)
    csvp = tmp_path / "meta.csv"
    csvp.write_text("session,user,stage,start,end,side,filename,moving_hand,held_out_user,held_out_stage,split,generalization\n"
                    "s1,u7,st,0,1,left,rec-0_left.hdf5,both,True,False,val,user\n")
    spec = importlib.util.spec_from_file_location("ingest_meta_bracelet", Path(__file__).resolve().parents[2] / "scripts" / "ingest_meta_bracelet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _no_full_scan(tf):
        raise AssertionError("l'ingest non deve leggere l'indice intero del tar")

    monkeypatch.setattr(mod.E2P, "scan_tar", _no_full_scan)
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "emg2pose", "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(tmp_path / "out"),
                                      "--report", str(tmp_path / "rep.json"), "--max-recordings", "2"])
    mod.main()
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["n_recordings"] == 2 and rep["stopped"] == "max-recordings"
    monkeypatch.setattr(sys, "argv", sys.argv[:-2] + ["--skip-existing"])
    mod.main()
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["n_recordings"] == 1 and rep["n_skipped_existing"] == 2  # ripresa: le due gia' fatte si saltano, la terza si fa
