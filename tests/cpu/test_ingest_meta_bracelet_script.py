import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_meta_bracelet_ingest import _p_bytes, _q_bytes, _targz  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest_meta_bracelet.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_qwerty_script_isolates_failures_and_resumes(tmp_path):
    members = {
        "2020-08-13-1000-keystrokes.hdf5": _q_bytes(n=1000, user="u1", seed=1),
        "2020-08-14-2000-keystrokes.hdf5": _q_bytes(n=1000, rate=1000.0, seed=2),  # rotto: frequenza sbagliata
        "2020-08-15-3000-keystrokes.hdf5": _q_bytes(n=1000, user="u2", seed=3),
    }
    tar = _targz(tmp_path, members)
    out, rep = tmp_path / "out", tmp_path / "rep.json"
    r = _run("--dataset", "emg2qwerty", "--tar", tar, "--out-root", out, "--report", rep)
    assert r.returncode == 1  # c'e' un fallimento, ma gli altri sono stati fatti
    d = json.loads(rep.read_text())
    assert d["n_recordings"] == 2 and d["n_failed"] == 1 and "daq_sample_rate" in list(d["failed"].values())[0]
    assert (out / "uu1" / "2020-08-13-1000-keystrokes" / "metadata.json").exists()
    # secondo giro: riprende (salta i gia' fatti) e ritenta solo il rotto
    r2 = _run("--dataset", "emg2qwerty", "--tar", tar, "--out-root", out, "--report", rep, "--skip-existing")
    d2 = json.loads(rep.read_text())
    assert d2["n_skipped_existing"] == 2 and d2["n_recordings"] == 0 and d2["n_failed"] == 1 and r2.returncode == 1


def test_qwerty_script_max_recordings_and_time_budget(tmp_path):
    members = {f"2020-08-1{i}-{i}000-keystrokes.hdf5": _q_bytes(n=800, user=f"u{i}", seed=i) for i in range(4)}
    tar = _targz(tmp_path, members)
    rep = tmp_path / "rep.json"
    r = _run("--dataset", "emg2qwerty", "--tar", tar, "--out-root", tmp_path / "o1", "--report", rep, "--max-recordings", "2")
    assert r.returncode == 0
    d = json.loads(rep.read_text())
    assert d["n_recordings"] == 2 and d["stopped"] == "max-recordings"
    r = _run("--dataset", "emg2qwerty", "--tar", tar, "--out-root", tmp_path / "o2", "--report", rep, "--time-budget-s", "0")
    assert r.returncode == 0
    d = json.loads(rep.read_text())
    assert d["stopped"] == "time-budget" and d["n_recordings"] == 0


def test_pose_script_end_to_end_with_csv_and_resume(tmp_path):
    tarp = tmp_path / "e.tar"
    with tarfile.open(tarp, "w") as tf:
        for nm in ("emg2pose_data/rec-a_left.hdf5", "emg2pose_data/rec-b_right.hdf5", "emg2pose_data/rec-c_left.hdf5"):
            data = _p_bytes(n=600)
            info = tarfile.TarInfo(nm)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    csvp = tmp_path / "meta.csv"
    csvp.write_text(
        "session,user,stage,start,end,side,filename,moving_hand,held_out_user,held_out_stage,split,generalization\n"
        "s1,u7,st,0,1,left,rec-a_left.hdf5,both,True,False,val,user\n"
        "s2,u8,st,0,1,right,rec-b_right.hdf5,both,False,False,train,none\n"  # rec-c non ha la riga
    )
    out, rep = tmp_path / "out", tmp_path / "rep.json"
    r = _run("--dataset", "emg2pose", "--tar", tarp, "--csv", csvp, "--out-root", out, "--report", rep)
    assert r.returncode == 0, r.stderr
    d = json.loads(rep.read_text())
    assert d["n_recordings"] == 3 and d["n_users"] == 3  # u7, u8, sconosciuto
    assert sorted(p.name for p in out.iterdir()) == ["usconosciuto", "uu7", "uu8"]
    r2 = _run("--dataset", "emg2pose", "--tar", tarp, "--csv", csvp, "--out-root", out, "--report", rep, "--skip-existing")
    assert r2.returncode == 0 and json.loads(rep.read_text())["n_skipped_existing"] == 3
