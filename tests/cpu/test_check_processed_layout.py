import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_processed import _data, _write  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_processed_layout.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_clean_scan_exits_zero_and_reports_hours(tmp_path):
    _write(tmp_path / "ds1" / "s01" / "session1", _data((30, 1000, 4)))
    _write(tmp_path / "ds1" / "s02" / "session1", _data((100, 4)), scale=[1, 2, 3, 4])
    _write(tmp_path / "ds2" / "s01", _data())
    rep = tmp_path / "rep.json"
    r = _run("--scan", tmp_path, "--report", rep)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(rep.read_text())
    assert d["ds1"]["n_sessions"] == 2 and d["ds1"]["n_subjects"] == 2 and d["ds2"]["n_sessions"] == 1
    assert d["ds1"]["layouts"] == {"trials3d": 1, "continuous": 1} and d["ds1"]["scales"] == {"none": 1, "per_channel": 1}
    assert d["ds1"]["n_sessions_conforming"] == 2 and d["ds1"]["hours_conforming"] > 0


def test_problems_make_exit_one_and_are_listed(tmp_path):
    _write(tmp_path / "ds" / "s01" / "a", _data())
    _write(tmp_path / "ds" / "s02" / "a", _data(), shape=[9, 9, 9])
    rep = tmp_path / "rep.json"
    r = _run("--root", f"ds={tmp_path / 'ds'}", "--report", rep)
    assert r.returncode == 1
    d = json.loads(rep.read_text())["ds"]
    assert d["n_sessions_with_problems"] == 1 and "s02/a" in d["problems_sample"]
    assert d["n_sessions_conforming"] == 1  # la sessione conforme conta comunque


def test_empty_dataset_is_flagged(tmp_path):
    (tmp_path / "vuoto").mkdir()
    r = _run("--root", f"vuoto={tmp_path / 'vuoto'}")
    assert r.returncode == 1 and "nessuna sessione" in r.stderr


def test_same_name_on_two_disks_is_not_overwritten(tmp_path):
    _write(tmp_path / "a" / "ds" / "s01", _data())
    _write(tmp_path / "b" / "ds" / "s01", _data())
    rep = tmp_path / "rep.json"
    r = _run("--scan", tmp_path / "a", "--scan", tmp_path / "b", "--report", rep)
    assert r.returncode == 0 and len(json.loads(rep.read_text())) == 2
