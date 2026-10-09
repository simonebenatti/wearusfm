"""`scripts/probe_memory.py` (collaudo dell'infrastruttura): sulla CPU, col rung piu' piccolo e la griglia di CapgMyo dei dati di prova, un
aggiornamento completo sul caso peggiore e il report scritto; la memoria si misura solo su CUDA."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from test_pretraining_loader import _tree  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_memory.py"


def test_probe_runs_one_full_update_on_the_worst_case(tmp_path):
    root, mpath = _tree(tmp_path)
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"window_s": {"emg2pose/u1/sessA": 3.9, "ninapro_db2/s01/session1": 3.0, "capgmyo/s01/s": 1.0}}))
    out = tmp_path / "probe.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(mpath), "--root", str(root), "--window-seconds", str(ws),
                        "--out", str(out), "--rungs", "30m", "--per-gpu", "1", "--dataset", "capgmyo", "--window-s", "1.0", "--steps", "1",
                        "--device", "cpu"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    case, = json.loads(out.read_text())["cases"]
    assert case["rung"] == "30m" and not case["oom"] and case["adam_states"] and case["channels"] == [128]  # griglia intera, nessun virtuale
    assert case["n_patches"] == [40] and abs(case["parameters"] - 28.3e6) < 1e6 and len(case["step_s"]) == 1
