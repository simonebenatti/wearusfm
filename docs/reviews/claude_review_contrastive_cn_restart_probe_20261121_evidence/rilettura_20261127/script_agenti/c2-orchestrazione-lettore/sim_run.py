"""Simulate run() with the REAL checked_command and real child processes.

Only require_binding, commands, validate_allocation, nvidia-smi and env are faked.
Usage: python sim_run.py CLONE WORKDIR
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CLONE, WORK = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(CLONE / "src"), str(CLONE / "scripts"), str(CLONE / "scripts/diagnostics")]
spec = importlib.util.spec_from_file_location("gate", CLONE / "scripts/diagnostics/contrastive_cn_restart.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
import contrastive_job

PY = sys.executable
OK = [PY, "-c", "pass"]
FAIL = [PY, "-c", "import sys; sys.exit(3)"]
SLEEP = [PY, "-c", "import time; time.sleep(30)"]


def report_writer(root):
    return [PY, "-c", f"open({str(root / 'report.json')!r}, 'w').write('{{}}')"]


def case(name, legs, compare_kind, end_offset=3600, deadline_short=None):
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    cells = [legs[0], legs[1]]
    compare = {"ok": report_writer(root), "noreport": OK, "fail": FAIL, "sleep": SLEEP}[compare_kind]
    gate.require_binding = lambda: {"legs": ["ordinary", "warm"]}
    gate.commands = lambda plan: (root, cells, compare, [])
    contrastive_job.validate_allocation = lambda kind: None
    gate.file_sha = lambda p: "3" * 64
    gate.subprocess.check_output = lambda cmd, text: "0, GPU-0, 0000:00, A100\n1, GPU-1, 0000:01, A100"
    os.environ.update(TORCH_DIST_INIT_BARRIER="1", SLURM_JOB_ID="sim", WEARUSFM_CODE_SHA="6" * 40,
                      SLURM_JOB_END_TIME=str(int(time.time()) + end_offset))
    exc = None
    t0 = time.monotonic()
    try:
        gate.run(False)
    except BaseException as e:  # noqa
        exc = e
    out = {"case": name, "exception": None if exc is None else f"{type(exc).__name__}: {exc}",
           "wall_s": round(time.monotonic() - t0, 2),
           "files": sorted(p.name for p in root.iterdir())}
    timings = root / "phase_timings.jsonl"
    out["timings"] = [json.loads(l) for l in timings.read_text().splitlines()] if timings.exists() else None
    failure = root / "execution_failure.json"
    out["failure"] = json.loads(failure.read_text()) if failure.exists() else None
    for t in out["timings"] or []:
        t.pop("unix_s", None)
        if "elapsed_s" in t:
            t["elapsed_s"] = round(t["elapsed_s"], 2)
    if out["failure"]:
        out["failure"]["elapsed_s"] = round(out["failure"]["elapsed_s"], 2)
    print(json.dumps(out))


case("all_ok", [OK, OK], "ok")
case("ordinary_exit3", [FAIL, OK], "ok")
case("warm_exit3", [OK, FAIL], "ok")
case("compare_exit3", [OK, OK], "fail")
case("report_missing", [OK, OK], "noreport")
# SLURM_JOB_END_TIME 28 s ahead -> deadline about 3 s: the first sleeping leg hits TimeoutExpired.
case("ordinary_timeout", [SLEEP, OK], "ok", end_offset=28)
case("warm_timeout", [OK, SLEEP], "ok", end_offset=28)
case("compare_timeout", [OK, OK], "sleep", end_offset=28)
# Deadline already spent: the leg uses all, the next phase meets remaining <= 0 -> TimeoutError.
case("deadline_exhausted_before_compare", [OK, [PY, "-c", "import time; time.sleep(3.5)"]], "ok", end_offset=28)
case("end_time_in_past", [OK, OK], "ok", end_offset=-100)
