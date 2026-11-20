"""Apply one mutation at a time in CLONE, run pytest on TESTS, restore and check git cleanliness.

Usage: run_muts.py CLONE BASETEMP_ROOT MUTS_JSON IDS(comma|all) TESTLIST_FILE OUT_JSONL
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

clone, btroot, mutfile = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ids = sys.argv[4]
tests = [t for t in Path(sys.argv[5]).read_text().split() if t]
out = Path(sys.argv[6])
muts = json.loads(mutfile.read_text())
if ids != "all":
    wanted = ids.split(",")
    muts = [m for m in muts if m["id"] in wanted]
    muts.sort(key=lambda m: wanted.index(m["id"]))
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
py = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
for m in muts:
    path = clone / m["file"]
    original = path.read_bytes()
    text = original.decode()
    if text.count(m["old"]) != 1:
        print(json.dumps({"id": m["id"], "error": "old not unique"}), flush=True)
        continue
    path.write_text(text.replace(m["old"], m["new"]))
    bt = btroot / ("bt_" + m["id"])
    t0 = time.monotonic()
    try:
        r = subprocess.run([py, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            f"--basetemp={bt}", *tests], cwd=clone, env=env, capture_output=True,
                           text=True, timeout=1800)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        failed = [l for l in lines if l.startswith("FAILED") or l.startswith("ERROR")]
        row = {"id": m["id"], "file": m["file"], "line": m["line"], "desc": m["desc"], "rc": r.returncode,
               "survived": r.returncode == 0, "summary": lines[-1] if lines else "", "failed": failed[:2],
               "elapsed_s": round(time.monotonic() - t0, 1), "ntests": len(tests)}
    finally:
        path.write_bytes(original)
        shutil.rmtree(bt, ignore_errors=True)
    dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    row["clean_after_restore"] = dirty == ""
    with out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if dirty:
        raise SystemExit("clone dirty after restore: " + dirty)
