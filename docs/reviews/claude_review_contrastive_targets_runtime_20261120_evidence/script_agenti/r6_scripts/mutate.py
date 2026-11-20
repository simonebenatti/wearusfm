"""Apply one mutation at a time in the temporary clone, run tests, restore. Usage: mutate.py CLONE BASETEMP MUTFILE"""
import json, subprocess, sys, os, shutil
from pathlib import Path
clone, basetemp, mutfile = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tests = sys.argv[4].split(",")
muts = json.loads(mutfile.read_text())
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
py = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
for m in muts:
    path = clone / m["file"]
    original = path.read_bytes()
    text = original.decode()
    count = text.count(m["old"])
    if count != 1:
        print(json.dumps({"id": m["id"], "error": f"old occurs {count} times"})); continue
    path.write_text(text.replace(m["old"], m["new"]))
    bt = basetemp / m["id"]
    try:
        r = subprocess.run([py, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={bt}", *tests],
                           cwd=clone, env=env, capture_output=True, text=True, timeout=900)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
        failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
        print(json.dumps({"id": m["id"], "desc": m["desc"], "rc": r.returncode,
                          "survived": r.returncode == 0, "failed": failed[:3], "tail": tail[-1:]}), flush=True)
    finally:
        path.write_bytes(original)
        shutil.rmtree(bt, ignore_errors=True)
