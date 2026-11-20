"""My own mutation runner: apply one mutation, run pytest, restore, verify clean. Usage: mutrun.py CLONE BTROOT MUTFILE IDS TESTFILELIST OUTJSONL"""
import json, subprocess, sys, os, shutil
from pathlib import Path
clone, btroot, mutfile, ids, tests, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","), Path(sys.argv[5]).read_text().split(), Path(sys.argv[6])
muts = {m["id"]: m for m in json.loads(mutfile.read_text())}
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
py = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
for mid in ids:
    m = muts[mid]
    path = clone / m["file"]
    original = path.read_bytes()
    text = original.decode()
    n = text.count(m["old"])
    if n != 1:
        rec = {"id": mid, "error": f"anchor occurs {n} times"}
    else:
        path.write_text(text.replace(m["old"], m["new"]))
        compile(path.read_text(), str(path), "exec")
        bt = btroot / ("bt_" + clone.name + "_" + mid + "_" + str(os.getpid()))
        try:
            r = subprocess.run([py, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                                f"--basetemp={bt}", *tests], cwd=clone, env=env, capture_output=True, text=True, timeout=1800)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            rec = {"id": mid, "desc": m["desc"], "rc": r.returncode, "survived": r.returncode == 0,
                   "failed": [l for l in lines if l.startswith(("FAILED", "ERROR"))][:2], "tail": lines[-1:] if lines else []}
        finally:
            path.write_bytes(original)
            shutil.rmtree(bt, ignore_errors=True)
    dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    rec["clean_after_restore"] = dirty == ""
    with out.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
