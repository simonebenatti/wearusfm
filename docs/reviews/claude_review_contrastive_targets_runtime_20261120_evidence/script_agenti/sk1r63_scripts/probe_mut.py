"""For each mutation id, apply it in CLONE, run the probe file, record which probes fail, restore."""
import json, os, subprocess, sys, shutil
from pathlib import Path
clone, mutfile, probe, btroot, out = map(Path, sys.argv[1:6])
ids = sys.argv[6].split(",")
muts = {m["id"]: m for m in json.loads(mutfile.read_text())}
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
for mid in ids:
    m = muts[mid]
    path = clone / m["file"]; original = path.read_bytes(); text = original.decode()
    assert text.count(m["old"]) == 1
    path.write_text(text.replace(m["old"], m["new"]))
    bt = btroot / ("p_" + mid)
    try:
        r = subprocess.run(["/Users/simonebenatti/.venvs/wearusfm-tok/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            f"--basetemp={bt}", "-rf", str(probe)], cwd=probe.parent, env=env, capture_output=True, text=True, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        failed = [l.split("::")[1].split(" ")[0] for l in lines if l.startswith("FAILED")]
        row = {"id": mid, "rc": r.returncode, "summary": lines[-1] if lines else "", "probes_failed": failed}
    finally:
        path.write_bytes(original); shutil.rmtree(bt, ignore_errors=True)
    dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    row["clean_after_restore"] = dirty == ""
    with out.open("a") as s: s.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if dirty: raise SystemExit("dirty")
