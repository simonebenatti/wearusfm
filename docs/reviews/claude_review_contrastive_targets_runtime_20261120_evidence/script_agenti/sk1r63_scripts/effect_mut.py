import json, os, subprocess, sys, shutil
from pathlib import Path
clone, mutfile, probe, bt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
muts = {m["id"]: m for m in json.loads(mutfile.read_text())}
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
for mid in sys.argv[5].split(","):
    m = muts[mid]; path = clone / m["file"]; original = path.read_bytes(); text = original.decode()
    assert text.count(m["old"]) == 1
    path.write_text(text.replace(m["old"], m["new"]))
    try:
        r = subprocess.run(["/Users/simonebenatti/.venvs/wearusfm-tok/bin/python", "-B", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
                            f"--basetemp={bt}/e_{mid}", str(probe)], cwd=probe.parent, env=env, capture_output=True, text=True, timeout=600)
        for line in r.stdout.splitlines():
            if "EFFECT" in line: print(mid, line[line.index("EFFECT"):])
        print(mid, "pytest rc", r.returncode, [l for l in r.stdout.splitlines() if l.strip()][-1])
    finally:
        path.write_bytes(original); shutil.rmtree(bt / ("e_" + mid), ignore_errors=True)
    print(mid, "clean", subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == "")
