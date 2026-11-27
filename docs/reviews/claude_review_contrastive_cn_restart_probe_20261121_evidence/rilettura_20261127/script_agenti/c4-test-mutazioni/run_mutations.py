import json, os, shutil, subprocess, sys, threading, queue, re, time
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W))
from mutations import M

CLONE = W.parent.parent / "clone"
TESTS = ["tests/cpu/test_cn_restart_compare.py", "tests/cpu/test_cn_restart_end_to_end.py",
         "tests/cpu/test_cn_restart_observe.py", "tests/cpu/test_cn_restart_package.py",
         "tests/cpu/test_cn_restart_leg.py"]
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
only = set(sys.argv[1:])
items = [m for m in M if not only or m[0] in only]

# Validate uniqueness on the pristine clone.
bad = []
for mid, f, old, new, cat, label in items:
    n = (CLONE / f).read_text().count(old)
    if n != 1:
        bad.append((mid, n))
if bad:
    print("BAD", bad); sys.exit(1)

NW = 2
trees = []
for i in range(NW):
    t = W / f"tree{i}"
    if not t.exists():
        subprocess.run(["cp", "-cR", str(CLONE), str(t)], check=True)
    trees.append(t)

env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:scripts:scripts/diagnostics")
results = {}
q = queue.Queue()
for m in items:
    q.put(m)
lock = threading.Lock()

def worker(i):
    tree = trees[i]
    while True:
        try:
            mid, f, old, new, cat, label = q.get_nowait()
        except queue.Empty:
            return
        target = tree / f
        original = (CLONE / f).read_text()
        target.write_text(original.replace(old, new, 1))
        bt = W / f"bt_mut{i}"
        shutil.rmtree(bt, ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(bt),
                            "-o", "addopts=", *TESTS], cwd=tree, env=env, capture_output=True, text=True, timeout=600)
        target.write_text(original)
        out = p.stdout + p.stderr
        failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)
        summary = out.strip().splitlines()[-1] if out.strip() else ""
        rec = {"id": mid, "file": f, "category": cat, "label": label, "killed": p.returncode != 0,
               "rc": p.returncode, "failed": failed[:3], "summary": summary, "seconds": round(time.time() - t0, 1)}
        with lock:
            results[mid] = rec
            print(json.dumps(rec), flush=True)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(NW)]
for t in threads: t.start()
for t in threads: t.join()
out = W / ("mutation_results.json" if not only else "mutation_results_subset.json")
out.write_text(json.dumps([results[m[0]] for m in items], indent=1))
surv = [r for r in results.values() if not r["killed"]]
print("TOTAL", len(results), "SURVIVED", len(surv))
