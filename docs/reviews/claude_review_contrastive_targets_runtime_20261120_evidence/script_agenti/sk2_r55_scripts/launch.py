import subprocess, sys
T = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp"
jobs = [(T + "/sk2_r55_clone", "NONE,C4d,C4g"), (T + "/sk2_r55_clone2", "C4h,C5c"), (T + "/sk2_r55_clone3", "C5f,C5m")]
procs = [subprocess.Popen([sys.executable, T + "/sk2_r55_scripts/mutants.py", clone, T + "/sk2_r55_scripts/results_" + clone.rsplit("_", 1)[-1] + ".jsonl", ids]) for clone, ids in jobs]
print([p.wait() for p in procs])
