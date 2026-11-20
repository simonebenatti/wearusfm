"""Unmocked provenance.preflight on real-format files, with per-phase wall time and the process peak RSS after each phase.

Only timing wrappers are installed (each calls the original function and returns its result unchanged).
"""
import functools, json, resource, sys, threading, time
from pathlib import Path
import numpy as np

sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data as D, metrics as M
import consolidation_long_pose as H
import tokenizer_pose_readout as R

root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text())
M.HISTORICAL_REFERENCE = plan.pop("historical_reference_fixture")
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)
rows, depth = [], [0]


def current_rss():
    import subprocess, os
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip())/2**20


def timed(module, name, label=None):
    original = getattr(module, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter(); depth[0] += 1
        try:
            return original(*args, **kwargs)
        finally:
            depth[0] -= 1
            size = ""
            if name in ("verify_pin", "verify_file"):
                pin = args[0]
                size = Path(pin["path"] if isinstance(pin, dict) else pin).stat().st_size
            rows.append((label or name, depth[0], time.perf_counter()-started, peak(), size))
    setattr(module, name, wrapper)


timed(P, "verify_pin", "P.verify_pin(hash)")
timed(R, "verify_file", "R.verify_file(hash)")
timed(R, "read_targets")
timed(H, "read_cache", "H.read_cache(inflate+validate)")
timed(H, "validate_cache", "H.validate_cache")
timed(D, "validate_cache", "D.validate_cache(=H.validate_cache+central)")
timed(D, "read_cache_pin")
timed(D, "read_predictions")
timed(M, "historical_reference_gate")
timed(P, "read_verified_inputs")

# peak sampler for the resident set between phases (the maximum is what matters; ru_maxrss is monotone)
print("numpy", np.__version__, "| baseline peak RSS GiB after imports", round(peak(), 3), flush=True)
started = time.perf_counter()
result = P.preflight(plan)
total = time.perf_counter()-started
json.dumps(result, allow_nan=False, sort_keys=True)
print("preflight passed:", result["integrity"], result["historical_R1"]["passed"], "| total seconds", round(total, 2), "| peak RSS GiB", round(peak(), 3))
for label, level, seconds, rss, size in rows:
    print("  %s%-46s %8.2f s   peak-so-far %.3f GiB   %s" % ("  "*level, label, seconds, rss, ("bytes %d" % size) if size != "" else ""))
windows = {s: plan["inputs"]["cache_"+s]["windows"] for s in ("train", "val")}
print("windows", windows, "| cache bytes", {s: plan["inputs"]["cache_"+s]["bytes"] for s in ("train", "val")},
      "| reference bytes", {s: plan["inputs"]["target_reference_"+s]["bytes"] for s in ("train", "val")})
hashed = sum(size for label, level, seconds, rss, size in rows if size != "")
hash_seconds = sum(seconds for label, level, seconds, rss, size in rows if size != "")
print("hashed bytes %d in %.2f s = %.0f MB/s" % (hashed, hash_seconds, hashed/hash_seconds/1e6))
