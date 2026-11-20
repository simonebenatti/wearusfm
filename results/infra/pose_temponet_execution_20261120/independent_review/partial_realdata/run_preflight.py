"""Unmocked provenance.preflight on the real-shaped inputs; phase timings and peak RSS."""
import json, resource, sys, time
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data as D, metrics as M
import consolidation_long_pose as H
root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text())
policy = plan.pop("historical_reference_fixture")
M.HISTORICAL_REFERENCE = policy
scale = 1 if sys.platform == "darwin" else 1024
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale/2**20
print("numpy", np.__version__, "| python", sys.version.split()[0], "| baseline RSS MiB", round(rss()))
started = time.perf_counter()
result = P.preflight(plan)
observed = result["historical_R1"]["observed"]
print("preflight passed:", result["integrity"], result["historical_R1"]["passed"], "| folds", [len(f) for f in result["train"]["folds"]],
      "| train supported windows", result["train"]["supported_windows"], "of", result["train"]["input_verification"]["n_windows"])
print("R-1 E_dyn bitwise equal to the historical summary:", {k: observed[k]["E_dyn"] == policy[k+"_E_dyn"] for k in observed},
      "| counts", observed["post_ridge"]["selected_windows"], observed["post_ridge"]["centered_windows"], observed["post_ridge"]["users"])
json.dumps(result, allow_nan=False, sort_keys=True)
print("strict JSON: True | seconds", round(time.perf_counter()-started, 2), "| peak RSS MiB", round(rss()))
# same call the first job made, with the first delivery's producer list: must reproduce its error
broken = json.loads(json.dumps(plan)); broken["producers"]["train"]["users"] = broken["producers"]["train"]["storage_directories"]
try:
    P.read_verified_inputs(broken, "train", stage="preflight", depth="post")
except ValueError as error:
    print("first delivery's list reproduces job 64931573:", error)
