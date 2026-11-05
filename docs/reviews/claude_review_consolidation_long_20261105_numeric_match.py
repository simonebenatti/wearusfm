"""Second pass, R-1: can the exact numeric equality required by e2f19ee hold on Leonardo?

Builds the context the new classification process is expected to report there:
the pinned historical context (no torch in that process) plus the OpenMP runtime
that every archived torch-loaded process on Leonardo lists (libomp, 8 threads),
and feeds it to GPT's own _numeric_match against the pinned historical reports.
No job, no fit: only GPT's comparison function on archived contexts."""
import copy, json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import consolidation_long_classification as M
import consolidation_long_classification_runtime as R

base = ROOT / "results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/completed_aggregates/T100-G"
hist = {t: json.loads((base / f"{t}.json").read_text()) for t in ("p1", "p2")}
assert all(M.jhash(h["numeric_context"]) == h["numeric_sha256"] for h in hist.values())
reference = {"numeric_sha256": hist["p1"]["numeric_sha256"], "reports": {"p1": [], "p2": []}}
assert hist["p2"]["numeric_sha256"] == reference["numeric_sha256"]
print("historical numeric_sha256:", reference["numeric_sha256"])
print("historical pools:", [(p["prefix"], p["version"], p["num_threads"]) for p in hist["p1"]["numeric_context"]["threadpools"]])

# archived torch-loaded process on Leonardo, same python/numpy/torch stack
r3c = json.loads((ROOT / "results/infra/pose_slots_20261102/execution/R3c/receipt.json").read_text())["runtime"]
print("archived torch+scipy process pools:", [(p["prefix"], p["version"], p["num_threads"]) for p in r3c["threadpools"]])
extra = [p for p in r3c["threadpools"] if p["prefix"] == "libomp"]
assert len(extra) == 1 and extra[0]["num_threads"] == 8

same = copy.deepcopy(hist["p1"]["numeric_context"])
print("identical context ->", R._numeric_match(same, reference)["exact_match"])

expected = copy.deepcopy(hist["p1"]["numeric_context"])
keys = ("internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer")
expected["threadpools"] = sorted(expected["threadpools"] + [{k: extra[0].get(k) for k in keys}],
                                 key=lambda p: json.dumps(p, sort_keys=True))
print("expected new-process hash:", M.jhash(expected))
try:
    R._numeric_match(expected, reference)
    print("historical context + torch runtime -> ACCEPTED")
except ValueError as e:
    print("historical context + torch runtime -> REFUSED:", e)
print("torch loaded in this (new-code) process:", "torch" in sys.modules)
