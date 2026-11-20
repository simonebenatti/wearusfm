"""Unmocked benchmark.benchmark_cpu at the REAL train size. Only the thread guard is replaced (one thread on the Mac)."""
import json, resource, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data as D, benchmark, runner as R, train

root = Path(sys.argv[1]).resolve(); repair = sys.argv[2] == "repair"
plan = json.loads((root/"plan.json").read_text())
allowed = plan["producers"]["train"]["users"]
identity = {"execution_sha256": "e"*64}
train.configure_numerics = lambda: train.numerical_profile()
loader = lambda: P.read_verified_inputs(plan, "train", stage="cpu_benchmark")
cache, first = benchmark._load_measurement(loader, identity)
support = D.central_support(cache["mask"], D.lengths_for(cache), cache["key_time_valid"])
firsts = {u: [i for i, m in enumerate(cache["meta"]) if m["user"] == u] for u in sorted({m["user"] for m in cache["meta"]})[:15]}
print("first windows of the 15 surrogate users with support:", [bool(support[v[0]].any()) for v in firsts.values()], flush=True)
if repair:
    for user, own in firsts.items():
        if not support[own[0]].any():
            donor = next(i for i in own if support[i].any())
            for key in ("mask", "target", "pred", "pose_var", "central_mask"):
                cache[key][own[0]] = cache[key][donor]
            print("gave valid pose to the first window of", user, flush=True)
out = root/("bench_cpu_"+sys.argv[2])
started = time.perf_counter()
try:
    result = benchmark.benchmark_cpu(cache, input_loader=loader, input_measurement=first, output_dir=out/"cost_only",
                                     allowed_train_users=allowed, input_identity=identity)
    text = json.dumps(R._plain(result), sort_keys=True, indent=2, allow_nan=False)
    (out/"benchmark.json").write_text(text)
    print("benchmark_cpu OK: measured", result["measured"], "| elapsed s", round(result["elapsed_seconds"], 1),
          "| fold projections s", [round(x, 1) for x in result["fold_projection_seconds"]],
          "| report projection s", round(result["report_projection_seconds"], 1),
          "| peak RSS GiB", round(result["memory"]["peak_rss_bytes"]/2**30, 2),
          "| projected peak GiB", round(result["memory"]["projected_peak_bytes"]/2**30, 2), "| strict JSON bytes", len(text), flush=True)
    for fold in result["folds"]:
        print("  fold", fold["outer_fold"], "generator s", round(fold["generator_seconds"], 1), "scan s", round(fold["scan"]["seconds"], 1), flush=True)
        for design in fold["designs"]:
            for pattern in design["patterns"]:
                print("    ", design["kind"], "context", design["context"], "dim", pattern["dimension"], "joints", len(pattern["joints"]),
                      "blocks", len(pattern["measured_blocks"]), "of", pattern["total_blocks"],
                      "| qr s", [round(b["qr_seconds"], 1) for b in pattern["measured_blocks"]],
                      "| solves s", round(sum(pattern["solve_seconds"].values()), 1), "| projected s", round(pattern["projected_seconds"], 1), flush=True)
    print("  report:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in result["report"].items()}, flush=True)
except Exception as error:
    frame = traceback.extract_tb(error.__traceback__)[-1]
    print("benchmark_cpu FAILED after", round(time.perf_counter()-started, 1), "s:", type(error).__name__, error,
          "| at", Path(frame.filename).name, frame.lineno, frame.name, flush=True)
print("process peak RSS GiB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30, 2), flush=True)
