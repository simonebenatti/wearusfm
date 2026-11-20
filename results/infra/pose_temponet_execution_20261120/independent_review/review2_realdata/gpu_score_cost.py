"""CPU-side cost of the scoring passes that benchmark_gpu repeats 11 times per cell (workflow._score_model with include_clean=True),
on the real-size train cache. The model forward is an L0 on CPU (negligible): what is timed is view_inputs + the numpy moments."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, benchmark as B, heads, train as T
root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text())
allowed, identity = plan["producers"]["train"]["users"], {"execution_sha256": "e"*64}
data, first = B._load_measurement(lambda: P.read_verified_inputs(plan, "train", stage="gpu_benchmark"), identity)
folds, population = B._validate_train(data, allowed, identity, None)
bundle = B._scoring_bundle(data, folds)
full = np.arange(population["train_windows"])
model = heads.make_head("L0", 0)
times = {}
for part, held, fold in (("fold", folds[1], 0), ("full", full, 2)):
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        B._production_score(model, bundle, fold, held, "post", device="cpu", include_clean=True)
        samples.append(time.perf_counter()-started)
    times[part] = min(samples)
started = time.perf_counter(); fitting = B._dataset(data, full, "post"); fitting.identity; times["dataset_identity_full"] = time.perf_counter()-started
started = time.perf_counter(); B._validation_surrogate(data); times["validation_surrogate"] = time.perf_counter()-started
per_head = 2*11*times["fold"] + 11*times["full"]
print("seconds per scoring pass", {k: round(v, 2) for k, v in times.items()}, "| scoring per head (2 fold cells + full, 11 passes each)", round(per_head, 1),
      "| five heads", round(5*per_head, 1), "| input load", round(first["input_load_seconds"], 1))
