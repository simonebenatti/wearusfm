"""Per-cell statements of benchmark.benchmark_gpu (benchmark.py:288-345) for ALL five heads on the real-size, real-support
train cache, device redirected to CPU, ONE epoch instead of 1+10 and fold0 + full only for the first head. Timings mean nothing."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, benchmark as B, heads, train as T, runner as R

root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text())
allowed, identity = plan["producers"]["train"]["users"], {"execution_sha256": "e"*64}
loader = lambda: P.read_verified_inputs(plan, "train", stage="gpu_benchmark")
data, first = B._load_measurement(loader, identity)
folds, population = B._validate_train(data, allowed, identity, None)
out = root/"gpu_smoke"; out.mkdir()
io = B._input_io(data, loader, identity, first)
bundle = B._scoring_bundle(data, folds)
full = np.arange(population["train_windows"])
print("population", population, "| thresholds", [round(x, 4) for x in bundle.thresholds], flush=True)
for name in heads.HEADS:
    depth = "pre" if name == "TCN_pre" else "post"
    parts = (("fold0", folds[0], folds[1], 0), ("full", full, full, 2)) if name == "L0" else (("fold0", folds[0], folds[1], 0),)
    for part, fit, held, fold in parts:
        started = time.perf_counter()
        fitting = B._dataset(data, fit, depth)
        fitting.identity
        trajectory = T.Trajectory(lambda seed, f: heads.make_head(name, seed), fitting, lr=1e-3, seed=0, fold=fold, device="cpu")
        trajectory.train_to_step(trajectory.steps_per_epoch)
        observed, clean = B._production_score(trajectory.model, bundle, fold, held, depth, device="cpu", include_clean=True)
        state = trajectory.state(); trajectory.restore(state)
        pin = P.atomic_torch(out/f"cost-{name}-{part}-warm.pt", state, root=out)
        B._read_state_artifact(pin)
        batches = [B._prediction_batch(trajectory.model, data, depth, np.arange(size) % len(data["meta"]), device="cpu").shape for size in (32, 18548 % 32)]
        print(f"{name:9s} {part:5s} depth {depth:4s} train_windows {fitting.size} excluded {fitting.excluded_windows} steps/epoch {trajectory.steps_per_epoch}"
              f" | epoch loss {trajectory.epoch_losses[-1]['loss_sum']/trajectory.epoch_losses[-1]['window_count']:.4f}"
              f" | held windows {len(held)} scored {observed['window_count']} E_dyn {observed['dynamic_sse']/observed['dynamic_target_ss']:.4f}"
              f" | state bytes {pin['bytes']} | batches {batches} | s {time.perf_counter()-started:.1f}", flush=True)
        json.dumps(R._plain({"observed": observed, "clean": clean, "curve": trajectory.epoch_losses[-1]}), allow_nan=False)
print("reference proxy", {k: v for k, v in B._reference_proxy(plan["inputs"]["cache_train"]["path"], 652549847).items() if k.endswith("bytes") or k.endswith("multiplier")})
