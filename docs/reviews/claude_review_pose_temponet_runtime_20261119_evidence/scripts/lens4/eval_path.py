"""Lens 4 check: per-batch host-side work of the production evaluator (_score_model: view_inputs,
_metric_panel, crop) versus the benchmark evaluator (_eval_model: pre-cast FP16 slice, subset)."""
import time
import numpy as np, torch
from pose_temponet import data as D, metrics
torch.set_num_threads(1)
rng = np.random.default_rng(0)
n, users = 1907, 40
lengths = rng.integers(140, 161, n)
mask = np.broadcast_to(np.arange(160)[None,:,None] < lengths[:,None,None], (n,160,20)).copy()
per = -(-n//users)
cache = {"lat": rng.standard_normal((n,160,384)).astype(np.float16), "target": rng.standard_normal((n,160,20)).astype(np.float32),
         "mask": mask, "std": np.ones((n,20),np.float32), "pose_var": rng.uniform(0,.6,n),
         "key_time_valid": np.arange(160)[None,:] < lengths[:,None],
         "meta": [{"user": f"u{i//per:02d}", "n_patches": int(lengths[i])} for i in range(n)]}
held = np.arange(0, n, 2)
feat16 = torch.from_numpy(cache["lat"][held].copy())
panel = D.subset(cache, held); panel["pose_var"] = cache["pose_var"][held]
pred = rng.standard_normal((32,160,20)).astype(np.float32)
def prod(rows):   # host side of workflow._score_model per batch (no model)
    x = torch.from_numpy(D.view_inputs(cache, "post", rows))
    p = pred[:, 6:-6]
    np.isfinite(p).all()
    pan = {"target": cache["target"][rows], "pose_var": cache["pose_var"][rows], "mask": cache["mask"][rows],
           "std": cache["std"][rows], "meta": [cache["meta"][i] for i in rows], "key_time_valid": cache["key_time_valid"][rows]}
    metrics.selection_statistics(p, pan, .2)
def bench(start):  # host side of benchmark._eval_model per batch (no model)
    x = feat16[start:start+32].to(dtype=torch.float32)
    metrics.selection_statistics(pred, D.subset(panel, np.arange(start, start+32)), .2)
for name, fn, args in (("production", prod, [held[s:s+32] for s in range(0, 928, 32)]),
                       ("benchmark", bench, list(range(0, 928, 32)))):
    t0 = time.perf_counter()
    for a in args: fn(a)
    print(name, "host seconds per batch", round((time.perf_counter()-t0)/len(args), 5))
t0=time.perf_counter()
for _ in range(50): D.lengths_for(cache)
print("lengths_for over 1907 meta, s/call", round((time.perf_counter()-t0)/50, 6))
big = {"meta": [{"n_patches": 150}]*18548, "target": np.broadcast_to(np.float32(0), (18548,160,20))}
t0=time.perf_counter()
for _ in range(20): D.lengths_for(big)
print("lengths_for over 18548 meta (inference view_inputs per batch), s/call", round((time.perf_counter()-t0)/20, 5))
