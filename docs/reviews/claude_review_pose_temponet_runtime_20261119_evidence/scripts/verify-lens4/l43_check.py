"""L4-3: host-side per-batch work, benchmark._eval_model vs workflow._score_model.

Cache mapping mirrors read_cache_pin(depth=None) on train: pre, lat, pred, target,
mask, std, pose_var, key_time_valid, meta, lengths, central_mask. The model forward
is replaced by a fixed prediction (it is identical in both paths). Interleaved reps.
"""
import time
import numpy as np, torch
from pose_temponet import data as D, metrics, benchmark as B, workflow as W
torch.set_num_threads(1)
rng = np.random.default_rng(1)
n, users = 1907, 40
lengths = rng.integers(140, 161, n)
avail = np.arange(160)[None, :] < lengths[:, None]
mask = np.broadcast_to(avail[:, :, None], (n, 160, 20)).copy()
per = -(-n // users)
cache = {"lat": rng.standard_normal((n, 160, 384)).astype(np.float16),
         "pre": rng.standard_normal((n, 160, 384)).astype(np.float16),
         "pred": rng.standard_normal((n, 160, 20)).astype(np.float32),
         "target": rng.standard_normal((n, 160, 20)).astype(np.float32),
         "mask": mask, "std": np.ones((n, 20), np.float32), "pose_var": rng.uniform(0, .6, n),
         "key_time_valid": avail.copy(), "config": {"split": "train"},
         "meta": [{"user": f"u{i // per:02d}", "n_patches": int(lengths[i]), "w": i} for i in range(n)]}
cache["lengths"] = D.lengths_for(cache)
cache["central_mask"] = D.central_support(cache["mask"], cache["lengths"], cache["key_time_valid"])
held = np.arange(0, n, 2)
fit = np.arange(1, n, 2)
# benchmark: _dataset(data, held, "post") and panel = subset(data, held)
validation = B._dataset(cache, held, "post")
panel = D.subset(cache, held)
panel["pose_var"] = metrics.dynamic_variance(panel["target"], panel["mask"])
thr = 0.2
pv = metrics.dynamic_variance(cache["target"], cache["mask"])
bundle = W.FoldBundle(cache, "real", (fit, held), (held, fit), (cache["target"],) * 2, (cache["target"],) * 2,
                      (pv, pv), (thr, thr), cache["central_mask"], {})
pred160 = rng.standard_normal((32, 160, 20)).astype(np.float32)

def bench_batch(start):
    stop = min(start + 32, len(held))
    x = validation.features[start:stop].to(dtype=torch.float32)          # benchmark.py:155 (CPU cast)
    p = pred160[: stop - start]
    metrics.selection_statistics(p, D.subset(panel, np.arange(start, stop)), thr)   # benchmark.py:159

def prod_batch(start):
    rows = held[start:start + 32]
    x = torch.from_numpy(D.view_inputs(bundle.cache, "post", rows))       # workflow.py:241
    p = pred160[: len(rows), D.CROP:-D.CROP]
    assert np.isfinite(p).all()
    metrics.selection_statistics(p, W._metric_panel(bundle, 0, rows), thr)  # workflow.py:244

starts = list(range(0, len(held), 32))
res = {"benchmark": [], "production": []}
for rep in range(4):
    order = (("benchmark", bench_batch), ("production", prod_batch)) if rep % 2 == 0 else \
            (("production", prod_batch), ("benchmark", bench_batch))
    for name, fn in order:
        t0 = time.perf_counter()
        for s in starts:
            fn(s)
        res[name].append((time.perf_counter() - t0) / len(starts))
for k, v in res.items():
    print(k, "s/batch per rep", [round(x, 5) for x in v], "median", round(float(np.median(v)), 5))
print("ratio production/benchmark (median)", round(np.median(res["production"]) / np.median(res["benchmark"]), 3))
# components
def t(f, k=30):
    t0 = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter() - t0) / k
rows = held[:32]
print("view_inputs", round(t(lambda: D.view_inputs(cache, "post", rows)), 5),
      "| fp16 slice->fp32", round(t(lambda: validation.features[:32].to(dtype=torch.float32)), 5),
      "| subset(panel)", round(t(lambda: D.subset(panel, np.arange(32))), 5),
      "| _metric_panel", round(t(lambda: W._metric_panel(bundle, 0, rows)), 5),
      "| selection_statistics", round(t(lambda: metrics.selection_statistics(pred160, D.subset(panel, np.arange(32)), thr)), 5),
      "| lengths_for(1907)", round(t(lambda: D.lengths_for(cache)), 5))
