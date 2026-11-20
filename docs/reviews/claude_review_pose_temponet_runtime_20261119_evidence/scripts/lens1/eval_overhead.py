"""Host-side cost per held evaluation: production _score_model path vs benchmark _eval_model path.

Synthetic fixture with the production geometry (1907 windows, 160x384 FP16, 40 users, 20 joints).
The model forward is excluded (identical in both); only the CPU work around it is timed.
"""
import time
import numpy as np
import torch

from pose_temponet import data as D, metrics, workflow as W, controls, train

rng = np.random.default_rng(0)
N, TT, F, J = 1907, 160, 384, 20
users = [f"user{u:02d}" for u in range(40)]
meta = [{"user": users[i % 40], "n_patches": 160, "recording": f"r{i}", "start": i} for i in range(N)]
cache = {
    "lat": rng.standard_normal((N, TT, F), dtype=np.float32).astype(np.float16),
    "pre": rng.standard_normal((N, TT, F), dtype=np.float32).astype(np.float16),
    "target": rng.standard_normal((N, TT, J)).astype(np.float32),
    "mask": rng.random((N, TT, J)) < 0.95,
    "std": np.ones((N, J), dtype=np.float32),
    "key_time_valid": np.ones((N, TT), dtype=bool),
    "meta": meta,
}
cache["pose_var"] = metrics.dynamic_variance(cache["target"], cache["mask"])
fit = np.flatnonzero(np.isin([m["user"] for m in meta], users[::2]))
held = np.flatnonzero(np.isin([m["user"] for m in meta], users[1::2]))
support = D.central_support(cache["mask"], D.lengths_for(cache), cache["key_time_valid"])
thr = metrics.training_threshold(cache["target"][fit], cache["mask"][fit])
pv = metrics.dynamic_variance(cache["target"], cache["mask"])
bundle = W.FoldBundle(cache, "real", (fit, held), (held, fit), (cache["target"],) * 2, (cache["target"],) * 2,
                      (pv, pv), (thr, thr), support, {})
pred148 = rng.standard_normal((32, 148, J)).astype(np.float32)
pred160 = rng.standard_normal((32, 160, J)).astype(np.float32)


def production_once():
    records = []
    for start in range(0, len(held), 32):
        rows = held[start:start + 32]
        x = torch.from_numpy(D.view_inputs(bundle.cache, "post", rows))
        p = pred148[:len(rows)]
        records.append(metrics.selection_statistics(p, W._metric_panel(bundle, 0, rows), bundle.thresholds[0]))
    return controls.pool_statistics(records)


# benchmark path (benchmark._dataset/_eval_model), reproduced without CUDA
from pose_temponet import benchmark as B
validation = B._dataset(cache, held, "post")
panel = D.subset(cache, held)
panel["pose_var"] = metrics.dynamic_variance(panel["target"], panel["mask"])


def benchmark_once():
    records = []
    n = len(validation.features)
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        x = validation.features[start:stop].to(dtype=torch.float32)
        p = pred160[:stop - start]
        records.append(metrics.selection_statistics(p, D.subset(panel, np.arange(start, stop)), thr))
    return controls.pool_statistics(records)


for name, fn in (("production _score_model host work", production_once), ("benchmark _eval_model host work", benchmark_once)):
    fn()
    times = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    print(f"{name}: median {np.median(times)*1e3:.1f} ms per held-fold evaluation ({len(held)} windows)")

t0 = time.perf_counter()
for _ in range(30):
    D.lengths_for(cache)
print("lengths_for(full cache) x30 (one production evaluation):", f"{(time.perf_counter()-t0)*1e3:.1f} ms")
