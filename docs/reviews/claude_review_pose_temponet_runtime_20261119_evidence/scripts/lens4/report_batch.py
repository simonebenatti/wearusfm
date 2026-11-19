"""Lens 4 check: cost of metrics.evaluate(per_joint=True) at the benchmark's default batch (64)
versus the metrics stage's batch_windows=32, on a train-like (40 users) and a val-like (15 users) population."""
import time, sys
import numpy as np
from pose_temponet import metrics
rng = np.random.default_rng(0)
def fake(n, users):
    target = rng.standard_normal((n,160,20)).astype(np.float32)
    lengths = rng.integers(140, 161, n)
    mask = np.arange(160)[None,:,None] < lengths[:,None,None]
    mask = np.broadcast_to(mask, (n,160,20)).copy()
    per = -(-n//users)
    meta = [{"user": f"u{i//per:02d}", "n_patches": int(lengths[i])} for i in range(n)]
    return {"target": target, "mask": mask, "meta": meta, "std": np.ones((n,20), np.float32),
            "pose_var": rng.uniform(0, .6, n), "key_time_valid": np.arange(160)[None,:] < lengths[:,None]}
for n, users in ((1907, 40), (4000, 15)):
    data = fake(n, users)
    pred = data["target"]*.75
    out = {}
    for bw in (64, 32):
        t0 = time.perf_counter(); metrics.evaluate(pred, data, dynamic_threshold=metrics.THRESHOLD, per_joint=True, batch_windows=bw)
        out[bw] = time.perf_counter()-t0
    print(f"n={n} users={users}: batch64 {out[64]:.3f}s  batch32 {out[32]:.3f}s  ratio32/64 {out[32]/out[64]:.2f}")
