"""L4-4: is the charged '17 x evaluate(per_joint=True, batch 64)' enough to cover the
real metrics stage's 15 x per_joint(batch 32) + 2 x plain(batch 32) + 2 x plain(batch 64, R-1)?
I/O terms are separate (not measured here)."""
import time
import numpy as np
from pose_temponet import metrics, data as D
rng = np.random.default_rng(2)
n, users = 1907, 15
lengths = rng.integers(140, 161, n)
avail = np.arange(160)[None, :] < lengths[:, None]
per = -(-n // users)
data = {"target": rng.standard_normal((n, 160, 20)).astype(np.float32),
        "mask": np.broadcast_to(avail[:, :, None], (n, 160, 20)).copy(), "key_time_valid": avail.copy(),
        "std": np.ones((n, 20), np.float32), "pose_var": rng.uniform(0, .6, n),
        "meta": [{"user": f"u{i // per:02d}", "n_patches": int(lengths[i])} for i in range(n)]}
pred160 = (data["target"] * .75).astype(np.float32)
pred148 = pred160[:, 6:-6].astype(np.float64)
def t(f, k=2):
    v = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
    return min(v)
charged = t(lambda: metrics.evaluate(pred160, data, dynamic_threshold=metrics.THRESHOLD, per_joint=True))  # benchmark.py:546
pj32 = t(lambda: metrics.evaluate(pred148.astype(np.float32), data, batch_windows=32, per_joint=True))      # runner.py:362
plain32 = t(lambda: metrics.evaluate(pred148, data, batch_windows=32))                                     # runner.py:342
plain64 = t(lambda: metrics.evaluate(pred148, data))                                                       # R-1 gate, metrics.py:178
print(f"per_joint b64 {charged:.3f}s  per_joint b32 {pj32:.3f}s  plain b32 {plain32:.3f}s  plain b64 {plain64:.3f}s")
print(f"charged 17*b64pj = {17*charged:.2f}s   real 15*b32pj + 2*b32 + 2*b64 = {15*pj32 + 2*plain32 + 2*plain64:.2f}s")
