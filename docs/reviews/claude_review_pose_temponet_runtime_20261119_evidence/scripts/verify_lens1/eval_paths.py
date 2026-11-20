"""Independent check of L1-1: call the REAL production evaluator (workflow._score_model)
and the REAL benchmark evaluator (benchmark._eval_model) on the same CPU fixture and the
same zero-cost module, so only the host work around the forward differs.
Fixture: 1907 windows grouped by user (contiguous blocks), FP16 lat/pre, pred present,
variable n_patches (>12), 40 users, alternating user folds as in production."""
import sys, time
import numpy as np, torch
from torch import nn
from pose_temponet import data as D, metrics, workflow as W, benchmark as B, controls

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
rng = np.random.default_rng(1)
N, TT, F, J = 1907, 160, 384, 20
users = [f"u{u:02d}" for u in range(40)]
user_of = np.sort(rng.integers(0, 40, N)); user_of[:40] = np.arange(40); user_of = np.sort(user_of)
lengths = rng.integers(140, 161, N); lengths[rng.random(N) < .6] = 160
meta = [{"user": users[user_of[i]], "n_patches": int(lengths[i]), "recording": f"r{i//10}", "start": i % 10} for i in range(N)]
avail = np.arange(TT)[None, :] < lengths[:, None]
mask = (rng.random((N, TT, J)) < .97) & avail[..., None]
cache = {"lat": rng.standard_normal((N, TT, F), dtype=np.float32).astype(np.float16),
         "pre": rng.standard_normal((N, TT, F), dtype=np.float32).astype(np.float16),
         "pred": rng.standard_normal((N, TT, J), dtype=np.float32),
         "target": rng.standard_normal((N, TT, J), dtype=np.float32), "mask": mask,
         "std": np.ones((N, J), np.float32), "key_time_valid": avail.copy(), "meta": meta,
         "config": {"split": "train", "which": "teacher", "checkpoint_step": 49046}}
cache["pose_var"] = metrics.dynamic_variance(cache["target"], cache["mask"])
folds = D.alternating_user_folds(cache)
fit = np.flatnonzero(np.isin([m["user"] for m in meta], folds[0]))
held = np.flatnonzero(np.isin([m["user"] for m in meta], folds[1]))
support = D.central_support(cache["mask"], D.lengths_for(cache), cache["key_time_valid"])
thr = metrics.training_threshold(cache["target"][fit], cache["mask"][fit])
pv = metrics.dynamic_variance(cache["target"], cache["mask"])
bundle = W.FoldBundle(cache, "real", (fit, held), (held, fit), (cache["target"],)*2, (cache["target"],)*2,
                      (pv, pv), (thr, thr), support, {})

class Zero(nn.Module):
    def forward(self, x):
        return x[..., :J] * 0.0
model = Zero()

validation = B._dataset(cache, held, "post")          # benchmark: built once, outside the timed eval
panel = D.subset(cache, held)
panel["pose_var"] = metrics.dynamic_variance(panel["target"], panel["mask"])

def prod():
    return W._score_model(model, bundle, 0, held, "post", "cpu")
def bench():
    return B._eval_model(model, validation, panel, thr, device="cpu")

def med(fn, reps=7):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return np.median(ts)*1e3, min(ts)*1e3

rp, rb = prod(), bench()[1]
print("held windows", len(held), "torch threads", torch.get_num_threads())
print("same pooled stats (MSE sums equal):", np.isclose(rp["mse_sum"], rb["mse_sum"]) if isinstance(rp, dict) else (rp, rb))
for name, fn in (("PROD  workflow._score_model", prod), ("BENCH benchmark._eval_model", bench)):
    m, lo = med(fn); print(f"{name}: median {m:.1f} ms, min {lo:.1f} ms per held-fold evaluation")
rows = [held[s:s+32] for s in range(0, len(held), 32)]
pos = [(s, min(s+32, len(held))) for s in range(0, len(held), 32)]
m, _ = med(lambda: [torch.from_numpy(D.view_inputs(cache, "post", r)) for r in rows]); print(f"  prod input prep (view_inputs) {m:.1f} ms")
m, _ = med(lambda: [validation.features[a:b].to(dtype=torch.float32) for a, b in pos]); print(f"  bench input prep (torch cast) {m:.1f} ms")
m, _ = med(lambda: [W._metric_panel(bundle, 0, r) for r in rows]); print(f"  prod _metric_panel {m:.1f} ms")
m, _ = med(lambda: [D.subset(panel, np.arange(a, b)) for a, b in pos]); print(f"  bench D.subset(panel) {m:.1f} ms")
