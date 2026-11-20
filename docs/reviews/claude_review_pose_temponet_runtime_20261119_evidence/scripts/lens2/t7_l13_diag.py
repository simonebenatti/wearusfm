"""Is the descriptive 'temporal_margin_recovery_fitting' inflated by in-sample noise fitting?"""
import numpy as np
from pose_temponet import controls as C
from pose_temponet.metrics import dynamic_variance
rng = np.random.default_rng(3)
N, T, D, J = 120, 40, 24, 2
x = rng.normal(size=(N, T, D)).astype(np.float16)
z = x.astype(np.float64); z -= z.mean(-1, keepdims=True); z /= np.sqrt((z*z).mean(-1, keepdims=True)+1e-5)
L = z @ rng.normal(size=(D, J)) / np.sqrt(D)
Tm = (np.roll(z, 3, axis=1) - np.roll(z, -3, axis=1)) @ rng.normal(size=(D, J)) / np.sqrt(D)
L = (L - L.mean((0, 1)))/L.std((0, 1)); Tm = (Tm - Tm.mean((0, 1)))/Tm.std((0, 1))
users = [f"u{i:03d}" for i in range(N)]
meta = [{"user": u, "n_patches": T} for u in users]
eps = C.window_noise(7, meta, T, J)
mask = np.ones((N, T, J), dtype=bool)
data = {"lat": x, "target": L + 0.3*Tm + rng.normal(size=(N, T, J)), "mask": mask, "std": np.ones((N, J)), "meta": meta}
data["pose_var"] = dynamic_variance(data["target"], mask)
fit, held = np.arange(0, N, 2), np.arange(1, N, 2)
comps = {"L": L, "T": Tm, "epsilon": eps}
sel = {"linear": {"a": 0.0, "sigma": 1.0, "alpha": 1.0}, "temporal": {"a": 0.6, "sigma": 1.5, "alpha": 1.0}}
out = C.evaluate_outer_references(data, comps, fit, held, sel, block_windows=8)
d = out["linear_diagnostics"]
for name in ("L0", "L13"):
    print(name, "temporal_margin_recovery_fitting =", round(d[name]["temporal_margin_recovery_fitting"], 4))
# A held-side analogue (fit on observed, score on outer held) for comparison
from pose_temponet.metrics import selection_statistics
ystar, y = C.synthetic_target(comps, 0.6, 1.5)
thr = C._threshold(dynamic_variance(ystar[fit], mask[fit]))
def E(pred, idx):
    return C.statistic_values(selection_statistics(pred, {"target": y[idx], "mask": mask[idx], "pose_var": dynamic_variance(ystar[idx], mask[idx]), "std": np.ones((len(idx), J)), "meta": [meta[i] for i in idx]}, thr))["E_dyn"]
print("note: for L0 the baseline IS the L0 WLS on clean y*, so a pure temporal-recovery reading would be 0")
