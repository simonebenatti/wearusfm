"""Verifier check for L2-2 at production-like p/N ratios (L13 p/N ~3.5%, L0 ~0.27%).

Uses the real generate_components (teacher TCN), window_noise, WindowedBasisFit and
_outer_linear_diagnostics. Compares the code's temporal_margin_recovery_fitting with a
like-for-like alternative (same design fitted on clean y*, scored on observed y).
"""
import sys, time
import numpy as np
from pose_temponet import controls as C
from pose_temponet.linear import WindowedBasisFit
from pose_temponet.metrics import selection_statistics, training_threshold, dynamic_variance

rho_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
rng = np.random.default_rng(11)
U, W, Tn, D, J = 40, 8, 160, 64, 4          # 320 windows; 160 fit windows -> 23,680 rows/joint
N = U*W
x = np.empty((N, Tn, D))
x[:, 0] = rng.normal(size=(N, D))
for t in range(1, Tn):
    x[:, t] = rho_x*x[:, t-1] + np.sqrt(1-rho_x**2)*rng.normal(size=(N, D))
x += 0.5*rng.normal(size=(N, 1, D))           # per-window offsets
x = x.astype(np.float16)
users = [f"u{u:03d}" for u in range(U) for _ in range(W)]
meta = [{"user": users[i], "window": i, "n_patches": Tn} for i in range(N)]
mask = np.ones((N, Tn, J), dtype=bool)
support = np.ones((N, Tn-12, J), dtype=bool)
fit = np.asarray([i for i in range(N) if int(users[i][1:]) % 2 == 0])
held = np.asarray([i for i in range(N) if int(users[i][1:]) % 2 == 1])
t0 = time.time()
comps = C.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=0, n_joints=J, block_windows=32)
data = {"lat": x, "target": comps["L"] + 0.5*comps["T"] + comps["epsilon"], "mask": mask,
        "std": np.ones((N, J)), "meta": meta}
basis = {"real": data["target"], "L": comps["L"], "T": comps["T"], "epsilon": comps["epsilon"]}
f0 = WindowedBasisFit(x, basis, support, fit, block_windows=32)
f13 = WindowedBasisFit(x, basis, support, fit, context=6, block_windows=32)
print(f"setup {time.time()-t0:.1f}s; fit rows/joint={len(fit)*148}; p(L0)={f0.dimension+1}; p(L13)={f13.dimension+1}")

def E(pred, idx, y, ystar, thr):
    return C.statistic_values(selection_statistics(pred, C._metric_panel(data, idx, y, ystar), thr))["E_dyn"]

# scan a few worlds and pick one in the fitting band [0.06,0.10] near E~0.78
best = None
for a in (0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0):
    clean0, _ = f0.predict_mixture(fit, {"L": 1., "T": a})
    for sigma in np.arange(0.5, 6.01, 0.25):
        ystar, y = C.synthetic_target(comps, a, sigma)
        thr = training_threshold(ystar[fit], mask[fit])
        ecl = E(clean0, fit, y, ystar, thr); eor = E(ystar[fit, 6:-6], fit, y, ystar, thr)
        gain = (ecl-eor)/ecl
        if 0.06 <= gain <= 0.10:
            score = (abs(gain-.08), abs(ecl-.78))
            if best is None or score < best[0]:
                best = (score, a, float(sigma), ecl, eor, gain)
_, a, sigma, ecl, eor, gain = best
print(f"chosen world a={a} sigma={sigma}: E_cleanLS(fit)={ecl:.4f} E_oracle(fit)={eor:.4f} gain={gain:.4f}")
sel = {"linear": {"a": 0.0, "sigma": sigma, "alpha": 1.0}, "temporal": {"a": a, "sigma": sigma, "alpha": 1.0}}
diag = C._outer_linear_diagnostics(data, comps, fit, held, sel, f0, block_windows=32, include_l13=True)
ystar, y = C.synthetic_target(comps, a, sigma)
thr = training_threshold(ystar[fit], mask[fit])
for name, fac in (("L0", f0), ("L13", f13)):
    code_val = diag[name]["temporal_margin_recovery_fitting"]
    clean_pred, _ = fac.predict_mixture(fit, {"L": 1., "T": a})          # like-for-like: fit on y*
    obs_pred, _ = fac.predict_mixture(fit, {"L": 1., "T": a, "epsilon": sigma})
    held_obs, _ = fac.predict_mixture(held, {"L": 1., "T": a, "epsilon": sigma})
    held_clean, _ = fac.predict_mixture(held, {"L": 1., "T": a})
    alt = C._recovery(ecl, E(clean_pred, fit, y, ystar, thr), eor)
    chk = C._recovery(ecl, E(obs_pred, fit, y, ystar, thr), eor)
    # held-outer view of the same two fits, baseline = L0 clean on held
    h0, _ = f0.predict_mixture(held, {"L": 1., "T": a})
    thr_h = thr
    eclh = E(h0, held, y, ystar, thr_h); eorh = E(ystar[held, 6:-6], held, y, ystar, thr_h)
    print(f"{name}: code temporal_margin_recovery_fitting={code_val:.4f} (recomputed {chk:.4f}) | "
          f"fit-on-y* (like E_cleanLS) recovery={alt:.4f} | inflation={code_val-alt:+.4f} | "
          f"held-outer: obs-fit={C._recovery(eclh, E(held_obs, held, y, ystar, thr_h), eorh):.4f} "
          f"clean-fit={C._recovery(eclh, E(held_clean, held, y, ystar, thr_h), eorh):.4f}")
print("T_unread_fraction_fitting:", {k: round(diag[k]["T_unread_fraction_fitting"], 4) for k in ("L0", "L13")})
