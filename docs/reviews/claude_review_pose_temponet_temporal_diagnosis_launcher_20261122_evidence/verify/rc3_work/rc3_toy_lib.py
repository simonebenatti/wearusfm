"""Dati del giocattolo di rc3_toy.py (stessi latenti, seme 777) come modulo importabile."""
import numpy as np, torch
torch.set_num_threads(1)
from pose_temponet import controls, linear
NU, WPU, TL, D, J, CROP = 40, 30, 160, 384, 20, 6
A, ST, SL = 0.4, 1.3, 1.35
def make_latents(seed=777, K=48):
    rng = np.random.default_rng(seed)
    lam = np.arange(1, K+1)**-0.75; lam = lam/np.sqrt((lam**2).sum())
    rho = np.linspace(0.5, 0.98, K)
    base = rng.normal(size=(D, K))
    x = np.empty((NU*WPU, TL, D), dtype=np.float16); users = []
    for u in range(NU):
        Au = (base + 0.3*rng.normal(size=(D, K)))*lam
        cu = 0.3*rng.normal(size=D)
        s = np.empty((WPU, TL, K)); s[:, 0] = rng.normal(size=(WPU, K))
        inn = rng.normal(size=(WPU, TL, K))*np.sqrt(1-rho**2)
        for t in range(1, TL): s[:, t] = rho*s[:, t-1] + inn[:, t]
        x[u*WPU:(u+1)*WPU] = (s@Au.T + cu + 0.12*rng.normal(size=(WPU, TL, D))).astype(np.float16)
        users += ["u%02d" % u]*WPU
    return x, users
def pose_var(y): return y.var(axis=1).mean(axis=1)
def stats(pred, y, dyn):
    e = pred-y; mse = float((e**2).mean(axis=(1, 2)).mean())
    e = e-e.mean(axis=1, keepdims=True); c = y-y.mean(axis=1, keepdims=True)
    return float((e[dyn]**2).sum()/(c[dyn]**2).sum()), mse
x, users = make_latents(); n = len(x)
low = np.array([int(u[1:]) < 20 for u in users]); fit, held = np.flatnonzero(low), np.flatnonzero(~low)
meta = [{"user": users[i], "window": i} for i in range(n)]
full = np.ones((n, TL-2*CROP, J), dtype=bool)
c = controls.generate_components(x, meta, full, fit, held, sorted(set(users)), outer_fold=0, block_windows=16)
F = linear.WindowedBasisFit(x, {k: c[k] for k in ("L", "T", "epsilon")}, full, fit, block_windows=16)
