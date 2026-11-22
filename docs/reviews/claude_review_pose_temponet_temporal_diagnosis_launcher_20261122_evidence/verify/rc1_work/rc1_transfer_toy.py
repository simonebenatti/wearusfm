"""GIOCATTOLO (latenti SINTETICI, codice vero): la stima «ridge temporale = WLS temporale * (ridge/WLS nel mondo lineare)»
predice la ridge calcolata direttamente nel mondo temporale? Uso: rc1_transfer_toy.py <seme> <p_spettro>
Codice vero: controls.generate_components, controls.synthetic_target, linear.WindowedBasisFit (alpha 0 = WLS)."""
import sys, time, json
import numpy as np
from pose_temponet import controls, linear
seed, p = int(sys.argv[1]), float(sys.argv[2])
N_USERS, WPU, T, D, J, CROP = 40, 16, 160, 128, 20, 6
rng = np.random.default_rng(seed)
lam = (1+np.arange(D))**(-p)
Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
rho = rng.uniform(0.6, 0.98, size=D)
x = np.empty((N_USERS*WPU, T, D), dtype=np.float16); users = []
i = 0
for u in range(N_USERS):
    cu = 0.3*rng.normal(size=D)*lam
    for w in range(WPU):
        s = np.empty((T, D)); s[0] = rng.normal(size=D)
        innov = rng.normal(size=(T, D))*np.sqrt(1-rho**2)
        for t in range(1, T):
            s[t] = rho*s[t-1]+innov[t]
        x[i] = ((s*lam+cu)@Q.T).astype(np.float16); users.append("u%02d" % u); i += 1
n = len(x)
low = np.array([int(u[1:]) < 20 for u in users]); fit, held = np.flatnonzero(low), np.flatnonzero(~low)
meta = [{"user": users[k], "window": k} for k in range(n)]
support = np.ones((n, T-2*CROP, J), dtype=bool)
t0 = time.time()
c = controls.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=0, block_windows=16)
factor = linear.WindowedBasisFit(x, {k: c[k] for k in ("L", "T", "epsilon")}, support, fit, block_windows=16)
def pv(y): return y.var(axis=1).mean(axis=1)
def E(pred, y, dyn):
    e = pred-y; e = e-e.mean(axis=1, keepdims=True); cc = y-y.mean(axis=1, keepdims=True)
    return float((e[dyn]**2).sum()/(cc[dyn]**2).sum())
out = {"seed": seed, "p": p}
W = {}
for world, a, sigma in (("linear", 0.0, 1.35), ("temporal", 0.4, 1.3)):
    ystar, y = controls.synthetic_target(c, a, sigma)
    thr = float(np.median(pv(ystar[fit]))); dyn = pv(ystar[held]) >= thr
    yh, ysh = y[held][:, CROP:-CROP], ystar[held][:, CROP:-CROP]
    e_or = E(ysh, yh, dyn)
    e_cl = E(factor.predict_mixture(held, {"L": 1., "T": a}, 0.)[0], yh, dyn)
    row = {"E_oracle": e_or, "E_cleanLS": e_cl}
    for alpha in (0.0, 0.01, 0.1, 1.0, 10.0):
        row["E_alpha_%g" % alpha] = E(factor.predict_mixture(held, {"L": 1., "T": a, "epsilon": sigma}, alpha)[0], yh, dyn)
    W[world] = row
lin, tem = W["linear"], W["temporal"]
margin = tem["E_cleanLS"]-tem["E_oracle"]
print("seme %d p %.2f  (%.0f s) margine temporale %.6f (%.2f%% di clean_ls)" % (seed, p, time.time()-t0, margin, 100*margin/tem["E_cleanLS"]))
ex_w = lin["E_alpha_0"]-lin["E_oracle"]; ex_wt = tem["E_alpha_0"]-tem["E_cleanLS"]
print("  WLS0: R temporale %.2f%% ; eccesso lineare %.6f" % (-100*ex_wt/margin, ex_w))
for alpha in (0.01, 0.1, 1.0, 10.0):
    ex_r = lin["E_alpha_%g" % alpha]-lin["E_oracle"]
    direct = -(tem["E_alpha_%g" % alpha]-tem["E_cleanLS"])/margin
    est = -ex_wt*(ex_r/ex_w)/margin
    print("  alpha %-5g rapporto ridge/WLS0 (lineare) %.4f | R temporale diretto %+.2f%% | stimato col metodo %+.2f%% | scarto %+.2f punti" % (
        alpha, ex_r/ex_w, 100*direct, 100*est, 100*(est-direct)))
    out["alpha_%g" % alpha] = {"ratio_lin": ex_r/ex_w, "R_direct": direct, "R_est": est}
out["R_wls0"] = -ex_wt/margin
print(json.dumps(out))
