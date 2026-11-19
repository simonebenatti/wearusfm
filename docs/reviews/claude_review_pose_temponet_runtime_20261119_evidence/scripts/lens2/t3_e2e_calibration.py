"""Independent dense recomputation of E_cal, E_cleanLS, E_oracle, alpha for a few grid pairs."""
import numpy as np
from pose_temponet import controls as C
from pose_temponet.metrics import dynamic_variance

rng = np.random.default_rng(7)
N, T, D, J, crop = 12, 20, 4, 2, 6
x = rng.normal(size=(N, T, D)).astype(np.float16)
mask = np.ones((N, T, J), dtype=bool)
mask[1, :9, 0] = False      # joint-specific pattern
mask[4, 15:, 1] = False
z = x.astype(np.float64); z = z - z.mean(-1, keepdims=True); z = z/np.sqrt((z*z).mean(-1, keepdims=True)+1e-5)
L = z @ rng.normal(size=(D, J)); Tm = np.roll(z, 2, axis=1) @ rng.normal(size=(D, J)) + 0.3*np.roll(z, -1, axis=1) @ rng.normal(size=(D, J))
eps = rng.normal(size=(N, T, J))
real = L + 0.5*Tm + 0.8*rng.normal(size=(N, T, J))
users = [f"u{i//1:02d}" for i in range(N)]  # one window per user
data = {"lat": x, "target": real, "mask": mask, "std": np.ones((N, J)), "pose_var": dynamic_variance(real, mask),
        "meta": [{"user": u, "n_patches": T, "i": i} for i, u in enumerate(users)]}
comps = {"L": L, "T": Tm, "epsilon": eps}
fit = np.arange(0, N, 2); held = np.arange(1, N, 2)
res = C.build_positive_control_fold(data, fit, held, users, outer_fold=0, components=comps, block_windows=3)
rows = {(r["a"], r["sigma"]): r for r in res["calibration"]["curve"]}

support = C.central_support if False else None
from pose_temponet.data import central_support
sup = central_support(mask, np.full(N, T, dtype=np.int64))

def fit_ridge(idx, y, alpha):
    coefs = np.zeros((D+1, J))
    for j in range(J):
        Xs, Ys, Ws = [], [], []
        for i in idx:
            q = sup[i].sum()
            for t in range(sup.shape[1]):
                if sup[i, t, j]:
                    Xs.append(z[i, t+crop]); Ys.append(y[i, t+crop, j]); Ws.append(1.0/q)
        X, Y, W = map(np.asarray, (Xs, Ys, Ws))
        xm = (W[:, None]*X).sum(0)/W.sum(); ym = (W*Y).sum()/W.sum()
        Xc, Yc = X-xm, Y-ym
        if alpha == 0:
            A = np.sqrt(W)[:, None]*np.concatenate([np.ones((len(X), 1)), X], 1)
            sol = np.linalg.lstsq(A, np.sqrt(W)*Y, rcond=None)[0]
            coefs[:, j] = sol
        else:
            b = np.linalg.solve(Xc.T @ (W[:, None]*Xc) + alpha*np.eye(D), Xc.T @ (W*Yc))
            coefs[0, j] = ym - xm @ b; coefs[1:, j] = b
    return coefs

def predict(idx, c):
    return z[idx, crop:T-crop] @ c[1:] + c[0]

def score(idx, pred, y, pv_source, thr):
    s = sup[idx]; yy = y[idx, crop:T-crop]
    q = s.sum((1, 2)); keep = q > 0
    mse = (((pred-yy)**2*s).sum((1, 2))[keep]/q[keep]).sum(); n = keep.sum()
    pv = dynamic_variance(pv_source[idx], mask[idx]); dyn = np.isfinite(pv) & (pv >= thr)
    num = den = 0.0
    for w in np.flatnonzero(dyn):
        for j in range(J):
            m = s[w, :, j]
            if m.sum() >= 2:
                e = (pred-yy)[w, m, j]; yv = yy[w, m, j]
                num += ((e-e.mean())**2).sum(); den += ((yv-yv.mean())**2).sum()
    return mse, n, num, den

def thr_of(idx, ysrc):
    pv = dynamic_variance(ysrc[idx], mask[idx]); return max(float(np.median(pv[np.isfinite(pv)])), 1e-12)

inner = C.inner_partitions(data["meta"], fit)
def pair(a, sigma):
    ystar = L + a*Tm; y = ystar + sigma*eps
    def pooled(alpha, ytrain, which):
        tot = np.zeros(4)
        for f, h in inner:
            c = fit_ridge(f, ytrain, alpha)
            tot += np.asarray(score(h, predict(h, c), y, ystar, thr_of(f, ystar)))
        return tot
    curve = {}
    for k in range(-5, 8):
        t = pooled(10.**k, y, "ridge"); curve[k] = t[0]/t[1]
    # module's bounded selection replicated: start -2..4, extend
    ks = list(range(-2, 5))
    while True:
        w = min(ks, key=lambda k: (curve[k], k))
        if w == min(ks) and w > -5: ks.append(min(ks)-1)
        elif w == max(ks) and w < 7: ks.append(max(ks)+1)
        else: break
    t = pooled(10.**w, y, "ridge"); e_cal = t[2]/t[3]
    t = pooled(0.0, ystar, "clean"); e_cls = t[2]/t[3]
    tot = np.zeros(4)
    for f, h in inner:
        tot += np.asarray(score(h, ystar[h, crop:T-crop], y, ystar, thr_of(f, ystar)))
    e_or = tot[2]/tot[3]
    return 10.**w, e_cal, e_cls, e_or

for a, sigma in [(0.0, 0.5), (0.4, 1.0), (1.2, 2.5), (3.0, 6.0), (0.2, 0.0)]:
    alpha, e_cal, e_cls, e_or = pair(a, sigma)
    r = rows[(a, sigma)]
    print(f"a={a} s={sigma}: alpha {alpha:g} vs {r['alpha']:g} | E_cal {e_cal:.12f} vs {r['E_cal']:.12f} | "
          f"E_cleanLS {e_cls:.12f} vs {r['E_cleanLS']:.12f} | E_oracle {e_or:.12f} vs {r['E_oracle']:.12f}")
# E_real
tot = np.zeros(4); curve = {}
for k in range(-5, 8):
    t = np.zeros(4)
    for f, h in inner:
        c = fit_ridge(f, real, 10.**k); t += np.asarray(score(h, predict(h, c), real, real, thr_of(f, real)))
    curve[k] = t
ks = list(range(-2, 5))
while True:
    w = min(ks, key=lambda k: (curve[k][0]/curve[k][1], k))
    if w == min(ks) and w > -5: ks.append(min(ks)-1)
    elif w == max(ks) and w < 7: ks.append(max(ks)+1)
    else: break
print("E_real", curve[w][2]/curve[w][3], "vs", res["real_reference"]["E_dyn"], "alpha", 10.**w, res["real_reference"]["alpha"])
print("status", res["calibration"]["status"], "evaluations", res["calibration"]["evaluations"])
