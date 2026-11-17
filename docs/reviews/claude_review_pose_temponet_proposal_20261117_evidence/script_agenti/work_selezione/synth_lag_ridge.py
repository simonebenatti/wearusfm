"""SINTETICO, mondo temporale: l'informazione temporale e' imparabile da 20 utenti da un lettore temporale REGOLARIZZATO?
Ridge in forma chiusa (LN senza stato, crop6, intercetta non penalizzata) con ritardi {-6,-4,-2,0,2,4,6} patch (7 x 384 = 2688 feature),
allenata sul fold 0 e letta sul fold 1, griglia di alpha a decadi; confronto con la ridge istantanea (ritardo 0) sullo stesso fold.
Stessi dati di synth_selection.generate. Nessun dato del progetto."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
T, D, J, C = S.T, S.D, S.J, S.CROP
p = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25, temporal=True)
world = S.make_world(1001)
xtr, ytr, utr = S.generate(world, 2001, 40, 48, p)
xva, yva, uva = S.generate(world, 3001, 15, 48, p)
ytr, yva = S.standardize(ytr, yva)
thr = float(np.median(S.pose_var(yva))); dtr = S.pose_var(ytr) >= thr
ids = np.arange(40); fm = [np.isin(utr, f) for f in (ids[::2], ids[1::2])]
(x0, y0), (x1, y1, d1) = (xtr[fm[0]], ytr[fm[0]]), (xtr[fm[1]], ytr[fm[1]], dtr[fm[1]])

def feats(x, lags):
    z = torch.nn.functional.layer_norm(torch.from_numpy(x.astype(np.float32)), (D,), eps=1e-5).double()
    return torch.cat([z[:, C + l:T - C + l] for l in lags], dim=2)  # (B, 148, D*len(lags))

def stats(x, y, lags, chunk=32):
    k = D * len(lags); sx = torch.zeros(k, dtype=torch.float64); sy = torch.zeros(J, dtype=torch.float64)
    xx = torch.zeros(k, k, dtype=torch.float64); xy = torch.zeros(k, J, dtype=torch.float64); n = 0
    for i in range(0, len(x), chunk):
        a = feats(x[i:i + chunk], lags).reshape(-1, k); b = torch.from_numpy(y[i:i + chunk, C:T - C]).double().reshape(-1, J)
        sx += a.sum(0); sy += b.sum(0); xx += a.T @ a; xy += a.T @ b; n += len(a)
    return sx, sy, xx, xy, n

def evaluate(x, y, dyn, lags, w, b, chunk=32):
    mse, n, csse, css = 0.0, 0, 0.0, 0.0
    for i in range(0, len(x), chunk):
        pr = feats(x[i:i + chunk], lags) @ w + b; t = torch.from_numpy(y[i:i + chunk, C:T - C]).double(); d = torch.from_numpy(dyn[i:i + chunk])
        mse += float(((pr - t) ** 2).mean(dim=(1, 2)).sum()); n += len(pr)
        cp, ct = pr - pr.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
        csse += float(((cp - ct) ** 2).sum(dim=(1, 2))[d].sum()); css += float((ct ** 2).sum(dim=(1, 2))[d].sum())
    return mse / n, csse / css

for name, lags in (("istantanea", [0]), ("ritardi -6..6 passo 2", [-6, -4, -2, 0, 2, 4, 6])):
    t0 = time.time()
    sx, sy, xx, xy, n = stats(x0, y0, lags)
    mx, my = sx / n, sy / n
    gram = xx - torch.outer(sx, sx) / n
    eig, vec = torch.linalg.eigh((gram + gram.T) / 2); eig = eig.clamp_min(0)
    cross = vec.T @ (xy - torch.outer(sx, my))
    rows = []
    for e in range(2, 9):
        w = vec @ (cross / (eig[:, None] + 10.0 ** e)); b = my - mx @ w
        rows.append((e,) + evaluate(x1, y1, d1, lags, w, b))
    best_mse = min(rows, key=lambda r: r[1]); best_dyn = min(rows, key=lambda r: r[2])
    print(f"ridge {name}: " + ", ".join(f"1e{e}: {m:.4f}/{dd:.4f}" for e, m, dd in rows) +
          f" | miglior MSE alpha 1e{best_mse[0]} ({best_mse[1]:.4f}, E_dyn {best_mse[2]:.4f}); miglior E_dyn 1e{best_dyn[0]} ({best_dyn[2]:.4f}) ({time.time() - t0:.0f}s)", flush=True)
