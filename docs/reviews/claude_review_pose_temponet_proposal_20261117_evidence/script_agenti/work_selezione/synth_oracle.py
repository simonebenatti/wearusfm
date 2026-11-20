"""Tetto teorico di E_dyn nei due mondi SINTETICI: predittore oracolo = bersaglio senza il rumore eta (la parte non prevedibile).
Rigenera gli stessi dati di synth_selection.generate (stesse chiamate al generatore nello stesso ordine) tenendo da parte eta.
Verifica che x e y coincidano con quelli di S.generate. Nessun dato del progetto."""
import sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S
T, D, J = S.T, S.D, S.J

def generate_parts(world, seed, n_users, w, p):
    rng = np.random.default_rng(seed)
    n = n_users * w
    x = np.empty((n, T, D), np.float16); y = np.empty((n, T, J), np.float32); e = np.empty((n, T, J), np.float32)
    pad = 3
    for u in range(n_users):
        g = rng.normal(0, p["sig_g"], D); o = rng.normal(0, p["sig_o"], J)
        c = rng.normal(0, 1, (w, world["Bc"].shape[0])); a = np.where(rng.random(w) < 0.5, 1.0, p["a_static"])
        f = S.ar1(rng, w, T + 2 * pad, world["tau"])
        nw = rng.normal(0, p["sig_n"], (w, D)); lw = rng.normal(0, p["sig_l"], (w, J))
        eta = S.ar1(rng, w * J, T, [5.0]).reshape(w, J, T).transpose(0, 2, 1) * p["sig_eta"]
        fc = f[:, pad:pad + T]
        z = (world["M"] + g)[None, None, :] + nw[:, None, :] + p["s_c"] * (c @ world["Bc"])[:, None, :] \
            + p["s_f"] * a[:, None, None] * (fc @ world["Bf"]) + rng.normal(0, p["sig_eps"], (w, T, D))
        h = np.tanh(2.0 * np.stack([f[:, k:k + T] for k in range(2 * pad + 1)]).mean(0)) if p["temporal"] else fc
        t = o[None, None, :] + lw[:, None, :] + p["t_c"] * (c @ world["Ac"])[:, None, :] + p["t_f"] * a[:, None, None] * (h @ world["Af"]) + eta
        sl = slice(u * w, (u + 1) * w)
        x[sl], y[sl], e[sl] = z.astype(np.float16), t.astype(np.float32), eta.astype(np.float32)
    return x, y, e

base = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25)
for temporal in (False, True):
    p = dict(base, temporal=temporal)
    world = S.make_world(1001)
    xtr, ytr, etr = generate_parts(world, 2001, 40, 48, p)
    xva, yva, eva = generate_parts(world, 3001, 15, 48, p)
    x2, y2, _ = S.generate(world, 3001, 15, 48, p)
    same = np.array_equal(x2, xva) and np.array_equal(y2, yva)
    flat = ytr.reshape(-1, J); mean, std = flat.mean(0), flat.std(0)
    ys = (yva - mean) / std; es = eva / std           # stessa standardizzazione di S.standardize
    thr = float(np.median(ys.var(axis=1).mean(axis=1))); dyn = ys.var(axis=1).mean(axis=1) >= thr
    sl = S.SL
    oracle = ys - es                                   # tutto tranne eta
    def edyn(pred, tgt, sel):
        cp = pred[sel][:, sl] - pred[sel][:, sl].mean(1, keepdims=True); ct = tgt[sel][:, sl] - tgt[sel][:, sl].mean(1, keepdims=True)
        return float(((cp - ct) ** 2).sum() / (ct ** 2).sum())
    print(f"mondo {'temporale' if temporal else 'lineare'}: dati identici a S.generate: {same}; soglia {thr:.4f}; "
          f"E_dyn dell'oracolo (bersaglio senza eta) sui val dinamici {edyn(oracle, ys, dyn):.4f}; E_dyn del predittore nullo {edyn(np.zeros_like(ys), ys, dyn):.4f}")
