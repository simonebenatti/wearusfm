#!/usr/bin/env python
"""Esperimento SINTETICO per la review della dimensione 3 (selezione) della proposta TEMPONet-lettore.

NESSUN dato del progetto: feature e bersagli sono generati qui. Serve a mostrare un MECCANISMO
(bilancio di ottimizzazione, granularita' dell'arresto, regola del rifit), non a stimare numeri veri.

Ricetta delle teste e dell'ottimizzazione: quella del protocollo
docs/pose_temponet_frozen_protocol_20261117.md (righe 29-35, 61-69):
- input (B,160,384) -> LayerNorm affine eps 1e-5 -> testa; uscita 20; ultimo strato a zero;
- loss: media dell'errore quadratico standardizzato sugli elementi validi di ogni finestra (crop6), poi media sulle finestre;
- AdamW betas (0.9,0.999), eps 1e-8, weight decay 0.01 solo su matrici/kernel, clipping norma 1, FP32;
- batch 32 finestre, ultimo batch conservato, shuffle per epoca da un generatore dedicato;
- LR {1e-4, 1e-3} costante, 50 epoche; due fold di utenti (ID ordinati, alternati); seme 0;
- scelta (LR, epoca) sul minimo della MSE per finestra messa in comune sui due fold; pareggi: epoca precedente, poi LR minore;
- rifit sui 40 utenti con lo stesso numero di epoche.

Generatore (tutto inventato, dichiarato nel rapporto):
z_t = M + g_u + n_w + c_w B_c + a_w f_t B_f + eps_t           (feature prima della LayerNorm, salvate in float16)
y_t = o_u + l_w + c_w A_c + a_w h(f)_t A_f + eta_t              (20 bersagli, poi standardizzati con media/dev. del train)
g_u, o_u: firma e scostamento dell'utente; n_w, l_w: disturbo e livello non osservato della finestra;
c_w: postura della finestra (livello leggibile); f_t: fattori AR(1) (dinamica); a_w: attivita' (finestre statiche/dinamiche);
h = identita' (bersaglio lineare istantaneo) oppure tanh(2 * media mobile a 7 patch) (bersaglio temporale non lineare).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

T, D, J, CROP = 160, 384, 20, 6
SL = slice(CROP, T - CROP)


# --------------------------------------------------------------------------------------- dati sintetici
def make_world(seed, kc=6, kf=10):
    rng = np.random.default_rng(seed)
    return {"M": rng.normal(0, 1, D), "Bc": rng.normal(0, 1, (kc, D)), "Bf": rng.normal(0, 1, (kf, D)),
            "Ac": rng.normal(0, 1, (kc, J)) / math.sqrt(kc), "Af": rng.normal(0, 1, (kf, J)) / math.sqrt(kf),
            "tau": np.geomspace(2, 40, kf)}


def ar1(rng, n, length, tau):
    """(n, length, k) AR(1) stazionari a varianza 1, un tempo caratteristico per fattore."""
    rho = np.exp(-1.0 / np.asarray(tau))
    out = np.empty((n, length, len(rho)))
    x = rng.normal(0, 1, (n, len(rho)))
    for t in range(length):
        x = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, x.shape)
        out[:, t] = x
    return out


def generate(world, seed, n_users, win_per_user, p):
    rng = np.random.default_rng(seed)
    kc, kf = world["Bc"].shape[0], world["Bf"].shape[0]
    n = n_users * win_per_user
    x = np.empty((n, T, D), np.float16)
    y = np.empty((n, T, J), np.float32)
    users = np.repeat(np.arange(n_users), win_per_user)
    pad = 3
    for u in range(n_users):
        w = win_per_user
        g = rng.normal(0, p["sig_g"], D)
        o = rng.normal(0, p["sig_o"], J)
        c = rng.normal(0, 1, (w, kc))
        a = np.where(rng.random(w) < 0.5, 1.0, p["a_static"])
        f = ar1(rng, w, T + 2 * pad, world["tau"])  # (w, T+6, kf)
        nw = rng.normal(0, p["sig_n"], (w, D))
        lw = rng.normal(0, p["sig_l"], (w, J))
        eta = ar1(rng, w * J, T, [5.0]).reshape(w, J, T).transpose(0, 2, 1) * p["sig_eta"]
        fc = f[:, pad:pad + T]
        z = (world["M"] + g)[None, None, :] + nw[:, None, :] + p["s_c"] * (c @ world["Bc"])[:, None, :] \
            + p["s_f"] * a[:, None, None] * (fc @ world["Bf"]) + rng.normal(0, p["sig_eps"], (w, T, D))
        if p["temporal"]:
            sm = np.stack([f[:, k:k + T] for k in range(2 * pad + 1)]).mean(0)  # media mobile a 7 patch
            h = np.tanh(2.0 * sm)
        else:
            h = fc
        t = o[None, None, :] + lw[:, None, :] + p["t_c"] * (c @ world["Ac"])[:, None, :] \
            + p["t_f"] * a[:, None, None] * (h @ world["Af"]) + eta
        sl = slice(u * w, (u + 1) * w)
        x[sl], y[sl] = z.astype(np.float16), t.astype(np.float32)
    return x, y, users


def standardize(y_train, *others):
    mean = y_train.reshape(-1, J).mean(0)
    std = y_train.reshape(-1, J).std(0)
    return [((a - mean) / std).astype(np.float32) for a in (y_train, *others)]


def pose_var(y):  # come pose_eval.extract: varianza nel tempo per giunto, media sui giunti, finestra intera
    return y.var(axis=1).mean(axis=1)


# --------------------------------------------------------------------------------------- teste
def _zero(layer):
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    return layer


class L0(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.out = _zero(nn.Linear(D, J))

    def forward(self, x):
        return self.out(self.norm(x))


class L13(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.out = _zero(nn.Conv1d(D, J, 13, padding=6))

    def forward(self, x):
        return self.out(self.norm(x).transpose(1, 2)).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.net = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(),
                                 nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1))
        self.out = _zero(nn.Linear(32, J))

    def forward(self, x):
        return self.out(self.net(self.norm(x)))


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.net = nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(),
                                 nn.Conv1d(64, 64, 3, padding=1, dilation=1), nn.ReLU(),
                                 nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv1d(64, 64, 3, padding=3, dilation=3), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1))
        self.out = _zero(nn.Conv1d(32, J, 1))

    def forward(self, x):
        return self.out(self.net(self.norm(x).transpose(1, 2))).transpose(1, 2)


ARMS = {"L0": L0, "L13": L13, "MLP": MLP, "TCN": TCN}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------- metriche
@torch.no_grad()
def evaluate(model, x, y, dynamic, chunk=128):
    """Somme sul crop6: MSE per finestra (somma delle medie), parti centrata e di livello, tutte e dinamiche."""
    model.eval()
    out = {"mse_sum": 0.0, "n": 0, "csse_all": 0.0, "css_all": 0.0, "csse_dyn": 0.0, "css_dyn": 0.0, "level_sum": 0.0}
    for i in range(0, len(x), chunk):
        p = model(torch.from_numpy(x[i:i + chunk].astype(np.float32)))[:, SL].double()
        t = torch.from_numpy(y[i:i + chunk, SL]).double()
        d = torch.from_numpy(dynamic[i:i + chunk])
        out["mse_sum"] += float(((p - t) ** 2).mean(dim=(1, 2)).sum())
        out["n"] += len(p)
        cp, ct = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
        csse, css = ((cp - ct) ** 2).sum(dim=(1, 2)), (ct ** 2).sum(dim=(1, 2))
        out["csse_all"] += float(csse.sum()); out["css_all"] += float(css.sum())
        out["csse_dyn"] += float(csse[d].sum()); out["css_dyn"] += float(css[d].sum())
        out["level_sum"] += float(((p.mean(1) - t.mean(1)) ** 2).mean(1).sum())
    return out


def fit(arm, lr, x, y, evals, *, seed, shuffle_seed, epochs, batch=32, log=None):
    """Un fit: ritorna per ogni epoca le somme di `evaluate` su ogni insieme di `evals` {nome: (x, y, dinamiche)}."""
    torch.manual_seed(seed)
    model = ARMS[arm]()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(shuffle_seed)
    curves = {name: [] for name in evals}
    train_curve = []
    n = len(x)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).numpy()
        tot, steps = 0.0, 0
        for i in range(0, n, batch):
            ix = np.sort(order[i:i + batch])
            xb = torch.from_numpy(x[ix].astype(np.float32))
            yb = torch.from_numpy(y[ix][:, SL])
            loss = ((model(xb)[:, SL] - yb) ** 2).mean(dim=(1, 2)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); steps += 1
        train_curve.append(tot / steps)
        for name, (xe, ye, de) in evals.items():
            curves[name].append(evaluate(model, xe, ye, de))
        if log:
            log(f"    {arm} lr={lr:g} epoca {epoch:2d} passi/epoca {steps} loss train {train_curve[-1]:.4f} " +
                " ".join(f"{k}: mse {v[-1]['mse_sum'] / v[-1]['n']:.4f} E_dyn {v[-1]['csse_dyn'] / v[-1]['css_dyn']:.4f}" for k, v in curves.items()))
    return {"params": n_params(model), "train_loss": train_curve, "curves": curves, "steps_per_epoch": steps}


# --------------------------------------------------------------------------------------- ridge di riferimento
def ln_stateless(x):
    return torch.nn.functional.layer_norm(torch.from_numpy(x.astype(np.float32)), (D,), eps=1e-5).double()


def ridge_stats(x, y, chunk=64):
    sx, sy, xx, xy, n = torch.zeros(D, dtype=torch.float64), torch.zeros(J, dtype=torch.float64), \
        torch.zeros(D, D, dtype=torch.float64), torch.zeros(D, J, dtype=torch.float64), 0
    for i in range(0, len(x), chunk):
        a = ln_stateless(x[i:i + chunk])[:, SL].reshape(-1, D)
        b = torch.from_numpy(y[i:i + chunk, SL]).double().reshape(-1, J)
        sx += a.sum(0); sy += b.sum(0); xx += a.T @ a; xy += a.T @ b; n += len(a)
    return sx, sy, xx, xy, n


def ridge_solver(stats):
    sx, sy, xx, xy, n = stats
    mx, my = sx / n, sy / n
    gram = xx - torch.outer(sx, sx) / n
    eig, vec = torch.linalg.eigh((gram + gram.T) / 2)
    eig = eig.clamp_min(0)
    cross = vec.T @ (xy - torch.outer(sx, my))
    def solve(alpha):
        w = vec @ (cross / (eig[:, None] + alpha))
        return w, my - mx @ w
    return solve, eig


class RidgeModel(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w, self.b = w, b

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (D,), eps=1e-5).double() @ self.w + self.b


def ridge_cv(folds, dyn, exps=range(-2, 9)):
    """folds: [(x, y), (x, y)]. MSE messa in comune sui due fold tenuti fuori, per alpha; come la ridge storica."""
    solvers = [ridge_solver(ridge_stats(*f))[0] for f in folds]
    rows = []
    for e in exps:
        tot = {"mse_sum": 0.0, "n": 0, "csse_dyn": 0.0, "css_dyn": 0.0, "level_sum": 0.0}
        for i in range(2):
            w, b = solvers[i](10.0 ** e)
            r = evaluate(RidgeModel(w, b), folds[1 - i][0], folds[1 - i][1], dyn[1 - i])
            for k in tot:
                tot[k] += r[k]
        rows.append({"alpha": 10.0 ** e, "mse": tot["mse_sum"] / tot["n"], "E_dyn": tot["csse_dyn"] / tot["css_dyn"],
                     "level": tot["level_sum"] / tot["n"]})
    best = min(rows, key=lambda r: r["mse"])
    return rows, best


# --------------------------------------------------------------------------------------- selezione
def select(curves_by_lr, key="mse"):
    """curves_by_lr: {lr: [curva fold0, curva fold1]} con le somme per epoca. Ritorna la scelta (LR, epoca) e la curva messa in comune."""
    pooled = {}
    for lr, folds in curves_by_lr.items():
        vals = []
        for e in range(len(folds[0])):
            if key == "mse":
                vals.append(sum(f[e]["mse_sum"] for f in folds) / sum(f[e]["n"] for f in folds))
            else:  # E_dyn sulle finestre dinamiche, somme messe in comune
                vals.append(sum(f[e]["csse_dyn"] for f in folds) / sum(f[e]["css_dyn"] for f in folds))
        pooled[lr] = vals
    best = None
    for lr in sorted(pooled):  # LR minore prima: a parita' vince il minore
        for e, v in enumerate(pooled[lr], start=1):
            if best is None or v < best[2] or (v == best[2] and e < best[1]):
                best = (lr, e, v)
    return {"lr": best[0], "epoch": best[1], "value": best[2]}, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="L0")
    ap.add_argument("--lrs", default="1e-4,1e-3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--temporal", type=int, default=0)
    ap.add_argument("--win-per-user", type=int, default=48)
    ap.add_argument("--val-users", type=int, default=15)
    ap.add_argument("--sig-g", type=float, default=0.3)
    ap.add_argument("--sig-n", type=float, default=0.3)
    ap.add_argument("--sig-o", type=float, default=0.6)
    ap.add_argument("--sig-l", type=float, default=0.4)
    ap.add_argument("--sig-eps", type=float, default=0.5)
    ap.add_argument("--sig-eta", type=float, default=0.6)
    ap.add_argument("--s-c", type=float, default=0.25)
    ap.add_argument("--s-f", type=float, default=0.2)
    ap.add_argument("--t-c", type=float, default=0.5)
    ap.add_argument("--t-f", type=float, default=0.35)
    ap.add_argument("--a-static", type=float, default=0.25)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--ridge-only", type=int, default=0)
    ap.add_argument("--refit", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    p = {k: getattr(args, k) for k in ("sig_g", "sig_n", "sig_o", "sig_l", "sig_eps", "sig_eta", "s_c", "s_f", "t_c", "t_f", "a_static")}
    p["temporal"] = bool(args.temporal)
    world = make_world(1000 + args.data_seed)
    xtr, ytr, utr = generate(world, 2000 + args.data_seed, 40, args.win_per_user, p)
    xva, yva, uva = generate(world, 3000 + args.data_seed, args.val_users, args.win_per_user, p)
    ytr, yva = standardize(ytr, yva)
    thr = float(np.median(pose_var(yva)))  # come la soglia storica: mediana di pose_var sui val
    dtr, dva = pose_var(ytr) >= thr, pose_var(yva) >= thr
    # due fold: ID utente ordinati e alternati (come la ridge storica: sorted(users)[::2], [1::2])
    ids = np.arange(40)
    fold_users = [ids[::2], ids[1::2]]
    fmask = [np.isin(utr, f) for f in fold_users]
    folds = [(xtr[m], ytr[m], dtr[m]) for m in fmask]
    log = lambda s: print(s, flush=True)
    log(f"dati: train {xtr.shape} val {xva.shape}; fold {[len(f[0]) for f in folds]}; soglia dinamica {thr:.4f}; "
        f"dinamiche train {int(dtr.sum())} val {int(dva.sum())}; generati in {time.time() - t0:.1f}s")
    res = {"args": vars(args), "generator": p, "n_train": len(xtr), "n_val": len(xva), "fold_sizes": [len(f[0]) for f in folds]}
    # zero predittore
    z = ytr[:, SL]
    res["zero_predictor_mse_train"] = float((z ** 2).mean())
    res["zero_predictor_mse_val"] = float((yva[:, SL] ** 2).mean())
    # ridge
    rows, best = ridge_cv([(f[0], f[1]) for f in folds], [f[2] for f in folds])
    res["ridge_cv"] = {"grid": rows, "best": best}
    log("ridge CV (MSE in comune sui due fold): " + ", ".join(f"1e{int(round(math.log10(r['alpha'])))}: {r['mse']:.4f}" for r in rows))
    log(f"ridge migliore alpha {best['alpha']:g} mse {best['mse']:.4f} E_dyn {best['E_dyn']:.4f} livello {best['level']:.4f}")
    solve, eig = ridge_solver(ridge_stats(xtr, ytr))
    w, b = solve(best["alpha"])
    rv = evaluate(RidgeModel(w, b), xva, yva, dva)
    res["ridge_val"] = {"mse": rv["mse_sum"] / rv["n"], "E_dyn": rv["csse_dyn"] / rv["css_dyn"], "level": rv["level_sum"] / rv["n"],
                        "weights_abs_mean": float(w.abs().mean()), "weights_abs_max": float(w.abs().max())}
    res["feature_gram_eig_top"] = [float(v) for v in torch.flip(eig, [0])[:12] / (len(xtr) * (T - 2 * CROP))]
    log(f"ridge sui val sintetici: mse {res['ridge_val']['mse']:.4f} E_dyn {res['ridge_val']['E_dyn']:.4f}; "
        f"|w| medio {res['ridge_val']['weights_abs_mean']:.4f} max {res['ridge_val']['weights_abs_max']:.4f}")
    json.dump(res, open(args.out, "w"), indent=1)
    if args.ridge_only:
        return
    lrs = [float(v) for v in args.lrs.split(",")]
    res["arms"] = {}
    for arm in args.arms.split(","):
        ta = time.time()
        cv = {}
        for lr in lrs:
            cv[lr] = []
            for k in range(2):
                r = fit(arm, lr, folds[k][0], folds[k][1], {"held": folds[1 - k]}, seed=0, shuffle_seed=10_000 + k,
                        epochs=args.epochs)
                cv[lr].append(r)
            log(f"  {arm} lr={lr:g}: CV fatta in {time.time() - ta:.0f}s, passi/epoca {[r['steps_per_epoch'] for r in cv[lr]]}")
        sel, pooled = select({lr: [r["curves"]["held"] for r in rs] for lr, rs in cv.items()}, "mse")
        sel_dyn, pooled_dyn = select({lr: [r["curves"]["held"] for r in rs] for lr, rs in cv.items()}, "E_dyn")
        out = {"params": cv[lrs[0]][0]["params"], "selection_mse": sel, "selection_E_dyn": sel_dyn,
               "pooled_mse": {str(k): v for k, v in pooled.items()}, "pooled_E_dyn": {str(k): v for k, v in pooled_dyn.items()},
               "train_loss": {str(lr): [r["train_loss"] for r in rs] for lr, rs in cv.items()},
               "steps_per_epoch_cv": cv[lrs[0]][0]["steps_per_epoch"]}
        for lr in lrs:
            v = pooled[lr]
            e = int(np.argmin(v)) + 1
            log(f"  {arm} lr={lr:g}: minimo CV mse {min(v):.4f} all'epoca {e} (epoca 1: {v[0]:.4f}, epoca {len(v)}: {v[-1]:.4f}); "
                f"E_dyn CV a quell'epoca {pooled_dyn[lr][e - 1]:.4f}; minimo E_dyn {min(pooled_dyn[lr]):.4f} all'epoca {int(np.argmin(pooled_dyn[lr])) + 1}")
        log(f"  {arm}: scelta su MSE = LR {sel['lr']:g}, epoca {sel['epoch']}, {sel['value']:.4f} (ridge CV {best['mse']:.4f}); "
            f"scelta su E_dyn = LR {sel_dyn['lr']:g}, epoca {sel_dyn['epoch']}, {sel_dyn['value']:.4f} (ridge CV {best['E_dyn']:.4f})")
        if args.refit:
            # rifit sui 40 utenti con il LR scelto, fino a 50 epoche, letto a ogni epoca sui val sintetici (oracolo, possibile solo qui)
            r = fit(arm, sel["lr"], xtr, ytr, {"val": (xva, yva, dva)}, seed=0, shuffle_seed=20_000, epochs=args.epochs)
            vm = [c["mse_sum"] / c["n"] for c in r["curves"]["val"]]
            vd = [c["csse_dyn"] / c["css_dyn"] for c in r["curves"]["val"]]
            e = sel["epoch"]
            half = max(1, int(round(e / 2)))
            out["refit"] = {"lr": sel["lr"], "steps_per_epoch": r["steps_per_epoch"], "val_mse": vm, "val_E_dyn": vd,
                            "at_selected_epoch": {"epoch": e, "mse": vm[e - 1], "E_dyn": vd[e - 1]},
                            "at_half_epochs_same_steps": {"epoch": half, "mse": vm[half - 1], "E_dyn": vd[half - 1]},
                            "oracle_mse": {"epoch": int(np.argmin(vm)) + 1, "mse": min(vm)},
                            "oracle_E_dyn": {"epoch": int(np.argmin(vd)) + 1, "E_dyn": min(vd)}}
            log(f"  {arm} rifit (LR {sel['lr']:g}, {r['steps_per_epoch']} passi/epoca): val all'epoca scelta {e}: mse {vm[e - 1]:.4f} E_dyn {vd[e - 1]:.4f}; "
                f"a meta' epoche ({half}): mse {vm[half - 1]:.4f} E_dyn {vd[half - 1]:.4f}; oracolo mse {min(vm):.4f} all'epoca {int(np.argmin(vm)) + 1}; "
                f"oracolo E_dyn {min(vd):.4f} all'epoca {int(np.argmin(vd)) + 1}; ridge val mse {res['ridge_val']['mse']:.4f} E_dyn {res['ridge_val']['E_dyn']:.4f}")
        out["seconds"] = time.time() - ta
        res["arms"][arm] = out
        json.dump(res, open(args.out, "w"), indent=1)
    log(f"fatto in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
