"""GIOCATTOLO (revisore training_loop, 22/11/2026). Latenti SINTETICI (stessa costruzione del giocattolo STAT del 21/11),
generatore VERO (controls.generate_components), testa VERA (heads.make_head), loss VERA (metrics.per_window_mse),
ciclo VERO (train.Trajectory: AdamW/gruppi/clip/batch/shuffle/dropout RNG). Un fold. NON sono i latenti veri.
Uso: toy_tl.py gen | toy_tl.py bench | toy_tl.py run <nome> <variante> <seed> <epoche> <max_secondi> [lr]
Varianti: base | noclip | nodrop | wd0 | lnfrozen | outinit
"""
import sys, os, json, time, math
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
HERE = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/tl1_work/"
SIGMA = 1.3
CROP = 6
N_USERS, WIN_PER_USER, T_LEN, D, J = 40, 46, 160, 384, 20


def make_latents(seed=12345, K=32):
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.60, 0.98, K)
    lam = 1.0/(1.0+np.arange(K)/4.0)
    A = rng.normal(size=(D, K))/np.sqrt(K)
    n = N_USERS*WIN_PER_USER
    x = np.empty((n, T_LEN, D), dtype=np.float16)
    users = []
    i = 0
    for u in range(N_USERS):
        Au = A + 0.35*rng.normal(size=(D, K))/np.sqrt(K)
        cu = 0.5*rng.normal(size=D)
        for w in range(WIN_PER_USER):
            s = np.empty((T_LEN, K))
            s[0] = rng.normal(size=K)
            innov = rng.normal(size=(T_LEN, K))*np.sqrt(1-rho**2)
            for t in range(1, T_LEN):
                s[t] = rho*s[t-1] + innov[t]
            s = s*lam + 0.3*rng.normal(size=K)*lam
            x[i] = (s@Au.T + cu + 0.15*rng.normal(size=(T_LEN, D))).astype(np.float16)
            users.append("u%02d" % u)
            i += 1
    return x, users


def ln64(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x-m)/np.sqrt(v+1e-5)


def pose_var(y):
    return y.var(axis=1).mean(axis=1)


def edyn_stats(pred, y, dyn):
    e = pred-y
    e = e-e.mean(axis=1, keepdims=True)
    c = y-y.mean(axis=1, keepdims=True)
    return float((e[dyn]**2).sum()), float((c[dyn]**2).sum()), float(((pred-y)**2).mean(axis=(1, 2)).mean())


def gen():
    from pose_temponet import controls
    t0 = time.time()
    x, users = make_latents()
    n = len(x)
    fit = np.flatnonzero(np.array([int(u[1:]) < 20 for u in users]))
    held = np.flatnonzero(np.array([int(u[1:]) >= 20 for u in users]))
    meta = [{"user": users[i], "window": i} for i in range(n)]
    support = np.ones((n, T_LEN-2*CROP, J), dtype=bool)
    c = controls.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=0, block_windows=16)
    print("components generated", time.time()-t0, "s; hidden active", c["hidden_active_fraction_fitting"], flush=True)
    G = np.zeros((D+1, D+1)); B = {k: np.zeros((D+1, J)) for k in ("L", "T", "epsilon")}
    for s in range(0, len(fit), 32):
        ix = fit[s:s+32]
        z = ln64(x[ix])[:, CROP:-CROP].reshape(-1, D)
        z = np.concatenate([z, np.ones((len(z), 1))], 1)
        G += z.T@z
        for k in B:
            B[k] += z.T@c[k][ix][:, CROP:-CROP].reshape(-1, J)
    w, V = np.linalg.eigh(G)
    keep = w > w.max()*1e-11
    rank = int(keep.sum())
    Ginv = (V[:, keep]/w[keep])@V[:, keep].T
    beta = {k: Ginv@B[k] for k in B}
    P = {k: {} for k in B}
    for name, idx in (("fit", fit), ("held", held)):
        for k in B:
            out = np.empty((len(idx), T_LEN-2*CROP, J))
            for s in range(0, len(idx), 32):
                ix = idx[s:s+32]
                z = ln64(x[ix])[:, CROP:-CROP]
                out[s:s+32] = z@beta[k][:-1] + beta[k][-1]
            P[k][name] = out
    rows = []
    for a in [k/5 for k in range(1, 16)]:
        ystar = c["L"]+a*c["T"]
        y = ystar+SIGMA*c["epsilon"]
        thr = float(np.median(pose_var(ystar[fit])))
        res = {"a": a, "threshold": thr}
        for name, idx in (("fit", fit), ("held", held)):
            dyn = pose_var(ystar[idx]) >= thr
            yc, ysc = y[idx][:, CROP:-CROP], ystar[idx][:, CROP:-CROP]
            cl = P["L"][name]+a*P["T"][name]
            ny = cl+SIGMA*P["epsilon"][name]
            s_or, ss, m_or = edyn_stats(ysc, yc, dyn)
            s_cl, _, m_cl = edyn_stats(cl, yc, dyn)
            s_ny, _, m_ny = edyn_stats(ny, yc, dyn)
            res[name] = {"E_oracle": s_or/ss, "E_cleanLS": s_cl/ss, "E_L0noisy": s_ny/ss, "gap": (s_cl-s_or)/s_cl,
                         "L0noisy_recovery": (s_cl-s_ny)/(s_cl-s_or), "n_dyn": int(dyn.sum()), "target_ss": ss,
                         "sse_oracle": s_or, "sse_cleanLS": s_cl, "mse_oracle": m_or, "mse_cleanLS": m_cl, "mse_L0noisy": m_ny}
        rows.append(res)
    best = min(rows, key=lambda r: abs(r["fit"]["gap"]-0.08))
    np.savez(HERE+"toy_data.npz", x=x, L=c["L"].astype(np.float32), T=c["T"].astype(np.float32),
             eps=c["epsilon"].astype(np.float32), fit=fit, held=held)
    json.dump({"rank": rank, "selected": best, "hidden_active": c["hidden_active_fraction_fitting"],
               "seconds": time.time()-t0}, open(HERE+"toy_refs.json", "w"), indent=1)
    yv = (c["L"]+best["a"]*c["T"]+SIGMA*c["epsilon"])[fit][:, CROP:-CROP]
    print("rank", rank, "SELECTED a", best["a"], "target var (fit, crop)", float(yv.var()),
          "var L", float(c["L"][fit][:, CROP:-CROP].var()), "var aT", float((best["a"]*c["T"])[fit][:, CROP:-CROP].var()),
          "var sigma*eps", float((SIGMA*c["epsilon"])[fit][:, CROP:-CROP].var()))
    print(json.dumps(best["held"]), "seconds", time.time()-t0)


def load():
    d = np.load(HERE+"toy_data.npz")
    refs = json.load(open(HERE+"toy_refs.json"))
    a = refs["selected"]["a"]; thr = refs["selected"]["threshold"]
    x, fit, held = d["x"], d["fit"], d["held"]
    ystar = d["L"].astype(np.float64)+a*d["T"].astype(np.float64)
    y = ystar+SIGMA*d["eps"].astype(np.float64)
    return x, fit, held, ystar, y, refs, a, thr


def ols_base(x, fit, y, ridge=0.0):
    """Base affine in forma chiusa su y RUMOROSO del solo fitting (LN float64, crop 6, intercetta non penalizzata)."""
    G = np.zeros((D+1, D+1)); B = np.zeros((D+1, J))
    for s in range(0, len(fit), 32):
        ix = fit[s:s+32]
        z = ln64(x[ix])[:, CROP:-CROP].reshape(-1, D)
        z = np.concatenate([z, np.ones((len(z), 1))], 1)
        G += z.T@z
        B += z.T@y[ix][:, CROP:-CROP].reshape(-1, J)
    if ridge > 0:
        pen = np.eye(D+1)*ridge*G.shape[0]; pen[-1, -1] = 0.0
        return np.linalg.solve(G+pen*np.trace(G[:-1, :-1])/D/G.shape[0], B)
    w, V = np.linalg.eigh(G)
    keep = w > w.max()*1e-11
    return ((V[:, keep]/w[keep])@V[:, keep].T)@B


def make_factory(variant, beta=None):
    from pose_temponet import heads

    def factory(seed, fold):
        m = heads.make_head("TCN_post", seed=seed)
        if variant.startswith("residual"):
            with torch.no_grad():
                m.affine.weight.copy_(torch.from_numpy(beta[:-1].T.astype(np.float32)))
                m.affine.bias.copy_(torch.from_numpy(beta[-1].astype(np.float32)))
            for p in (m.affine.weight, m.affine.bias, m.norm.weight, m.norm.bias):
                p.requires_grad_(False)
        if variant in ("nodrop", "drop03"):
            for mod in m.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = 0.0 if variant == "nodrop" else 0.3
        if variant == "lnfrozen":
            m.norm.weight.requires_grad_(False); m.norm.bias.requires_grad_(False)
        if variant == "outinit":
            last = [mm for mm in m.deep if isinstance(mm, nn.Conv1d)][-1]
            g = torch.Generator().manual_seed(424242+seed)
            with torch.no_grad():
                bound = 1/math.sqrt(last.in_channels)       # = default PyTorch (kaiming_uniform a=sqrt(5)) per k=1
                last.weight.uniform_(-bound, bound, generator=g)
        return m
    return factory


def make_recipe(variant):
    from pose_temponet import train as T
    if variant == "noclip":
        return T._TestRecipe(initial_epochs=150, maximum_epochs=300, last_epochs=5, batch_size=32, clip_norm=1e9)
    if variant == "wd0":
        return T._TestRecipe(initial_epochs=150, maximum_epochs=300, last_epochs=5, batch_size=32, weight_decay=0.0)
    return T.Recipe()       # ricetta di produzione esatta


def build(variant, seed, lr):
    from pose_temponet import train as T
    x, fit, held, ystar, y, refs, a, thr = load()
    tr = fit
    if variant == "tail6":        # 902 = 28*32 + 6: ultimo batch da 6 finestre come nel fold 1 reale (934 = 29*32 + 6)
        tr = fit[:902]
    elif variant == "notail":     # 896 = 28*32: nessun batch parziale
        tr = fit[:896]
    y_train = ystar if os.environ.get("TOY_TARGET") == "clean" else y      # TOY_TARGET=clean: bersaglio senza rumore
    data = T.TensorTrainingSet(torch.from_numpy(x[tr]), torch.from_numpy(y_train[tr]),
                               torch.ones((len(tr), T_LEN-2*CROP, J), dtype=torch.bool),
                               ["w%05d" % i for i in tr])
    beta = None
    if variant == "residual":
        beta = ols_base(x, fit, y)
    elif variant == "residualclean":      # base stimata sul bersaglio pulito (non realizzabile sui dati veri: solo riferimento)
        beta = ols_base(x, fit, ystar)
    traj = T.Trajectory(make_factory(variant, beta), data, lr=lr, seed=seed, fold=int(os.environ.get("TOY_FOLD", "0")), device="cpu", _recipe=make_recipe(variant))
    return traj, (x, fit, held, ystar, y, refs, a, thr)


class ClipSpy:
    """Registra la norma PRIMA del clip (totale e per ramo) senza cambiare il comportamento."""
    def __init__(self, model):
        self.model = model
        self.rows = []
        self.orig = torch.nn.utils.clip_grad_norm_
        torch.nn.utils.clip_grad_norm_ = self

    def __call__(self, parameters, max_norm, **kw):
        sq = {"affine": 0.0, "deep_hidden": 0.0, "deep_out": 0.0, "norm": 0.0}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            v = float(p.grad.detach().pow(2).sum())
            if name.startswith("affine"):
                sq["affine"] += v
            elif name.startswith("norm"):
                sq["norm"] += v
            elif name.startswith("deep.12"):
                sq["deep_out"] += v
            else:
                sq["deep_hidden"] += v
        total = self.orig(parameters, max_norm, **kw)
        self.rows.append((float(total), math.sqrt(sq["affine"]), math.sqrt(sq["deep_hidden"]), math.sqrt(sq["deep_out"]), math.sqrt(sq["norm"])))
        return total


def run(name, variant, seed, epochs, max_seconds, lr=0.003):
    traj, (x, fit, held, ystar, y, refs, a, thr) = build(variant, seed, lr)
    model = traj.model
    spy = ClipSpy(model)
    R = {k: refs["selected"][k] for k in ("fit", "held")}
    dyn = {"fit": pose_var(ystar[fit]) >= thr, "held": pose_var(ystar[held]) >= thr}
    idx = {"fit": fit, "held": held}
    last_name = [n for n, _ in model.named_parameters() if n.startswith("deep.")][-1]
    assert last_name.startswith("deep.12"), last_name

    def score(split):
        mode = model.training
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx[split]), 64):
                out.append(model(torch.from_numpy(x[idx[split][s:s+64]].astype(np.float32)))[:, CROP:-CROP].numpy().astype(np.float64))
        model.train(mode)
        p = np.concatenate(out)
        yc, ysc = y[idx[split]][:, CROP:-CROP], ystar[idx[split]][:, CROP:-CROP]
        s_n, ss, mse = edyn_stats(p, yc, dyn[split]); s_c, _, _ = edyn_stats(p, ysc, dyn[split])
        r = R[split]; margin = r["sse_cleanLS"]-r["sse_oracle"]
        return {"E": s_n/ss, "mse": mse, "rec": (r["sse_cleanLS"]-s_n)/margin, "clean_rec": 1-s_c/margin}

    log = {"name": name, "variant": variant, "seed": seed, "lr": lr, "a": a, "recipe": str(traj.recipe),
           "steps_per_epoch": traj.steps_per_epoch, "points": [], "clip": []}
    t0 = time.time()
    h0 = score("held")
    log["points"].append({"epoch": 0, "held": h0, "t": 0.0})
    print(name, "epoch 0 held rec %.3f mse %.5f" % (h0["rec"], h0["mse"]), flush=True)
    spe = traj.steps_per_epoch
    for ep in range(1, epochs+1):
        n_before = len(spy.rows)
        if variant == "cosine":       # LR per epoca: coseno da lr a 0 sull'orizzonte dichiarato (NON e' la ricetta congelata)
            for group in traj.optimizer.param_groups:
                group["lr"] = lr*0.5*(1+math.cos(math.pi*(ep-1)/epochs))
        traj.train_to_step(ep*spe)
        rows = np.array(spy.rows[n_before:])
        tr_loss = traj.epoch_losses[-1]["loss_sum"]/traj.epoch_losses[-1]["window_count"]
        cl = {"epoch": ep, "clipped_fraction": float((rows[:, 0] > 1.0).mean()), "norm_median": float(np.median(rows[:, 0])),
              "norm_min": float(rows[:, 0].min()), "norm_max": float(rows[:, 0].max()),
              "affine_median": float(np.median(rows[:, 1])), "deep_hidden_median": float(np.median(rows[:, 2])),
              "deep_out_median": float(np.median(rows[:, 3])), "ln_median": float(np.median(rows[:, 4])), "train_loss": tr_loss}
        log["clip"].append(cl)
        h = score("held")
        pt = {"epoch": ep, "held": h, "t": time.time()-t0}
        if ep % 10 == 0 or ep == epochs:
            pt["fit"] = score("fit")
        log["points"].append(pt)
        msg = "%s ep %d held mse %.5f rec %.3f clean_rec %.3f | train loss %.4f clip frac %.2f norm med %.2f (aff %.2f hid %.2f out %.2f ln %.2f)" % (
            name, ep, h["mse"], h["rec"], h["clean_rec"], tr_loss, cl["clipped_fraction"], cl["norm_median"],
            cl["affine_median"], cl["deep_hidden_median"], cl["deep_out_median"], cl["ln_median"])
        if "fit" in pt:
            msg += " | fit rec %.3f clean_rec %.3f" % (pt["fit"]["rec"], pt["fit"]["clean_rec"])
        print(msg, "t %.0fs" % (time.time()-t0), flush=True)
        if ep == 1:
            log["first_steps"] = [list(r) for r in spy.rows[:5]]
        json.dump(log, open(HERE+"toy_run_%s.json" % name, "w"))
        if time.time()-t0 > max_seconds:
            print(name, "STOP: time budget reached at epoch", ep, flush=True)
            break
    adm = [p for p in log["points"] if p["epoch"] > 0]
    best = min(adm, key=lambda p: (p["held"]["mse"], p["epoch"]))
    bestrec = max(adm, key=lambda p: p["held"]["rec"])
    allrows = np.array(spy.rows)
    summ = {"selected_epoch": best["epoch"], "selected_rec": best["held"]["rec"], "selected_clean_rec": best["held"]["clean_rec"],
            "selected_mse": best["held"]["mse"], "best_rec_epoch": bestrec["epoch"], "best_rec": bestrec["held"]["rec"],
            "last_epoch": adm[-1]["epoch"], "last_rec": adm[-1]["held"]["rec"],
            "clipped_fraction_all_steps": float((allrows[:, 0] > 1.0).mean()), "steps": int(len(allrows)),
            "norm_quantiles_5_50_95": [float(q) for q in np.quantile(allrows[:, 0], [0.05, 0.5, 0.95])]}
    log["summary"] = summ
    json.dump(log, open(HERE+"toy_run_%s.json" % name, "w"))
    print(name, "SUMMARY", json.dumps(summ), flush=True)


def bench():
    for nn_flag in (True, False):
        torch.backends.nnpack.set_flags(nn_flag)
        traj, _ = build("base", 0, 0.003)
        t0 = time.time(); traj.train_to_step(3); t1 = time.time()
        print("nnpack", nn_flag, "3 steps %.2fs" % (t1-t0), "loss0", traj.loss_sum/traj.loss_windows, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "bench":
        bench()
    else:
        _, _, name, variant, seed, epochs, max_seconds = sys.argv[:7]
        lr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.003
        if os.environ.get("TOY_NNPACK") == "0":
            torch.backends.nnpack.set_flags(False)
        run(name, variant, int(seed), int(epochs), float(max_seconds), lr)
