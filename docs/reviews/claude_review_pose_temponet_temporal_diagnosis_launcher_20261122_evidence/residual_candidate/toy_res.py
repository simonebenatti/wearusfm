"""GIOCATTOLO CPU del revisore (candidato residuale). Latenti SINTETICI (make_latents di ieri, seme 12345),
generatore VERO (controls.generate_components), riferimenti lineari VERI (linear.WindowedBasisFit, select_alpha,
controls.inner_partitions), testa VERA (heads.make_head TCN_post), loss VERA (metrics.per_window_mse),
ricetta VERA (AdamW gruppi veri, lr 0.003, wd 0.01, batch 32, clip 1.0, dropout 0.1). Un fold. NON sono i latenti veri.
Uso: toy_res.py gen <fold> | toy_res.py run <fold> <variante> <seme> <epoche> <max_secondi>
Varianti: cur | res (LN fissa + base ridge congelata) | lnfix | base | wd<valore> | drop<valore> | resw (res con base WLS alpha=0)
"""
import sys, json, time, math
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
WORK = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate/toy/"
A, SIGMA, CROP = 0.4, 1.3, 6
N_USERS, WIN_PER_USER, T_LEN, D, J = 40, 46, 160, 384, 20


def make_latents(seed=12345, K=32):
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.60, 0.98, K)
    lam = 1.0/(1.0+np.arange(K)/4.0)
    Amix = rng.normal(size=(D, K))/np.sqrt(K)
    n = N_USERS*WIN_PER_USER
    x = np.empty((n, T_LEN, D), dtype=np.float16)
    users = []
    i = 0
    for u in range(N_USERS):
        Au = Amix + 0.35*rng.normal(size=(D, K))/np.sqrt(K)
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


def pose_var(y):
    return y.var(axis=1).mean(axis=1)


def edyn_stats(pred, y, dyn):
    e = pred-y
    mse = float((e**2).mean(axis=(1, 2)).mean())
    e = e-e.mean(axis=1, keepdims=True)
    c = y-y.mean(axis=1, keepdims=True)
    return float((e[dyn]**2).sum()), float((c[dyn]**2).sum()), mse


def gen(fold):
    from pose_temponet import controls, linear
    t0 = time.time()
    x, users = make_latents()
    n = len(x)
    low = np.array([int(u[1:]) < 20 for u in users])
    fit, held = (np.flatnonzero(low), np.flatnonzero(~low)) if fold == 0 else (np.flatnonzero(~low), np.flatnonzero(low))
    meta = [{"user": users[i], "window": i} for i in range(n)]
    support = np.ones((n, T_LEN-2*CROP, J), dtype=bool)
    c = controls.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=fold, block_windows=16)
    ystar, y = controls.synthetic_target(c, A, SIGMA)
    print("components", round(time.time()-t0, 1), "s", flush=True)
    basis = {k: c[k] for k in ("L", "T", "epsilon")}
    mix_y, mix_clean = {"L": 1., "T": A, "epsilon": SIGMA}, {"L": 1., "T": A}
    factor = linear.WindowedBasisFit(x, basis, support, fit, block_windows=16)
    inner = [(linear.WindowedBasisFit(x, basis, support, ifit, block_windows=16), iheld)
             for ifit, iheld in controls.inner_partitions(meta, fit)]
    print("factors", round(time.time()-t0, 1), "s", flush=True)

    def inner_mse(alpha):
        tot, cnt = 0.0, 0
        for f, iheld in inner:
            p, _ = f.predict_mixture(iheld, mix_y, alpha)
            tot += float(((p-y[iheld][:, CROP:-CROP])**2).mean(axis=(1, 2)).sum()); cnt += len(iheld)
        return tot/cnt
    sel = linear.select_alpha(inner_mse)
    alpha = sel["alpha"]
    print("alpha selected", alpha, "boundary", sel["boundary_unresolved"], [(r["alpha"], round(r["mse"], 5)) for r in sel["curve"]], flush=True)
    P = {}
    for name, idx in (("fit", fit), ("held", held)):
        P["clean_"+name] = factor.predict_mixture(idx, mix_clean, 0.)[0]
        P["wls_"+name] = factor.predict_mixture(idx, mix_y, 0.)[0]
        P["ridge_"+name] = factor.predict_mixture(idx, mix_y, alpha)[0]
    thr = float(np.median(pose_var(ystar[fit])))
    refs = {"fold": fold, "a": A, "sigma": SIGMA, "alpha": alpha, "alpha_curve": sel["curve"], "threshold": thr,
            "hidden_active": c["hidden_active_fraction_fitting"]}
    for name, idx in (("fit", fit), ("held", held)):
        dyn = pose_var(ystar[idx]) >= thr
        yc, ysc = y[idx][:, CROP:-CROP], ystar[idx][:, CROP:-CROP]
        s_or, ss, m_or = edyn_stats(ysc, yc, dyn)
        s_cl, _, m_cl = edyn_stats(P["clean_"+name], yc, dyn)
        s_w, _, m_w = edyn_stats(P["wls_"+name], yc, dyn)
        s_r, _, m_r = edyn_stats(P["ridge_"+name], yc, dyn)
        margin = s_cl-s_or
        refs[name] = {"n_dyn": int(dyn.sum()), "n": int(len(idx)), "target_ss": ss, "sse_oracle": s_or, "sse_cleanLS": s_cl,
                      "E_oracle": s_or/ss, "E_cleanLS": s_cl/ss, "gap": margin/s_cl, "E_wls_noisy": s_w/ss, "E_ridge_noisy": s_r/ss,
                      "R_wls_noisy": (s_cl-s_w)/margin, "R_ridge_noisy": (s_cl-s_r)/margin,
                      "mse_oracle": m_or, "mse_cleanLS": m_cl, "mse_wls_noisy": m_w, "mse_ridge_noisy": m_r}
    # quota di T non letta linearmente (fitting, tutte le finestre, centrata)
    pt = factor.predict_mixture(fit, {"T": 1.}, 0.)[0]; tf = c["T"][fit][:, CROP:-CROP]
    e = pt-tf; e = e-e.mean(1, keepdims=True); cc = tf-tf.mean(1, keepdims=True)
    refs["T_unread_fraction_fitting_all_windows"] = float((e**2).sum()/(cc**2).sum())
    refs["seconds"] = time.time()-t0
    np.savez(WORK+"data_f%d.npz" % fold, x=x, y=y.astype(np.float32), ystar=ystar.astype(np.float32), fit=fit, held=held,
             **{k: v.astype(np.float32) for k, v in P.items() if k.startswith(("ridge", "wls"))})
    json.dump(refs, open(WORK+"refs_f%d.json" % fold, "w"), indent=1)
    print(json.dumps({k: refs[k] for k in ("alpha", "threshold", "T_unread_fraction_fitting_all_windows", "seconds")}))
    print("HELD", json.dumps(refs["held"])); print("FIT", json.dumps(refs["fit"]))


def run(fold, variant, seed, epochs, max_seconds, lr=0.003):
    from pose_temponet import heads, metrics
    d = np.load(WORK+"data_f%d.npz" % fold)
    refs = json.load(open(WORK+"refs_f%d.json" % fold))
    x, fit, held = d["x"], d["fit"], d["held"]
    y, ystar = d["y"].astype(np.float64), d["ystar"].astype(np.float64)
    thr = refs["threshold"]
    dyn_h = pose_var(ystar[held]) >= thr
    Rh = refs["held"]; margin = Rh["sse_cleanLS"]-Rh["sse_oracle"]
    wd, drop, freeze_ln, base_kind = 0.01, None, False, None
    if variant == "res": freeze_ln, base_kind = True, "ridge"
    elif variant == "resw": freeze_ln, base_kind = True, "wls"
    elif variant == "lnfix": freeze_ln = True
    elif variant == "base": base_kind = "ridge"
    elif variant.startswith("wd"): wd = float(variant[2:])
    elif variant.startswith("drop"): drop = float(variant[4:])
    elif variant == "warm": pass      # ramo affine INIZIALIZZATO alla ridge (non congelato), LN apprendibile, bersaglio y
    elif variant != "cur": raise SystemExit("variante sconosciuta")
    name = "f%d_%s_s%d" % (fold, variant, seed)
    seeds = heads.trajectory_seeds(seed, fold)
    model = heads.make_head("TCN_post", seed=seed)
    if drop is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.p = drop
    if freeze_ln:
        for p in model.norm.parameters(): p.requires_grad_(False)
    base_fit = base_held = None
    if base_kind:
        for p in model.affine.parameters(): p.requires_grad_(False)     # resta a zero: uscita = solo ramo profondo
        base_fit = d[base_kind+"_fit"].astype(np.float64); base_held = d[base_kind+"_held"].astype(np.float64)
    if variant == "warm":
        G = np.zeros((D+1, D+1)); Bm = np.zeros((D+1, J)); rf = d["ridge_fit"].astype(np.float64)
        for s0 in range(0, len(fit), 32):
            z = torch.nn.functional.layer_norm(torch.from_numpy(x[fit[s0:s0+32]]).float(), (D,), eps=1e-5).numpy().astype(np.float64)
            z = z[:, CROP:-CROP].reshape(-1, D); z = np.concatenate([z, np.ones((len(z), 1))], 1)
            G += z.T@z; Bm += z.T@rf[s0:s0+32].reshape(-1, J)
        w_, V_ = np.linalg.eigh(G); keep = w_ > w_.max()*1e-11
        beta = (V_[:, keep]/w_[keep])@V_[:, keep].T@Bm
        with torch.no_grad():
            model.affine.weight.copy_(torch.from_numpy(beta[:-1].T.astype(np.float32)))
            model.affine.bias.copy_(torch.from_numpy(beta[-1].astype(np.float32)))
            zh = torch.nn.functional.layer_norm(torch.from_numpy(x[held[:64]]).float(), (D,), eps=1e-5)
            diff = (model.affine(zh)[:, CROP:-CROP].numpy()-d["ridge_held"][:64]).__abs__().max()
        print("warm: rango", int(keep.sum()), "scarto massimo dalla ridge sugli esclusi (64 finestre)", float(diff), flush=True)
    groups = [{"params": [p for p in g["params"] if p.requires_grad], "weight_decay": g["weight_decay"]}
              for g in heads.optimizer_groups(model, wd)]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.manual_seed(seeds["dropout"])
    shuffle = torch.Generator().manual_seed(seeds["shuffle"])
    target = y[fit][:, CROP:-CROP]
    if base_kind: target = target-base_fit
    xt = torch.from_numpy(x[fit]); yt = torch.from_numpy(target.astype(np.float32))
    xh = torch.from_numpy(x[held])
    sup = torch.ones((32, T_LEN-2*CROP, J), dtype=torch.bool)
    yc, ysc = y[held][:, CROP:-CROP], ystar[held][:, CROP:-CROP]

    def score():
        model.eval(); tot, aff = [], []
        with torch.no_grad():
            for s in range(0, len(held), 128):
                h = model.norm(xh[s:s+128].float())
                a_ = model.affine(h)
                dp = model.deep(h.transpose(1, 2)).transpose(1, 2)
                tot.append((a_+dp)[:, CROP:-CROP].numpy().astype(np.float64)); aff.append(a_[:, CROP:-CROP].numpy().astype(np.float64))
        model.train()
        p = np.concatenate(tot); pa = np.concatenate(aff)
        if base_kind: p = p+base_held; pa = base_held
        s_n, ss, mse = edyn_stats(p, yc, dyn_h); s_c, _, mse_c = edyn_stats(p, ysc, dyn_h)
        s_a, _, _ = edyn_stats(pa, yc, dyn_h)
        return {"E": s_n/ss, "mse": mse, "rec": (Rh["sse_cleanLS"]-s_n)/margin, "clean_rec": 1-s_c/margin,
                "rec_linear_part": (Rh["sse_cleanLS"]-s_a)/margin, "mse_clean": mse_c}

    log = {"name": name, "fold": fold, "variant": variant, "seed": seed, "lr": lr, "wd": wd, "dropout": drop, "freeze_ln": freeze_ln,
           "base": base_kind, "trainable": ntrain, "points": []}
    t0 = time.time()
    h0 = score(); log["points"].append({"epoch": 0, "held": h0, "t": 0.0})
    print(name, "trainable", ntrain, "ep 0 rec %.3f" % h0["rec"], flush=True)
    model.train()
    for ep in range(1, epochs+1):
        perm = torch.randperm(len(fit), generator=shuffle)
        loss_sum = 0.0
        for s in range(0, len(fit), 32):
            b = perm[s:s+32]
            opt.zero_grad(set_to_none=True)
            pred = model(xt[b].float())[:, CROP:-CROP]
            loss = metrics.per_window_mse(pred, yt[b], sup[:len(b)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss)*len(b)
        h = score()
        log["points"].append({"epoch": ep, "held": h, "train_loss": loss_sum/len(fit), "t": time.time()-t0})
        print("%s ep %d train %.5f held mse %.5f rec %.3f clean %.3f lin %.3f t %.0fs" % (
            name, ep, loss_sum/len(fit), h["mse"], h["rec"], h["clean_rec"], h["rec_linear_part"], time.time()-t0), flush=True)
        json.dump(log, open(WORK+"run_%s.json" % name, "w"))
        if time.time()-t0 > max_seconds:
            print(name, "STOP tempo all'epoca", ep, flush=True); break
    adm = [p for p in log["points"] if p["epoch"] > 0]
    best = min(adm, key=lambda p: (p["held"]["mse"], p["epoch"]))
    print("%s SUMMARY sel ep %d rec %.4f clean %.4f lin %.4f | max rec %.4f | last ep %d rec %.4f" % (
        name, best["epoch"], best["held"]["rec"], best["held"]["clean_rec"], best["held"]["rec_linear_part"],
        max(p["held"]["rec"] for p in adm), adm[-1]["epoch"], adm[-1]["held"]["rec"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]))
    else:
        run(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
