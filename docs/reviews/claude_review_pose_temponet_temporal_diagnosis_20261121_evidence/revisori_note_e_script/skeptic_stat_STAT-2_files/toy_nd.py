"""TOY simulation (reviewer STAT). Synthetic latents, REAL generator (controls.generate_components),
REAL heads (heads.make_head) and REAL loss (metrics.per_window_mse). One fold only. Not real latents.
Usage: toy.py gen | toy.py run <name> <head> <target> <init> <epochs> <max_seconds> [lr]
"""
import sys, json, time, math
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
WORK = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat2_work/"
SIGMA = 1.3
CROP = 6
N_USERS, WIN_PER_USER, T_LEN, D, J = 40, 46, 160, 384, 20


def make_latents(seed=12345, K=32):
    """Synthetic latents: K AR(1) factors with heterogeneous time constants, a shared mixing matrix
    plus a user-specific perturbation, user offset, per-window level offset and white noise."""
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
    sys.path.insert(0, "scripts")
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
    # closed-form L0 (OLS, full support => equal window weights), float64 LN design, intercept
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
            s_clclean, ss_clean, _ = edyn_stats(cl, ysc, dyn)
            tdyn_unread = None
            res[name] = {"E_oracle": s_or/ss, "E_cleanLS": s_cl/ss, "E_L0noisy": s_ny/ss, "gap": (s_cl-s_or)/s_cl,
                         "L0noisy_recovery": (s_cl-s_ny)/(s_cl-s_or), "margin_over_noise_ss": (s_cl-s_or)/s_or,
                         "clean_over_noisy_ss": ss_clean/ss, "n_dyn": int(dyn.sum()), "target_ss": ss,
                         "sse_oracle": s_or, "sse_cleanLS": s_cl, "mse_oracle": m_or, "mse_cleanLS": m_cl}
        rows.append(res)
        print("a=%.1f held: E_or %.4f E_cl %.4f gap %.4f margin/noise %.4f L0noisy rec %.3f | fit gap %.4f" % (
            a, res["held"]["E_oracle"], res["held"]["E_cleanLS"], res["held"]["gap"], res["held"]["margin_over_noise_ss"],
            res["held"]["L0noisy_recovery"], res["fit"]["gap"]), flush=True)
    best = min(rows, key=lambda r: abs(r["fit"]["gap"]-0.08))   # chosen on the fitting half only
    # T unread by L0 on fitting (dynamic share of T not explained by instantaneous linear readout)
    tf = c["T"][fit][:, CROP:-CROP]; pt = P["T"]["fit"]
    e = (pt-tf); e = e-e.mean(1, keepdims=True); cc = tf-tf.mean(1, keepdims=True)
    unread = float((e**2).sum()/(cc**2).sum())
    np.savez(WORK+"toy_data.npz", x=x, L=c["L"].astype(np.float32), T=c["T"].astype(np.float32),
             eps=c["epsilon"].astype(np.float32), fit=fit, held=held)
    torch.save({"states": c["generator_states"], "normalization": c["normalization"]}, WORK+"toy_gen.pt")
    json.dump({"rank": rank, "selected": best, "rows": rows, "T_unread_fraction_fitting_all_windows": unread,
               "hidden_active": c["hidden_active_fraction_fitting"], "seconds": time.time()-t0},
              open(WORK+"toy_refs.json", "w"), indent=1)
    print("rank", rank, "T unread (fit, all windows)", unread)
    print("SELECTED a", best["a"], json.dumps(best["held"]), "seconds", time.time()-t0)


class Small(nn.Module):
    """Generator-family head: same structure as TCN_post (LN + affine + deep), hidden width 8."""
    def __init__(self, width=8):
        super().__init__()
        self.norm = nn.LayerNorm(384, eps=1e-5)
        tr = [nn.Conv1d(384, width, 1), nn.Conv1d(width, width, 3, dilation=1, padding=1),
              nn.Conv1d(width, width, 3, dilation=2, padding=2), nn.Conv1d(width, width, 3, dilation=3, padding=3),
              nn.Conv1d(width, width, 1), nn.Conv1d(width, 20, 1)]
        layers = []
        for i, layer in enumerate(tr):
            layers.append(layer)
            if i < 5:
                layers.append(nn.ReLU())
                if i in (3, 4):
                    layers.append(nn.Dropout(p=0.1))
        self.deep = nn.Sequential(*layers)
        self.affine = nn.Linear(384, 20)
        convs = [m for m in self.deep if isinstance(m, nn.Conv1d)]
        for layer in convs[:-1]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu"); nn.init.zeros_(layer.bias)
        nn.init.zeros_(convs[-1].weight); nn.init.zeros_(convs[-1].bias)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)

    def forward(self, f):
        h = self.norm(f.float())
        return self.affine(h)+self.deep(h.transpose(1, 2)).transpose(1, 2)


def embed_truth(model, a, mode="zero"):
    """Copy the generator into the first 8 channels. mode 'zero': other channels all zero (GPT's embedding).
    mode 'free': other channels keep their random incoming weights, only their outgoing weights are zero."""
    g = torch.load(WORK+"toy_gen.pt")
    ts, ln_, tn = g["states"]["T"], g["normalization"]["L"], g["normalization"]["T"]
    convs = [m for m in model.deep if isinstance(m, nn.Conv1d)]
    keys = ["0", "2", "4", "6", "8", "10"]
    with torch.no_grad():
        for i, (layer, key) in enumerate(zip(convs, keys)):
            w, b = ts[key+".weight"], ts[key+".bias"]
            if i < 5:
                if mode == "zero":
                    layer.weight.zero_(); layer.bias.zero_()
                else:
                    # true channels read only true channels; extra channels keep their random incoming weights
                    layer.weight[:8].zero_(); layer.bias[:8].zero_()
                layer.weight[:w.shape[0], :w.shape[1]].copy_(w); layer.bias[:b.shape[0]].copy_(b)
            else:
                scale = torch.tensor(tn["std"], dtype=torch.float32); mean = torch.tensor(tn["mean"], dtype=torch.float32)
                layer.weight.zero_()
                layer.weight[:, :8].copy_(a*w/scale[:, None, None]); layer.bias.copy_(a*(b-mean)/scale)
        ls = torch.tensor(ln_["std"], dtype=torch.float32); lm = torch.tensor(ln_["mean"], dtype=torch.float32)
        model.affine.weight.copy_(g["states"]["L"]["weight"]/ls[:, None])
        model.affine.bias.copy_((g["states"]["L"]["bias"]-lm)/ls)


def run(name, head, target, init, epochs, max_seconds, lr=0.003, eval_fit_every=5):
    sys.path.insert(0, "scripts")
    from pose_temponet import heads, metrics
    d = np.load(WORK+"toy_data.npz")
    refs = json.load(open(WORK+"toy_refs.json"))
    a = refs["selected"]["a"]; thr = refs["selected"]["threshold"]
    x, fit, held = d["x"], d["fit"], d["held"]
    ystar = d["L"].astype(np.float64)+a*d["T"].astype(np.float64)
    y = ystar+SIGMA*d["eps"].astype(np.float64)
    R = {k: refs["selected"][k] for k in ("fit", "held")}
    dyn = {"fit": pose_var(ystar[fit]) >= thr, "held": pose_var(ystar[held]) >= thr}
    idx = {"fit": fit, "held": held}
    torch.manual_seed(100000)
    if head == "TCN_post":
        model = heads.make_head("TCN_post", seed=0)
    elif head == "L0":
        model = heads.make_head("L0", seed=0)
    else:
        model = Small(int(head[1:]))
    if init == "truth":
        embed_truth(model, a, "zero")
    elif init == "truthfree":
        embed_truth(model, a, "free")
    import os
    if os.environ.get('TOY_DROPOUT') is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = float(os.environ['TOY_DROPOUT'])
        print(name, 'dropout p overridden to', os.environ['TOY_DROPOUT'], flush=True)
    if os.environ.get('TOY_WD') is not None:
        print(name, 'weight decay overridden to', os.environ['TOY_WD'], flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(heads.optimizer_groups(model, float(__import__('os').environ.get('TOY_WD', 0.01))), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    shuffle = torch.Generator().manual_seed(300000)
    train_target = (ystar if target == "clean" else y)[fit][:, CROP:-CROP].astype(np.float32)
    xt = torch.from_numpy(x[fit])            # float16, cast per batch as in the runtime
    yt = torch.from_numpy(train_target)
    sup = torch.ones((32, T_LEN-2*CROP, J), dtype=torch.bool)

    def score(split):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx[split]), 64):
                out.append(model(torch.from_numpy(x[idx[split][s:s+64]].astype(np.float32)))[:, CROP:-CROP].numpy().astype(np.float64))
        model.train()
        p = np.concatenate(out)
        yc, ysc = y[idx[split]][:, CROP:-CROP], ystar[idx[split]][:, CROP:-CROP]
        s_n, ss, mse = edyn_stats(p, yc, dyn[split]); s_c, _, _ = edyn_stats(p, ysc, dyn[split])
        r = R[split]; margin = r["sse_cleanLS"]-r["sse_oracle"]
        return {"E": s_n/ss, "mse": mse, "rec": (r["sse_cleanLS"]-s_n)/margin, "clean_rec": 1-s_c/margin}

    log = {"name": name, "head": head, "target": target, "init": init, "lr": lr, "params": nparam, "a": a, "points": []}
    t0 = time.time(); steps = math.ceil(len(fit)/32)
    h0 = score("held"); f0 = score("fit")
    log["points"].append({"epoch": 0, "held": h0, "fit": f0, "t": 0.0})
    print(name, "params", nparam, "epoch 0 held rec %.3f clean %.3f | fit rec %.3f clean %.3f" % (h0["rec"], h0["clean_rec"], f0["rec"], f0["clean_rec"]), flush=True)
    model.train()
    for ep in range(1, epochs+1):
        perm = torch.randperm(len(fit), generator=shuffle)
        for s in range(0, len(fit), 32):
            b = perm[s:s+32]
            opt.zero_grad(set_to_none=True)
            pred = model(xt[b].float())[:, CROP:-CROP]
            loss = metrics.per_window_mse(pred, yt[b], sup[:len(b)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        h = score("held")
        pt = {"epoch": ep, "held": h, "t": time.time()-t0}
        if ep % eval_fit_every == 0 or ep in (1, 2, 3) or ep == epochs:
            pt["fit"] = score("fit")
        log["points"].append(pt)
        msg = "%s ep %d held E %.5f mse %.5f rec %.3f clean_rec %.3f" % (name, ep, h["E"], h["mse"], h["rec"], h["clean_rec"])
        if "fit" in pt:
            msg += " | fit E %.5f rec %.3f clean_rec %.3f" % (pt["fit"]["E"], pt["fit"]["rec"], pt["fit"]["clean_rec"])
        print(msg, "t %.0fs" % (time.time()-t0), flush=True)
        json.dump(log, open(WORK+"toy_run_%s.json" % name, "w"))
        if time.time()-t0 > max_seconds:
            print(name, "STOP: time budget reached at epoch", ep, flush=True)
            break
    adm = [p for p in log["points"] if p["epoch"] > 0]
    best = min(adm, key=lambda p: (p["held"]["mse"], p["epoch"]))
    bestrec = max(adm, key=lambda p: p["held"]["rec"])
    print("%s SUMMARY: selected(min held MSE) ep %d rec %.3f clean %.3f | best rec ep %d %.3f | last ep %d rec %.3f clean %.3f" % (
        name, best["epoch"], best["held"]["rec"], best["held"]["clean_rec"], bestrec["epoch"], bestrec["held"]["rec"],
        adm[-1]["epoch"], adm[-1]["held"]["rec"], adm[-1]["held"]["clean_rec"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        _, _, name, head, target, init, epochs, max_seconds = sys.argv[:8]
        lr = float(sys.argv[8]) if len(sys.argv) > 8 else 0.003
        run(name, head, target, init, int(epochs), float(max_seconds), lr)
