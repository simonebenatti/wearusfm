"""Skeptic STAT-1: does a CLEAN-trained TCN_post still have its held minimum in the last 5 epochs of 150?
TOY: synthetic latents (same generator as reviewer STAT's toy.py), REAL controls.generate_components,
REAL heads.make_head("TCN_post"), REAL metrics.per_window_mse. Data generated in memory, nothing big on disk.
Usage: toy150.py <name> <t_len> <target clean|noisy> <epochs> <max_seconds> <shuffle_seed> <init_seed>
"""
import sys, json, time, math
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "scripts")
sys.path.insert(0, "/Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/stat_work_files")
import toy as T  # reviewer STAT's module: make_latents, ln64, pose_var, edyn_stats (pure functions)
from pose_temponet import controls, heads, metrics

OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat1_work/"
SIGMA, CROP, D, J = 1.3, 6, 384, 20


def main():
    name, t_len, target, epochs, max_seconds, shuffle_seed, init_seed = sys.argv[1:8]
    t_len, epochs, max_seconds, shuffle_seed, init_seed = int(t_len), int(epochs), float(max_seconds), int(shuffle_seed), int(init_seed)
    T.T_LEN = t_len
    t0 = time.time()
    x, users = T.make_latents()
    n = len(x)
    fit = np.flatnonzero(np.array([int(u[1:]) < 20 for u in users]))
    held = np.flatnonzero(np.array([int(u[1:]) >= 20 for u in users]))
    meta = [{"user": users[i], "window": i} for i in range(n)]
    support = np.ones((n, t_len-2*CROP, J), dtype=bool)
    c = controls.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=0, block_windows=16)
    # closed-form L0 (OLS, float64 LN design, intercept), as in toy.gen
    G = np.zeros((D+1, D+1)); B = {k: np.zeros((D+1, J)) for k in ("L", "T")}
    for s in range(0, len(fit), 32):
        ix = fit[s:s+32]
        z = T.ln64(x[ix])[:, CROP:-CROP].reshape(-1, D)
        z = np.concatenate([z, np.ones((len(z), 1))], 1)
        G += z.T@z
        for k in B:
            B[k] += z.T@c[k][ix][:, CROP:-CROP].reshape(-1, J)
    w, V = np.linalg.eigh(G)
    keep = w > w.max()*1e-11
    Ginv = (V[:, keep]/w[keep])@V[:, keep].T
    beta = {k: Ginv@B[k] for k in B}
    P = {k: {} for k in B}
    idx = {"fit": fit, "held": held}
    for split in idx:
        for k in B:
            out = np.empty((len(idx[split]), t_len-2*CROP, J))
            for s in range(0, len(idx[split]), 32):
                ix = idx[split][s:s+32]
                out[s:s+32] = T.ln64(x[ix])[:, CROP:-CROP]@beta[k][:-1] + beta[k][-1]
            P[k][split] = out
    rows = []
    for a in [k/5 for k in range(1, 16)]:
        ystar = c["L"]+a*c["T"]; y = ystar+SIGMA*c["epsilon"]
        thr = float(np.median(T.pose_var(ystar[fit])))
        res = {"a": a, "threshold": thr}
        for split in idx:
            dyn = T.pose_var(ystar[idx[split]]) >= thr
            yc, ysc = y[idx[split]][:, CROP:-CROP], ystar[idx[split]][:, CROP:-CROP]
            cl = P["L"][split]+a*P["T"][split]
            s_or, ss, _ = T.edyn_stats(ysc, yc, dyn)
            s_cl, _, _ = T.edyn_stats(cl, yc, dyn)
            res[split] = {"sse_oracle": s_or, "sse_cleanLS": s_cl, "ss": ss, "gap": (s_cl-s_or)/s_cl}
        rows.append(res)
    sel = min(rows, key=lambda r: abs(r["fit"]["gap"]-0.08))
    a, thr = sel["a"], sel["threshold"]
    print(name, "gen %.0fs t_len %d a %.1f held gap %.4f fit gap %.4f E_cleanLS %.4f E_oracle %.4f" % (
        time.time()-t0, t_len, a, sel["held"]["gap"], sel["fit"]["gap"], sel["held"]["sse_cleanLS"]/sel["held"]["ss"],
        sel["held"]["sse_oracle"]/sel["held"]["ss"]), flush=True)
    ystar = c["L"]+a*c["T"]; y = ystar+SIGMA*c["epsilon"]
    dyn = {k: T.pose_var(ystar[idx[k]]) >= thr for k in idx}
    del c, P, G, B
    torch.manual_seed(100000+init_seed)
    model = heads.make_head("TCN_post", seed=init_seed)
    opt = torch.optim.AdamW(heads.optimizer_groups(model, 0.01), lr=0.003, betas=(0.9, 0.999), eps=1e-8)
    shuffle = torch.Generator().manual_seed(shuffle_seed)
    xt = torch.from_numpy(x[fit])
    yt = torch.from_numpy((ystar if target == "clean" else y)[fit][:, CROP:-CROP].astype(np.float32))
    sup = torch.ones((32, t_len-2*CROP, J), dtype=torch.bool)

    def score(split):
        model.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx[split]), 64):
                out.append(model(torch.from_numpy(x[idx[split][s:s+64]].astype(np.float32)))[:, CROP:-CROP].numpy().astype(np.float64))
        model.train()
        p = np.concatenate(out)
        yc, ysc = y[idx[split]][:, CROP:-CROP], ystar[idx[split]][:, CROP:-CROP]
        s_n, ss, mse = T.edyn_stats(p, yc, dyn[split]); s_c, _, mse_c = T.edyn_stats(p, ysc, dyn[split])
        r = sel[split]; margin = r["sse_cleanLS"]-r["sse_oracle"]
        return {"E": s_n/ss, "mse": mse, "mse_clean": mse_c, "rec": (r["sse_cleanLS"]-s_n)/margin, "clean_rec": 1-s_c/margin}

    pts = []
    t0 = time.time()
    model.train()
    for ep in range(1, epochs+1):
        perm = torch.randperm(len(fit), generator=shuffle)
        for s in range(0, len(fit), 32):
            b = perm[s:s+32]
            opt.zero_grad(set_to_none=True)
            loss = metrics.per_window_mse(model(xt[b].float())[:, CROP:-CROP], yt[b], sup[:len(b)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        h = score("held")
        pt = {"epoch": ep, "held": h, "t": time.time()-t0}
        msg = "%s ep %d held mse %.6f E %.6f rec %.4f clean_rec %.4f" % (name, ep, h["mse"], h["E"], h["rec"], h["clean_rec"])
        if ep % 10 == 0 or ep == epochs:
            pt["fit"] = score("fit")
            msg += " | fit rec %.4f clean_rec %.4f" % (pt["fit"]["rec"], pt["fit"]["clean_rec"])
        pts.append(pt)
        bm = min(pts, key=lambda p: (p["held"]["mse"], p["epoch"])); be = min(pts, key=lambda p: (p["held"]["E"], p["epoch"]))
        print(msg, "| argmin mse ep %d, argmin E ep %d, t %.0fs" % (bm["epoch"], be["epoch"], time.time()-t0), flush=True)
        json.dump({"name": name, "t_len": t_len, "target": target, "a": a, "points": pts}, open(OUT+"run_%s.json" % name, "w"))
        if time.time()-t0 > max_seconds:
            print(name, "STOP time budget at epoch", ep, flush=True)
            break
    last = pts[-1]["epoch"]
    bm = min(pts, key=lambda p: (p["held"]["mse"], p["epoch"])); be = min(pts, key=lambda p: (p["held"]["E"], p["epoch"]))
    print("%s SUMMARY last ep %d | argmin held MSE ep %d (rec %.4f) | argmin held E_dyn ep %d (rec %.4f) | last rec %.4f | in last 5 epochs run: mse %s edyn %s" % (
        name, last, bm["epoch"], bm["held"]["rec"], be["epoch"], be["held"]["rec"], pts[-1]["held"]["rec"],
        bm["epoch"] >= last-4, be["epoch"] >= last-4), flush=True)


if __name__ == "__main__":
    main()
