"""RC-3, verifica scettica: giocattolo CPU indipendente (latenti SINTETICI miei, seme 777; codice VERO:
controls.generate_components, controls.inner_partitions, linear.WindowedBasisFit/select_alpha, metrics.per_window_mse, heads.make_head).
T1: identita' di ortogonalita' del residuo nella metrica della loss vera (anche con maschere parziali).
T2: validazione dello stimatore «costo ridge nel mondo temporale = costo WLS temporale * (ridge/WLS nel mondo lineare)».
T3: base ridge (alpha da CV interna) contro L0 neurale con arresto anticipato (ricetta vera), sugli stessi esclusi.
Uso: rc3_toy.py [t3]
"""
import sys, json, time
import numpy as np, torch
torch.set_num_threads(1)
from pose_temponet import controls, linear, metrics, heads
OUT = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work/"
NU, WPU, TL, D, J, CROP = 40, 30, 160, 384, 20, 6
A, ST, SL = 0.4, 1.3, 1.35
t0 = time.time()

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
basis = {k: c[k] for k in ("L", "T", "epsilon")}
F = linear.WindowedBasisFit(x, basis, full, fit, block_windows=16)
print("componenti e fattore: %.0f s" % (time.time()-t0), flush=True)
res = {}
# ---------------- T2
worlds = {"temporal": ({"L": 1., "T": A}, ST), "linear": ({"L": 1.}, SL)}
inner = [(linear.WindowedBasisFit(x, basis, full, i_fit, block_windows=16), i_held) for i_fit, i_held in controls.inner_partitions(meta, fit)]
T2 = {}
for w, (mix_clean, sig) in worlds.items():
    mix_y = {**mix_clean, "epsilon": sig}
    ystar = sum(v*c[k] for k, v in mix_clean.items()); y = ystar + sig*c["epsilon"]
    thr = float(np.median(pose_var(ystar[fit]))); dyn = pose_var(ystar[held]) >= thr
    yh, ysh = y[held][:, CROP:-CROP], ystar[held][:, CROP:-CROP]
    def inner_mse(alpha):
        tot = cnt = 0
        for f, ih in inner:
            p, _ = f.predict_mixture(ih, mix_y, alpha); tot += float(((p-y[ih][:, CROP:-CROP])**2).mean(axis=(1, 2)).sum()); cnt += len(ih)
        return tot/cnt
    sel = linear.select_alpha(inner_mse)
    row = {"alpha_cv": sel["alpha"], "boundary": sel["boundary_unresolved"], "curve": sel["curve"]}
    row["E_oracle"], row["mse_oracle"] = stats(ysh, yh, dyn)
    row["E_clean"], row["mse_clean"] = stats(F.predict_mixture(held, mix_clean, 0.)[0], yh, dyn)
    for alpha in (0., 0.1, 1., 10., 100.):
        row["E_%g" % alpha], row["mse_%g" % alpha] = stats(F.predict_mixture(held, mix_y, alpha)[0], yh, dyn)
    T2[w] = row
    print(w, "alpha CV", sel["alpha"], "E_oracle %.6f E_clean %.6f" % (row["E_oracle"], row["E_clean"]),
          " ".join("E_%g %.6f" % (a_, row["E_%g" % a_]) for a_ in (0., 0.1, 1., 10., 100.)), flush=True)
t, l = T2["temporal"], T2["linear"]; margin = t["E_clean"]-t["E_oracle"]
print("margine temporale %.6f (relativo %.4f)" % (margin, margin/t["E_clean"]))
T2["check"] = []
for alpha in (0.1, 1., 10., 100.):
    k = "E_%g" % alpha
    true_cost = t[k]-t["E_clean"]; est = (t["E_0"]-t["E_clean"])*(l[k]-l["E_clean"])/(l["E_0"]-l["E_clean"])
    T2["check"].append({"alpha": alpha, "true_cost": true_cost, "estimated_cost": est, "true_R": -true_cost/margin, "estimated_R": -est/margin})
    print("alpha %g: costo vero %.6f stimato %.6f | R vero %.4f stimato %.4f | R WLS0 %.4f" % (
        alpha, true_cost, est, -true_cost/margin, -est/margin, -(t["E_0"]-t["E_clean"])/margin), flush=True)
res["T2"] = T2
# ---------------- T1: gradiente della loss vera rispetto a un lettore affine, bersaglio = residuo di fitting
def grad_affine(target, support, idx):
    W = torch.zeros((D, J), dtype=torch.float64, requires_grad=True); b = torch.zeros(J, dtype=torch.float64, requires_grad=True)
    tot = 0.
    for s in range(0, len(idx), 64):
        take = idx[s:s+64]
        z = torch.from_numpy(linear.layer_norm64(x[take])[:, CROP:-CROP])
        loss = metrics.per_window_mse(z@W+b, torch.from_numpy(target[s:s+64]), torch.from_numpy(support[take]))*len(take)/len(idx)
        loss.backward(); tot += float(loss)
    return float(torch.sqrt((W.grad**2).sum()+(b.grad**2).sum())), tot
yt = c["L"]+A*c["T"]+ST*c["epsilon"]; mix_y = {"L": 1., "T": A, "epsilon": ST}
T1 = {}
gref, _ = grad_affine(yt[fit][:, CROP:-CROP], full, fit)
for alpha in (0., 0.1, 1., 10.):
    r = yt[fit][:, CROP:-CROP]-F.predict_mixture(fit, mix_y, alpha)[0]
    g, loss = grad_affine(r, full, fit)
    T1["full_alpha_%g" % alpha] = {"grad_norm": g, "relative_to_raw_target": g/gref, "residual_loss": loss}
    print("supporto pieno, alpha %g: ||grad affine|| %.3e (rispetto al bersaglio y: %.3e) loss residuo %.5f" % (alpha, g, g/gref, loss), flush=True)
# maschere parziali: giunti 16-19 assenti nel 15% delle finestre, ultime 40 posizioni assenti nel 10%
rng = np.random.default_rng(5); sup = full.copy()
sup[rng.random(n) < 0.15, :, 16:] = False; sup[rng.random(n) < 0.10, -40:, :] = False
sub = fit[:240]
Fm = linear.WindowedBasisFit(x, {"y": yt}, sup, sub, block_windows=16)
gref_m, _ = grad_affine(yt[sub][:, CROP:-CROP], sup, sub)
for alpha in (0., 1.):
    r = yt[sub][:, CROP:-CROP]-Fm.predict_mixture(sub, {"y": 1.}, alpha)[0]
    g, loss = grad_affine(r, sup, sub)
    T1["masked_alpha_%g" % alpha] = {"grad_norm": g, "relative_to_raw_target": g/gref_m, "groups": [len(gr) for gr, _ in Fm.factors]}
    print("maschere parziali (%d gruppi), alpha %g: ||grad affine|| %.3e (rispetto al bersaglio y: %.3e)" % (len(Fm.factors), alpha, g, g/gref_m), flush=True)
res["T1"] = T1
json.dump(res, open(OUT+"rc3_toy_T1T2.json", "w"), indent=1)
print("T1+T2: %.0f s" % (time.time()-t0), flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "t3":
    # ---------------- T3: L0 neurale (ricetta vera) contro ridge CV, mondo temporale, stessi esclusi
    ystar = c["L"]+A*c["T"]; thr = float(np.median(pose_var(ystar[fit]))); dyn = pose_var(ystar[held]) >= thr
    yh = yt[held][:, CROP:-CROP]
    xt, xh = torch.from_numpy(x[fit]).float(), torch.from_numpy(x[held]).float()
    ytf = torch.from_numpy(yt[fit][:, CROP:-CROP].astype(np.float32)); sup32 = torch.ones((32, TL-2*CROP, J), dtype=torch.bool)
    T3 = {"margin": margin, "R_wls0": -(t["E_0"]-t["E_clean"])/margin, "alpha_cv": t["alpha_cv"],
          "R_ridge_cv": -(stats(F.predict_mixture(held, mix_y, t["alpha_cv"])[0], yh, dyn)[0]-t["E_clean"])/margin, "runs": {}}
    print("T3: R WLS0 %.4f | R ridge CV (alpha %g) %.4f" % (T3["R_wls0"], t["alpha_cv"], T3["R_ridge_cv"]), flush=True)
    for lr, epochs in ((0.003, 150), (0.001, 150), (0.0003, 300)):
        model = heads.make_head("L0", seed=0); seeds = heads.trajectory_seeds(0, 0)
        opt = torch.optim.AdamW(heads.optimizer_groups(model, 0.01), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        gen = torch.Generator().manual_seed(seeds["shuffle"]); pts = []
        for ep in range(1, epochs+1):
            perm = torch.randperm(len(fit), generator=gen)
            for s in range(0, len(fit), 32):
                bidx = perm[s:s+32]; opt.zero_grad(set_to_none=True)
                loss = metrics.per_window_mse(model(xt[bidx])[:, CROP:-CROP], ytf[bidx], sup32[:len(bidx)])
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            with torch.no_grad(): p = model(xh)[:, CROP:-CROP].numpy().astype(np.float64)
            E, mse = stats(p, yh, dyn); pts.append({"epoch": ep, "R": (t["E_clean"]-E)/margin, "mse": mse})
            if time.time()-t0 > 1100: break
        best = min(pts, key=lambda q: (q["mse"], q["epoch"]))
        T3["runs"]["%g" % lr] = {"selected_epoch": best["epoch"], "R_selected": best["R"], "R_last": pts[-1]["R"], "last_epoch": pts[-1]["epoch"],
                                 "R_max": max(q["R"] for q in pts), "curve": {str(q["epoch"]): q["R"] for q in pts if q["epoch"] in (1, 5, 10, 20, 31, 50, 75, 100, 150, 200, 300)}}
        print("L0 neurale LR %g: scelta epoca %d R %.4f | ultima epoca %d R %.4f | max R %.4f | %s | %.0f s" % (
            lr, best["epoch"], best["R"], pts[-1]["epoch"], pts[-1]["R"], T3["runs"]["%g" % lr]["R_max"],
            " ".join("%s:%.3f" % kv for kv in T3["runs"]["%g" % lr]["curve"].items()), time.time()-t0), flush=True)
        json.dump(T3, open(OUT+"rc3_toy_T3.json", "w"), indent=1)
