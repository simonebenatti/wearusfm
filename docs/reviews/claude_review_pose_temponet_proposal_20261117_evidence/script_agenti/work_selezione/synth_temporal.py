"""Variante dell'esperimento SINTETICO synth_selection.py (stesse funzioni, importate senza modifiche) nel mondo con bersaglio
TEMPORALE NON LINEARE (temporal=1: tanh(2 x media mobile a 7 patch dei fattori AR(1))), dove una testa temporale non lineare dovrebbe
battere L0 se la ricetta glielo permette. Unica differenza: numero di thread di torch come argomento (solo velocita').
Nessun dato del progetto."""
import argparse, json, math, sys, time
import numpy as np
import torch
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="L0,TCN")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--refit", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
p = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25, temporal=True)
t0 = time.time()
world = S.make_world(1001)
xtr, ytr, utr = S.generate(world, 2001, 40, 48, p)
xva, yva, uva = S.generate(world, 3001, 15, 48, p)
ytr, yva = S.standardize(ytr, yva)
thr = float(np.median(S.pose_var(yva)))
dtr, dva = S.pose_var(ytr) >= thr, S.pose_var(yva) >= thr
ids = np.arange(40); fmask = [np.isin(utr, f) for f in (ids[::2], ids[1::2])]
folds = [(xtr[m], ytr[m], dtr[m]) for m in fmask]
log = lambda s: print(s, flush=True)
log(f"mondo temporale: train {xtr.shape} val {xva.shape}; soglia {thr:.4f}; generati in {time.time()-t0:.1f}s")
res = {"generator": p, "threads": a.threads}
rows, best = S.ridge_cv([(f[0], f[1]) for f in folds], [f[2] for f in folds])
res["ridge_cv"] = {"grid": rows, "best": best}
log("ridge CV: " + ", ".join(f"1e{int(round(math.log10(r['alpha'])))}: {r['mse']:.4f}/{r['E_dyn']:.4f}" for r in rows))
solve, eig = S.ridge_solver(S.ridge_stats(xtr, ytr)); w, b = solve(best["alpha"])
rv = S.evaluate(S.RidgeModel(w, b), xva, yva, dva)
res["ridge_val"] = {"mse": rv["mse_sum"]/rv["n"], "E_dyn": rv["csse_dyn"]/rv["css_dyn"]}
log(f"ridge alpha {best['alpha']:g}: CV mse {best['mse']:.4f} E_dyn {best['E_dyn']:.4f}; val mse {res['ridge_val']['mse']:.4f} E_dyn {res['ridge_val']['E_dyn']:.4f}")
res["arms"] = {}
for arm in a.arms.split(","):
    ta = time.time(); cv = {}
    for lr in (1e-4, 1e-3):
        cv[lr] = [S.fit(arm, lr, folds[k][0], folds[k][1], {"held": folds[1-k]}, seed=0, shuffle_seed=10_000+k, epochs=50) for k in range(2)]
        log(f"  {arm} lr={lr:g}: CV in {time.time()-ta:.0f}s")
    sel, pooled = S.select({lr: [r["curves"]["held"] for r in rs] for lr, rs in cv.items()}, "mse")
    seld, pooled_d = S.select({lr: [r["curves"]["held"] for r in rs] for lr, rs in cv.items()}, "E_dyn")
    out = {"selection_mse": sel, "selection_E_dyn": seld, "pooled_mse": {str(k): v for k, v in pooled.items()},
           "pooled_E_dyn": {str(k): v for k, v in pooled_d.items()}}
    for lr in (1e-4, 1e-3):
        v, vd = pooled[lr], pooled_d[lr]
        log(f"  {arm} lr={lr:g}: CV mse min {min(v):.4f} ep {int(np.argmin(v))+1} (ep50 {v[-1]:.4f}); E_dyn min {min(vd):.4f} ep {int(np.argmin(vd))+1} (ep50 {vd[-1]:.4f}); E_dyn all'epoca scelta su mse {vd[int(np.argmin(v))]:.4f}")
    log(f"  {arm}: scelta su MSE LR {sel['lr']:g} ep {sel['epoch']} ({sel['value']:.4f}); su E_dyn LR {seld['lr']:g} ep {seld['epoch']} ({seld['value']:.4f})")
    if a.refit:
        r = S.fit(arm, sel["lr"], xtr, ytr, {"val": (xva, yva, dva)}, seed=0, shuffle_seed=20_000, epochs=50)
        vm = [c["mse_sum"]/c["n"] for c in r["curves"]["val"]]; vdd = [c["csse_dyn"]/c["css_dyn"] for c in r["curves"]["val"]]
        e = sel["epoch"]
        out["refit"] = {"lr": sel["lr"], "val_mse": vm, "val_E_dyn": vdd, "at_selected": {"epoch": e, "mse": vm[e-1], "E_dyn": vdd[e-1]},
                        "oracle_mse": {"epoch": int(np.argmin(vm))+1, "mse": min(vm)}, "oracle_E_dyn": {"epoch": int(np.argmin(vdd))+1, "E_dyn": min(vdd)}}
        log(f"  {arm} rifit LR {sel['lr']:g}: val all'epoca scelta {e}: mse {vm[e-1]:.4f} E_dyn {vdd[e-1]:.4f}; oracolo mse {min(vm):.4f} ep {int(np.argmin(vm))+1}; oracolo E_dyn {min(vdd):.4f} ep {int(np.argmin(vdd))+1}; ep50 E_dyn {vdd[-1]:.4f}")
    out["seconds"] = time.time()-ta
    res["arms"][arm] = out
    json.dump(res, open(a.out, "w"), indent=1)
log(f"fatto in {time.time()-t0:.0f}s")
