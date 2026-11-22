"""(i) La calibrazione misura E_cleanLS/E_oracle sullo stesso bersaglio y = L + a*T + sigma*eps?
Confronto la riga (a=0.4, sigma=1.3) prodotta da controls.build_positive_control_fold (funzione vera, pannelli quadratici)
con un calcolo diretto: WLS su y* dell'inner-fit, predizione sull'inner-held, metrics.selection_statistics su y osservato."""
import json
import numpy as np, torch
from pose_temponet import controls as C, data as D, metrics as M
from pose_temponet.linear import WindowedBasisFit, layer_norm64
torch.set_num_threads(1)
rng = np.random.default_rng(11)
U, PER = 8, 6; N = U*PER
base = rng.standard_normal((N, 160, 384))
for t in range(1, 160):
    base[:, t] = 0.9*base[:, t-1] + np.sqrt(1-0.81)*base[:, t]
lat = (base*1.7 + 0.3).astype(np.float16)
users = [f"u{i:02d}" for i in range(U)]
meta = [{"user": users[i//PER], "window": int(i), "n_patches": 160} for i in range(N)]
mask = np.ones((N, 160, 20), dtype=bool); mask[4, 30:50, 1] = False; mask[9] = False
W0 = rng.standard_normal((384, 20))/np.sqrt(384)
real = layer_norm64(lat) @ W0 + 0.8*rng.standard_normal((N, 160, 20))
cache = {"lat": lat, "target": real, "mask": mask, "meta": meta, "std": np.ones((N, 20))}
support = D.central_support(mask, D.lengths_for(cache))
folds = D.alternating_user_folds(cache); ucol = np.asarray([m["user"] for m in meta])
out = {}
for f in (0, 1):
    fit = np.flatnonzero(np.isin(ucol, folds[f])); held = np.flatnonzero(np.isin(ucol, folds[1-f]))
    res = C.build_positive_control_fold(cache, fit, held, users, outer_fold=f, block_windows=16)
    cal = res["calibration"]; comp = res["components"]
    rec = {"status": cal["status"], "reason": cal.get("reason"), "evaluations": cal["evaluations"],
           "real_reference": {k: res["real_reference"][k] for k in ("E_dyn", "MSE", "alpha", "boundary_unresolved")}}
    if cal["evaluations"]:
        row = [r for r in cal["curve"] if abs(r["a"]-0.4) < 1e-9 and abs(r["sigma"]-1.3) < 1e-9][0]
        a, sigma = 0.4, 1.3
        clean, obs = C.synthetic_target(comp, a, sigma)
        stats = {"clean_ls": [], "oracle": [], "ridge": []}
        for inner_fit, inner_held in C.inner_partitions(meta, fit):
            factor = WindowedBasisFit(lat, {"real": real, **{k: comp[k] for k in ("L", "T", "epsilon")}}, support, inner_fit, block_windows=16)
            thr = M.training_threshold(clean[inner_fit], mask[inner_fit])
            panel = C._metric_panel(cache, inner_held, obs, clean)
            pred, _ = factor.predict_mixture(inner_held, {"L": 1., "T": a}, 0.)
            stats["clean_ls"].append(M.selection_statistics(pred, panel, thr))
            stats["oracle"].append(M.selection_statistics(clean[inner_held, 6:-6], panel, thr))
            predr, _ = factor.predict_mixture(inner_held, {"L": 1., "T": a, "epsilon": sigma}, float(row["alpha"]))
            stats["ridge"].append(M.selection_statistics(predr, panel, thr))
        direct = {k: C.statistic_values(C.pool_statistics(v))["E_dyn"] for k, v in stats.items()}
        rec["row_from_build_positive_control_fold"] = {k: row[k] for k in ("E_cal", "E_cleanLS", "E_oracle", "alpha", "oracle_gain_fraction", "admissible")}
        rec["direct_selection_statistics"] = direct
        rec["abs_diff"] = {"E_cleanLS": abs(row["E_cleanLS"]-direct["clean_ls"]), "E_oracle": abs(row["E_oracle"]-direct["oracle"]),
                           "E_cal": abs(row["E_cal"]-direct["ridge"])}
        rec["n_admissible"] = sum(r["admissible"] for r in cal["curve"])
    out[f"fold{f}"] = rec
print(json.dumps(out, indent=1, default=float))
