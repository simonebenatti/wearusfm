"""Prova CPU della catena generazione -> bersaglio con le funzioni VERE (latenti sintetici).
Non tocca il tree; scrive solo JSON su stdout."""
import json, sys
import numpy as np, torch
from pose_temponet import controls as C, heads, workflow as W, data as D, metrics as M, train as T

torch.set_num_threads(1)
out = {}
rng = np.random.default_rng(7)
U, PER = 6, 8
N = U*PER
# latenti con correlazione temporale (AR) e una componente comune, FP16 come la cache
base = rng.standard_normal((N, 160, 384))
for t in range(1, 160):
    base[:, t] = 0.9*base[:, t-1] + np.sqrt(1-0.81)*base[:, t]
lat = (base*1.7 + 0.3).astype(np.float16)
users = [f"u{i:02d}" for i in range(U)]
meta = [{"user": users[i//PER], "window": int(i), "n_patches": 160} for i in range(N)]
# due finestre corte con SPAZZATURA nelle patch di padding (il generatore non azzera, la testa si')
short = [5, 13]
for i in short:
    meta[i]["n_patches"] = 150
    lat[i, 150:] = (rng.standard_normal((10, 384))*5).astype(np.float16)
mask = np.ones((N, 160, 20), dtype=bool)
for i in short:
    mask[i, 150:] = False
mask[3, 40:60, 2] = False          # buco di posa su un giunto
mask[7] = False                    # finestra senza posa valida
real = rng.standard_normal((N, 160, 20))
cache = {"lat": lat, "target": real, "mask": mask, "meta": meta, "std": np.ones((N, 20)),
         "config": {"split": "train", "which": "teacher", "checkpoint_step": 49046}}
lengths = D.lengths_for(cache)
support = D.central_support(mask, lengths)
folds = D.alternating_user_folds(cache)
ucol = np.asarray([m["user"] for m in meta])
fitting = [np.flatnonzero(np.isin(ucol, h)) for h in folds]
held = [fitting[1], fitting[0]]

# ---------- 1. rumore AR(1) ----------
big_meta = [{"user": "x", "window": i, "n_patches": 160} for i in range(400)]
e = C.window_noise(123, big_meta, 160, 20)
rho = float(np.exp(-0.25))
ac = lambda k: float(np.mean(e[:, k:]*e[:, :160-k]))
out["noise"] = {"shape": list(e.shape), "mean": float(e.mean()), "var_all": float(e.var()),
    "var_t0": float(e[:, 0].var()), "var_t1": float(e[:, 1].var()), "var_t159": float(e[:, 159].var()),
    "rho_expected": rho, "lag1": ac(1), "lag2": ac(2), "lag4": ac(4), "lag8": ac(8),
    "rho^2": rho**2, "rho^4": rho**4, "rho^8": rho**8,
    "corr_joint0_joint1": float(np.mean(e[..., 0]*e[..., 1])),
    "corr_window_i_iplus1": float(np.mean(e[:-1]*e[1:])),
    "deterministic": bool(np.array_equal(e[:5], C.window_noise(123, big_meta[:5], 160, 20))),
    "seed_changes_noise": bool(not np.allclose(e[:5], C.window_noise(124, big_meta[:5], 160, 20))),
    "depends_on_meta_not_position": bool(np.array_equal(e[3], C.window_noise(123, [big_meta[3]], 160, 20)[0]))}
# stesso rumore se cambia solo l'ordine delle chiavi del dizionario (canonical ordina le chiavi)
m2 = {"window": 3, "n_patches": 160, "user": "x"}
out["noise"]["key_order_invariant"] = bool(np.array_equal(e[3], C.window_noise(123, [m2], 160, 20)[0]))

# ---------- 2. componenti e normalizzazione ----------
arts = []
sel = {"linear": {"a": 0.0, "sigma": 1.35, "alpha": 1.0}, "temporal": {"a": 0.4, "sigma": 1.3, "alpha": 1.0}}
out["components"] = {}
for f in (0, 1):
    comp = C.generate_components(lat, meta, support, fitting[f], held[f], users, outer_fold=f, block_windows=16)
    rec = {"seed": comp["seed"], "hidden_active_fraction_fitting": comp["hidden_active_fraction_fitting"]}
    s = support[fitting[f]]; q = s.sum(axis=(1, 2)); w = np.divide(s, q[:, None, None], out=np.zeros(s.shape), where=q[:, None, None] > 0)
    for k in ("L", "T"):
        v = comp[k][fitting[f], 6:-6]
        mean = (v*w).sum(axis=(0, 1))/w.sum(axis=(0, 1)); var = ((v-mean)**2*w).sum(axis=(0, 1))/w.sum(axis=(0, 1))
        vh = comp[k][held[f], 6:-6]; sh = support[held[f]]
        rec[k] = {"fit_weighted_mean_maxabs": float(np.abs(mean).max()), "fit_weighted_var_min": float(var.min()),
                  "fit_weighted_var_max": float(var.max()),
                  "held_plain_mean_maxabs": float(np.abs(vh[sh].mean()) if sh.any() else np.nan),
                  "held_plain_var": float(vh[sh].var())}
    # held non entra nelle statistiche: perturbo i latenti held e rigenero
    lat2 = lat.copy(); lat2[held[f]] = (lat2[held[f]].astype(np.float32)*-0.5 + 1).astype(np.float16)
    comp2 = C.generate_components(lat2, meta, support, fitting[f], held[f], users, outer_fold=f, block_windows=16)
    rec["held_perturbation_changes_fit_LT"] = float(max(np.abs(comp2[k][fitting[f]]-comp[k][fitting[f]]).max() for k in ("L", "T")))
    rec["held_perturbation_changes_normalization"] = bool(comp2["normalization"] != comp["normalization"])
    rec["generator_hash_equal_after_perturbation"] = bool(comp2["generator_hashes"] == comp["generator_hashes"])
    # block_windows non cambia il risultato?
    comp3 = C.generate_components(lat, meta, support, fitting[f], held[f], users, outer_fold=f, block_windows=4)
    rec["block_windows_4_vs_16_maxabs"] = float(max(np.abs(comp3[k]-comp[k]).max() for k in ("L", "T")))
    rec["epsilon_equals_window_noise_base_plus_2"] = bool(np.array_equal(comp["epsilon"], C.window_noise(20261118+1000*f+2, meta, 160, 20)))
    # y - y* = sigma*eps
    clean, obs = C.synthetic_target(comp, 0.4, 1.3)
    rec["y_minus_ystar_minus_sigma_eps_maxabs"] = float(np.abs((obs-clean)-1.3*comp["epsilon"]).max())
    rec["ystar_minus_L_minus_aT_maxabs"] = float(np.abs(clean-(comp["L"]+0.4*comp["T"])).max())
    rec["var_ystar_fit_supported"] = float(clean[fitting[f], 6:-6][support[fitting[f]]].var())
    rec["var_noise_term_fit_supported"] = float((obs-clean)[fitting[f], 6:-6][support[fitting[f]]].var())
    # LN del generatore == LN della testa all'inizio (stessi FP32)
    head = heads.make_head("TCN_post", 0).eval()
    x32 = torch.as_tensor(np.array(lat[:8], dtype=np.float32))
    with torch.no_grad():
        z_gen = torch.nn.functional.layer_norm(x32, (384,), eps=1e-5); z_head = head.norm(x32)
    rec["LN_generator_vs_head_init_maxabs"] = float((z_gen-z_head).abs().max())
    out["components"][f"fold{f}"] = rec
    ref = C.evaluate_outer_references(cache, comp, fitting[f], held[f], sel, block_windows=16, include_l13_diagnostics=False)
    arts.append({"outer_fold": f, "partition": comp["partition"], "components": comp,
                 "calibration": {"status": "calibrated", "selected": sel},
                 "outer_reference_statistics": ref, "outer_scored_after_inner_selection": True})
print(json.dumps(out, indent=1)); sys.stdout.flush()

# ---------- 3. coerenza riferimenti <-> job dei controlli <-> cancello ----------
C_orig = C.combine_reference_folds
C.combine_reference_folds = lambda a: {"status": "diagnostic", "note": "monkeypatch in memoria solo per la prova"}
bundle = W._prepare_control_folds(cache, arts, "temporal", allowed_train_users=users,
                                  source_identity={"test": "synthetic"}, expected_user_count=U)
res = {}
def embed_teacher(comp, a):
    pupil = heads.make_head("TCN_post", 0).eval()
    layers = [m for m in pupil.deep if isinstance(m, torch.nn.Conv1d)]
    teacher, linear = comp["generator_states"]["T"], comp["generator_states"]["L"]
    wk = [k for k in teacher if k.endswith(".weight")]
    with torch.no_grad():
        for layer, key in zip(layers, wk):
            layer.weight.zero_(); layer.bias.zero_()
            w = teacher[key]; b = teacher[key[:-6]+"bias"]
            layer.weight[:w.shape[0], :w.shape[1]].copy_(w); layer.bias[:len(b)].copy_(b)
        n = comp["normalization"]
        lstd, lmean = (torch.tensor(n["L"][k], dtype=torch.float32) for k in ("std", "mean"))
        tstd, tmean = (torch.tensor(n["T"][k], dtype=torch.float32) for k in ("std", "mean"))
        pupil.affine.weight.copy_(linear["weight"]/lstd[:, None]); pupil.affine.bias.copy_((linear["bias"]-lmean)/lstd)
        layers[-1].weight.mul_((a/tstd)[:, None, None]); layers[-1].bias.copy_((layers[-1].bias-tmean)*a/tstd)
    return pupil
for f in (0, 1):
    comp = arts[f]["components"]; ref = arts[f]["outer_reference_statistics"]["temporal"]
    clean, obs = C.synthetic_target(comp, 0.4, 1.3)
    r = {"bundle_target_is_observed_y": bool(np.array_equal(bundle.targets[f], obs)),
         "bundle_clean_is_ystar": bool(np.array_equal(bundle.clean_targets[f], clean)),
         "bundle_threshold_equals_reference": bool(bundle.thresholds[f] == arts[f]["outer_reference_statistics"]["temporal_threshold"])}
    ts = W._training_set(bundle, bundle.fitting[f], "post", f)
    x, y, m = ts.batch(torch.arange(ts.size), torch.device("cpu"))
    rows = ts._rows.numpy(); idx = bundle.fitting[f][rows]
    r["train_rows"] = int(ts.size); r["train_excluded_windows"] = int(ts.excluded_windows)
    r["train_target_equals_y_crop_fp32"] = bool(torch.equal(y, torch.from_numpy(obs[idx, 6:-6]).float()))
    r["train_mask_equals_support"] = bool(torch.equal(m, torch.from_numpy(support[idx])))
    r["train_x_equals_view_inputs"] = bool(torch.equal(x, torch.from_numpy(D.view_inputs(cache, "post", idx))))
    r["train_x_padding_zeroed"] = bool(all(float(x[list(idx).index(i), 150:].abs().max()) == 0 for i in short if i in idx))
    # allievo con i pesi veri = oracolo, valutato dal percorso del job (workflow._score_model)
    pupil = embed_teacher(comp, 0.4)
    with torch.no_grad():
        pred = pupil(torch.from_numpy(D.view_inputs(cache, "post", np.arange(N)))).numpy()
    err = pred[:, 6:-6]-clean[:, 6:-6]
    r["pupil_vs_ystar_supported_maxabs"] = float(np.abs(err[support]).max())
    r["pupil_vs_ystar_supported_rel_rmse"] = float(np.sqrt((err[support]**2).mean()/(clean[:, 6:-6][support]**2).mean()))
    full_err = pred-clean
    sh = [i for i in short]
    r["short_windows_pupil_vs_ystar_real_but_unsupported_t144_149_maxabs"] = float(np.abs(full_err[sh, 144:150]).max())
    r["short_windows_supported_maxabs"] = float(np.abs(err[sh][support[sh]]).max())
    r["edge_t0_5_maxabs_full_windows"] = float(np.abs(full_err[[i for i in range(N) if i not in short], :6]).max())
    so, sc = W._score_model(pupil, bundle, f, bundle.held[f], "post", "cpu", include_clean=True)
    r["oracle_reference_stats"] = ref["oracle"]; r["pupil_scored_by_workflow"] = so
    r["oracle_Edyn_ref"] = ref["oracle"]["dynamic_sse"]/ref["oracle"]["dynamic_target_ss"]
    r["oracle_Edyn_pupil"] = so["dynamic_sse"]/so["dynamic_target_ss"]
    r["pupil_clean_mse"] = sc["mse_sum"]/sc["window_count"]
    # L0 con i coefficienti clean_ls = riferimento clean_ls
    factor = __import__("pose_temponet.linear", fromlist=["x"]).WindowedBasisFit(
        lat, {"real": real, **{k: comp[k] for k in ("L", "T", "epsilon")}}, support, fitting[f], block_windows=16)
    coef, _ = factor.mixed_coefficients({"L": 1., "T": 0.4}, 0.)
    l0 = heads.make_head("L0", 0).eval()
    with torch.no_grad():
        l0.out.weight.copy_(torch.from_numpy(coef[1:].T).float()); l0.out.bias.copy_(torch.from_numpy(coef[0]).float())
    s0 = W._score_model(l0, bundle, f, bundle.held[f], "post", "cpu")
    r["cleanls_Edyn_ref"] = ref["clean_ls"]["dynamic_sse"]/ref["clean_ls"]["dynamic_target_ss"]
    r["cleanls_Edyn_L0head"] = s0["dynamic_sse"]/s0["dynamic_target_ss"]
    r["cleanls_mse_ref"] = ref["clean_ls"]["mse_sum"]/ref["clean_ls"]["window_count"]
    r["cleanls_mse_L0head"] = s0["mse_sum"]/s0["window_count"]
    r["window_count_ref_vs_head"] = [ref["oracle"]["window_count"], so["window_count"]]
    # definizione di E_dyn della calibrazione (pannelli quadratici) contro selection_statistics, stessi fit/held
    act = [comp[k][held[f], 6:-6] for k in ("L", "T", "epsilon")]
    zero = np.zeros_like(act[0])
    qp = C._quadratic_panel([zero, zero, -act[2]], act, support[held[f]])
    fitg = C._pose_variance_gram([comp[k][fitting[f]] for k in ("L", "T")], mask[fitting[f]])
    heldg = C._pose_variance_gram([comp[k][held[f]] for k in ("L", "T")], mask[held[f]])
    thr = C._threshold(C._quad(fitg, [1., 0.4])); pv = C._quad(heldg, [1., 0.4])
    selw = np.isfinite(pv) & (pv >= thr)
    st = C.score_aggregate(C.aggregate_quadratic_panel(qp, selw), [1., 0.4, 1.3])
    r["calibration_style_oracle"] = st
    r["calibration_style_oracle_Edyn"] = st["dynamic_sse"]/st["dynamic_target_ss"]
    r["threshold_quadratic_vs_training_threshold"] = [thr, bundle.thresholds[f]]
    r["dynamic_selected_quadratic_vs_bundle"] = [int(selw.sum()), int((np.isfinite(bundle.pose_variances[f][held[f]]) & (bundle.pose_variances[f][held[f]] >= bundle.thresholds[f])).sum())]
    res[f"fold{f}"] = r
C.combine_reference_folds = C_orig
print(json.dumps(res, indent=1, default=float))
