"""RC-3, verifica scettica: ricalcolo dai JSON archiviati (sola lettura). Uso: python -I rc3_json.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
g = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
ref_t = g["reference_gate"]["temporal_records"]
ref_l = g["reference_gate"]["linear_records"]
sel = g["selected_panels"]["temporal"]
def E(s): return s["dynamic_sse"]/s["dynamic_target_ss"]
def M(s): return s["mse_sum"]/s["window_count"]
cal = {f: json.load(open(B+"calibration_%d.json" % f)) for f in (0, 1)}
pool = {}
def add(name, st):
    p = pool.setdefault(name, {"dynamic_sse": 0., "dynamic_target_ss": 0., "mse_sum": 0., "window_count": 0})
    for k in p: p[k] += st[k]
print("== 1. numeri del claim (mondo temporale, esclusi esterni, 20 utenti di fitting)")
rows = {}
for f in (0, 1):
    fold = "fold%d" % f
    cl, orc = ref_t[fold]["clean_ls"], ref_t[fold]["oracle"]; m = cl-orc
    ld = g["linear_diagnostics"][fold]["L0"]["worlds"]
    wt = ld["temporal"]["held"]["statistics"]; wl = ld["linear"]["held"]["statistics"]
    ors = cal[f]["outer_reference_statistics"]
    # coerenza fra gate e calibrazione
    assert abs(E(ors["temporal"]["clean_ls"])-cl) < 1e-12 and abs(E(ors["temporal"]["oracle"])-orc) < 1e-12
    add("t_wls0", wt); add("t_clean", ors["temporal"]["clean_ls"]); add("t_oracle", ors["temporal"]["oracle"])
    add("l_wls0", wl); add("l_ridge", ors["linear"]["ridge"]); add("l_clean", ors["linear"]["clean_ls"])
    add("t_L0", g["selected_panels"]["temporal"]["L0"][fold] if "dynamic_sse" in g["selected_panels"]["temporal"]["L0"][fold] else {"dynamic_sse": 0, "dynamic_target_ss": 1, "mse_sum": 0, "window_count": 1})
    R_wls0 = (cl-E(wt))/m; R_L0 = (cl-sel["L0"][fold]["E_dyn"])/m; R_T = (cl-sel["TCN_post"][fold]["E_dyn"])/m
    # mondo lineare
    El_clean = E(ors["linear"]["clean_ls"]); El_ridge = E(ors["linear"]["ridge"]); El_wls0 = E(wl)
    Ml_clean = M(ors["linear"]["clean_ls"]); Ml_ridge = M(ors["linear"]["ridge"]); Ml_wls0 = M(wl)
    Mt_clean = M(ors["temporal"]["clean_ls"]); Mt_wls0 = M(wt)
    selt = cal[f]["calibration"]["selected"]["temporal"]; sell = cal[f]["calibration"]["selected"]["linear"]
    ratio_E = (El_ridge-El_clean)/(El_wls0-El_clean); ratio_M = (Ml_ridge-Ml_clean)/(Ml_wls0-Ml_clean)
    # scala fra i mondi misurata sull'alpha=0 a 20 utenti (stesso eps, sigma diverso, finestre dinamiche diverse)
    world_scale_E = (E(wt)-cl)/(El_wls0-El_clean)
    est_delta_E = (El_ridge-El_clean)*world_scale_E     # = (E_wls0_t - clean_t) * ratio_E
    rows[fold] = dict(clean=cl, oracle=orc, margin=m, E_wls0_t=E(wt), R_wls0=R_wls0, R_L0=R_L0, R_TCN=R_T,
                      L0_minus_wls0_points=100*(R_L0-R_wls0),
                      sigma_t=selt["sigma"], sigma_l=sell["sigma"], alpha_t=selt["alpha"], alpha_l=sell["alpha"],
                      El_clean=El_clean, El_wls0=El_wls0, El_ridge=El_ridge,
                      lin_cost_wls0=El_wls0-El_clean, lin_cost_ridge=El_ridge-El_clean, ratio_E=ratio_E, ratio_MSE=ratio_M,
                      t_cost_wls0=E(wt)-cl, world_scale_E=world_scale_E, sigma_ratio_sq=(selt["sigma"]/sell["sigma"])**2,
                      est_R_ridge_t=-est_delta_E/m, est_R_ridge_t_MSEratio=R_wls0*ratio_M,
                      Mt_cost_wls0=Mt_wls0-Mt_clean, Ml_cost_wls0=Ml_wls0-Ml_clean, Ml_cost_ridge=Ml_ridge-Ml_clean,
                      inner_R_ridge=(selt["E_cleanLS"]-selt["E_cal"])/(selt["E_cleanLS"]-selt["E_oracle"]),
                      inner_cost_t=selt["E_cal"]-selt["E_cleanLS"], inner_cost_l=sell["E_cal"]-sell["E_cleanLS"])
    rows[fold]["est_minus_L0_points"] = 100*(rows[fold]["est_R_ridge_t"]-R_L0)
cl, orc = ref_t["pooled"]["clean_ls"], ref_t["pooled"]["oracle"]; m = cl-orc
R_wls0 = (cl-E(pool["t_wls0"]))/m; R_L0 = (cl-sel["L0"]["pooled"]["E_dyn"])/m
El_clean, El_ridge, El_wls0 = E(pool["l_clean"]), E(pool["l_ridge"]), E(pool["l_wls0"])
ratio_E = (El_ridge-El_clean)/(El_wls0-El_clean)
rows["pooled"] = dict(clean=cl, oracle=orc, margin=m, E_wls0_t=E(pool["t_wls0"]), R_wls0=R_wls0, R_L0=R_L0,
                      R_TCN=(cl-sel["TCN_post"]["pooled"]["E_dyn"])/m, L0_minus_wls0_points=100*(R_L0-R_wls0),
                      El_clean=El_clean, El_wls0=El_wls0, El_ridge=El_ridge, lin_cost_wls0=El_wls0-El_clean,
                      lin_cost_ridge=El_ridge-El_clean, ratio_E=ratio_E, est_R_ridge_t=R_wls0*ratio_E,
                      est_minus_L0_points=100*(R_wls0*ratio_E-R_L0))
for fold, r in rows.items():
    print(fold)
    for k, v in r.items(): print("   %-28s %.6f" % (k, v))
print("== 2. curve L0 neurale, mondo temporale (pooled esclusi interni al train = held dei due fold)")
s = json.load(open(B+"selection_temporal_L0.json"))
cl, orc = ref_t["pooled"]["clean_ls"], ref_t["pooled"]["oracle"]; m = cl-orc
print("selezionato", s["selected"], "R %.4f" % ((cl-s["selected"]["edyn"])/m))
for c in s["curves"]:
    pts = [p for p in c["points"] if p["exposure"] > 0]
    best = min(pts, key=lambda p: (p["mse"], p["exposure"])); last = pts[-1]; bestE = min(pts, key=lambda p: p["edyn"])
    def R(p): return (cl-p["edyn"])/m
    marks = [e for e in (1, 5, 10, 20, 31, 50, 75, 100, 150, 200, 300)]
    byexp = {p["exposure"]: p for p in pts}
    print("LR %g orizzonte %s n punti %d | min MSE a esp %g R %.4f mse %.6f | ultimo esp %g R %.4f mse %.6f | max R %.4f a esp %g" % (
        c["lr"], c["horizon"], len(pts), best["exposure"], R(best), best["mse"], last["exposure"], R(last), last["mse"], R(bestE), bestE["exposure"]))
    print("    ", " ".join("%g:%.4f" % (e, R(byexp[e])) for e in marks if e in byexp))
