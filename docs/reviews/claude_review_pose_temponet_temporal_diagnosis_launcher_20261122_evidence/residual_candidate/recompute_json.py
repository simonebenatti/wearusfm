"""Ricalcolo dai JSON archiviati (sola lettura). Uso: python -I recompute_json.py"""
import json
from fractions import Fraction
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
g = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
ref = g["reference_gate"]["temporal_records"]
sel = g["selected_panels"]["temporal"]
def E(s): return s["dynamic_sse"]/s["dynamic_target_ss"]
out = {}
pool = {"dynamic_sse": 0.0, "dynamic_target_ss": 0.0}
for fold in ("fold0", "fold1", "pooled"):
    cl, orc = ref[fold]["clean_ls"], ref[fold]["oracle"]
    m = cl-orc
    row = {"clean_ls": cl, "oracle": orc, "margin": m, "threshold50": cl-0.5*m}
    for h in ("L0", "MLP", "TCN_post"):
        row["R_"+h] = (cl-sel[h][fold]["E_dyn"])/m
    if fold != "pooled":
        w = g["linear_diagnostics"][fold]["L0"]["worlds"]["temporal"]
        st = w["held"]["statistics"]
        row["E_closed_noisy_held"] = E(st)
        row["R_closed_noisy_held"] = (cl-E(st))/m
        row["E_closed_noisy_fit"] = E(w["fitting"]["statistics"])
        for k in pool: pool[k] += st[k]
        row["T_unread_L0_fit"] = g["linear_diagnostics"][fold]["L0"]["T_unread_fraction_fitting"]
        row["T_unread_L13_fit"] = g["linear_diagnostics"][fold]["L13"]["T_unread_fraction_fitting"]
        w13 = g["linear_diagnostics"][fold]["L13"]["worlds"]["temporal"]["held"]["statistics"]
        row["R_L13_closed_noisy_held"] = (cl-E(w13))/m
    else:
        row["E_closed_noisy_held"] = E(pool)
        row["R_closed_noisy_held"] = (cl-E(pool))/m
    # quanto deve recuperare il ramo temporale partendo dalla base in forma chiusa su y
    row["needed_from_closed_base_points"] = 0.5-row["R_closed_noisy_held"]
    row["tcn_minus_closed_points"] = row["R_TCN_post"]-row["R_closed_noisy_held"]
    row["tcn_minus_L0neural_points"] = row["R_TCN_post"]-row["R_L0"]
    row["ratio_needed_over_obtained_vs_closed"] = row["needed_from_closed_base_points"]/row["tcn_minus_closed_points"]
    out[fold] = row
for fold, row in out.items():
    print(fold)
    for k, v in row.items():
        print("   %-40s %.6f" % (k, v))
print("recipe", g["recipe"]) 
for f in (0, 1):
    c = json.load(open(B+"calibration_%d.json" % f))
    print("calibration", f, "keys", list(c.keys())[:20])
    cal = c.get("calibration", c)
    def find(o, key, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    yield path+"/"+k, v
                else:
                    yield from find(v, key, path+"/"+k)
    for p, v in find(c, "selected"):
        if isinstance(v, dict) and "temporal" in v:
            t = v["temporal"]
            small = {k: t[k] for k in t if k != "alpha_curve"}
            print("  ", p, json.dumps(small))
            Rin = (t["E_cleanLS"]-t["E_cal"])/(t["E_cleanLS"]-t["E_oracle"])
            print("   inner-CV ridge(alpha=%g) on gate scale: %.6f ; alpha curve:" % (t["alpha"], Rin))
            for r in t["alpha_curve"]:
                print("      alpha %g inner MSE %.6f" % (r["alpha"], r["mse"]))
            lin = v["linear"]
            print("   linear world:", json.dumps({k: lin[k] for k in lin if k != "alpha_curve"}))
