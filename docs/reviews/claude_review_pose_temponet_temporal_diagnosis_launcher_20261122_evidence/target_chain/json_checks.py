"""Numeri ricalcolati dai soli JSON archiviati (nessun dato vero). Uso: python json_checks.py <cartella launcher>"""
import json, math, sys
B = sys.argv[1].rstrip("/") + "/"
cal = [json.load(open(B+f"calibration_{f}.json")) for f in (0, 1)]
ev = lambda st: (st["mse_sum"]/st["window_count"], st["dynamic_sse"]/st["dynamic_target_ss"])
out = {"calibration": {}, "gate": {}, "coherence": {}, "noise": {}, "fit_vs_held": {}}
for f, d in enumerate(cal):
    c = d["calibration"]; t = d["outer_reference_statistics"]["temporal"]; l = d["outer_reference_statistics"]["linear"]
    adm = [r for r in c["curve"] if r["admissible"] and r["a"] > 0]
    s = c["selected"]["temporal"]; cl, orc = ev(t["clean_ls"])[1], ev(t["oracle"])[1]
    ld = d["outer_reference_statistics"]["linear_diagnostics"]
    out["calibration"][f"fold{f}"] = {
        "evaluations": c["evaluations"], "a_values": sorted({r["a"] for r in c["curve"]}),
        "n_sigma": len({r["sigma"] for r in c["curve"]}),
        "admissible_temporal": [(r["a"], r["sigma"], round(r["oracle_gain_fraction"], 5)) for r in adm],
        "selected_temporal": {k: s[k] for k in ("a", "sigma", "alpha", "E_cal", "E_cleanLS", "E_oracle", "oracle_gain_fraction")},
        "selected_linear": {k: c["selected"]["linear"][k] for k in ("a", "sigma", "alpha", "E_cal")},
        "E_real": c["E_real"], "inner_margin_abs": s["E_cleanLS"]-s["E_oracle"],
        "outer_clean_ls": cl, "outer_oracle": orc, "outer_gain": (cl-orc)/cl, "outer_margin_abs": cl-orc,
        "oracle_mse_temporal": ev(t["oracle"])[0], "sigma2_temporal": s["sigma"]**2,
        "oracle_mse_linear": ev(l["oracle"])[0], "sigma2_linear": c["selected"]["linear"]["sigma"]**2,
        "noisy_WLS_held_R": {n: (cl-ev(ld[n]["worlds"]["temporal"]["held"]["statistics"])[1])/(cl-orc) for n in ld},
        "T_unread_fraction_fitting": {n: ld[n]["T_unread_fraction_fitting"] for n in ld},
        "L13_clean_margin_recovery_fitting": ld["L13"]["temporal_margin_recovery_fitting"]}
T = [d["outer_reference_statistics"]["temporal"] for d in cal]
pool = lambda name, key: sum(t[name][key] for t in T)
pc = pool("clean_ls", "dynamic_sse")/pool("clean_ls", "dynamic_target_ss"); po = pool("oracle", "dynamic_sse")/pool("oracle", "dynamic_target_ss")
out["gate"]["pooled_refs"] = {"clean_ls": pc, "oracle": po, "gain": (pc-po)/pc, "threshold_50": (pc+po)/2}
for h in ("L0", "MLP", "TCN_post"):
    s = json.load(open(B+f"selection_temporal_{h}.json")); r = {}
    for f in (0, 1):
        st, cs = s["selected_outer_statistics"][f"fold{f}"], s["selected_outer_clean_statistics"][f"fold{f}"]
        cl, orc = T[f]["clean_ls"], T[f]["oracle"]
        r[f"fold{f}"] = (ev(cl)[1]-ev(st)[1])/(ev(cl)[1]-ev(orc)[1])
        cross = (cs["dynamic_sse"]+orc["dynamic_sse"]-st["dynamic_sse"])/2
        out["noise"][f"{h}_fold{f}_corr_prederr_heldnoise"] = cross/math.sqrt(cs["dynamic_sse"]*orc["dynamic_sse"])
        out["coherence"][f"{h}_fold{f}"] = {"threshold_equal": s["bundle"]["thresholds"][f] == cal[f]["outer_reference_statistics"]["temporal_threshold"],
            "target_ss_rel_diff": abs(st["dynamic_target_ss"]-orc["dynamic_target_ss"])/orc["dynamic_target_ss"],
            "window_count_equal": st["window_count"] == orc["window_count"], "target_sha256": s["bundle"]["target_sha256"][f][:16]}
    e = ev(s["selected_outer_statistics"]["pooled"])[1]; r["pooled"] = (pc-e)/(pc-po); r["selected"] = s["selected"]
    out["gate"][h] = r
    if h == "TCN_post":
        pts = [(p["edyn"], c["lr"], p["exposure"]) for c in s["curves"] for p in c["points"] if p["exposure"] > 0]
        b = min(pts); out["gate"]["best_edyn_all_curves"] = {"edyn": b[0], "lr": b[1], "exposure": b[2], "R": (pc-b[0])/(pc-po)}
        out["gate"]["end_of_horizon_R"] = {str(c["lr"]): (pc-c["points"][-1]["edyn"])/(pc-po) for c in s["curves"]}
        for f in (0, 1):
            cf = cal[f]["outer_reference_statistics"]["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"]
            fit = s["selected_fitting_diagnostics"][str(f)]; held = s["selected_outer_statistics"][f"fold{f}"]
            gf = [ev(cf)[i]-ev(fit)[i] for i in (0, 1)]; gh = [ev(T[f]["clean_ls"])[i]-ev(held)[i] for i in (0, 1)]
            out["fit_vs_held"][f"fold{f}"] = {"fitting_clean_ls": ev(cf), "fitting_TCN": ev(fit), "held_clean_ls": ev(T[f]["clean_ls"]), "held_TCN": ev(held),
                "gain_fit_mse_edyn": gf, "gain_held_mse_edyn": gh, "transferred_mse": gh[0]/gf[0], "transferred_edyn": gh[1]/gf[1],
                "fitting_target_ss_rel_diff": abs(fit["dynamic_target_ss"]-cf["dynamic_target_ss"])/cf["dynamic_target_ss"]}
print(json.dumps(out, indent=1))
