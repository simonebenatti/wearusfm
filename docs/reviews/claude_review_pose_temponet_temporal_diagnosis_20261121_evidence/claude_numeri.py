"""Reviewer's own numbers for the temporal control (raw JSON only)."""
import json
from pathlib import Path
L = Path("results/infra/pose_temponet_execution_20261121_A1/launcher")
sel = {h: json.loads((L/f"selection_temporal_{h}.json").read_text()) for h in ("TCN_post", "L0", "MLP")}
cal = {f: json.loads((L/f"calibration_{f}.json").read_text()) for f in (0, 1)}
E = lambda s: s["dynamic_sse"]/s["dynamic_target_ss"]
out = {}
for f in (0, 1):
    t = cal[f]["outer_reference_statistics"]["temporal"]
    ss = t["clean_ls"]["dynamic_target_ss"]
    margin = t["clean_ls"]["dynamic_sse"] - t["oracle"]["dynamic_sse"]
    row = {"target_ss": ss, "margin_sse": margin, "margin_E": margin/ss, "cleanLS": E(t["clean_ls"]), "oracle": E(t["oracle"]),
           "oracle_gain_fraction": margin/t["clean_ls"]["dynamic_sse"]}
    for h in sel:
        st = sel[h]["selected_outer_statistics"][f"fold{f}"]
        cl = sel[h]["selected_outer_clean_statistics"][f"fold{f}"]
        row[h] = {"E": E(st), "recovery": (t["clean_ls"]["dynamic_sse"] - st["dynamic_sse"])/margin,
                  "clean_E": E(cl), "clean_sse": cl["dynamic_sse"], "clean_ss": cl["dynamic_target_ss"],
                  "fit_E": E(sel[h]["selected_fitting_diagnostics"][str(f)]),
                  "selected": sel[h]["selected"]}
    # closed-form linear readers fitted on the noisy y: fitting and held
    d = cal[f]["outer_reference_statistics"]["linear_diagnostics"]
    for name in ("L0", "L13"):
        w = d[name]["worlds"]["temporal"]
        row["closed_"+name] = {"fit_on_y_fitting": E(w["fitting"]["statistics"]) if "statistics" in w["fitting"] else None,
                               "keys": sorted(w.keys()), "fitting_keys": sorted(w["fitting"].keys())}
    out[f] = row
print(json.dumps(out[0]["closed_L0"], indent=0)[:600])
for f in (0, 1):
    r = out[f]
    print(f"\n=== fold {f}: target_ss {r['target_ss']:.0f} | margine {r['margin_sse']:.0f} SSE = {r['margin_E']:.6f} E_dyn | cleanLS {r['cleanLS']:.6f} oracolo {r['oracle']:.6f} | guadagno {r['oracle_gain_fraction']:.4%}")
    for h in sel:
        x = r[h]
        print(f"   {h:9s} held {x['E']:.6f} recupero {x['recovery']:+.2%} | fitting {x['fit_E']:.6f} | held pulito {x['clean_E']:.6f} (SSE {x['clean_sse']:.0f} su {x['clean_ss']:.0f}) | scelta lr {x['selected']['lr']} epoca {x['selected']['exposure']:.0f}")
    clean_ls_clean = r["margin_sse"]  # cleanLS clean SSE ~ noisy SSE - noise SS, with noise SS ~ oracle SSE
    for h in sel:
        print(f"   {h:9s} recupero sul bersaglio pulito (riferimento cleanLS-oracolo): {(clean_ls_clean - r[h]['clean_sse'])/clean_ls_clean:+.2%}")
json.dump(out, open("../diag_mine.json", "w"), indent=1, default=str)
