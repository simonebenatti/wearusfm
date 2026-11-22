"""Grandezze DERIVATE (non presenti nella diagnosi): recupero sul fitting con riferimenti omologhi,
e posizione dei riferimenti sul bersaglio pulito. Solo aritmetica sui JSON archiviati."""
import json, math
from pathlib import Path
ROOT = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1")
L = ROOT/"launcher"
J = lambda name: json.loads((L/name).read_text())
edyn = lambda r: r["dynamic_sse"]/r["dynamic_target_ss"]
mse = lambda r: r["mse_sum"]/r["window_count"]
cal = [J("calibration_0.json"), J("calibration_1.json")]
sel = {h: J(f"selection_temporal_{h}.json") for h in ("L0", "MLP", "TCN_post")}

print("=== I. fitting: clean_ls (stimato su y* del fitting, valutato su y del fitting), oracolo (derivato da L13), teste")
fit = {"cls": [], "orc": [], "l13": [], "den": []}
for f, c in enumerate(cal):
    d = c["outer_reference_statistics"]["linear_diagnostics"]
    base = d["L0"]["temporal_margin_clean_fit_statistics"]; l13 = d["L13"]["temporal_margin_clean_fit_statistics"]
    assert d["L0"]["temporal_margin_recovery_fitting"] == 0.0
    assert base["dynamic_target_ss"] == l13["dynamic_target_ss"]
    R13 = d["L13"]["temporal_margin_recovery_fitting"]
    e_base, e_l13 = edyn(base), edyn(l13)
    e_orc = e_base - (e_base-e_l13)/R13      # dalla definizione di _recovery (controls.py:251-260, 633)
    den = base["dynamic_target_ss"]
    fit["cls"].append(base["dynamic_sse"]); fit["l13"].append(l13["dynamic_sse"]); fit["orc"].append(e_orc*den); fit["den"].append(den)
    print(f" fold{f}: clean_ls_fit {e_base:.9f} (MSE {mse(base):.6f}) L13 clean fit {e_l13:.9f} R13 {R13:.6f} -> oracolo_fit {e_orc:.9f} | margine {e_base-e_orc:.9f} ({100*(e_base-e_orc)/e_base:.4f}% di clean_ls)")
    for h in ("L0", "MLP", "TCN_post"):
        s = sel[h]["selected_fitting_diagnostics"][str(f)]
        assert s["dynamic_target_ss"] == den, (h, s["dynamic_target_ss"], den)
        held = sel[h]["selected_outer_statistics"]["fold"+str(f)]
        hc, ho = (edyn(c["outer_reference_statistics"]["temporal"][k]) for k in ("clean_ls", "oracle"))
        print(f"    {h:8s}: fitting E_dyn {edyn(s):.9f} recupero fitting {100*(e_base-edyn(s))/(e_base-e_orc):8.4f}% | esclusi E_dyn {edyn(held):.9f} recupero esclusi {100*(hc-edyn(held))/(hc-ho):8.4f}%")
D = math.fsum(fit["den"]); pc, po = math.fsum(fit["cls"])/D, math.fsum(fit["orc"])/D
print(f" pooled fitting: clean_ls {pc:.9f} oracolo {po:.9f} L13(pulito) {math.fsum(fit['l13'])/D:.9f} -> recupero L13 {100*(pc-math.fsum(fit['l13'])/D)/(pc-po):.4f}%")
for h in ("L0", "MLP", "TCN_post"):
    s = sel[h]["selected_fitting_diagnostics"]
    e = math.fsum(s[k]["dynamic_sse"] for k in "01")/math.fsum(s[k]["dynamic_target_ss"] for k in "01")
    print(f"    {h:8s}: pooled fitting E_dyn {e:.9f} recupero fitting pooled {100*(pc-e)/(pc-po):.4f}%")

print("\n=== L. bersaglio pulito sugli esclusi: dove stanno clean_ls e oracolo (derivato: sse_oss - sse_oracolo, a meno del termine incrociato)")
for split in ("fold0", "fold1", "pooled"):
    def obs(name):
        rows = [c["outer_reference_statistics"]["temporal"][name] for c in cal]
        if split == "pooled": return math.fsum(r["dynamic_sse"] for r in rows)
        return rows[int(split[-1])]["dynamic_sse"]
    cden = sel["TCN_post"]["selected_outer_clean_statistics"][split]["dynamic_target_ss"]
    approx_cls = (obs("clean_ls")-obs("oracle"))/cden
    line = f" {split}: clean_ls sul pulito ~ {approx_cls:.6f} (approssimato)"
    for h in ("TCN_post", "MLP", "L0"):
        st = sel[h]["selected_outer_statistics"][split]; cs = sel[h]["selected_outer_clean_statistics"][split]
        cross = (st["dynamic_sse"]-obs("oracle")) - cs["dynamic_sse"]   # = -2 sigma <e, eps>
        line += f" | {h} pulito {edyn(cs):.6f} (termine incrociato {cross:+.1f}, {100*cross/cs['dynamic_sse']:+.2f}% della sse pulita)"
    print(line)
print("\n=== M. differenze E_dyn finale - minimo per curva (TCN temporale) e posizione relativa alla soglia")
t = sel["TCN_post"]
for c in t["curves"]:
    pos = [p for p in c["points"] if p["exposure"] > 0]
    bm = min(pos, key=lambda p: (p["mse"], p["exposure"]))
    print(f" lr {c['lr']:g}: exp del minimo MSE {bm['exposure']}, edyn {bm['edyn']:.6f}; finale edyn {pos[-1]['edyn']:.6f}; punti sotto la soglia 0,699569: {sum(p['edyn'] <= 0.699568976 for p in pos)}")
print(" punti totali sotto soglia su tutte le curve:", sum(p["edyn"] <= 0.699568976 for c in t["curves"] for p in c["points"] if p["exposure"] > 0), "su", sum(1 for c in t["curves"] for p in c["points"] if p["exposure"] > 0))
