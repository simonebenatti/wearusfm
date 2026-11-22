"""RC-3: mondo LINEARE sui latenti veri (JSON archiviati): ridge con alpha da CV contro L0 neurale, stessi esclusi, 20 utenti. Uso: python -I rc3_json2.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
g = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
rl, rt = g["reference_gate"]["linear_records"], g["reference_gate"]["temporal_records"]
sel = g["selected_panels"]["linear"]
def E(s): return s["dynamic_sse"]/s["dynamic_target_ss"]
pool = {"dynamic_sse": 0., "dynamic_target_ss": 0.}
print("chiavi selected_panels:", list(g["selected_panels"].keys()), "| teste lineare:", list(sel.keys()))
for fold in ("fold0", "fold1", "pooled"):
    orc, ridge, const = rl[fold]["E_dyn"]["oracle"], rl[fold]["E_dyn"]["ridge"], rl[fold]["E_dyn"]["constant"]
    mt = rt[fold]["clean_ls"]-rt[fold]["oracle"]
    if fold != "pooled":
        st = g["linear_diagnostics"][fold]["L0"]["worlds"]["linear"]["held"]["statistics"]
        for k in pool: pool[k] += st[k]
        wls0 = E(st)
    else:
        wls0 = E(pool)
    out = {"oracolo": orc, "ridge alpha CV": ridge, "WLS alpha 0": wls0}
    for h in ("L0", "MLP", "TCN_post"):
        out[h+" neurale"] = sel[h][fold]["E_dyn"]
    print(fold, "margine temporale %.6f" % mt)
    for k, v in out.items():
        print("   %-16s E_dyn %.6f | sopra l'oracolo %.6f = %6.2f%% del margine temporale | recupero costante->ridge %.4f" % (
            k, v, v-orc, 100*(v-orc)/mt, (const-v)/(const-ridge)))
    print("   L0 neurale meno ridge (E_dyn): %+.6f = %+.2f punti del margine temporale (positivo = ridge migliore)" % (
        out["L0 neurale"]-ridge, 100*(out["L0 neurale"]-ridge)/mt))
    print("   WLS0 meno L0 neurale: %+.2f punti; WLS0 meno ridge: %+.2f punti" % (100*(wls0-out["L0 neurale"])/mt, 100*(wls0-ridge)/mt))
s = json.load(open(B+"selection_linear_L0.json"))
print("L0 lineare selezionata:", s["selected"])
for c in s["curves"]:
    pts = [p for p in c["points"] if p["exposure"] > 0]; best = min(pts, key=lambda p: (p["mse"], p["exposure"]))
    print("  LR %g: min MSE a esp %g edyn %.6f | ultimo esp %g edyn %.6f" % (c["lr"], best["exposure"], best["edyn"], pts[-1]["exposure"], pts[-1]["edyn"]))
