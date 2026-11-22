"""Ricalcolo del cancello dai soli accumulatori archiviati."""
import json, sys
from fractions import Fraction as F
L = sys.argv[1]
def load(name):
    return json.load(open(L + "/" + name))
cal = [load("calibration_%d.json" % k) for k in (0, 1)]
gate = load("positive_controls_gate_20261121.json")["gate"]
tcn = load("selection_temporal_TCN_post.json")
mlp = load("selection_temporal_MLP.json")
l0 = load("selection_temporal_L0.json")
def pool(rows):
    return {k: sum(r[k] for r in rows) for k in ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")}
def edyn(r):
    return r["dynamic_sse"]/r["dynamic_target_ss"]
ref = {"fold0": cal[0]["outer_reference_statistics"]["temporal"], "fold1": cal[1]["outer_reference_statistics"]["temporal"]}
ref["pooled"] = {n: pool([ref["fold0"][n], ref["fold1"][n]]) for n in ("clean_ls", "oracle", "constant")}
print("== cancello temporale ricalcolato dagli accumulatori ==")
for split in ("fold0", "fold1", "pooled"):
    ls, orc = edyn(ref[split]["clean_ls"]), edyn(ref[split]["oracle"])
    for name, s in (("TCN_post", tcn), ("MLP", mlp), ("L0", l0)):
        st = s["selected_outer_statistics"][split]
        e = edyn(st)
        R = (ls - e)/(ls - orc)
        print("%-6s %-8s E_cleanLS=%.16f E_oracle=%.16f E_head=%.16f R=%.16f" % (split, name, ls, orc, e, R))
    print("   gate.temporal_records:", gate["reference_gate"]["temporal_records"][split])
    print("   gate.recovery TCN:", repr(gate["temporal_head_gate"]["recovery"][split]), " MLP:", repr(gate["temporal_head_gate"]["mlp_recovery"][split]))
    print("   oracle_gain_fraction:", repr(gate["reference_gate"]["temporal_gate"]["oracle_gain_fraction"][split]), " ricalcolato:", repr((ls-orc)/ls))
# pooled come media dei rapporti (per confronto)
R = {}
for split in ("fold0", "fold1"):
    ls, orc = edyn(ref[split]["clean_ls"]), edyn(ref[split]["oracle"])
    R[split] = (ls - edyn(tcn["selected_outer_statistics"][split]))/(ls - orc)
print("media dei due R di fold (NON usata dal codice):", (R["fold0"] + R["fold1"])/2)
print("media degli E_dyn di fold TCN (NON usata):", (edyn(tcn["selected_outer_statistics"]["fold0"]) + edyn(tcn["selected_outer_statistics"]["fold1"]))/2,
      " pooled per somma:", edyn(tcn["selected_outer_statistics"]["pooled"]))
print()
print("== pooled archiviato = somma dei due fold? (selection_temporal_TCN_post.selected_outer_statistics) ==")
for s, name in ((tcn, "TCN_post"), (mlp, "MLP"), (l0, "L0")):
    st = s["selected_outer_statistics"]
    p = pool([st["fold0"], st["fold1"]])
    print(name, {k: (p[k], st["pooled"][k], p[k] == st["pooled"][k]) for k in p})
print()
print("== selected (curva) vs pooled dai checkpoint ==")
for s, name in ((tcn, "TCN_post"), (mlp, "MLP"), (l0, "L0")):
    st = s["selected_outer_statistics"]["pooled"]
    print(name, "selected:", s["selected"], " pooled MSE:", repr(st["mse_sum"]/st["window_count"]), " pooled E_dyn:", repr(edyn(st)))
print()
print("== identita' su y e y* dagli accumulatori archiviati (termine incrociato implicito) ==")
for split in ("fold0", "fold1", "pooled"):
    sy = tcn["selected_outer_statistics"][split]["dynamic_sse"]
    sc = tcn["selected_outer_clean_statistics"][split]["dynamic_sse"]
    so = ref[split]["oracle"]["dynamic_sse"]
    sl = ref[split]["clean_ls"]["dynamic_sse"]
    cross2 = sc + so - sy
    print("%-6s SSE_y(TCN)=%.6f SSE_y*(TCN)=%.6f SSE_oracle=%.6f -> 2<e,n>=%.6f (%.4f%% di SSE_y; %.3f%% del margine SSE cleanLS-oracle=%.6f)" % (
        split, sy, sc, so, cross2, 100*cross2/sy, 100*cross2/(sl-so), sl-so))
    print("        denominatori: SS(y)=%.6f SS(y*)=%.6f ; E su y*: TCN=%.6f ; cleanLS (stima, cross=0)=%.6f" % (
        tcn["selected_outer_statistics"][split]["dynamic_target_ss"], tcn["selected_outer_clean_statistics"][split]["dynamic_target_ss"],
        sc/tcn["selected_outer_clean_statistics"][split]["dynamic_target_ss"], (sl-so)/tcn["selected_outer_clean_statistics"][split]["dynamic_target_ss"]))
print()
print("== passi per epoca (fold_steps a esposizione 1.0) e punto selezionato ==")
for c in tcn["curves"]:
    p1 = [p for p in c["points"] if p["exposure"] == 1.0][0]
    best = min((p for p in c["points"] if p["exposure"] > 0), key=lambda p: (p["mse"], p["exposure"]))
    bd = min((p for p in c["points"] if p["exposure"] > 0), key=lambda p: (p["edyn"], p["exposure"]))
    print("lr=%g horizon=%d n_points=%d steps@1=%s  min MSE: exp=%g mse=%.10f edyn=%.10f | min Edyn: exp=%g edyn=%.10f mse=%.10f" % (
        c["lr"], c["horizon"], len(c["points"]), p1["fold_steps"], best["exposure"], best["mse"], best["edyn"], bd["exposure"], bd["edyn"], bd["mse"]))
print("selected_fitting_diagnostics:", tcn["selected_fitting_diagnostics"])
for k in ("0", "1"):
    d = tcn["selected_fitting_diagnostics"][k]
    print("  fitting fold", k, "MSE=%.10f E_dyn=%.10f" % (d["mse_sum"]/d["window_count"], d["dynamic_sse"]/d["dynamic_target_ss"]))
print()
print("== mondi calibrati ==")
for k in (0, 1):
    for w in ("linear", "temporal"):
        s = cal[k]["calibration"]["selected"][w]
        print("fold%d %-8s" % (k, w), {q: s[q] for q in s if q not in ("alpha_curve",)})
    print("   E_real", cal[k]["calibration"]["E_real"], "real_reference", {q: v for q, v in cal[k]["real_reference"].items() if q != "curve"})
