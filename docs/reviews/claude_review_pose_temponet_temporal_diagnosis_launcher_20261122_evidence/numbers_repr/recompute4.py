"""Verifica dei numeri dell'altra diagnosi (chat 2, 8fac3a82) dagli stessi JSON."""
import json, math
from pathlib import Path
L = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher")
J = lambda n: json.loads((L/n).read_text())
edyn = lambda r: r["dynamic_sse"]/r["dynamic_target_ss"]
def pool(rows): return {k: math.fsum(r[k] for r in rows) for k in ("dynamic_sse", "dynamic_target_ss", "mse_sum", "window_count")}
cal = [J("calibration_0.json"), J("calibration_1.json")]
sel = {h: J(f"selection_temporal_{h}.json") for h in ("L0", "MLP", "TCN_post")}
lin = {h: J(f"selection_linear_{h}.json") for h in ("L0", "MLP", "TCN_post")}
ref = {n: [c["outer_reference_statistics"]["temporal"][n] for c in cal] for n in ("clean_ls", "oracle")}
def panels(rows): return {"fold0": rows[0], "fold1": rows[1], "pooled": pool(rows)}
R = {n: panels(v) for n, v in ref.items()}
def rec(e, s): return (edyn(R["clean_ls"][s])-e)/(edyn(R["clean_ls"][s])-edyn(R["oracle"][s]))
print("== L0 e L13 in forma chiusa su y (worlds.temporal) fitting -> esclusi")
for name in ("L0", "L13"):
    f = panels([c["outer_reference_statistics"]["linear_diagnostics"][name]["worlds"]["temporal"]["fitting"]["statistics"] for c in cal])
    h = panels([c["outer_reference_statistics"]["linear_diagnostics"][name]["worlds"]["temporal"]["held"]["statistics"] for c in cal])
    for s in ("fold0", "fold1", "pooled"):
        print(f" {name} {s}: fitting {edyn(f[s]):.6f} esclusi {edyn(h[s]):.6f} divario {edyn(h[s])-edyn(f[s]):+.5f} recupero esclusi {100*rec(edyn(h[s]), s):.2f}%  den esclusi {h[s]['dynamic_target_ss']!r}")
print("== scala alternativa: da L0 neurale all'oracolo")
for s in ("fold0", "fold1", "pooled"):
    st = lambda h: (sel[h]["selected_outer_statistics"][s] if s != "pooled" else pool([sel[h]["selected_outer_statistics"][k] for k in ("fold0", "fold1")]))
    l0, tcn, orc, cls = edyn(st("L0")), edyn(st("TCN_post")), edyn(R["oracle"][s]), edyn(R["clean_ls"][s])
    need = (l0-(cls+orc)/2)/(l0-orc)
    print(f" {s}: (L0-TCN)/(L0-oracolo) = {100*(l0-tcn)/(l0-orc):.2f}% ; soglia equivalente {100*need:.2f}% ; TCN contro L0 neurale {100*(l0-tcn)/l0:.3f}%")
print("== divari fitting->esclusi (E_dyn)")
for h in ("TCN_post", "MLP", "L0"):
    fd = sel[h]["selected_fitting_diagnostics"]; so = sel[h]["selected_outer_statistics"]
    print(f" {h}: fold0 {edyn(so['fold0'])-edyn(fd['0']):+.5f} fold1 {edyn(so['fold1'])-edyn(fd['1']):+.5f} | fitting {edyn(fd['0']):.6f}/{edyn(fd['1']):.6f} esclusi {edyn(so['fold0']):.6f}/{edyn(so['fold1']):.6f}")
for f, c in enumerate(cal):
    d = c["outer_reference_statistics"]["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"]
    print(f" clean_ls fold{f}: fitting {edyn(d):.6f} esclusi {edyn(ref['clean_ls'][f]):.6f} divario {edyn(ref['clean_ls'][f])-edyn(d):+.5f}")
t = sel["TCN_post"]; c3 = next(c for c in t["curves"] if c["lr"] == 0.003)
fin = c3["points"][-1]; print("== LR 0,003 a 150:", fin, " peggioramento rispetto al selezionato: %.3f%%" % (100*(fin["edyn"]-t["selected"]["edyn"])/t["selected"]["edyn"]), " recupero a 150: %.2f%%" % (100*rec(fin["edyn"], "pooled")))
for c in t["curves"]:
    print("   finale lr %g: recupero pooled %.2f%%" % (c["lr"], 100*rec(c["points"][-1]["edyn"], "pooled")))
print("== euristica fra mondi: eccesso TCN lineare sull'oracolo lineare / margine temporale")
for f in (0, 1):
    lo = cal[f]["outer_reference_statistics"]["linear"]["oracle"]; lt = lin["TCN_post"]["selected_outer_statistics"]["fold"+str(f)]
    m = edyn(ref["clean_ls"][f])-edyn(ref["oracle"][f])
    print(f" fold{f}: eccesso {edyn(lt)-edyn(lo):.6f} = {100*(edyn(lt)-edyn(lo))/m:.2f}% del margine temporale; somma col recupero {100*((edyn(lt)-edyn(lo))/m+rec(edyn(sel['TCN_post']['selected_outer_statistics']['fold'+str(f)]), 'fold'+str(f))):.2f}%")
print("== sigma lineari:", [c["calibration"]["selected"]["linear"]["sigma"] for c in cal], "temporali:", [c["calibration"]["selected"]["temporal"]["sigma"] for c in cal])
print("== 150/900*4427 s =", 150/900*4427, "s =", 150/900*4427/3600, "GPUh; epoche totali della ricerca TCN temporale:", sum(c["horizon"] for c in t["curves"]))
print("== soglie dinamiche:", [c["outer_reference_statistics"]["temporal_threshold"] for c in cal], t["bundle"]["thresholds"])
