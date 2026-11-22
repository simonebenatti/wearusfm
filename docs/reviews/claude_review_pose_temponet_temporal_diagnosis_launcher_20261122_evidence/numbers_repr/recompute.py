"""Ricalcolo indipendente dei numeri della diagnosi GPT dai JSON grezzi (accumulatori).
Sola lettura. Nessun import dal pacchetto pose_temponet: le formule sono riscritte qui.
"""
import json, math
from pathlib import Path
from fractions import Fraction

ROOT = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1")
L = ROOT/"launcher"
J = lambda name: json.loads((L/name).read_text())

def pool(records):
    return {k: (math.fsum(r[k] for r in records) if k != "window_count" else sum(r[k] for r in records))
            for k in ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")}
def edyn(r): return r["dynamic_sse"]/r["dynamic_target_ss"]
def mse(r): return r["mse_sum"]/r["window_count"]
def rec(base, cand, ref): return (base-cand)/(base-ref)

cal = [J("calibration_0.json"), J("calibration_1.json")]
sel = {(w, h): J(f"selection_{w}_{h}.json") for w in ("linear", "temporal") for h in ("L0", "MLP", "TCN_post")}
gate = J("positive_controls_gate_20261121.json")

print("=== A. calibrazione scelta nei due fold")
for c in cal:
    s = c["calibration"]["selected"]
    print(" fold", c["outer_fold"], "temporal a=%r sigma=%r alpha=%r gain=%.6f | linear a=%r sigma=%r alpha=%r" % (
        s["temporal"]["a"], s["temporal"]["sigma"], s["temporal"]["alpha"], s["temporal"]["oracle_gain_fraction"],
        s["linear"]["a"], s["linear"]["sigma"], s["linear"]["alpha"]), "status", c["calibration"]["status"],
        "E_real %.6f" % c["calibration"]["E_real"])

print("\n=== B. riferimenti temporali dagli accumulatori grezzi (calibration_*.json outer_reference_statistics.temporal)")
ref = {}
for name in ("clean_ls", "oracle", "constant"):
    rows = [c["outer_reference_statistics"]["temporal"][name] for c in cal]
    ref[name] = {"fold0": rows[0], "fold1": rows[1], "pooled": pool(rows)}
tcn = sel[("temporal", "TCN_post")]
mlp = sel[("temporal", "MLP")]
l0 = sel[("temporal", "L0")]
for split in ("fold0", "fold1", "pooled"):
    cl, orc = edyn(ref["clean_ls"][split]), edyn(ref["oracle"][split])
    t = tcn["selected_outer_statistics"][split]
    # pooled ricalcolato da me dai due fold, non letto
    if split == "pooled":
        t2 = pool([tcn["selected_outer_statistics"]["fold0"], tcn["selected_outer_statistics"]["fold1"]])
        assert all(abs(t2[k]-t[k]) <= 1e-9*max(1, abs(t[k])) for k in t), (t2, t)
        t = t2
    te = edyn(t)
    print(f" {split}: clean_ls {cl:.9f} oracle {orc:.9f} soglia50 {(cl+orc)/2:.9f} TCN {te:.9f} recupero {100*rec(cl, te, orc):.6f}% | gain oracolo {(cl-orc)/cl:.6f}")
    print(f"    denominatori dynamic_target_ss: clean_ls {ref['clean_ls'][split]['dynamic_target_ss']!r} oracle {ref['oracle'][split]['dynamic_target_ss']!r} TCN {t['dynamic_target_ss']!r} | finestre {ref['clean_ls'][split]['window_count']} {ref['oracle'][split]['window_count']} {t['window_count']}")
    print(f"    gate JSON: recovery {gate['gate']['temporal_head_gate']['recovery'][split]!r}  records {gate['gate']['reference_gate']['temporal_records'][split]}")

print("\n=== C. MLP e L0 temporali (descrittivi) dagli accumulatori")
for name, s in (("MLP", mlp), ("L0", l0), ("TCN_post", tcn)):
    p = pool([s["selected_outer_statistics"]["fold0"], s["selected_outer_statistics"]["fold1"]])
    cl, orc = edyn(ref["clean_ls"]["pooled"]), edyn(ref["oracle"]["pooled"])
    print(f" {name}: pooled E_dyn {edyn(p):.9f} MSE {mse(p):.9f} recupero {100*rec(cl, edyn(p), orc):.4f}%  selected {s['selected']} alarms {s['optimizer_alarms']} unresolved {s['optimization_unresolved']}")
    for f in ("fold0", "fold1"):
        r = s["selected_outer_statistics"][f]
        print(f"     {f}: E_dyn {edyn(r):.9f} MSE {mse(r):.9f} recupero {100*rec(edyn(ref['clean_ls'][f]), edyn(r), edyn(ref['oracle'][f])):.4f}%  target_ss {r['dynamic_target_ss']!r} n {r['window_count']}")
tp = pool([tcn["selected_outer_statistics"][f] for f in ("fold0", "fold1")]); mp = pool([mlp["selected_outer_statistics"][f] for f in ("fold0", "fold1")])
print(f" TCN contro MLP: (MLP-TCN)/MLP = {100*(edyn(mp)-edyn(tp))/edyn(mp):.6f}%   (MLP-TCN)/TCN = {100*(edyn(mp)-edyn(tp))/edyn(tp):.6f}%   differenza assoluta {edyn(mp)-edyn(tp):.9f}")

print("\n=== D. fitting al checkpoint scelto (selected_fitting_diagnostics) contro esclusi")
for name, s in (("TCN_post", tcn), ("MLP", mlp), ("L0", l0)):
    fd = s.get("selected_fitting_diagnostics")
    if not fd: print(" ", name, "nessun selected_fitting_diagnostics"); continue
    fp = pool([fd["0"], fd["1"]])
    hp = pool([s["selected_outer_statistics"][f] for f in ("fold0", "fold1")])
    print(f" {name}: fitting pooled E_dyn {edyn(fp):.9f} MSE {mse(fp):.9f} (n={fp['window_count']}) | esclusi pooled E_dyn {edyn(hp):.9f} MSE {mse(hp):.9f} (n={hp['window_count']})")
    for k in ("0", "1"):
        print(f"     fold{k} fitting: E_dyn {edyn(fd[k]):.9f} MSE {mse(fd[k]):.9f} n {fd[k]['window_count']} target_ss {fd[k]['dynamic_target_ss']!r}")

print("\n=== E. bersaglio pulito sugli esclusi (selected_outer_clean_statistics)")
for name, s in (("TCN_post", tcn), ("MLP", mlp), ("L0", l0)):
    cs = s.get("selected_outer_clean_statistics")
    if not cs: print(" ", name, "nessun selected_outer_clean_statistics"); continue
    cp = pool([cs["fold0"], cs["fold1"]])
    print(f" {name}: pooled clean E_dyn {edyn(cp):.9f} MSE {mse(cp):.9f} | fold0 {edyn(cs['fold0']):.9f} fold1 {edyn(cs['fold1']):.9f} | target_ss pooled {cp['dynamic_target_ss']!r}")

print("\n=== F. curve: orizzonti, completezza, minimi")
for key, s in sel.items():
    print(" ", key, "selected", s["selected"], "min_exposures", s["minimum_exposures"], "alarms", s["optimizer_alarms"], "unresolved", s["optimization_unresolved"])
    best_all = None
    for c in s["curves"]:
        pts = c["points"]; ex = [p["exposure"] for p in pts]
        integer = [e for e in ex if e >= 1]
        complete = (ex[0] == 0.0 and ex[-1] == c["horizon"] and integer == [float(k) for k in range(1, c["horizon"]+1)])
        pos = [p for p in pts if p["exposure"] > 0]
        bm = min(pos, key=lambda p: (p["mse"], p["exposure"])); bd = min(pos, key=lambda p: (p["edyn"], p["exposure"]))
        print(f"     lr {c['lr']:<7g} horizon {c['horizon']:>3} punti {len(pts)} (frazionari {[e for e in ex if 0 < e < 1]}) completo {complete} ultimo {ex[-1]} passi finali {pts[-1]['fold_steps']} | min MSE {bm['mse']:.9f} @ {bm['exposure']} (edyn {bm['edyn']:.9f}) | min E_dyn {bd['edyn']:.9f} @ {bd['exposure']} (mse {bd['mse']:.9f}) | finale edyn {pts[-1]['edyn']:.6f} mse {pts[-1]['mse']:.6f}")
        if best_all is None or bd["edyn"] < best_all[1]["edyn"]:
            best_all = (c["lr"], bd)
    print(f"     -> minimo E_dyn di TUTTE le curve: lr {best_all[0]} esposizione {best_all[1]['exposure']} edyn {best_all[1]['edyn']!r} mse {best_all[1]['mse']!r}")
    if key[0] == "temporal":
        cl, orc = edyn(ref["clean_ls"]["pooled"]), edyn(ref["oracle"]["pooled"])
        print(f"        recupero pooled retrospettivo: {100*rec(cl, best_all[1]['edyn'], orc):.6f}%")
