"""Riepilogo delle mie corse (python -I rc2_summ.py). Scelta = minimo MSE rumorosa sugli esclusi, a parita' l'epoca minore."""
import json, glob, os
W = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc2_work/"
O = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate/toy/"
def pick(pts, cap=None):
    pts = [p for p in pts if p["epoch"] > 0 and (cap is None or p["epoch"] <= cap)]
    sel = min(pts, key=lambda p: (p["held"]["mse"], p["epoch"])); mx = max(pts, key=lambda p: p["held"]["rec"])
    return pts, sel, mx
runs = {}
print("%-22s %6s %4s %8s %8s %9s %8s %5s %8s %s" % ("corsa", "epoche", "sel", "R@sel", "lin@sel", "MSE@sel", "Rmax", "(ep)", "R ultima", "penultimo vive (min dopo ep 0 / ultima)"))
for f in sorted(glob.glob(W+"run_*.json")):
    r = json.load(open(f)); pts, sel, mx = pick(r["points"]); runs[r["name"]] = r
    alive = [p["held"]["relu"][-1][0] for p in pts]
    print("%-22s %6d %4d %+8.4f %+8.4f %9.5f %+8.4f %5d %+8.4f %d / %d" % (r["name"], pts[-1]["epoch"], sel["epoch"], sel["held"]["rec"], sel["held"]["rec_linear_part"],
          sel["held"]["mse"], mx["held"]["rec"], mx["epoch"], pts[-1]["held"]["rec"], min(alive), alive[-1]))
print("\nDeterminismo rispetto ai JSON dell'altro revisore (epoche comuni):")
for mine, theirs in (("cur_s0_lr0.003", "f0_cur_s0"), ("res_s0_lr0.003", "f0_res_s0"), ("cur_s2_lr0.003", "f0_cur_s2"), ("cur_s1_lr0.003", "f0_cur_s1")):
    if mine not in runs: continue
    a = {p["epoch"]: p["held"] for p in runs[mine]["points"]}
    b = {p["epoch"]: p["held"] for p in json.load(open(O+"run_%s.json" % theirs))["points"]}
    c = sorted(set(a) & set(b))
    print("  %-16s epoche 0..%d: max |dMSE| %.2e, max |dR| %.2e" % (mine, c[-1], max(abs(a[e]["mse"]-b[e]["mse"]) for e in c), max(abs(a[e]["rec"]-b[e]["rec"]) for e in c)))
def pair(x, y, label):
    if x not in runs or y not in runs: return
    a, b = runs[x]["points"], runs[y]["points"]
    common = min(a[-1]["epoch"], b[-1]["epoch"])
    _, sa, ma = pick(a, common); _, sb, mb = pick(b, common)
    ea = [p for p in a if p["epoch"] == common][0]["held"]["rec"]; eb = [p for p in b if p["epoch"] == common][0]["held"]["rec"]
    print("  %s | epoche comuni %d | alla scelta: %+.4f (ep %d) - %+.4f (ep %d) = %+.4f | ai massimi: %+.4f - %+.4f = %+.4f | all'epoca %d: %+.4f" % (
        label, common, sa["held"]["rec"], sa["epoch"], sb["held"]["rec"], sb["epoch"], sa["held"]["rec"]-sb["held"]["rec"],
        ma["held"]["rec"], mb["held"]["rec"], ma["held"]["rec"]-mb["held"]["rec"], common, ea-eb))
    _, sa, _ = pick(a); _, sb, _ = pick(b)
    print("      senza tetto comune: %+.4f (ep %d su %d) - %+.4f (ep %d su %d) = %+.4f" % (sa["held"]["rec"], sa["epoch"], a[-1]["epoch"], sb["held"]["rec"], sb["epoch"], b[-1]["epoch"], sa["held"]["rec"]-sb["held"]["rec"]))
print("\nAppaiate (prima - seconda):")
pair("res_s0_lr0.003", "cur_s0_lr0.003", "res - cur, seme 0, LR 0,003")
pair("res_s2_lr0.003", "cur_s2_lr0.003", "res - cur, seme 2, LR 0,003")
pair("f1_res_s0_lr0.003", "f1_cur_s0_lr0.003", "res - cur, seme 0, LR 0,003, FOLD 1")
pair("res_s0_lr0.001", "res_s0_lr0.003", "res LR 0,001 - res LR 0,003, seme 0")
pair("cur_s0_lr0.001", "cur_s0_lr0.003", "cur LR 0,001 - cur LR 0,003, seme 0")
pair("res_s0_lr0.001", "cur_s0_lr0.001", "res - cur, seme 0, LR 0,001")
# res s2 mio contro cur s2 dell'altro revisore (22 epoche)
if "res_s2_lr0.003" in runs:
    b = json.load(open(O+"run_f0_cur_s2.json"))["points"]; a = runs["res_s2_lr0.003"]["points"]
    _, sb, mb = pick(b); _, sa, ma = pick(a)
    print("  res s2 (mio, %d epoche) - cur s2 (altro revisore, 22 epoche): alla scelta %+.4f (ep %d) - %+.4f (ep %d) = %+.4f ; ai massimi %+.4f - %+.4f = %+.4f" % (
        a[-1]["epoch"], sa["held"]["rec"], sa["epoch"], sb["held"]["rec"], sb["epoch"], sa["held"]["rec"]-sb["held"]["rec"], ma["held"]["rec"], mb["held"]["rec"], ma["held"]["rec"]-mb["held"]["rec"]))
for name in sorted(runs):
    by = {p["epoch"]: p["held"]["rec"] for p in runs[name]["points"]}
    print("%-22s " % name + " ".join("%d:%+.3f" % (e, by[e]) for e in sorted(by) if e in (5, 10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 35, 40)))
