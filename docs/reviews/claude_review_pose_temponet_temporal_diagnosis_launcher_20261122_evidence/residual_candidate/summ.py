"""Riepilogo del giocattolo dai run_*.json (python -I summ.py). Selezione: minimo MSE rumorosa sugli esclusi fra le epoche fatte."""
import json, glob, os, sys
W = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate/toy/"
cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
runs = {}
for p in sorted(glob.glob(W+"run_f*.json")):
    r = json.load(open(p)); pts = [q for q in r["points"] if q["epoch"] > 0 and (cap is None or q["epoch"] <= cap)]
    if not pts: continue
    sel = min(pts, key=lambda q: (q["held"]["mse"], q["epoch"]))
    best = max(pts, key=lambda q: q["held"]["rec"])
    by = {q["epoch"]: q["held"] for q in pts}
    runs[(r["variant"], r["seed"])] = dict(n=pts[-1]["epoch"], sel=sel["epoch"], rec=sel["held"]["rec"], mse=sel["held"]["mse"],
                                           lin=sel["held"]["rec_linear_part"], clean=sel["held"]["clean_rec"], best=best["held"]["rec"],
                                           beste=best["epoch"], last=pts[-1]["held"]["rec"], by=by, trainable=r["trainable"],
                                           sec=pts[-1]["t"])
print("tetto epoche:", cap)
print("%-9s %4s %6s %4s %8s %8s %8s %8s %8s %6s %8s %6s" % ("variante", "seme", "epoche", "sel", "R@sel", "lin@sel", "prof@sel", "MSE@sel", "Rmax", "(ep)", "R ultima", "s/ep"))
for (v, s), d in sorted(runs.items()):
    print("%-9s %4d %6d %4d %+8.4f %+8.4f %+8.4f %8.5f %+8.4f %6d %+8.4f %6.1f" % (v, s, d["n"], d["sel"], d["rec"], d["lin"], d["rec"]-d["lin"], d["mse"], d["best"], d["beste"], d["last"], d["sec"]/d["n"]))
print()
eps = (5, 10, 15, 19, 20, 25, 30, 35, 40)
print("%-9s %4s " % ("variante", "seme") + " ".join("%7s" % ("ep%d" % e) for e in eps))
for (v, s), d in sorted(runs.items()):
    print("%-9s %4d " % (v, s) + " ".join(("%+7.3f" % d["by"][e]["rec"]) if e in d["by"] else "%7s" % "-" for e in eps))
print()
for s in (0, 1, 2):
    if ("cur", s) in runs:
        for v in ("res", "base", "lnfix", "warm", "wd0.3", "drop0.3"):
            if (v, s) in runs:
                a, b = runs[("cur", s)], runs[(v, s)]
                common = min(a["n"], b["n"])
                sa = min((q for q in a["by"].items() if q[0] <= common), key=lambda q: (q[1]["mse"], q[0]))
                sb = min((q for q in b["by"].items() if q[0] <= common), key=lambda q: (q[1]["mse"], q[0]))
                print("seme %d %-8s - cur: epoche comuni %d | R@sel %+.4f (ep %d) contro %+.4f (ep %d): differenza %+.4f | all'epoca comune %d: %+.4f contro %+.4f: differenza %+.4f | massimo scarto |diff| per epoca: %.4f" % (
                    s, v, common, sb[1]["rec"], sb[0], sa[1]["rec"], sa[0], sb[1]["rec"]-sa[1]["rec"], common, b["by"][common]["rec"], a["by"][common]["rec"],
                    b["by"][common]["rec"]-a["by"][common]["rec"], max(abs(b["by"][e]["rec"]-a["by"][e]["rec"]) for e in range(1, common+1))))
