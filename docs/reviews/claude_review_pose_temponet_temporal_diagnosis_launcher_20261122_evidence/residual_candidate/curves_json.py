"""Curve archiviate (dati VERI, JSON): velocita' di convergenza di L0, MLP e TCN sul controllo temporale. python -I curves_json.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
CL, OR = 0.7284790286546209, 0.6706589239838878      # pooled clean_ls / oracle (positive_controls_gate)
def R(e): return (CL-e)/(CL-OR)
for head in ("L0", "MLP", "TCN_post"):
    s = json.load(open(B+"selection_temporal_%s.json" % head))
    sel = s["selected"]
    print("==", head, "selected lr", sel["lr"], "exposure", sel["exposure"], "mse %.6f edyn %.6f R %.4f" % (sel["mse"], sel["edyn"], R(sel["edyn"])),
          "| min exposures", s["minimum_exposures"], "| alarms", s["optimizer_alarms"])
    for c in s["curves"]:
        pts = c["points"]
        best = min(pts[1:], key=lambda p: (p["mse"], p["exposure"]))
        bestd = min(pts[1:], key=lambda p: (p["edyn"], p["exposure"]))
        byexp = {p["exposure"]: p for p in pts}
        row = []
        for e in (1.0, 5.0, 10.0, 20.0, 31.0, 50.0, 75.0, 100.0, 150.0, 300.0):
            if e in byexp:
                row.append("%g: mse %.4f R %+.3f" % (e, byexp[e]["mse"], R(byexp[e]["edyn"])))
        print("  lr %g horizon %s n_points %d | argmin MSE exp %g (mse %.6f, R %+.4f) | argmin E_dyn exp %g (R %+.4f) | last exp %g mse %.4f R %+.3f" % (
            c["lr"], c["horizon"], len(pts), best["exposure"], best["mse"], R(best["edyn"]), bestd["exposure"], R(bestd["edyn"]),
            pts[-1]["exposure"], pts[-1]["mse"], R(pts[-1]["edyn"])))
        print("     " + " | ".join(row))
