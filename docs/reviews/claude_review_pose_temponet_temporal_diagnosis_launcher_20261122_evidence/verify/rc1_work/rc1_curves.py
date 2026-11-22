"""Controllo indipendente: livelli della parte lineare (L0 neurale) lungo le curve archiviate. python -I rc1_curves.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
G = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
ref = G["reference_gate"]["temporal_records"]["pooled"]
R = lambda e: (ref["clean_ls"]-e)/(ref["clean_ls"]-ref["oracle"])
for head in ("L0", "TCN_post"):
    s = json.load(open(B+"selection_temporal_%s.json" % head))
    print(head, "scelta:", s["selected"], "R pooled %.4f%%" % (100*R(s["selected"]["edyn"])))
    for c in s["curves"]:
        pts = [p for p in c["points"] if p["exposure"] > 0]
        best = min(pts, key=lambda p: (p["mse"], p["exposure"]))
        bestR = max(pts, key=lambda p: R(p["edyn"]))
        at = {p["exposure"]: p for p in pts}
        txt = " ".join("%g:%+.2f%%" % (e, 100*R(at[e]["edyn"])) for e in (10.0, 20.0, 31.0, 50.0, 100.0, 150.0, 300.0) if e in at)
        print("  lr %-7g orizzonte %-4s argmin MSE esp %-5g R %+.2f%% | max R %+.2f%% (esp %g) | %s" % (
            c["lr"], c["horizon"], best["exposure"], 100*R(best["edyn"]), 100*R(bestR["edyn"]), bestR["exposure"], txt))
