"""Stima pooled della base ridge (alpha=1, 20 utenti) nel mondo temporale e controlli di contorno. python -I rc1_pooled_est.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
G = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
C = {f: json.load(open(B+"calibration_%d.json" % f)) for f in (0, 1)}
E = lambda s: s["dynamic_sse"]/s["dynamic_target_ss"]; M = lambda s: s["mse_sum"]/s["window_count"]
num = den = cl = orc = 0.0
for f in (0, 1):
    o = C[f]["outer_reference_statistics"]; ld = G["linear_diagnostics"]["fold%d" % f]["L0"]["worlds"]
    ratio = (E(o["linear"]["ridge"])-E(o["linear"]["oracle"]))/(E(ld["linear"]["held"]["statistics"])-E(o["linear"]["oracle"]))
    t = o["temporal"]; w = ld["temporal"]["held"]["statistics"]
    sse_excess_wls = w["dynamic_sse"]-t["clean_ls"]["dynamic_sse"]
    num += sse_excess_wls*ratio; den += w["dynamic_target_ss"]; cl += t["clean_ls"]["dynamic_sse"]; orc += t["oracle"]["dynamic_sse"]
    print("fold%d: MSE esterna clean_ls %.6f oracolo %.6f margine %.6f | E_dyn WLS0 mondo lineare %.6f | costante: %s" % (
        f, M(t["clean_ls"]), M(t["oracle"]), M(t["clean_ls"])-M(t["oracle"]), E(ld["linear"]["held"]["statistics"]),
        "E_dyn %.6f" % E(t["constant"])))
margin = (cl-orc)/den
est = -(num/den)/margin
rt = G["temporal_head_gate"]["recovery"]["pooled"]
print("pooled: eccesso stimato della ridge su clean_ls %.6f ; margine %.6f ; R stimato della base %.2f%%" % (num/den, margin, 100*est))
print("pooled con quella base: servono %.2f punti; TCN sopra la base %.2f; rapporto %.3f" % (100*(0.5-est), 100*(rt-est), (0.5-est)/(rt-est)))
# scambio di base: guadagno «gratuito» rispetto a tre ipotesi sulla parte lineare attuale della TCN
for name, b in (("L0 neurale scelto", -0.122819), ("L0 a LR 0,003 esp. 31", -0.1529), ("WLS alpha=0", -0.178356)):
    print("   se la parte lineare attuale vale %s (%.2f%%): scambio di base = %+.2f punti su %.2f mancanti" % (name, 100*b, 100*(est-b), 100*(0.5-rt)))
