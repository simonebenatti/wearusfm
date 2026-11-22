"""Sui JSON veri: il rapporto «eccesso di un lettore / eccesso della WLS alpha=0» si trasferisce dal mondo lineare a quello temporale?
Controllo su una quantita' archiviata in ENTRAMBI i mondi: L0 neurale selezionato. python -I rc1_transfer_real.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
G = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
E = lambda s: s["dynamic_sse"]/s["dynamic_target_ss"]
for f in ("fold0", "fold1"):
    ld = G["linear_diagnostics"][f]["L0"]["worlds"]
    lin_or = G["reference_gate"]["linear_records"][f]["E_dyn"]["oracle"]; lin_ridge = G["reference_gate"]["linear_records"][f]["E_dyn"]["ridge"]
    cl = G["reference_gate"]["temporal_records"][f]["clean_ls"]; orc = G["reference_gate"]["temporal_records"][f]["oracle"]
    wl, wt = E(ld["linear"]["held"]["statistics"])-lin_or, E(ld["temporal"]["held"]["statistics"])-cl
    l0l = G["selected_panels"]["linear"]["L0"][f]["E_dyn"]-lin_or; l0t = G["selected_panels"]["temporal"]["L0"][f]["E_dyn"]-cl
    r_lin, r_tem = l0l/wl, l0t/wt
    rr = (lin_ridge-lin_or)/wl
    print("%s: L0 neurale/WLS0 mondo lineare %.4f, mondo temporale %.4f (scarto relativo %+.1f%%) | ridge/WLS0 mondo lineare %.4f" % (f, r_lin, r_tem, 100*(r_tem/r_lin-1), rr))
    m = cl-orc
    lo, hi = -wt*rr/m, -wt*rr*(r_tem/r_lin)/m
    print("     base ridge stimata: %.2f%% (rapporto invariato) ; %.2f%% (rapporto corretto come quello di L0 neurale)" % (100*lo, 100*hi))
