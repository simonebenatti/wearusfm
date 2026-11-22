"""Ricalcolo indipendente per RC-1 (sola lettura dei JSON archiviati). Uso: python -I rc1_recompute.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
G = json.load(open(B+"positive_controls_gate_20261121.json"))["gate"]
C = {f: json.load(open(B+"calibration_%d.json" % f)) for f in (0, 1)}
def E(s): return s["dynamic_sse"]/s["dynamic_target_ss"]
def M(s): return s["mse_sum"]/s["window_count"]
def pool(a, b): return {k: a[k]+b[k] for k in ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")}

print("== A. pannelli: stessi denominatori fra riferimenti del cancello e diagnostica lineare? ==")
ors = {f: C[f]["outer_reference_statistics"] for f in (0, 1)}
print("chiavi outer_reference_statistics:", sorted(ors[0].keys()))
print("chiavi temporal:", sorted(ors[0]["temporal"].keys()), "| linear:", sorted(ors[0]["linear"].keys()))
ld_gate = G["linear_diagnostics"]
for f in (0, 1):
    t = ors[f]["temporal"]
    d_cal = ors[f]["linear_diagnostics"]["L0"]["worlds"]["temporal"]["held"]["statistics"]
    d_gate = ld_gate["fold%d" % f]["L0"]["worlds"]["temporal"]["held"]["statistics"]
    print(" fold%d: den clean_ls %.9f | den oracle %.9f | den WLS(y) [calibration] %.9f | [gate] %.9f | finestre %d/%d/%d" % (
        f, t["clean_ls"]["dynamic_target_ss"], t["oracle"]["dynamic_target_ss"], d_cal["dynamic_target_ss"],
        d_gate["dynamic_target_ss"], t["clean_ls"]["window_count"], t["oracle"]["window_count"], d_cal["window_count"]))
    assert d_cal == d_gate
    assert t["clean_ls"]["dynamic_target_ss"] == d_cal["dynamic_target_ss"] == t["oracle"]["dynamic_target_ss"]
    # E_dyn di clean_ls/oracle ricalcolati dalle statistiche == temporal_records del gate
    ref = G["reference_gate"]["temporal_records"]["fold%d" % f]
    assert abs(E(t["clean_ls"])-ref["clean_ls"]) < 1e-15 and abs(E(t["oracle"])-ref["oracle"]) < 1e-15
print(" -> stessi denominatori e stesse finestre: confronto sullo stesso pannello (asserzioni superate)")

print("== B. scala del cancello, mondo temporale, esclusi esterni ==")
rows = {}
for name, f in (("fold0", 0), ("fold1", 1), ("pooled", None)):
    if f is None:
        cl = pool(ors[0]["temporal"]["clean_ls"], ors[1]["temporal"]["clean_ls"])
        orc = pool(ors[0]["temporal"]["oracle"], ors[1]["temporal"]["oracle"])
        wls = pool(*[ld_gate["fold%d" % k]["L0"]["worlds"]["temporal"]["held"]["statistics"] for k in (0, 1)])
    else:
        cl, orc = ors[f]["temporal"]["clean_ls"], ors[f]["temporal"]["oracle"]
        wls = ld_gate[name]["L0"]["worlds"]["temporal"]["held"]["statistics"]
    ref = G["reference_gate"]["temporal_records"][name]
    assert abs(E(cl)-ref["clean_ls"]) < 1e-14 and abs(E(orc)-ref["oracle"]) < 1e-14
    m = E(cl)-E(orc)
    R = lambda e: (E(cl)-e)/m
    sel = G["selected_panels"]["temporal"]
    r = {"margin": m, "R_wls0": R(E(wls)), "R_L0": R(sel["L0"][name]["E_dyn"]), "R_MLP": R(sel["MLP"][name]["E_dyn"]),
         "R_TCN": R(sel["TCN_post"][name]["E_dyn"])}
    assert abs(r["R_TCN"]-G["temporal_head_gate"]["recovery"][name]) < 1e-12
    r["need_from_wls0"] = 0.5-r["R_wls0"]; r["tcn_minus_wls0"] = r["R_TCN"]-r["R_wls0"]
    r["ratio_wls0"] = r["need_from_wls0"]/r["tcn_minus_wls0"]
    r["need_from_L0"] = 0.5-r["R_L0"]; r["tcn_minus_L0"] = r["R_TCN"]-r["R_L0"]; r["ratio_L0"] = r["need_from_L0"]/r["tcn_minus_L0"]
    r["shortfall"] = 0.5-r["R_TCN"]; r["ratio_base0"] = 0.5/r["R_TCN"]
    rows[name] = r
    print(name, " ".join("%s=%.4f" % (k, v*100 if not k.startswith("ratio") else v) for k, v in r.items()))

print("== C. mondo lineare a 20 utenti, esclusi esterni: WLS alpha=0 contro ridge alpha scelto (archiviati entrambi) ==")
est = {}
for f in (0, 1):
    lin = ors[f]["linear"]
    wls_lin = ld_gate["fold%d" % f]["L0"]["worlds"]["linear"]["held"]["statistics"]
    assert wls_lin["dynamic_target_ss"] == lin["ridge"]["dynamic_target_ss"] == lin["oracle"]["dynamic_target_ss"]
    sel = C[f]["calibration"]["selected"]
    ex_w, ex_r = E(wls_lin)-E(lin["oracle"]), E(lin["ridge"])-E(lin["oracle"])
    mx_w, mx_r = M(wls_lin)-M(lin["oracle"]), M(lin["ridge"])-M(lin["oracle"])
    print(" fold%d: alpha lineare %.3g sigma_lin %.2f | eccesso E_dyn sull'oracolo: WLS0 %.6f, ridge %.6f, rapporto ridge/WLS0 %.4f | eccesso MSE: WLS0 %.6f, ridge %.6f, rapporto %.4f" % (
        f, sel["linear"]["alpha"], sel["linear"]["sigma"], ex_w, ex_r, ex_r/ex_w, mx_w, mx_r, mx_r/mx_w))
    # trasferimento al mondo temporale: costo da rumore ~ sigma^2 * ||ridge(eps)||^2 / denominatore
    t = ors[f]["temporal"]; wls_t = ld_gate["fold%d" % f]["L0"]["worlds"]["temporal"]["held"]["statistics"]
    s_t, s_l = sel["temporal"]["sigma"], sel["linear"]["sigma"]
    ex_wt = E(wls_t)-E(t["clean_ls"])
    pred_wt = ex_w*(s_t/s_l)**2*lin["oracle"]["dynamic_target_ss"]/t["oracle"]["dynamic_target_ss"]
    m = E(t["clean_ls"])-E(t["oracle"])
    print("        temporale: eccesso WLS0 su clean_ls %.6f (osservato) contro %.6f (previsto dal lineare con sigma^2 e denominatori; senza correzione del selettore)" % (ex_wt, pred_wt))
    # stima della ridge alpha=1 a 20 utenti nel mondo temporale: stesso rapporto ridge/WLS0 del mondo lineare
    est_ridge = -ex_wt*(ex_r/ex_w)/m
    est[f] = est_ridge
    print("        stima R della base ridge(alpha lineare) nel mondo temporale: %.2f%% (WLS0 dimostrato: %.2f%%; L0 neurale %.2f%%)" % (
        100*est_ridge, 100*rows["fold%d" % f]["R_wls0"], 100*rows["fold%d" % f]["R_L0"]))
    need = 0.5-est_ridge; got = rows["fold%d" % f]["R_TCN"]-est_ridge
    print("        con quella base: servono %.2f punti; la TCN oggi sta %.2f punti sopra; rapporto %.3f" % (100*need, 100*got, need/got))

print("== D. calibrazione interna (10 utenti per meta'), mondo temporale scelto ==")
for f in (0, 1):
    t = C[f]["calibration"]["selected"]["temporal"]
    Rin = (t["E_cleanLS"]-t["E_cal"])/(t["E_cleanLS"]-t["E_oracle"])
    cur = {r["alpha"]: r["mse"] for r in t["alpha_curve"]}
    print(" fold%d: a=%.1f sigma=%.2f alpha=%.3g R_interno(ridge)=%.4f%% | MSE interna per alpha: %s" % (
        f, t["a"], t["sigma"], t["alpha"], 100*Rin, ", ".join("%g:%.6f" % (a, cur[a]) for a in sorted(cur))))
    # quanto conta alpha a 10 utenti: differenza di MSE fra alpha minimo valutato e alpha scelto,
    # in unita' del margine MSE esterno clean_ls-oracle (ordine di grandezza; i margini interni in MSE non sono archiviati)
    tt = ors[f]["temporal"]; mm = M(tt["clean_ls"])-M(tt["oracle"])
    amin = min(cur)
    print("        MSE(alpha=%g)-MSE(alpha=%g) = %.6f ; margine MSE esterno clean_ls-oracolo = %.6f ; rapporto = %.1f%% del margine" % (
        amin, t["alpha"], cur[amin]-cur[t["alpha"]], mm, 100*(cur[amin]-cur[t["alpha"]])/mm))
print("== E. sensibilita' del rapporto alla base (pooled): rapporto = (50 - b)/(R_TCN - b) ==")
rt = rows["pooled"]["R_TCN"]
for b in (-0.25, -0.1784, -0.15, -0.1228, -0.10, -0.05, 0.0):
    print("   base %+.2f%% -> servono %.2f punti, TCN sopra la base %.2f, rapporto %.3f" % (100*b, 100*(0.5-b), 100*(rt-b), (0.5-b)/(rt-b)))
