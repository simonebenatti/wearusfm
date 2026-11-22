"""Controlli incrociati aggiuntivi sui JSON: pannello di fitting e terza via (linear_diagnostics)."""
import json, sys
L = sys.argv[1]
load = lambda n: json.load(open(L + "/" + n))
cal = [load("calibration_%d.json" % k) for k in (0, 1)]
tcn, mlp, l0 = (load("selection_temporal_%s.json" % h) for h in ("TCN_post", "MLP", "L0"))
for k in (0, 1):
    d = cal[k]["outer_reference_statistics"]["linear_diagnostics"]
    for name in ("L0", "L13"):
        w = d[name]["worlds"]["temporal"]
        print("fold%d %s worlds.temporal held   :" % (k, name), {q: w["held"]["statistics"][q] for q in ("window_count", "dynamic_target_ss", "dynamic_sse")})
        print("fold%d %s worlds.temporal fitting:" % (k, name), {q: w["fitting"]["statistics"][q] for q in ("window_count", "dynamic_target_ss", "dynamic_sse")})
        m = d[name]["temporal_margin_clean_fit_statistics"]
        print("fold%d %s clean-fit margin (fitting, contro y):" % (k, name), {q: m[q] for q in ("window_count", "dynamic_target_ss", "dynamic_sse")},
              "recovery_fitting", d[name]["temporal_margin_recovery_fitting"], "T_unread", d[name]["T_unread_fraction_fitting"])
    for h, s in (("TCN_post", tcn), ("MLP", mlp), ("L0", l0)):
        f = s["selected_fitting_diagnostics"][str(k)]
        print("fold%d testa %-8s fitting:" % (k, h), {q: f[q] for q in ("window_count", "dynamic_target_ss", "dynamic_sse")}, "E_dyn=%.10f MSE=%.10f" % (f["dynamic_sse"]/f["dynamic_target_ss"], f["mse_sum"]/f["window_count"]))
    ref = cal[k]["outer_reference_statistics"]["temporal"]
    c = tcn["selected_outer_clean_statistics"]["fold%d" % k]
    o = tcn["selected_outer_statistics"]["fold%d" % k]
    print("fold%d SS(y) - SS(y*) - SSE_oracolo = 2<y*_c, n_c> = %.6f (%.4f%% di SS(y))" % (
        k, o["dynamic_target_ss"] - c["dynamic_target_ss"] - ref["oracle"]["dynamic_sse"],
        100*(o["dynamic_target_ss"] - c["dynamic_target_ss"] - ref["oracle"]["dynamic_sse"])/o["dynamic_target_ss"]))
    print("fold%d elementi dinamici stimati da SSE_oracolo/(1.69*0.9457): %.0f -> finestre piene equivalenti %.1f su %d; var. interna media di y*: %.4f (soglia %.4f)" % (
        k, ref["oracle"]["dynamic_sse"]/(1.69*0.9457), ref["oracle"]["dynamic_sse"]/(1.69*0.9457)/2960, ref["oracle"]["window_count"],
        c["dynamic_target_ss"]/(ref["oracle"]["dynamic_sse"]/(1.69*0.9457)), cal[k]["outer_reference_statistics"]["temporal_threshold"]))
    print("fold%d MSE oracolo su tutti gli esclusi = %.6f (sigma^2 = 1.69)" % (k, ref["oracle"]["mse_sum"]/ref["oracle"]["window_count"]))
# pooled sul fitting della TCN (controllo della cifra citata nella diagnosi)
f0, f1 = tcn["selected_fitting_diagnostics"]["0"], tcn["selected_fitting_diagnostics"]["1"]
print("TCN pooled fitting: E_dyn=%.6f MSE=%.6f" % ((f0["dynamic_sse"]+f1["dynamic_sse"])/(f0["dynamic_target_ss"]+f1["dynamic_target_ss"]),
                                                    (f0["mse_sum"]+f1["mse_sum"])/(f0["window_count"]+f1["window_count"])))
print("MLP held pulito pooled E_dyn:", mlp["selected_outer_clean_values"]["pooled"]["E_dyn"], " TCN:", tcn["selected_outer_clean_values"]["pooled"]["E_dyn"])
