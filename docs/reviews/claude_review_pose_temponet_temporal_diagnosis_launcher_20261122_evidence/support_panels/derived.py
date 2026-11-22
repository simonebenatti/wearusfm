"""Numeri derivati (descrittivi) dai JSON: margine sulla MSE e basi lineari stimate su y rumoroso."""
import json, sys
L = sys.argv[1]
load = lambda n: json.load(open(L + "/" + n))
cal = [load("calibration_%d.json" % k) for k in (0, 1)]
tcn, mlp, l0 = (load("selection_temporal_%s.json" % h) for h in ("TCN_post", "MLP", "L0"))
K = ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")
pool = lambda rows: {k: sum(r[k] for r in rows) for k in K}
E = lambda r: r["dynamic_sse"]/r["dynamic_target_ss"]
MSE = lambda r: r["mse_sum"]/r["window_count"]
ref = {n: pool([c["outer_reference_statistics"]["temporal"][n] for c in cal]) for n in ("clean_ls", "oracle", "constant")}
wls_noisy = pool([c["outer_reference_statistics"]["linear_diagnostics"]["L0"]["worlds"]["temporal"]["held"]["statistics"] for c in cal])
l13_noisy = pool([c["outer_reference_statistics"]["linear_diagnostics"]["L13"]["worlds"]["temporal"]["held"]["statistics"] for c in cal])
heads = {"TCN_post": tcn["selected_outer_statistics"]["pooled"], "MLP": mlp["selected_outer_statistics"]["pooled"], "L0 (testa)": l0["selected_outer_statistics"]["pooled"],
         "L0 WLS su y rumoroso (chiuso, alpha=0)": wls_noisy, "L13 WLS su y rumoroso (chiuso, alpha=0)": l13_noisy}
print("pooled held: clean_ls E_dyn=%.6f MSE=%.6f | oracolo E_dyn=%.6f MSE=%.6f | costante MSE=%.6f" % (E(ref["clean_ls"]), MSE(ref["clean_ls"]), E(ref["oracle"]), MSE(ref["oracle"]), MSE(ref["constant"])))
for name, r in heads.items():
    print("%-40s E_dyn=%.6f R_Edyn=%+.4f | MSE=%.6f R_MSE=%+.4f | finestre %d SS=%.3f" % (name, E(r), (E(ref["clean_ls"])-E(r))/(E(ref["clean_ls"])-E(ref["oracle"])),
          MSE(r), (MSE(ref["clean_ls"])-MSE(r))/(MSE(ref["clean_ls"])-MSE(ref["oracle"])), r["window_count"], r["dynamic_target_ss"]))
base = heads["L0 (testa)"]
print("TCN rispetto alla testa L0 allenata sullo stesso y rumoroso: quota del margine L0->oracolo su E_dyn = %.4f ; su MSE = %.4f" % (
    (E(base)-E(heads["TCN_post"]))/(E(base)-E(ref["oracle"])), (MSE(base)-MSE(heads["TCN_post"]))/(MSE(base)-MSE(ref["oracle"]))))
target = (E(ref["clean_ls"]) + E(ref["oracle"]))/2
print("soglia 50%% pooled: E_dyn <= %.6f ; da L0 (testa) servirebbe recuperare %.4f della distanza L0->oracolo" % (target, (E(base)-target)/(E(base)-E(ref["oracle"]))))
for k in (0, 1):
    r = cal[k]["outer_reference_statistics"]["temporal"]
    w = cal[k]["outer_reference_statistics"]["linear_diagnostics"]["L0"]["worlds"]["temporal"]["held"]["statistics"]
    print("fold%d: WLS L0 su y rumoroso E_dyn=%.6f (R=%+.4f); clean_ls %.6f; testa L0 %.6f" % (k, E(w), (E(r["clean_ls"])-E(w))/(E(r["clean_ls"])-E(r["oracle"])), E(r["clean_ls"]), E(l0["selected_outer_statistics"]["fold%d" % k])))
