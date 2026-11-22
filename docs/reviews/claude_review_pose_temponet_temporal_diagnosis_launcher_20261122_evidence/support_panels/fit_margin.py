"""Derivato descrittivo dai JSON: oracolo sul pannello di FITTING ricavato da temporal_margin_recovery_fitting (L13)."""
import json, sys
L = sys.argv[1]
load = lambda n: json.load(open(L + "/" + n))
tcn, mlp, l0 = (load("selection_temporal_%s.json" % h) for h in ("TCN_post", "MLP", "L0"))
for k in (0, 1):
    d = load("calibration_%d.json" % k)["outer_reference_statistics"]["linear_diagnostics"]
    base = d["L0"]["temporal_margin_clean_fit_statistics"]
    l13 = d["L13"]["temporal_margin_clean_fit_statistics"]
    R = d["L13"]["temporal_margin_recovery_fitting"]
    ss = base["dynamic_target_ss"]
    margin = (base["dynamic_sse"] - l13["dynamic_sse"]) / R
    oracle = base["dynamic_sse"] - margin
    print("fold%d FITTING: E_cleanLS=%.6f E_oracolo=%.6f (ricavato) margine SSE=%.3f" % (k, base["dynamic_sse"]/ss, oracle/ss, margin))
    for h, s in (("TCN_post", tcn), ("MLP", mlp), ("L0", l0)):
        f = s["selected_fitting_diagnostics"][str(k)]
        assert f["dynamic_target_ss"] == ss, (f["dynamic_target_ss"], ss)
        print("    %-8s E_dyn fitting=%.6f  quota del margine sul fitting (in campione, NON e' il cancello)=%.4f" % (
            h, f["dynamic_sse"]/ss, (base["dynamic_sse"] - f["dynamic_sse"])/margin))
