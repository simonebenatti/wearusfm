"""RC-3: il beneficio della ridge si trasferisce fra mondo lineare e temporale in proporzione a sigma^2? Latenti VERI, CV interna a 10 utenti (calibration_{0,1}.json). Uso: python -I rc3_json3.py"""
import json
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
for f in (0, 1):
    c = json.load(open(B+"calibration_%d.json" % f))["calibration"]
    t, l = c["selected"]["temporal"], c["selected"]["linear"]
    ct = {r["alpha"]: r["mse"] for r in t["alpha_curve"]}; cl = {r["alpha"]: r["mse"] for r in l["alpha_curve"]}
    s2 = (t["sigma"]/l["sigma"])**2
    print("fold %d sigma_t %.2f sigma_l %.2f (rapporto^2 %.4f) alpha scelto t/l %g/%g" % (f, t["sigma"], l["sigma"], s2, t["alpha"], l["alpha"]))
    for a in sorted(set(ct) & set(cl)):
        if a == 0.01: continue
        dt, dl = ct[a]-ct[0.01], cl[a]-cl[0.01]
        print("   alpha %-7g MSE interna t %.6f l %.6f | variazione da alpha 0,01: t %+.6f l %+.6f | rapporto t/l %.4f (atteso se solo rumore: %.4f)" % (a, ct[a], cl[a], dt, dl, dt/dl, s2))
    # stesso mondo lineare ma sigma del temporale: esiste nella griglia la coppia (a=0, sigma=sigma_t)?
    for r in c["curve"]:
        if r["a"] == 0.0 and abs(r["sigma"]-t["sigma"]) < 1e-9:
            cc = {q["alpha"]: q["mse"] for q in r["alpha_curve"]}
            print("   griglia (a=0, sigma=%.2f): alpha scelto %g E_cal %.6f E_cleanLS %.6f -> costo ridge %.6f ; mondo temporale costo ridge %.6f (rapporto %.4f)" % (
                r["sigma"], r["alpha"], r["E_cal"], r["E_cleanLS"], r["E_cal"]-r["E_cleanLS"], t["E_cal"]-t["E_cleanLS"], (t["E_cal"]-t["E_cleanLS"])/(r["E_cal"]-r["E_cleanLS"])))
            if 0.01 in cc and 1.0 in cc: print("      MSE interna alpha 0,01 -> 1: %+.6f (temporale %+.6f)" % (cc[1.0]-cc[0.01], ct[1.0]-ct[0.01]))
