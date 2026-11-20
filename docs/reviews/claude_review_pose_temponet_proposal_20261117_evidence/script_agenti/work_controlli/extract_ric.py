"""Controllo ESEGUITO (seconda stesura): estrae dalla ricevuta storica a 49.046 (estratta da git 2930d7d) i numeri usati nella review:
griglia della ridge, E_dyn dei tre lettori sulle dinamiche, supporti, identita' della testa storica."""
import json, sys
r = json.load(open(sys.argv[1]))
c = r["cells"]
for name in ("post_ridge", "pre_ridge"):
    s = c[name]["fit"]["selection"]
    print(name, "alpha scelto", s["alpha"], "cv_mse", s["cv_mse"], "estensioni", s["extensions"], "n_users", c[name]["fit"]["n_users"])
    best = s["cv_mse"]
    for g in s["grid"]:
        print("   alpha %-10g mse %.6f  scarto rispetto all'ottimo %+.2f%%" % (g["alpha"], g["mse"], 100 * (g["mse"] / best - 1)))
    print("   fold_users:", [len(f) for f in s["fold_users"]])
for name in ("post_ridge", "pre_ridge", "trained_head"):
    a = c[name]["analysis"]
    print(name, "which", c[name].get("which"), "| support", a["support_sha256"][:12], "| targets", a["targets_sha256"]["arrays"][:12])
    for grp in ("dynamic", "all", "static"):
        g = a["groups"][grp]
        keys = list(g.keys())
        cen = g.get("centered_support_n_ge_2", g.get("centered"))
        out = {k: v for k, v in cen.items() if not isinstance(v, (dict, list))}
        print("   ", grp, "chiavi", keys, "|", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()})
th = c["trained_head"]
print("trained_head fit:", {k: v for k, v in th.get("fit", {}).items() if not isinstance(v, (dict, list))})
