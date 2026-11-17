import json, sys
d = json.load(open(sys.argv[1]))
for name, cell in d["cells"].items():
    print("=====", name)
    for g, grp in cell["analysis"]["groups"].items():
        c = grp["centered"]
        m = c["means"]
        print(f"  {g:8s} n_el={c['n_valid_elements']:>9d} total={m['total']:.6f} level(mean_error)={m['mean_error']:.6f} centered={m['centered']:.6f} vary={m['vary']:.6f} "
              f"baseline_total={m['baseline_total']:.6f} mean_target_sq={m['mean_target_squared']:.6f} varp={m['varp']:.6f} | E_dyn={c['centered_mse_ratio']:.6f} corr={c['pooled_centered_correlation']:.4f} var_ratio={c['variance_ratio']:.4f}")
        print(f"           share centered/total = {m['centered']/m['total']:.4f} ; share level/total = {m['mean_error']/m['total']:.4f}")
    pu = cell["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
    print("  per-user dynamic n_windows:", {u: r["n_windows"] for u, r in pu.items()})
    print("  per-user dynamic E_dyn:", {u: round(r["sums"]["centered"]/r["sums"]["vary"], 4) for u, r in pu.items()})
