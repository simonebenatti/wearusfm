import json, sys
d = json.load(open(sys.argv[1]))
print("caches:")
print(json.dumps(d["caches"], indent=1)[:3000])
for name, cell in d["cells"].items():
    print("=====", name)
    print(" keys:", list(cell))
    fit = cell["fit"]
    print(" fit keys:", list(fit))
    if "selection" in fit:
        sel = fit["selection"]
        print(" selection keys:", list(sel))
        for k, v in sel.items():
            s = json.dumps(v)
            print("   ", k, ":", s[:600])
        print(" n_features", fit["n_features"], "n_users", fit["n_users"], "fit_support_sha256", fit["fit_support_sha256"])
    else:
        print(" fit:", fit)
    an = cell["analysis"]
    print(" analysis keys:", list(an))
    print(" support_sha256:", an["support_sha256"], "dynamic_threshold:", an["dynamic_threshold"])
    print(" targets_sha256:", an["targets_sha256"])
    print(" prediction_sha256:", an["prediction_sha256"])
    for g, grp in an["groups"].items():
        c = grp["centered"]
        print("  group", g, "n_selected_windows", grp["n_selected_windows"], "| centered n_windows", c["n_windows"], "n_window_joints", c["n_window_joints"],
              "n_valid_elements", c["n_valid_elements"], "n_users", c["n_users"], "centered_mse_ratio", c["centered_mse_ratio"], "mae_rad", grp["mae_rad"])
    print(" artifacts:", json.dumps(cell["artifacts"])[:800])
