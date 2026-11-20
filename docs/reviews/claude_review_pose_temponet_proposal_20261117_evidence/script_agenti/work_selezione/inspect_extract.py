import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    print("=====", p.split("/")[-1])
    for k, v in d.items():
        if k == "raw_inputs":
            print(" raw_inputs keys:", list(v), "n file_states:", len(v["file_states"]), "unchanged:", v.get("unchanged"))
            continue
        if k == "checkpoint_guard":
            g = dict(v); cfg = g.pop("config")
            print(" checkpoint_guard (no config):", json.dumps(g))
            print("  config.jepa:", json.dumps(cfg["jepa"]))
            print("  config.model:", json.dumps(cfg["model"]))
            print("  config.loader:", json.dumps(cfg["loader"]))
            print("  config other:", json.dumps({kk: vv for kk, vv in cfg.items() if kk not in ("jepa", "model", "loader")}))
            continue
        s = json.dumps(v)
        print(f" {k}: {s[:1500]}")
