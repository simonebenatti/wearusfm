import json, sys
tag = sys.argv[1]
for r in range(4):
    d = json.load(open(f"run_{tag}/sk1.rank{r}.json"))
    if "exception" in d:
        print(r, d["exception"]); print(d["trace"]); continue
    pts = []
    for p in d["points"]:
        lv = p["live_vs_probe"][0]
        pts.append((p["update"], p["n_probes"], p["live_has_rebuilt_buckets_attr"], p["live_layout_sizes"], p["live_layout_equals_initial"],
                    all(p["probe_layouts_equal_initial_live"]), sum(p["live_digest"] == x for x in p["probe_digests"]), len(set(p["probe_digests"])),
                    f"{lv['n_diff']}/{lv['n']}", f"{lv['max_abs']:.3g}", f"{lv['rel_l2']:.3g}"))
    print(f"{tag} rank{r} steps={d['steps']} gate={d['gate_status']} uniq_hash_per_point={[g['unique_gradient_hashes'] for g in d['gate_points']]} errors={[g['errors'] for g in d['gate_points']]} gc={d['gc']}")
    print("  (upd, probes, live_rebuilt_attr, live_layout, live==initial, probes==initial, #probe==live bytes, #uniq probe hashes, ndiff, max_abs, rel_l2)")
    for t in pts: print("  ", t)
