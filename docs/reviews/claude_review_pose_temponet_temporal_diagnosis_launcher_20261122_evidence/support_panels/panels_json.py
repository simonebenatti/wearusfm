"""Confronto dei pannelli nei JSON archiviati (solo lettura, niente codice del progetto)."""
import json, sys
from fractions import Fraction
L = sys.argv[1]
def load(name):
    return json.load(open(L + "/" + name))
cal = [load("calibration_%d.json" % k) for k in (0, 1)]
gate = load("positive_controls_gate_20261121.json")["gate"]
sel = {(w, h): load("selection_%s_%s.json" % (w, h)) for w in ("linear", "temporal") for h in ("L0", "MLP", "TCN_post")}
KEYS = ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")
print("== soglie ==")
for k in (0, 1):
    o = cal[k]["outer_reference_statistics"]
    print("fold%d cal linear_threshold=%r temporal_threshold=%r" % (k, o["linear_threshold"], o["temporal_threshold"]))
for (w, h), s in sel.items():
    print(w, h, "bundle.thresholds=", s["bundle"]["thresholds"])
print()
print("== identita' ==")
for (w, h), s in sel.items():
    b = s["bundle"]
    print(w, h, "support_sha256", b["support_sha256"][:16], "window_ids", b["window_ids_sha256"][:16], "pop", b["population_sha256"][:16],
          "eval_id", s["evaluation_identity"][:16], "feat", s["feature_array_sha256"][:16], "artifacts", b["artifacts_sha256"][:16])
    print("    target_sha256", [x[:16] for x in b["target_sha256"]], "clean", [x[:16] for x in b["clean_target_sha256"]])
    print("    folds", [{k: (v if not isinstance(v, list) else "list[%d]" % len(v)) for k, v in f.items()} for f in b["folds"]])
print()
print("== accumulatori riferimenti chiusi (calibration_*.json outer_reference_statistics) ==")
for w in ("linear", "temporal"):
    for k in (0, 1):
        for name, st in cal[k]["outer_reference_statistics"][w].items():
            print("%-8s fold%d %-9s" % (w, k, name), {q: st[q] for q in KEYS}, "E_dyn=%.16g" % (st["dynamic_sse"]/st["dynamic_target_ss"]), "MSE=%.16g" % (st["mse_sum"]/st["window_count"]))
print()
print("== accumulatori teste selezionate (selection_*.json selected_outer_statistics) ==")
for (w, h), s in sel.items():
    for split in ("fold0", "fold1", "pooled"):
        st = s["selected_outer_statistics"][split]
        print("%-8s %-8s %-6s" % (w, h, split), {q: st[q] for q in KEYS}, "E_dyn=%.16g" % (st["dynamic_sse"]/st["dynamic_target_ss"]), "MSE=%.16g" % (st["mse_sum"]/st["window_count"]))
print()
print("== uguaglianza esatta dei denominatori per pannello (mondo temporale) ==")
for k in (0, 1):
    ref = cal[k]["outer_reference_statistics"]["temporal"]
    rows = {("ref", n): ref[n] for n in ref}
    for h in ("L0", "MLP", "TCN_post"):
        rows[("head", h)] = sel[("temporal", h)]["selected_outer_statistics"]["fold%d" % k]
    den = {key: st["dynamic_target_ss"] for key, st in rows.items()}
    wc = {key: st["window_count"] for key, st in rows.items()}
    print("fold%d dynamic_target_ss:" % k)
    for key, v in den.items():
        print("   ", key, repr(v), "hex", float(v).hex())
    print("   distinti:", len(set(den.values())), " max-min:", max(den.values()) - min(den.values()), " rel:", (max(den.values()) - min(den.values()))/max(den.values()))
    print("   window_count:", wc, "distinti:", len(set(wc.values())))
print()
print("== uguaglianza esatta dei denominatori per pannello (mondo lineare) ==")
for k in (0, 1):
    ref = cal[k]["outer_reference_statistics"]["linear"]
    rows = {("ref", n): ref[n] for n in ref}
    for h in ("L0", "MLP", "TCN_post"):
        rows[("head", h)] = sel[("linear", h)]["selected_outer_statistics"]["fold%d" % k]
    den = {key: st["dynamic_target_ss"] for key, st in rows.items()}
    wc = {key: st["window_count"] for key, st in rows.items()}
    print("fold%d dynamic_target_ss:" % k)
    for key, v in den.items():
        print("   ", key, repr(v))
    print("   distinti:", len(set(den.values())), " max-min:", max(den.values()) - min(den.values()), " rel:", (max(den.values()) - min(den.values()))/max(den.values()))
    print("   window_count:", wc, "distinti:", len(set(wc.values())))
