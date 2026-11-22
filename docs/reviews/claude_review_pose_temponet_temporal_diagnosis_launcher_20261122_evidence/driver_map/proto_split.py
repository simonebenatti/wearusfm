"""Scomposizione dall'esterno: parte affine e parte profonda della testa; punteggi contro L, a*T, y*, y."""
import json, sys
from pathlib import Path
import numpy as np, torch
from pose_temponet import controls as C, data as D, heads, metrics as M, train as T, workflow as W
from proto_fixture import build_cache
OUT = Path(sys.argv[1]); torch.set_num_threads(1)
calibrated = sorted([torch.load(str(OUT / f"calibration_{k}.pt"), map_location="cpu", weights_only=False) for k in (0, 1)], key=lambda a: a["outer_fold"])
states = torch.load(str(OUT / "selection_states.pt"), map_location="cpu", weights_only=False)
selection = json.loads((OUT / "selection.json").read_text())
cache, users = build_cache()
bundle = W._prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=users, source_identity=selection["bundle"]["source_identity"], expected_user_count=40)
sel = selection["selected"]; fold = 0
state = states["candidate_states"][sel["lr"]]["best_states"][fold]
model = heads.make_head("TCN_post", seed=0); model.load_state_dict(state["model"], strict=True); model.eval()
rows = bundle.held[fold]
choice = calibrated[fold]["calibration"]["selected"]["temporal"]; comp = calibrated[fold]["components"]
records = {k: [] for k in ("full_y", "full_ystar", "lin_L", "deep_aT", "lin_ystar")}
same = True
with torch.no_grad():
    for start in range(0, len(rows), W.BATCH):
        take = rows[start:start + W.BATCH]
        x = torch.from_numpy(D.view_inputs(bundle.cache, "post", take))
        h = model.norm(x.float())
        lin = model.affine(h)
        deep = model.deep(h.transpose(1, 2)).transpose(1, 2)
        full = model(x)
        same &= bool(torch.equal(lin + deep, full))
        panel = W._metric_panel(bundle, fold, take)
        thr = bundle.thresholds[fold]
        crop = lambda t: t[:, D.CROP:-D.CROP].numpy()
        records["full_y"].append(M.selection_statistics(crop(full), panel, thr))
        records["full_ystar"].append(M.selection_statistics(crop(full), W._metric_panel(bundle, fold, take, clean=True), thr))
        records["lin_L"].append(M.selection_statistics(crop(lin), {**panel, "target": comp["L"][take]}, thr))
        records["deep_aT"].append(M.selection_statistics(crop(deep), {**panel, "target": choice["a"] * comp["T"][take]}, thr))
        records["lin_ystar"].append(M.selection_statistics(crop(lin), W._metric_panel(bundle, fold, take, clean=True), thr))
print("affine(h) + deep(h) == model(x) bit a bit:", same)
for k, v in records.items():
    s = C.pool_statistics(v); print(k, s, C.statistic_values(s))
print("full_y == archivio fold0:", C.pool_statistics(records["full_y"]) == selection["selected_outer_statistics"]["fold0"])
# oracolo e costante sul fitting e sugli esclusi senza alcun fit
for name, idx in (("held", bundle.held[fold]), ("fitting", bundle.fitting[fold])):
    oracle = M.selection_statistics(bundle.clean_targets[fold][idx][:, 6:-6], W._metric_panel(bundle, fold, idx), bundle.thresholds[fold])
    print("oracolo", name, C.statistic_values(oracle), "| archivio held:", C.statistic_values(calibrated[fold]["outer_reference_statistics"]["temporal"]["oracle"]) if name == "held" else "")
