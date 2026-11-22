"""Prototipo del driver D0/D1 sulla fixture piccola (CPU, ricetta di test).
Importa i moduli congelati senza modificarli. Uso: python proto_driver.py FIXTURE_DIR
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pose_temponet import controls as C, data as D, heads, metrics as M, selection as S, train as T, workflow as W
from proto_fixture import build_cache, recipe, SOURCE

OUT = Path(sys.argv[1])
torch.set_num_threads(1)
DEVICE = "cpu"


def describe(value, depth=0, limit=3, prefix=""):
    pad = "  " * depth
    if isinstance(value, dict):
        keys = list(value)
        print(f"{pad}{prefix}dict[{len(keys)}] chiavi={[(type(k).__name__, k) for k in keys][:12]}")
        if depth < limit:
            for k in keys[:40]:
                describe(value[k], depth + 1, limit, prefix=f"{k!r}: ")
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{prefix}{type(value).__name__}[{len(value)}]" + (f" di {type(value[0]).__name__}" if value else ""))
        if depth < limit and value and isinstance(value[0], (dict, list, tuple)):
            describe(value[0], depth + 1, limit, prefix="[0]: ")
    elif isinstance(value, torch.Tensor):
        print(f"{pad}{prefix}Tensor {value.dtype} {tuple(value.shape)} {value.device}")
    elif isinstance(value, np.ndarray):
        print(f"{pad}{prefix}ndarray {value.dtype} {value.shape}")
    else:
        text = repr(value)
        print(f"{pad}{prefix}{type(value).__name__} {text[:70]}")


# ---------------------------------------------------------------- 1. caricamento
print("== 1. caricamento ==")
for name in ("calibration_0.pt", "selection_states.pt"):
    for flag in (True, False):
        try:
            torch.load(str(OUT / name), map_location="cpu", weights_only=flag)
            print(name, "weights_only=", flag, "OK")
        except Exception as error:  # noqa: BLE001
            print(name, "weights_only=", flag, "ERRORE", type(error).__name__, str(error).splitlines()[0][:150])
calibrated = sorted([torch.load(str(OUT / f"calibration_{k}.pt"), map_location="cpu", weights_only=False) for k in (0, 1)],
                    key=lambda a: a["outer_fold"])                      # runner._calibrations:98-102
states = torch.load(str(OUT / "selection_states.pt"), map_location="cpu", weights_only=False)
selection = json.loads((OUT / "selection.json").read_text())

print("== 2. schema calibration.pt ==")
describe({k: v for k, v in calibrated[0].items() if k not in ("factor_diagnostics",)}, limit=2)
print("calibration.selected.temporal:", {k: v for k, v in calibrated[0]["calibration"]["selected"]["temporal"].items() if k != "alpha_curve"})
print("== 3. schema selection_states.pt ==")
describe(states, limit=1)
lr0 = sorted(states["candidate_states"])[0]
describe(states["candidate_states"][lr0], depth=1, limit=2)
print("  -- uno stato di traiettoria (best_states[0]) --")
one = states["candidate_states"][lr0]["best_states"][0]
for k, v in one.items():
    if k in ("model", "optimizer"):
        continue
    describe(v, depth=2, limit=2, prefix=f"{k!r}: ")
print("    model keys:", [(k, tuple(v.shape), str(v.dtype)) for k, v in one["model"].items()])
print("    optimizer keys:", list(one["optimizer"]), "; state keys:", list(one["optimizer"]["state"]),
      "; per-param:", {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else v) for k, v in one["optimizer"]["state"][0].items()})
print("    param_groups:", [{k: v for k, v in g.items()} for g in one["optimizer"]["param_groups"]])

# ---------------------------------------------------------------- 2. bundle e bersagli
print("== 4. ricostruzione di y, y*, maschere, soglie ==")
cache, users = build_cache()
bundle = W._prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=users,
                                  source_identity=selection["bundle"]["source_identity"], expected_user_count=40)
print("bundle.manifest == selection['bundle']:", dict(bundle.manifest) == selection["bundle"],
      "| bundle_sha256 uguale:", bundle.manifest["bundle_sha256"] == selection["bundle"]["bundle_sha256"])
for fold in (0, 1):
    choice = calibrated[fold]["calibration"]["selected"]["temporal"]
    ystar, y = C.synthetic_target(calibrated[fold]["components"], choice["a"], choice["sigma"])
    comp = calibrated[fold]["components"]
    manual = comp["L"] + choice["a"] * comp["T"]
    print("fold", fold, "a", choice["a"], "sigma", choice["sigma"],
          "| y bit a bit:", np.array_equal(y, bundle.targets[fold]) and W._array_hash(y) == selection["bundle"]["target_sha256"][fold],
          "| y* bit a bit:", np.array_equal(ystar, bundle.clean_targets[fold]) and W._array_hash(ystar) == selection["bundle"]["clean_target_sha256"][fold],
          "| L+a*T manuale == y*:", np.array_equal(manual, ystar),
          "| y - y* == sigma*eps (bit):", np.array_equal(y - ystar, choice["sigma"] * comp["epsilon"]),
          "| max|y-y*-sigma*eps|:", float(np.abs((y - ystar) - choice["sigma"] * comp["epsilon"]).max()))
    print("   soglia", bundle.thresholds[fold], "== archivio", bundle.thresholds[fold] == selection["bundle"]["thresholds"][fold],
          "| normalizzazione L mean[0], std[0]:", comp["normalization"]["L"]["mean"][0], comp["normalization"]["L"]["std"][0],
          "| eps dtype/shape:", comp["epsilon"].dtype, comp["epsilon"].shape)
depth = "post"
feature_hash = W._array_hash(W._features(bundle.cache, depth))
identity = W._identity({"bundle": bundle.manifest, "depth": depth, "feature_array_sha256": feature_hash})
print("feature_array_sha256 uguale:", feature_hash == selection["feature_array_sha256"],
      "| evaluation_identity uguale:", identity == selection["evaluation_identity"])

# ---------------------------------------------------------------- 3. D0
print("== 5. D0: sola inferenza su tutti gli stati ==")
curves = {c["lr"]: c for c in selection["curves"]}


def load_head(state):
    value = dict(state)
    digest = value.pop("payload_sha256")
    assert T.payload_sha256(value) == digest, "checksum dello stato"
    model = heads.make_head("TCN_post", seed=0).to(device=DEVICE, dtype=torch.float32)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def score(model, fold, indices):
    observed, clean = W._score_model(model, bundle, fold, indices, depth, DEVICE, include_clean=True)
    return observed, clean


exact, total, worst = 0, 0, 0.0
table = {}
for lr, pair in states["candidate_states"].items():
    points = {p["exposure"]: p for p in curves[lr]["points"]}
    for kind, exposure in (("best", pair["best_exposure"]), ("current", pair["points"][-1]["exposure"])):
        per_fold = {"held": [], "fit": [], "held_clean": [], "fit_clean": []}
        for fold in (0, 1):
            state = pair[kind + "_states"][fold]
            model = load_head(state)
            ho, hc = score(model, fold, bundle.held[fold])
            fo, fc = score(model, fold, bundle.fitting[fold])
            per_fold["held"].append(ho); per_fold["held_clean"].append(hc)
            per_fold["fit"].append(fo); per_fold["fit_clean"].append(fc)
        pooled = {k: C.statistic_values(C.pool_statistics(v)) for k, v in per_fold.items()}
        archived = points[exposure]
        total += 1
        same = pooled["held"]["MSE"] == archived["mse"] and pooled["held"]["E_dyn"] == archived["edyn"]
        exact += same
        worst = max(worst, abs(pooled["held"]["MSE"] / archived["mse"] - 1), abs(pooled["held"]["E_dyn"] / archived["edyn"] - 1))
        table[(lr, kind)] = (exposure, pooled, per_fold)
print(f"parita' pooled (MSE, E_dyn) con i punti delle curve: {exact}/{total} uguali bit a bit; scarto relativo massimo {worst:.3e}")
sel = selection["selected"]
exposure, pooled, per_fold = table[(sel["lr"], "best")]
print("punto scelto: per fold == selected_outer_statistics:",
      all(per_fold["held"][f] == selection["selected_outer_statistics"][f"fold{f}"] for f in (0, 1)),
      "| clean == selected_outer_clean_statistics:",
      all(per_fold["held_clean"][f] == selection["selected_outer_clean_statistics"][f"fold{f}"] for f in (0, 1)),
      "| fitting == selected_fitting_diagnostics:",
      all(per_fold["fit"][f] == selection["selected_fitting_diagnostics"][str(f)] for f in (0, 1)))
print("esempio (lr scelto, best): esposizione", exposure, {k: v for k, v in pooled.items()})
one = states["candidate_states"][sel["lr"]]["current_states"][0]
print("epoch_losses (fold0, lr scelto):", [(r["epoch"], r["loss_sum"] / r["window_count"], r["window_count"]) for r in one["epoch_losses"]])
print("contatori: updates", one["updates"], "steps_per_epoch", one["steps_per_epoch"], "position", one["position"],
      "permutation", one["permutation"], "len(epoch_order_sha256)", len(one["epoch_order_sha256"]),
      "dropout_cuda", one["dropout_cuda"], "device_type", one["device_type"])
print("numeric nello stato:", one["numeric"])

# --- via rigorosa: Trajectory.restore (stesso contesto numerico) e via dei blocchi diversi
fitting_sets = [W._training_set(bundle, ix, depth, fold) for fold, ix in enumerate(bundle.fitting)]
best = states["candidate_states"][sel["lr"]]["best_states"]
for fold in (0, 1):
    trajectory = T.Trajectory(lambda seed, f: heads.make_head("TCN_post", seed=seed), fitting_sets[fold], lr=sel["lr"],
                              seed=0, fold=fold, device=DEVICE, _recipe=recipe())
    trajectory.restore(best[fold])
    ok = W._score_model(trajectory.model, bundle, fold, bundle.held[fold], depth, DEVICE) == selection["selected_outer_statistics"][f"fold{fold}"]
    print("Trajectory.restore fold", fold, "-> punteggio identico:", ok, "| dataset_sha256 uguale:", trajectory.data.identity == best[fold]["dataset_sha256"])
torch.set_num_threads(2)
try:
    trajectory = T.Trajectory(lambda seed, f: heads.make_head("TCN_post", seed=seed), fitting_sets[0], lr=sel["lr"],
                              seed=0, fold=0, device=DEVICE, _recipe=recipe())
    trajectory.restore(best[0])
    print("restore con 2 thread: accettato")
except ValueError as error:
    print("restore con 2 thread (profilo numerico diverso): rifiutato ->", error)
torch.set_num_threads(1)

model = load_head(best[0])
reference = W._score_model(model, bundle, 0, bundle.held[0], depth, DEVICE)
original = W.BATCH
x_all = torch.from_numpy(D.view_inputs(bundle.cache, depth, bundle.held[0]))
with torch.no_grad():
    full = model(x_all)[:, 6:-6].numpy()
    parts = np.concatenate([model(x_all[i:i + 3])[:, 6:-6].numpy() for i in range(0, len(x_all), 3)])
print("predizioni FP32 uguali bit a bit con blocchi 20 e 3 (CPU):", np.array_equal(full, parts),
      "| max scarto:", float(np.abs(full - parts).max()))
for size in (3, 7):
    W.BATCH = size          # solo nel processo del prototipo, per misurare l'effetto dei blocchi
    other = W._score_model(model, bundle, 0, bundle.held[0], depth, DEVICE)
    print("BATCH", size, "statistiche identiche a BATCH 32:", other == reference,
          {k: (other[k] - reference[k]) for k in ("mse_sum", "dynamic_sse", "dynamic_target_ss")})
W.BATCH = original

# ---------------------------------------------------------------- 4. D1
print("== 6. D1: cancello di parita' (training su y) e fit sul bersaglio pulito ==")
factory = lambda seed, fold: heads.make_head("TCN_post", seed=seed)     # workflow.py:296
evaluator = lambda model, fold: W._score_model(model, bundle, fold, bundle.held[fold], depth, DEVICE)   # workflow.py:298
LR = sel["lr"]
rec = recipe()


def fresh(sets):
    return [T.Trajectory(factory, data, lr=LR, seed=0, fold=fold, device=DEVICE, _recipe=rec) for fold, data in enumerate(sets)]


gate = fresh(fitting_sets)
print("initial_sha256 uguale all'archivio:", [t.initial_sha256 for t in gate] == selection["initial_sha256_per_fold"])
schedule = S.checkpoint_schedule([t.steps_per_epoch for t in gate], rec.initial_epochs)
archived = curves[LR]["points"]
matches = []
for (exposure, fold_steps), point in zip(schedule[:5], archived[:5]):
    for trajectory, step in zip(gate, fold_steps):
        trajectory.train_to_step(step)
    got = S.pool_scores([t.evaluate(evaluator) for t in gate], exposure, fold_steps)
    matches.append((exposure, list(fold_steps), got.mse == point["mse"] and got.edyn == point["edyn"] and
                    list(got.fold_steps) == point["fold_steps"] and exposure == point["exposure"]))
print("cancello di parita' (primi 5 punti):", matches)

clean_bundle = dataclasses.replace(bundle, targets=bundle.clean_targets)   # solo per costruire il set di training
clean_sets = [W._training_set(clean_bundle, ix, depth, fold) for fold, ix in enumerate(bundle.fitting)]
for fold in (0, 1):
    print("fold", fold, "training target == y*[fit] (bit):",
          np.array_equal(clean_sets[fold].target.numpy(), bundle.clean_targets[fold][bundle.fitting[fold]]),
          "| features uguali al set originale:", torch.equal(clean_sets[fold].features, fitting_sets[fold].features),
          "| support uguale:", torch.equal(clean_sets[fold].support, fitting_sets[fold].support),
          "| identity diversa:", clean_sets[fold].identity != fitting_sets[fold].identity)
saved = {}


def keep(pair):
    exposure = pair.points[-1].exposure
    if exposure in (1.0, float(rec.initial_epochs)):
        saved[exposure] = [t.state() for t in pair.trajectories]


noisy_pair = T.PairedCandidate(fresh(fitting_sets), evaluator, selection["evaluation_identity"])
noisy_pair.advance(rec.initial_epochs)
clean_pair = T.PairedCandidate(fresh(clean_sets), evaluator, selection["evaluation_identity"])
clean_pair.advance(rec.initial_epochs, keep)
n = len(noisy_pair.points)
print("y-run via PairedCandidate riproduce l'archivio fino a", rec.initial_epochs, "epoche:",
      [dataclasses.asdict(p) for p in noisy_pair.points] == [{**p, "fold_steps": tuple(p["fold_steps"])} for p in archived[:n]])
for fold in (0, 1):
    a, b = noisy_pair.trajectories[fold].state(), clean_pair.trajectories[fold].state()
    print("fold", fold, "ordine dei batch uguale:", a["epoch_order_sha256"] == b["epoch_order_sha256"],
          "| RNG dropout uguale:", torch.equal(a["dropout_cpu"], b["dropout_cpu"]),
          "| RNG shuffle uguale:", torch.equal(a["shuffle_rng"], b["shuffle_rng"]),
          "| init uguale:", a["initial_sha256"] == b["initial_sha256"],
          "| pesi diversi:", T.payload_sha256(a["model"]) != T.payload_sha256(b["model"]),
          "| dataset_sha256 diverso:", a["dataset_sha256"] != b["dataset_sha256"])
print("stati salvati dal callback alle esposizioni:", sorted(saved), "updates:", {k: [s["updates"] for s in v] for k, v in saved.items()})
print("best del fit pulito (selezione su y):", clean_pair.best_exposure, [s["updates"] for s in clean_pair.best_states])
curve = clean_pair.curve()
print("curva pulita valutata su y: ", [(p.exposure, round(p.mse, 6), round(p.edyn, 6)) for p in curve.points])
print("curva rumorosa valutata su y:", [(p.exposure, round(p.mse, 6), round(p.edyn, 6)) for p in noisy_pair.points])
# la valutazione del fit pulito usa davvero y: ricalcolo indipendente dell'ultimo punto
last = [W._score_model(t.model, bundle, t.fold, bundle.held[t.fold], depth, DEVICE, include_clean=True) for t in clean_pair.trajectories]
obs = C.statistic_values(C.pool_statistics([x[0] for x in last])); cln = C.statistic_values(C.pool_statistics([x[1] for x in last]))
print("ultimo punto del fit pulito: contro y", obs, "(== curva:", obs["MSE"] == curve.points[-1].mse, ") ; contro y*", cln)
print("loss di training per epoca (fold0): rumoroso", [r["loss_sum"] / r["window_count"] for r in noisy_pair.trajectories[0].epoch_losses],
      "; pulito", [r["loss_sum"] / r["window_count"] for r in clean_pair.trajectories[0].epoch_losses])
state = clean_pair.state()
torch.save(state, str(OUT / "d1_clean_pair.pt"))
again = torch.load(str(OUT / "d1_clean_pair.pt"), map_location="cpu", weights_only=False)
digest = again.pop("payload_sha256")
print("stato della coppia pulita salvato e riletto, checksum ok:", T.payload_sha256(again) == digest)
