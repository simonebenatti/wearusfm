"""Un callback che calcola punteggi in piu' a ogni checkpoint (fitting, y e y*) non altera la traiettoria."""
import dataclasses, json, sys
from pathlib import Path
import torch
from pose_temponet import controls as C, heads, selection as S, train as T, workflow as W
from proto_fixture import build_cache, recipe
OUT = Path(sys.argv[1]); torch.set_num_threads(1)
calibrated = sorted([torch.load(str(OUT / f"calibration_{k}.pt"), map_location="cpu", weights_only=False) for k in (0, 1)], key=lambda a: a["outer_fold"])
selection = json.loads((OUT / "selection.json").read_text())
cache, users = build_cache()
bundle = W._prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=users, source_identity=selection["bundle"]["source_identity"], expected_user_count=40)
factory = lambda seed, fold: heads.make_head("TCN_post", seed=seed)
evaluator = lambda model, fold: W._score_model(model, bundle, fold, bundle.held[fold], "post", "cpu")
clean_bundle = dataclasses.replace(bundle, targets=bundle.clean_targets)
sets = [W._training_set(clean_bundle, ix, "post", f) for f, ix in enumerate(bundle.fitting)]
rec = recipe()
def run(callback):
    pair = T.PairedCandidate([T.Trajectory(factory, d, lr=0.003, seed=0, fold=f, device="cpu", _recipe=rec) for f, d in enumerate(sets)], evaluator, selection["evaluation_identity"])
    pair.advance(rec.initial_epochs, callback)
    return pair
extra = []
def diagnostics(pair):
    rows = {}
    for t in pair.trajectories:
        for name, idx in (("held", bundle.held[t.fold]), ("fit", bundle.fitting[t.fold])):
            rows[(t.fold, name)] = W._score_model(t.model, bundle, t.fold, idx, "post", "cpu", include_clean=True)
    pooled = {name + "_" + kind: C.statistic_values(C.pool_statistics([rows[(f, name)][k] for f in (0, 1)]))
              for name in ("held", "fit") for k, kind in ((0, "y"), (1, "ystar"))}
    extra.append((pair.points[-1].exposure, pooled))
plain, rich = run(None), run(diagnostics)
print("stato finale identico con e senza diagnostica nel callback:", T.payload_sha256(plain.state()) == T.payload_sha256(rich.state()))
print("curve identiche:", plain.curve() == rich.curve(), "| punti diagnostici:", len(extra))
print("held_y del callback == punto della curva:", all(e[1]["held_y"]["MSE"] == p.mse and e[1]["held_y"]["E_dyn"] == p.edyn for e, p in zip(extra, rich.points)))
print("ultimo punto:", extra[-1])
