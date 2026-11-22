"""Controlli mirati: ordine degli import, contesto numerico, effetto dei blocchi, RNG globale, restore fra contesti."""
import sys

order = {}
import pose_temponet.provenance as P                      # stessa testa di `python -m pose_temponet` (__main__.py:14)
order["dopo provenance"] = {m: (m in sys.modules) for m in ("numpy", "torch", "scipy", "sklearn")}
from pose_temponet import runner                          # __main__.py:245 (comando run)
order["dopo runner"] = {m: (m in sys.modules) for m in ("numpy", "torch", "scipy", "sklearn")}
print("ordine degli import:", order)

import json
from pathlib import Path

import numpy as np
import torch

from pose_temponet import controls as C, data as D, heads, metrics as M, selection as S, train as T, workflow as W
from proto_fixture import build_cache, recipe

OUT = Path(sys.argv[1])
torch.set_num_threads(1)
context = P.numeric_context("cpu")
print("numeric_context(cpu) sul Mac: chiavi", sorted(context), "| threadpools", [(p["prefix"], p["num_threads"]) for p in context["threadpools"]],
      "| digest", P.digest(context)[:16])
try:
    P.verify_numeric({"numeric_pins": {"gpu8": "0" * 64, "cpu8": "0" * 64}}, "controls", context)
except ValueError as error:
    print("verify_numeric sul Mac:", error)

before = torch.get_rng_state().clone()
model = heads.make_head("TCN_post", seed=0)
print("make_head lascia intatto l'RNG globale:", torch.equal(before, torch.get_rng_state()))

calibrated = sorted([torch.load(str(OUT / f"calibration_{k}.pt"), map_location="cpu", weights_only=False) for k in (0, 1)],
                    key=lambda a: a["outer_fold"])
states = torch.load(str(OUT / "selection_states.pt"), map_location="cpu", weights_only=False)
selection = json.loads((OUT / "selection.json").read_text())
cache, users = build_cache()
bundle = W._prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=users,
                                  source_identity=selection["bundle"]["source_identity"], expected_user_count=40)
sel = selection["selected"]
state = states["candidate_states"][sel["lr"]]["best_states"][0]
model.load_state_dict(state["model"], strict=True)
model.eval()
held = bundle.held[0]
x = torch.from_numpy(D.view_inputs(bundle.cache, "post", held))
with torch.no_grad():
    full = model(x)[:, 6:-6].numpy()
    for size in (1, 2, 3, 7, 16, 19):
        parts = np.concatenate([model(x[i:i + size])[:, 6:-6].numpy() for i in range(0, len(x), size)])
        print("blocchi da", size, "vs un blocco da", len(x), ": uguali bit a bit", np.array_equal(parts, full),
              "| finestre diverse", int((parts != full).any(axis=(1, 2)).sum()), "| max scarto", float(np.abs(parts - full).max()))
    one = np.concatenate([model(x[i:i + 1])[:, 6:-6].numpy() for i in range(len(x))])
    three = np.concatenate([model(x[i:i + 3])[:, 6:-6].numpy() for i in range(0, len(x), 3)])
    print("blocchi da 1 vs blocchi da 3 uguali:", np.array_equal(one, three))
reference = W._score_model(model, bundle, 0, held, "post", "cpu")
keep = W.BATCH
for size in (3, 7, 16):
    W.BATCH = size
    other = W._score_model(model, bundle, 0, held, "post", "cpu")
    print("BATCH", size, "scarti relativi:", {k: (other[k] - reference[k]) / reference[k] for k in ("mse_sum", "dynamic_sse", "dynamic_target_ss")},
          "| E_dyn", C.statistic_values(other)["E_dyn"], "vs", C.statistic_values(reference)["E_dyn"])
W.BATCH = keep

# chiave float di candidate_states e ricerca del LR
print("chiavi candidate_states:", [repr(k) for k in states["candidate_states"]], "| 0.003 in chiavi:", 0.003 in states["candidate_states"],
      "| 3e-3 == 0.003:", 3e-3 == 0.003, "| selezione JSON lr:", repr(sel["lr"]))
# stato: campi che Trajectory.restore confronta con il contesto corrente
print("campi di provenienza dello stato:", {k: state[k] for k in ("schema", "lr", "seed", "fold", "device_type", "steps_per_epoch", "updates")})
print("numeric dello stato == numerical_profile() corrente:", state["numeric"] == T.numerical_profile())
# selection_states.pt <-> selection.json
print("states['selection_sha256'] == selection['selection_sha256']:", states["selection_sha256"] == selection["selection_sha256"])
report = dict(selection); digest = report.pop("selection_sha256")
print("selection_sha256 ricalcolabile dal JSON:", T.payload_sha256(report) == digest)
pair = states["candidate_states"][sel["lr"]]
value = dict(pair); digest = value.pop("payload_sha256")
print("payload_sha256 della coppia ricalcolabile:", T.payload_sha256(value) == digest,
      "| punti della coppia == curva JSON:", [{**p, "fold_steps": list(p["fold_steps"])} for p in pair["points"]] ==
      next(c["points"] for c in selection["curves"] if c["lr"] == sel["lr"]))
print("best_states vs current_states: updates", [s["updates"] for s in pair["best_states"]], [s["updates"] for s in pair["current_states"]],
      "| best_exposure", pair["best_exposure"], "| approved_horizon", pair["approved_horizon"])
# configure_numerics con l'ambiente del Mac (OMP_NUM_THREADS=1)
try:
    T.configure_numerics()
    print("configure_numerics: accettato")
except ValueError as error:
    print("configure_numerics con OMP_NUM_THREADS=1:", error)
