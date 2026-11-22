"""Fixture piccola per provare il driver: cache sintetica 40 utenti x 1 finestra (160x384, FP16),
due calibrazioni vere (controls.build_positive_control_fold), bundle del mondo temporale e una
selezione vera TCN_post con la ricetta di test. Scrive calibration_{0,1}.pt, selection.json e
selection_states.pt in OUT, con le stesse chiamate di serializzazione del codice congelato
(provenance.atomic_torch = torch.save(value, stream); atomic_json = json.dumps(sort_keys, indent=2)).

La funzione _cache e' copiata da tests/cpu/test_pose_temponet_e2e_chain.py:28-59 (commit e8c98244), senza il ramo "val".
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pose_temponet import controls as C, data as D, metrics as M, train as T, workflow as W

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def _cache(users, windows, split, rng, weights):
    n = len(users) * windows
    lengths = np.full(n, 160, dtype=np.int64)
    lengths[3::7] = 150
    available = np.arange(160)[None, :] < lengths[:, None]
    x = np.tile(rng.normal(size=(n, 160, 16)), (1, 1, 24)).astype(np.float16)
    pre = (.7 * x.astype(float) + .5 * rng.normal(size=x.shape)).astype(np.float16)
    x[~available] = pre[~available] = 0
    z = x.astype(float)
    z -= z.mean(-1, keepdims=True)
    z /= np.sqrt(np.mean(z*z, axis=-1, keepdims=True) + 1e-5)
    signal = z @ weights
    target = signal + 1.9 * rng.normal(size=signal.shape)
    scales = np.ones(n)
    target = (target * scales[:, None, None]).astype(np.float32)
    mask = np.repeat(available[..., None], 20, axis=2)
    target[~mask] = 0
    return {
        "pre": pre, "lat": x, "key_time_valid": available,
        "pred": (.8 * signal * scales[:, None, None]).astype(np.float32),
        "target": target, "mask": mask, "std": np.full((n, 20), .3, np.float32),
        "pose_var": M.dynamic_variance(target, mask).astype(np.float32),
        "meta": [{"user": users[i // windows], "recording": i // windows,
                  "window": i % windows, "n_patches": int(lengths[i])} for i in range(n)],
        "config": {"split": split, "which": "teacher", "checkpoint_step": 49046},
        "has_head": True, "dim": 384, "n_recordings": len(users),
    }


def build_cache():
    rng = np.random.default_rng(20261119)
    weights = rng.normal(size=(384, 20)) / np.sqrt(384)
    users = [f"user{i:02d}" for i in range(40)]
    cache = _cache(users, 1, "train", rng, weights)
    # una finestra senza alcun supporto (come le 53 del train vero): maschera tutta falsa
    cache["mask"][5] = False
    cache["target"][5] = 0
    return cache, users


SOURCE = {"cache_sha256": "a" * 64, "protocol_sha256": "b" * 64}


def recipe():
    return T._TestRecipe(initial_epochs=3, maximum_epochs=6, last_epochs=1, batch_size=4)


def main():
    torch.set_num_threads(1)
    cache, users = build_cache()
    folds = D.alternating_user_folds(cache)
    t0 = time.perf_counter()
    artifacts = []
    for fold in (0, 1):
        fit_users = set(folds[fold])
        fit = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] in fit_users])
        held = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] not in fit_users])
        work = dict(cache)
        work.pop("pre", None)                                   # runner.py:287
        result = C.build_positive_control_fold(work, fit, held, users, outer_fold=fold, block_windows=32)  # runner.py:292
        print("fold", fold, "calibration", result["calibration"]["status"],
              {w: {k: v[k] for k in ("a", "sigma", "alpha")} for w, v in result["calibration"].get("selected", {}).items()})
        torch.save(result, OUT / f"calibration_{fold}.pt")       # provenance.atomic_torch
        artifacts.append(result)
    print("calibrazioni:", round(time.perf_counter() - t0, 1), "s")
    gate = C.combine_reference_folds(artifacts)
    print("combined gate:", gate["status"])
    if gate["status"] != "diagnostic":
        print("FIXTURE NON QUALIFICATA: il bundle di controllo non si puo' costruire con questa cache")
        return 1
    bundle = W._prepare_control_folds(cache, artifacts, "temporal", allowed_train_users=users,
                                      source_identity=SOURCE, expected_user_count=40)
    t0 = time.perf_counter()
    result = W._run_selection(bundle, "TCN_post", device="cpu", recipe=recipe())
    print("selezione:", round(time.perf_counter() - t0, 1), "s; scelto", result["selection"]["selected"])
    (OUT / "selection.json").write_text(json.dumps(result["selection"], sort_keys=True, indent=2, allow_nan=False) + "\n")
    torch.save(result["states"], OUT / "selection_states.pt")
    print("scritti:", sorted(p.name + ":" + str(p.stat().st_size) for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
