"""Train-only scientific adapters; no files, scheduler calls or development selection.

The caller verifies cache pins before calling these APIs and owns atomic JSON/PT
serialization. Only ``predict_locked`` accepts the 15-user development cache,
after all five selections and fifteen refits have been locked. Host feature
storage retains the cache dtype; conversion to FP32 occurs only inside training
or evaluation batches, never for an entire split.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from . import controls, data as D, heads, metrics, selection as S, train as T

HEADS = ("L0", "L13", "MLP", "TCN_post", "TCN_pre")
CONTROL_HEADS = ("L0", "MLP", "TCN_post")
WORLDS = ("linear", "temporal")
BATCH = 32


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _array_hash(value):
    value = np.asarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
    if value.ndim == 0:
        h.update(value.tobytes())
    else:
        for start in range(0, len(value), BATCH):
            h.update(np.ascontiguousarray(value[start:start+BATCH]).tobytes())
    return h.hexdigest()


def _compact_identity(value):
    """Hash large arrays without JSON lists or retaining temporary copies."""
    if isinstance(value, np.ndarray):
        return {"numpy_sha256": _array_hash(value)}
    if isinstance(value, torch.Tensor):
        return {"torch_sha256": T.payload_sha256(value)}
    if isinstance(value, Mapping):
        return {str(k): _compact_identity(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact_identity(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _identity(value):
    return T.payload_sha256(_json(_compact_identity(value)))


def _lock(value, key):
    out = _json(value)
    _require(key not in out, "cannot replace an existing lock")
    out[key] = T.payload_sha256(out)
    return out


def _verify(value, key):
    out = dict(value)
    digest = out.pop(key, None)
    _require(isinstance(digest, str) and T.payload_sha256(out) == digest, key + " mismatch")
    return out


def _metadata_ids(cache):
    return tuple(controls.canonical(meta) for meta in cache["meta"])


def _population_identity(cache):
    return _identity({"meta": cache["meta"], "config": cache["config"],
                      **{k: cache[k] for k in ("target", "mask", "std")}})


def _source(source_identity):
    _require(isinstance(source_identity, Mapping) and bool(source_identity), "pinned source identity is required")
    return _json(source_identity)


def _feature_depth(head):
    _require(head in HEADS, "unregistered head")
    return "pre" if head == "TCN_pre" else "post"


def _features(cache, depth):
    if "features" in cache:
        _require(cache.get("feature_depth") == depth, "requested feature depth was not retained")
        return cache["features"]
    return cache["pre" if depth == "pre" else "lat"]


@dataclass(frozen=True)
class FoldBundle:
    cache: Mapping
    world: str
    fitting: tuple[np.ndarray, np.ndarray]
    held: tuple[np.ndarray, np.ndarray]
    targets: tuple[np.ndarray, np.ndarray]
    clean_targets: tuple[np.ndarray, np.ndarray]
    pose_variances: tuple[np.ndarray, np.ndarray]
    thresholds: tuple[float, float]
    support: np.ndarray
    manifest: Mapping
    reference_gate: Mapping | None = None


def _prepare_real_folds(cache, *, allowed_train_users, source_identity, expected_folds=None,
                        expected_user_count=40):
    _require(cache.get("config", {}).get("split") == "train", "selection/refit require the train cache")
    _require(cache["config"].get("which") == "teacher" and cache["config"].get("checkpoint_step") == 49046,
             "the frozen teacher at 49046 is required")
    _require(np.asarray(cache["target"]).shape[1:] == (160, 20), "training target geometry differs")
    users = [m["user"] for m in cache["meta"]]
    _require(all(isinstance(u, str) and u for u in users), "train users must be normalized nonempty strings")
    actual, allowed = sorted(set(users)), sorted(set(allowed_train_users))
    _require(actual == allowed and len(actual) == expected_user_count, "train population differs from the allowed users")
    ids = _metadata_ids(cache)
    _require(len(set(ids)) == len(ids), "duplicate window metadata")
    halves = D.alternating_user_folds(cache, expected_folds)
    fitting = tuple(np.flatnonzero(np.isin(users, half)) for half in halves)
    held = (fitting[1], fitting[0])
    support = D.central_support(cache["mask"], D.lengths_for(cache), cache.get("key_time_valid"))
    y = np.asarray(cache["target"])
    pv = metrics.dynamic_variance(y, cache["mask"])
    thresholds = tuple(metrics.training_threshold(y[ix], cache["mask"][ix]) for ix in fitting)
    manifest = _lock({"schema": "pose-head-folds-v1", "world": "real", "source_identity": _source(source_identity),
                      "population_sha256": _population_identity(cache), "users": actual, "user_folds": halves,
                      "window_ids_sha256": _identity(ids), "thresholds": thresholds,
                      "support_sha256": _array_hash(support),
                      "folds": [{"fit_indices": f.tolist(), "held_indices": h.tolist(),
                                 "fitting_windows": len(f), "held_windows": len(h),
                                 "excluded_fitting_windows": int((~support[f].any(axis=(1, 2))).sum())}
                                for f, h in zip(fitting, held)]}, "bundle_sha256")
    return FoldBundle(cache, "real", fitting, held, (y, y), (y, y), (pv, pv), thresholds, support, manifest)


def prepare_real_folds(cache, *, allowed_train_users, source_identity, expected_folds=None):
    """Partition the verified 40-user train cache; no development cache argument."""
    return _prepare_real_folds(cache, allowed_train_users=allowed_train_users, source_identity=source_identity,
                               expected_folds=expected_folds)


def _prepare_control_folds(cache, artifacts, world, *, allowed_train_users, source_identity,
                           expected_folds=None, expected_user_count=40):
    _require(world in WORLDS, "unknown positive-control world")
    real = _prepare_real_folds(cache, allowed_train_users=allowed_train_users, source_identity=source_identity,
                               expected_folds=expected_folds, expected_user_count=expected_user_count)
    gate = controls.combine_reference_folds(artifacts)
    _require(gate["status"] == "diagnostic", "closed-form positive controls did not qualify")
    ordered = sorted(artifacts, key=lambda a: a["outer_fold"])
    targets, clean, pv, thresholds = [], [], [], []
    for fold, artifact in enumerate(ordered):
        expected = controls.validate_outer_partition(cache["meta"], real.fitting[fold], real.held[fold], allowed_train_users)
        _require(artifact["partition"] == expected, "control artifact belongs to another fold")
        choice = artifact["calibration"]["selected"][world]
        ystar, y = controls.synthetic_target(artifact["components"], choice["a"], choice["sigma"])
        _require(y.shape == cache["target"].shape, "control target shape mismatch")
        threshold = metrics.training_threshold(ystar[real.fitting[fold]], cache["mask"][real.fitting[fold]])
        _require(threshold == artifact["outer_reference_statistics"][world + "_threshold"],
                 "control threshold differs from its closed-form reference")
        targets.append(y); clean.append(ystar)
        pv.append(metrics.dynamic_variance(ystar, cache["mask"]))
        thresholds.append(threshold)
    manifest = dict(real.manifest)
    manifest.pop("bundle_sha256")
    manifest.update(world=world, thresholds=thresholds, artifacts_sha256=_identity(ordered),
                    target_sha256=[_array_hash(y) for y in targets],
                    clean_target_sha256=[_array_hash(y) for y in clean], reference_gate=gate)
    return FoldBundle(cache, world, real.fitting, real.held, tuple(targets), tuple(clean), tuple(pv),
                      tuple(thresholds), real.support, _lock(manifest, "bundle_sha256"), gate)


def prepare_control_folds(cache, artifacts, world, *, allowed_train_users, source_identity, expected_folds=None):
    """Replay the two fixed calibration artifacts; never recalibrate on outer held."""
    return _prepare_control_folds(cache, artifacts, world, allowed_train_users=allowed_train_users,
                                  source_identity=source_identity, expected_folds=expected_folds)


def _metric_panel(bundle, fold, indices, *, clean=False):
    c = bundle.cache
    panel = {"target": (bundle.clean_targets if clean else bundle.targets)[fold][indices],
             "pose_var": bundle.pose_variances[fold][indices],
             "mask": c["mask"][indices], "std": c["std"][indices],
             "meta": [c["meta"][i] for i in indices]}
    if "key_time_valid" in c:
        panel["key_time_valid"] = c["key_time_valid"][indices]
    return panel


def _assert_bundle(bundle):
    _verify(bundle.manifest, "bundle_sha256")
    _require(_population_identity(bundle.cache) == bundle.manifest["population_sha256"],
             "training cache changed after preparation")
    if bundle.world != "real":
        _require([_array_hash(y) for y in bundle.targets] == bundle.manifest["target_sha256"] and
                 [_array_hash(y) for y in bundle.clean_targets] == bundle.manifest["clean_target_sha256"],
                 "synthetic targets changed after calibration")


def _training_set(bundle, indices, depth, fold):
    # Advanced indexing copies only this fold, preserving the cache's FP16 dtype.
    # TensorTrainingSet.batch casts to FP32 after selecting the minibatch.
    x = np.asarray(_features(bundle.cache, depth))[indices]
    lengths = D.lengths_for(bundle.cache)[indices]
    available = np.arange(x.shape[1])[None, :] < lengths[:, None]
    _require(x.ndim == 3 and x.shape[-1] == 384 and x.dtype.kind == "f" and
             np.isfinite(x[available]).all(), "invalid real latent context")
    if not available.all():
        x[~available] = 0
    ids = _metadata_ids(bundle.cache)
    return T.TensorTrainingSet(torch.from_numpy(x),
                               torch.from_numpy(np.asarray(bundle.targets[fold][indices])),
                               torch.from_numpy(bundle.support[indices]),
                               [ids[i] for i in indices])


def _score_model(model, bundle, fold, indices, depth, device, *, include_clean=False):
    records, clean_records = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for start in range(0, len(indices), BATCH):
                rows = indices[start:start+BATCH]
                x = torch.from_numpy(D.view_inputs(bundle.cache, depth, rows)).to(device)
                prediction = model(x)[:, D.CROP:-D.CROP].detach().cpu().numpy()
                _require(np.isfinite(prediction).all(), "nonfinite fold prediction")
                records.append(metrics.selection_statistics(prediction, _metric_panel(bundle, fold, rows),
                                                             bundle.thresholds[fold]))
                if include_clean:
                    clean_records.append(metrics.selection_statistics(prediction, _metric_panel(bundle, fold, rows, clean=True),
                                                                       bundle.thresholds[fold]))
    finally:
        model.train(was_training)
    result = controls.pool_statistics(records)
    return (result, controls.pool_statistics(clean_records)) if include_clean else result


def _require_gate(gate, bundle=None, *, production=False):
    _require(isinstance(gate, Mapping), "positive controls must pass before real fits")
    _verify(gate, "gate_sha256")
    _require(gate.get("status") == "passed" and gate.get("complete") is True, "positive controls must pass before real fits")
    if production:
        _require(gate.get("recipe") == _json(T.Recipe().__dict__), "test gate cannot enter production")
    if bundle is not None:
        for key in ("population_sha256", "source_identity", "user_folds"):
            _require(gate[key] == bundle.manifest[key], "positive controls have different " + key)


def _check_selection(report, *, production=False):
    _verify(report, "selection_sha256")
    _require(report.get("selection_locked") is True and report.get("head") in HEADS,
             "head selection is not locked")
    _require(report.get("world") in ("real", *WORLDS), "invalid selection world")
    _require(S.choose([S.curve_from_dict(c) for c in report["curves"]]).__dict__ == report["selected"],
             "selection differs from the frozen MSE selector")
    if production:
        _require(report["recipe"] == _json(T.Recipe().__dict__), "test recipe cannot enter production")


def _run_selection(bundle, head, *, control_gate=None, device="cpu", recipe=T.Recipe(),
                   factory=None, on_checkpoint=None):
    T._check_recipe(recipe)
    _assert_bundle(bundle)
    depth = _feature_depth(head)
    if bundle.world == "real":
        _require_gate(control_gate, bundle, production=not recipe.test_only)
    else:
        _require(head in CONTROL_HEADS and bundle.reference_gate["status"] == "diagnostic",
                 "invalid control head or reference gate")
    factory = factory or (lambda seed, fold: heads.make_head(head, seed=seed))
    fitting = [_training_set(bundle, ix, depth, fold) for fold, ix in enumerate(bundle.fitting)]
    evaluator = lambda model, fold: _score_model(model, bundle, fold, bundle.held[fold], depth, device)
    fitting_evaluator = lambda model, fold: _score_model(model, bundle, fold, bundle.fitting[fold], depth, device)
    feature_hash = _array_hash(_features(bundle.cache, depth))
    identity = _identity({"bundle": bundle.manifest, "depth": depth, "feature_array_sha256": feature_hash})
    result = T._run_search(factory, fitting, evaluator, device=device, recipe=recipe,
                           on_checkpoint=on_checkpoint, evaluation_identity=identity, fitting_evaluator=fitting_evaluator)
    report = result["selection"]
    report.pop("selection_sha256")
    selected_stats, clean_stats = {}, {}
    pair = result["candidate_states"][report["selected"]["lr"]]
    for fold in (0, 1):
        trajectory = T.Trajectory(factory, fitting[fold], lr=report["selected"]["lr"], seed=0, fold=fold,
                                  device=device, _recipe=recipe)
        trajectory.restore(pair["best_states"][fold])
        observed, clean = _score_model(trajectory.model, bundle, fold, bundle.held[fold], depth, device, include_clean=True)
        selected_stats["fold" + str(fold)] = observed
        clean_stats["fold" + str(fold)] = clean
    for panel in (selected_stats, clean_stats):
        panel["pooled"] = controls.pool_statistics([panel["fold0"], panel["fold1"]])
    point = controls.statistic_values(selected_stats["pooled"])
    _require(np.isclose(point["MSE"], report["selected"]["mse"], atol=1e-12, rtol=1e-12) and
             np.isclose(point["E_dyn"], report["selected"]["edyn"], atol=1e-12, rtol=1e-12),
             "selected checkpoint does not reproduce the selected curve point")
    report.update(head=head, world=bundle.world, bundle=dict(bundle.manifest),
                  feature_array_sha256=feature_hash,
                  positive_gate_sha256=control_gate["gate_sha256"] if control_gate else None,
                  selected_outer_statistics=selected_stats, selected_outer_clean_statistics=clean_stats,
                  selected_outer_values={k: controls.statistic_values(v) for k, v in selected_stats.items()},
                  selected_outer_clean_values={k: controls.statistic_values(v) for k, v in clean_stats.items()},
                  selected_checkpoint="paired minimum pooled observed MSE; not minimum E_dyn",
                  closed_form_references=bundle.reference_gate,
                  closed_form_linear_diagnostics=(control_gate.get("linear_diagnostics", {}) if control_gate else {}))
    report = _lock(report, "selection_sha256")
    states = {"schema": "pose-head-selection-states-v1", "selection_sha256": report["selection_sha256"],
              "candidate_states": result["candidate_states"]}
    return {"selection": report, "states": states,
            "filenames": {"selection": "selection.json", "states": "selection_states.pt"}}


def run_selection(bundle, head, *, control_gate=None, device="cpu", on_checkpoint=None):
    """One fixed head/world search. Real searches require the complete positive gate."""
    T.configure_numerics()
    return _run_selection(bundle, head, control_gate=control_gate, device=device, on_checkpoint=on_checkpoint)


def qualify_controls(artifacts, selections):
    """All six searches, fold and pooled statistics, before any real selection."""
    reference = controls.combine_reference_folds(artifacts)
    if reference["status"] != "diagnostic":
        return _lock({"status": reference["status"], "complete": False, "reference_gate": reference}, "gate_sha256")
    _require(set(selections) == set(WORLDS) and all(set(selections[w]) == set(CONTROL_HEADS) for w in WORLDS),
             "six positive-control selections are required")
    identity = _identity(sorted(artifacts, key=lambda a: a["outer_fold"]))
    source, recipe = None, None
    linear_results, panels, locks = {}, {}, {}
    for world in WORLDS:
        panels[world], locks[world] = {}, {}
        for head in CONTROL_HEADS:
            report = selections[world][head]
            _check_selection(report)
            _require(recipe is None or recipe == report["recipe"], "control recipes differ")
            recipe = report["recipe"]
            _require(report["head"] == head and report["world"] == world, "control selection label mismatch")
            bundle = report["bundle"]
            _verify(bundle, "bundle_sha256")
            _require(bundle["artifacts_sha256"] == identity, "controls use another calibration artifact")
            current_source = {k: bundle[k] for k in ("population_sha256", "source_identity", "user_folds")}
            if source is not None:
                _require(source == current_source, "positive-control populations differ")
            source = current_source
            panels[world][head] = {k: controls.statistic_values(v) for k, v in report["selected_outer_statistics"].items()}
            locks[world][head] = report["selection_sha256"]
            if world == "linear":
                linear_results[head] = controls.linear_head_gate(reference["linear_records"], panels[world][head],
                                                                 optimization_alarms=report["optimizer_alarms"])
    temporal = controls.temporal_head_gate(reference["temporal_records"],
                    {k: v["E_dyn"] for k, v in panels["temporal"]["TCN_post"].items()},
                    optimization_alarms=selections["temporal"]["TCN_post"]["optimizer_alarms"],
                    mlp_records={k: v["E_dyn"] for k, v in panels["temporal"]["MLP"].items()})
    statuses = [r["status"] for r in linear_results.values()] + [temporal["status"]]
    status = "passed" if all(s == "passed" for s in statuses) else (
        "optimization_unresolved" if "optimization_unresolved" in statuses else "recipe_not_admitted")
    return _lock({"schema": "pose-head-positive-gate-v1", **source, "artifacts_sha256": identity,
                  "status": status, "complete": True, "reference_gate": reference,
                  "recipe": recipe,
                  "linear_head_gates": linear_results, "temporal_head_gate": temporal,
                  "selected_panels": panels, "selection_locks": locks,
                  "linear_diagnostics": {"fold" + str(a["outer_fold"]):
                      a["outer_reference_statistics"].get("linear_diagnostics", {}) for a in artifacts},
                  "objective_tradeoff_changes_gate": False,
                  "temporal_L0_MLP_descriptive_only": True}, "gate_sha256")


def _run_refits(bundle, selected, control_gate, *, device="cpu", recipe=T.Recipe(), factory=None):
    _require(bundle.world == "real", "synthetic controls have no full-40 refit")
    _assert_bundle(bundle)
    _require_gate(control_gate, bundle, production=not recipe.test_only)
    _check_selection(selected)
    _require(selected["world"] == "real" and selected["bundle"] == bundle.manifest and
             selected["positive_gate_sha256"] == control_gate["gate_sha256"], "refit selection provenance differs")
    head, depth = selected["head"], _feature_depth(selected["head"])
    _require(_array_hash(_features(bundle.cache, depth)) == selected["feature_array_sha256"],
             "refit features differ from selection")
    factory = factory or (lambda seed, fold: heads.make_head(head, seed=seed))
    indices = np.arange(len(bundle.cache["meta"]))
    fitting = _training_set(bundle, indices, depth, 0)
    # A full-fit diagnostic has a newly fixed, full-fit threshold; no held values.
    threshold = metrics.training_threshold(bundle.targets[0], bundle.cache["mask"])
    full_bundle = FoldBundle(bundle.cache, "real", bundle.fitting, bundle.held, bundle.targets, bundle.clean_targets,
                             bundle.pose_variances, (threshold, threshold), bundle.support, bundle.manifest)
    diagnostic = lambda model, fold: _score_model(model, full_bundle, 0, indices, depth, device)
    result = T._run_refits(factory, fitting, selected, device=device, recipe=recipe, fitting_evaluator=diagnostic)
    states = result.pop("refit_states")
    cells = {str(seed): {"payload_sha256": state["payload_sha256"],
                         "model_sha256": T.payload_sha256(state["model"]), "updates": state["updates"],
                         "dataset_sha256": state["dataset_sha256"], "initial_sha256": state["initial_sha256"]}
             for seed, state in states.items()}
    manifest = _lock({"schema": "pose-head-refits-v1", **result, "head": head, "world": "real",
                      "bundle": dict(bundle.manifest), "positive_gate_sha256": control_gate["gate_sha256"],
                      "full_fit_dynamic_threshold": threshold, "cells": cells}, "manifest_sha256")
    return {"manifest": manifest,
            "states": {"schema": "pose-head-refit-states-v1", "manifest_sha256": manifest["manifest_sha256"],
                       "refit_states": states},
            "filenames": {"manifest": "refits.json", "states": "refit_states.pt"}}


def run_refits(bundle, selected, control_gate, *, device="cpu"):
    """Three fresh full-training refits, at the one locked LR and exposure."""
    _check_selection(selected, production=True)
    T.configure_numerics()
    return _run_refits(bundle, selected, control_gate, device=device)


def _lock_inference_manifest(selections, refits, *, development_identity, production=True):
    _require(set(selections) == set(HEADS) == set(refits), "all five selections and fifteen refits are required")
    cells, source, gate = {}, None, None
    for head in HEADS:
        selected, result = selections[head], refits[head]
        _check_selection(selected, production=production)
        manifest, payload = result["manifest"], result["states"]
        _verify(manifest, "manifest_sha256")
        _require(selected["head"] == head and selected["world"] == "real" and manifest["head"] == head and
                 manifest["world"] == "real" and manifest["refits_locked"] is True, "invalid real-refit cell")
        _require(manifest["selection_sha256"] == selected["selection_sha256"] and
                 manifest["bundle"] == selected["bundle"] and manifest["recipe"] == selected["recipe"],
                 "refit does not descend from its selection")
        current = {k: selected["bundle"][k] for k in ("source_identity", "population_sha256", "users", "user_folds")}
        _require(source is None or source == current, "real selections have different training populations")
        source = current
        _require(gate is None or gate == selected["positive_gate_sha256"], "real selections use different control gates")
        gate = selected["positive_gate_sha256"]
        _require(gate == manifest["positive_gate_sha256"], "refit control gate differs")
        _require(payload["manifest_sha256"] == manifest["manifest_sha256"], "refit JSON/PT locks differ")
        states = payload["refit_states"]
        _require(set(states) == {0, 1, 2} and set(manifest["cells"]) == {"0", "1", "2"}, "all three refit seeds required")
        _require(T.payload_sha256(states) == manifest["refit_sha256"], "refit payload differs from manifest")
        for seed, state in states.items():
            _verify(state, "payload_sha256")
            cell = manifest["cells"][str(seed)]
            _require(state["seed"] == seed and state["fold"] == 2 and _json(state["recipe"]) == selected["recipe"],
                     "wrong refit seed, fold or recipe")
            _require(state["payload_sha256"] == cell["payload_sha256"] and
                     T.payload_sha256(state["model"]) == cell["model_sha256"], "model lock mismatch")
            _require(state["lr"] == selected["selected"]["lr"] and
                     state["updates"] == math.ceil(selected["selected"]["exposure"] * state["steps_per_epoch"]),
                     "refit dose differs from selected exposure")
            cells[f"{head}/{seed}"] = {**cell, "selection_sha256": selected["selection_sha256"],
                                      "refit_manifest_sha256": manifest["manifest_sha256"]}
    return _lock({"schema": "pose-head-inference-lock-v1", "complete": True, "cells": cells,
                  **source, "positive_gate_sha256": gate, "development_identity": _source(development_identity),
                  "production": production}, "inference_lock_sha256")


def lock_inference_manifest(selections, refits, *, development_identity):
    """Verify every JSON/PT link before the caller opens development data."""
    return _lock_inference_manifest(selections, refits, development_identity=development_identity)


def _predict_locked(cache, head, seed, checkpoint, lock, *, development_identity, device="cpu",
                     production=True, factory=None):
    _verify(lock, "inference_lock_sha256")
    expected = {f"{h}/{s}" for h in HEADS for s in (0, 1, 2)}
    _require(lock.get("complete") is True and set(lock["cells"]) == expected, "all fifteen locked refits must precede inference")
    _require(not production or lock.get("production") is True, "test lock cannot enter production inference")
    _require(_source(development_identity) == lock["development_identity"], "development pin differs from the lock")
    _require(head in HEADS and seed in (0, 1, 2), "unknown inference cell")
    _verify(checkpoint, "payload_sha256")
    cell = lock["cells"][f"{head}/{seed}"]
    _require(checkpoint["payload_sha256"] == cell["payload_sha256"] and
             T.payload_sha256(checkpoint["model"]) == cell["model_sha256"], "inference checkpoint is not locked")
    _require(cache.get("config", {}).get("split") == "val", "inference requires the pinned development split")
    users = {m["user"] for m in cache["meta"]}
    _require(not users.intersection(lock["users"]), "development overlaps training users")
    _require(not production or len(users) == 15, "development population must contain fifteen users")
    _require(not production or _json(checkpoint["recipe"]) == _json(T.Recipe().__dict__), "test checkpoint cannot enter production")
    depth = _feature_depth(head)
    model = (factory or (lambda seed, fold: heads.make_head(head, seed=seed)))(seed, 2).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    count = len(cache["meta"])
    prediction = np.empty((count, 148, 20), dtype=np.float32)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, count, BATCH):
            rows = np.arange(start, min(start+BATCH, count))
            x = torch.from_numpy(D.view_inputs(cache, depth, rows)).to(device)
            values = model(x)[:, D.CROP:-D.CROP].detach().cpu().numpy()
            _require(values.shape == prediction[rows].shape and np.isfinite(values).all(), "invalid development prediction")
            prediction[start:start+len(rows)] = values
    receipt = _lock({"schema": "pose-head-prediction-v1", "complete": True, "head": head, "seed": seed,
                     "inference_lock_sha256": lock["inference_lock_sha256"],
                     "checkpoint_payload_sha256": checkpoint["payload_sha256"],
                     "development_identity": lock["development_identity"], "windows": count, "users": len(users),
                     "shape": list(prediction.shape), "dtype": str(prediction.dtype), "crop": [6, 154],
                     "prediction_array_sha256": _array_hash(prediction), "batch_size": BATCH}, "receipt_sha256")
    return {"prediction": prediction, "receipt": receipt}


def predict_locked(cache, head, seed, checkpoint, lock, *, development_identity, device="cpu"):
    """One final prediction per locked cell; no fit, score, selection or held tuning."""
    T.configure_numerics()
    return _predict_locked(cache, head, seed, checkpoint, lock, development_identity=development_identity, device=device)
