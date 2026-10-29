"""Descriptive, paired PSD3 comparisons on the complete frozen-feature grid.

One subject bootstrap per dataset is shared by every encoder, view, reader and
readout seed. MLP scores are averaged across its two fixed seeds, never across
predicted labels. The bootstrap does not resample training seeds or retrain a
reader. Dataset weights and the set of comparisons are fixed before scoring.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral, Real
import re

import numpy as np

from wearusfm.harness.downstream_ab import fingerprint


MODELS = ("B", "R3c")
VIEWS = ("pool", "dec")
READERS = ("logistic", "mlp")
READOUT_SEEDS = {"logistic": ("0",), "mlp": ("0", "1")}
TASK_WEIGHTS = {
    "p1": {"ninapro_db2": 8 / 13, "ninapro_db3": 3 / 13, "ninapro_db6": 2 / 13},
    "p2": {"epn612": 0.5, "uci_emg": 0.5},
}
DATASETS = tuple(ds for weights in TASK_WEIGHTS.values() for ds in weights)
CELLS = tuple("/".join(parts) for parts in product(MODELS, VIEWS, READERS))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if arr.ndim != 1 or not len(arr):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    return arr


def _labels(value, name):
    arr = _vector(value, name)
    # Check the original scalars too: np.asarray([1, "1"]) silently turns both
    # into strings, which must not manufacture label or prediction agreement.
    if all(isinstance(v, (str, np.str_)) for v in value):
        if any(not str(v).strip() for v in value):
            raise ValueError(f"{name} contains an empty label")
        return arr.astype(str), "text"
    if not all(isinstance(v, (Real, np.bool_)) for v in value):
        raise ValueError(f"{name} labels must be uniformly numeric or textual")
    if arr.dtype.kind not in "biuf" or not np.isfinite(arr).all():
        raise ValueError(f"{name} contains invalid or nonfinite numeric labels")
    return arr, "numeric"


def _normalize_shard(shard):
    if not isinstance(shard, dict):
        raise ValueError("each shard must be a dictionary")
    required = ("status", "protocol", "recipe_sha256", "model", "dataset", "view", "input_sha256",
                "roles_sha256", "checkpoint_sha256", "rows", "y", "subjects", "predictions")
    missing = set(required) - shard.keys()
    if missing:
        raise ValueError(f"shard fields missing: {sorted(missing)}")
    if shard["status"] != "completed":
        raise ValueError("every shard must have status completed; timing-only or partial reports cannot be scored")
    key = (shard["model"], shard["dataset"], shard["view"])
    if key[0] not in MODELS or key[1] not in DATASETS or key[2] not in VIEWS:
        raise ValueError(f"unexpected grid cell: {key}")
    protocol = _identifier(shard["protocol"], "protocol")
    hashes = {name: _sha(shard[name], name)
              for name in ("recipe_sha256", "input_sha256", "roles_sha256", "checkpoint_sha256")}
    rows = _vector(shard["rows"], "rows")
    if (not all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in shard["rows"])
            or rows.dtype.kind not in "iu" or np.any(rows < 0)
            or np.any(rows > np.iinfo(np.int64).max)):
        raise ValueError("rows must contain nonnegative int64 indices")
    rows = rows.astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"duplicate rows in {key}")
    y, family = _labels(shard["y"], "y")
    subjects = _vector(shard["subjects"], "subjects")
    if not all(isinstance(s, (str, np.str_)) and str(s).strip() for s in shard["subjects"]):
        raise ValueError("subjects must be nonempty strings")
    subjects = subjects.astype(str)
    if len(y) != len(rows) or len(subjects) != len(rows):
        raise ValueError(f"row/label/subject lengths differ in {key}")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"fewer than two target classes in {key}")
    raw_predictions = shard["predictions"]
    if not isinstance(raw_predictions, dict) or set(raw_predictions) != set(READERS):
        raise ValueError(f"predictions must contain exactly {READERS}")
    predictions = {}
    for reader in READERS:
        seeds = raw_predictions[reader]
        if not isinstance(seeds, dict) or set(seeds) != set(READOUT_SEEDS[reader]):
            raise ValueError(f"{reader} requires readout seeds {READOUT_SEEDS[reader]}")
        predictions[reader] = {}
        for seed in READOUT_SEEDS[reader]:
            pred, pred_family = _labels(seeds[seed], f"{reader}/{seed} predictions")
            if pred_family != family or len(pred) != len(y):
                raise ValueError(f"prediction labels or length differ in {key}/{reader}/{seed}")
            # A class learned in a fold can be absent from the pooled held-out
            # targets. Such predictions are errors, not evidence of corruption.
            # Training-class membership belongs to the caller's fold validator.
            predictions[reader][seed] = pred
    order = np.argsort(rows, kind="stable")
    return {"key": key, "protocol": protocol, **hashes,
            "rows": rows[order], "y": y[order], "label_family": family,
            "subjects": subjects[order],
            "predictions": {reader: {seed: pred[order] for seed, pred in seeds.items()}
                            for reader, seeds in predictions.items()}}


def _validate(shards):
    if not isinstance(shards, (list, tuple)) or not shards:
        raise ValueError("shards must be a nonempty list")
    grid, checkpoints = {}, {}
    for raw in shards:
        shard = _normalize_shard(raw)
        key = shard["key"]
        if key in grid:
            raise ValueError(f"duplicate grid cell: {key}")
        grid[key] = shard
        model = key[0]
        if model in checkpoints and checkpoints[model] != shard["checkpoint_sha256"]:
            raise ValueError(f"inconsistent checkpoint for {model}")
        checkpoints[model] = shard["checkpoint_sha256"]
    expected = set(product(MODELS, DATASETS, VIEWS))
    if set(grid) != expected:
        raise ValueError(f"incomplete grid; missing cells: {sorted(expected - set(grid))}")
    protocols = {s["protocol"] for s in grid.values()}
    recipes = {s["recipe_sha256"] for s in grid.values()}
    if len(protocols) != 1 or len(recipes) != 1:
        raise ValueError("protocol or recipe_sha256 mismatch")
    for ds in DATASETS:
        ref = grid[(MODELS[0], ds, VIEWS[0])]
        for model, view in product(MODELS, VIEWS):
            cur = grid[(model, ds, view)]
            for name in ("input_sha256", "roles_sha256", "label_family"):
                if cur[name] != ref[name]:
                    raise ValueError(f"{ds}: {name} mismatch")
            for name in ("rows", "y", "subjects"):
                if not np.array_equal(cur[name], ref[name]):
                    raise ValueError(f"{ds}: paired {name} mismatch")
    return grid, checkpoints, protocols.pop(), recipes.pop()


def _contrast_definitions():
    contrasts = {}
    for model, reader in product(MODELS, READERS):
        contrasts[f"dec_minus_pool/{model}/{reader}"] = {
            f"{model}/dec/{reader}": 1.0, f"{model}/pool/{reader}": -1.0}
    for model, view in product(MODELS, VIEWS):
        contrasts[f"mlp_minus_logistic/{model}/{view}"] = {
            f"{model}/{view}/mlp": 1.0, f"{model}/{view}/logistic": -1.0}
    for view, reader in product(VIEWS, READERS):
        contrasts[f"B_minus_R3c/{view}/{reader}"] = {
            f"B/{view}/{reader}": 1.0, f"R3c/{view}/{reader}": -1.0}
    for model in MODELS:
        contrasts[f"interaction/{model}"] = {
            f"{model}/dec/mlp": 1.0, f"{model}/pool/mlp": -1.0,
            f"{model}/dec/logistic": -1.0, f"{model}/pool/logistic": 1.0}
    return contrasts


def _summary(point, draws, seed_points):
    cells = {}
    for i, cell in enumerate(CELLS):
        cells[cell] = {"bacc": float(point[i]),
                       "ci95": np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
                       "subject_bootstrap_se": float(np.std(draws[:, i], ddof=1)),
                       "by_readout_seed": seed_points[cell]}
    contrasts = {}
    for name, coefficients in _contrast_definitions().items():
        coeff = np.asarray([coefficients.get(cell, 0.0) for cell in CELLS])
        samples = (draws @ coeff) * 100.0
        contrasts[name] = {
            "delta_pp": float(point @ coeff * 100.0),
            "ci95_pp": np.quantile(samples, [0.025, 0.975]).tolist(),
            "paired_subject_se_pp": float(np.std(samples, ddof=1)),
            "coefficients": coefficients,
        }
    return {"cells": cells, "contrasts": contrasts}


def _dataset_scores(grid, ds, rng, n_boot):
    ref = grid[(MODELS[0], ds, VIEWS[0])]
    y, subjects = ref["y"], ref["subjects"]
    classes, class_index = np.unique(y, return_inverse=True)
    subject_names, subject_index = np.unique(subjects, return_inverse=True)
    ns, nc = len(subject_names), len(classes)
    # These counts have the same estimand as downstream_ab.subject_counts/bacc:
    # macro recall pooled across sampled subjects, NOT a mean of subject BAcc.
    flat = subject_index * nc + class_index
    n = np.bincount(flat, minlength=ns * nc).reshape(ns, nc)
    hits, seed_order, absent_target_predictions = [], [], {}
    for model, view, reader in product(MODELS, VIEWS, READERS):
        cell = f"{model}/{view}/{reader}"
        absent_target_predictions[cell] = {}
        for seed in READOUT_SEEDS[reader]:
            pred = grid[(model, ds, view)]["predictions"][reader][seed]
            hits.append(np.bincount(flat[pred == y], minlength=ns * nc).reshape(ns, nc))
            seed_order.append((cell, seed))
            absent_target_predictions[cell][seed] = int(np.count_nonzero(~np.isin(pred, classes)))
    hits = np.asarray(hits, dtype=np.float64)
    totals = n.sum(axis=0)
    seed_point = (hits.sum(axis=1) / totals).mean(axis=1)

    pick = rng.integers(ns, size=(n_boot, ns))
    multiplicity = np.zeros((n_boot, ns), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(n_boot)[:, None], pick), 1)
    denominators = multiplicity @ n
    correct = (multiplicity @ hits.transpose(1, 0, 2).reshape(ns, -1)).reshape(n_boot, len(hits), nc)
    present = denominators > 0
    # A sampled subject set can lack target classes. Use the identical present
    # class set for all predictions, including a one-class draw (as sklearn
    # balanced_accuracy_score does). Never discard or redraw such samples.
    recall = np.divide(correct, denominators[:, None, :], out=np.zeros_like(correct),
                       where=present[:, None, :])
    seed_draws = recall.sum(axis=2) / present.sum(axis=1)[:, None]
    point, draws, seed_points = [], [], {}
    for cell in CELLS:
        idx = [i for i, (name, _) in enumerate(seed_order) if name == cell]
        point.append(seed_point[idx].mean())
        draws.append(seed_draws[:, idx].mean(axis=1))
        seed_points[cell] = {seed_order[i][1]: float(seed_point[i]) for i in idx}
    point, draws = np.asarray(point), np.asarray(draws).T
    result = _summary(point, draws, seed_points)
    result.update({"n_rows": int(len(y)), "n_subjects": ns, "n_classes": nc,
                   "classes": classes.tolist(), "subject_ids": subject_names.tolist(),
                   "input_sha256": ref["input_sha256"], "roles_sha256": ref["roles_sha256"],
                   "paired_rows_sha256": fingerprint(ref["rows"]),
                   "paired_targets_subjects_sha256": fingerprint(ref["rows"], y, subjects),
                   "predictions_outside_observed_targets": absent_target_predictions,
                   "bootstrap_draws_with_missing_classes": int(np.count_nonzero(present.sum(axis=1) < nc)),
                   "bootstrap_single_class_draws": int(np.count_nonzero(present.sum(axis=1) == 1))})
    return result, point, draws, seed_points


def compare_shards(shards: list[dict], n_boot: int = 2000, seed: int = 20261029) -> dict:
    """Validate all 20 shards, then return dataset and fixed-weight P1/P2 scores.

    Input rows may be permuted but must match exactly after sorting by row index.
    Extra top-level shard metadata is ignored; scientific identity and the exact
    reader/seed prediction grid are mandatory. The caller is responsible for
    validating the four-fold training provenance represented by roles_sha256,
    including prediction membership in each fold's training classes.
    No cached result is edited and no learner is fitted here.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 2:
        raise ValueError("n_boot must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    grid, checkpoints, protocol, recipe_sha = _validate(shards)
    rng = np.random.default_rng(seed)
    output = {
        "protocol": protocol, "recipe_sha256": recipe_sha,
        "comparison": "psd3_paired_subject_bootstrap_v1", "descriptive_only": True,
        "n_boot": int(n_boot), "bootstrap_seed": int(seed),
        "units": {"bacc": "fraction", "ci95": "fraction", "contrasts": "percentage points"},
        "readout_seeds": {reader: list(seeds) for reader, seeds in READOUT_SEEDS.items()},
        "identity": {"checkpoints": checkpoints, "n_shards": len(grid)},
        "uncertainty": "Percentile subject-cluster bootstrap conditional on the single pretraining seed, "
                       "fixed folds and fitted readers. One draw per dataset is shared by all cells and readout "
                       "seeds; datasets are resampled independently. MLP BAcc is the mean of two fixed readout "
                       "seed scores, not an ensemble or resampling of training seeds. No multiple-comparison "
                       "correction and no adoption gate.",
        "bootstrap_class_support": "Recall is averaged over target classes present in each draw, identically "
                                   "for every cell; one-class draws are retained and counted.",
        "datasets": {}, "aggregates": {},
    }
    values = {}
    for ds in DATASETS:
        result, point, draws, seed_points = _dataset_scores(grid, ds, rng, int(n_boot))
        output["datasets"][ds] = result
        values[ds] = (point, draws, seed_points)
    for task, weights in TASK_WEIGHTS.items():
        point = sum(weight * values[ds][0] for ds, weight in weights.items())
        draws = sum(weight * values[ds][1] for ds, weight in weights.items())
        seed_points = {
            cell: {seed: float(sum(weight * values[ds][2][cell][seed] for ds, weight in weights.items()))
                   for seed in READOUT_SEEDS[cell.rsplit("/", 1)[-1]]}
            for cell in CELLS
        }
        output["aggregates"][task] = {"weights": dict(weights), **_summary(point, draws, seed_points)}
    return output
