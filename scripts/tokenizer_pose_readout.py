"""Three-arm teacher pose ridge, with frozen crop6 and train-only selection.

Plans are instantiated after verified extractions. Historical reference caches
pin only selection/target identity; their latents and predictions are never used.
validate and benchmark do not fit on validation; ridge fits all three arms and
writes the paired comparison only after all cells succeed. No jobs are submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_info

import pose_dynamics as D
import pose_e3_b as E
import pose_temporal_readout as T

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "tokenizer-pose-readout-20261103-v1"
ARMS = ("T25-L", "T100-L", "T100-G")
OFFSET, EPS, SEED, N_BOOT, BATCH = 6, 1e-5, 20261102, 2000, 64
THRESHOLD = 0.2574231028556824
SELECTION = {"train": {"n_windows": 1907, "n_users": 40, "n_recordings": 200},
             "val": {"n_windows": 18548, "n_users": 15, "n_recordings": 1950}}
REFERENCE_FIELDS = {"path", "sha256", "bytes", "n_windows", "n_users", "n_recordings"}
CACHE_FIELDS = {"path", "sha256", "bytes", "provenance_path", "provenance_sha256",
                "receipt_path", "receipt_sha256", "checkpoint_sha256", "extraction_code_sha"}
require = D.require
arrays_hash, json_hash = T.arrays_hash, T.json_hash


def recipe():
    return {"which": "teacher", "cached_features": "mean_K64_fp16", "dim": 384,
            "normalization": "stateless LayerNorm FP32", "normalization_eps": EPS,
            "offset": OFFSET, "common_centers": "6 <= t and t+6 < n_patches",
            "target_support": "central target mask only", "fit_precision": "float64",
            "intercept": "unpenalized", "cv": "alternating sorted train users; two folds; pooled masked standardized SSE",
            "alpha_exponents": [-2, -1, 0, 1, 2, 3, 4], "max_extensions_per_side": 3,
            "tie_break": "smallest alpha", "dynamic_threshold": THRESHOLD,
            "dynamic_selection": "original full-window pose_var", "min_centered_points": 2,
            "bootstrap": {"unit": "user", "replicates": N_BOOT, "seed": SEED},
            "batch_windows": BATCH}


def source_hashes():
    return {str(p.relative_to(ROOT)): D.file_sha256(p)
            for p in (Path(__file__), Path(T.__file__), Path(E.__file__), Path(D.__file__))}


def make_plan(caches, references, fold_users, step):
    plan = {"protocol": PROTOCOL, "step": step, "recipe": recipe(), "stats_sha256": E.STATS_SHA,
            "fold_users": fold_users, "sources": source_hashes(), "references": references, "caches": caches}
    validate_plan(plan)
    return plan


def _digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL, "plan protocol mismatch")
    require(type(plan.get("step")) is int and plan["step"] in (5000, 19618), "invalid checkpoint dose")
    require(plan.get("recipe") == recipe(), "readout recipe mismatch")
    require(plan.get("stats_sha256") == E.STATS_SHA, "pose stats pin mismatch")
    require(plan.get("sources") == source_hashes(), "readout source pins mismatch")
    folds = plan.get("fold_users")
    require(isinstance(folds, list) and len(folds) == 2 and all(isinstance(f, list) for f in folds), "invalid train folds")
    users = sorted(sum(folds, []))
    require(len(users) == len(set(users)) == 40 and all(isinstance(u, str) and u for u in users), "expected 40 unique train users")
    require(folds == [users[::2], users[1::2]], "folds must alternate sorted train users")
    require(set(plan.get("references", {})) == {"train", "val"}, "missing reference split")
    paths = []
    for split, pin in plan["references"].items():
        require(set(pin) == REFERENCE_FIELDS, "reference pin fields mismatch")
        require(all(pin[k] == v for k, v in SELECTION[split].items()), "historical selection count mismatch")
        require(Path(pin["path"]).is_absolute() and _digest(pin["sha256"]) and type(pin["bytes"]) is int and pin["bytes"] > 0,
                "invalid reference file pin")
        paths.append(str(Path(pin["path"]).resolve()))
    require(set(plan.get("caches", {})) == set(ARMS), "need exactly three tokenizer arms")
    for arm in ARMS:
        require(set(plan["caches"][arm]) == {"train", "val"}, "missing cache split")
        checkpoint = set()
        for pin in plan["caches"][arm].values():
            require(set(pin) == CACHE_FIELDS, "cache pin fields mismatch")
            require(all(Path(pin[k]).is_absolute() for k in ("path", "provenance_path", "receipt_path")), "artifact paths must be absolute")
            require(all(_digest(pin[k]) for k in ("sha256", "provenance_sha256", "receipt_sha256", "checkpoint_sha256")) and
                    _digest(pin["extraction_code_sha"], 40), "invalid cache digest")
            require(type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid cache size")
            paths.append(str(Path(pin["path"]).resolve()))
            checkpoint.add(pin["checkpoint_sha256"])
        require(len(checkpoint) == 1, "train/val checkpoint mismatch")
    require(len(paths) == len(set(paths)), "duplicate cache/reference paths")
    require(len({plan["caches"][a]["train"]["checkpoint_sha256"] for a in ARMS}) == 3, "duplicate arm checkpoints")


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_json_unique)


def runtime_identity(expected_code_sha):
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(_digest(expected_code_sha, 40) and code == expected_code_sha, "execution code SHA mismatch")
    return {"code_sha": code, "sources": source_hashes(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "torch_threads": torch.get_num_threads(),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer")}
                            for p in threadpool_info()]}


def verify_file(path, sha256, size=None):
    path = Path(path)
    state = T.file_state(path)
    require(path.is_file() and (size is None or path.stat().st_size == size), "artifact size mismatch: " + str(path))
    require(D.file_sha256(path) == sha256, "artifact SHA mismatch: " + str(path))
    require(T.file_state(path) == state, "artifact changed while hashing")
    return {str(path): state}


def read_targets(path):
    """Read only target/selection arrays, never reference latents or predictions."""
    before = T.file_state(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in ("target", "mask", "std", "pose_var")}
        data.update(meta=json.loads(str(z["meta_json"])), n_recordings=int(z["n_recordings"]))
    require(T.file_state(path) == before, "cache changed while reading targets")
    y, mask, std, pv = (data[k] for k in ("target", "mask", "std", "pose_var"))
    require(y.ndim == 3 and y.dtype == np.float32 and mask.shape == y.shape and mask.dtype == bool, "invalid target/mask shape or precision")
    require(len(data["meta"]) == len(y), "metadata length mismatch")
    require(all(isinstance(m.get("user"), str) and m["user"] and type(m.get("n_patches")) is int and
                2*OFFSET < m["n_patches"] <= y.shape[1] for m in data["meta"]), "invalid real window metadata")
    lengths = np.asarray([m["n_patches"] for m in data["meta"]], dtype=np.int64)
    require(not np.any(mask & (np.arange(y.shape[1])[None, :] >= lengths[:, None])[..., None]), "mask includes padding")
    require(np.isfinite(y[mask]).all(), "nonfinite supported target")
    require(std.shape == (len(y), y.shape[-1]) and std.dtype == np.float32 and np.isfinite(std).all() and (std > 0).all(), "invalid target std")
    require(pv.shape == (len(y),) and pv.dtype == np.float32 and not np.isinf(pv).any() and (pv[np.isfinite(pv)] >= 0).all(), "invalid original pose variance")
    data.update(lengths=lengths, central_mask=T.central_mask(mask, lengths, offset=OFFSET))
    return data


def target_signature(data):
    return {"metadata": json_hash(data["meta"]), "arrays": arrays_hash(*(data[k] for k in ("target", "mask", "std", "pose_var"))),
            "central_mask": arrays_hash(data["central_mask"])}


def check_selection(data, selection):
    users = sorted({m["user"] for m in data["meta"]})
    require(len(data["meta"]) == selection["n_windows"] and len(users) == selection["n_users"] and
            data["n_recordings"] == selection["n_recordings"], "selection counts differ")
    return users


def validate_reference(pin):
    states = verify_file(pin["path"], pin["sha256"], pin["bytes"])
    data = read_targets(pin["path"])
    users = check_selection(data, pin)
    require(data["target"].shape[1:] == (160, 20), "reference geometry mismatch")
    return {"users": users, "targets": target_signature(data), "states": states}


def validate_artifact(pin, arm, split, plan, reference):
    states = verify_file(pin["path"], pin["sha256"], pin["bytes"])
    states.update(verify_file(pin["provenance_path"], pin["provenance_sha256"]))
    states.update(verify_file(pin["receipt_path"], pin["receipt_sha256"]))
    prov, receipt = read_json(pin["provenance_path"]), read_json(pin["receipt_path"])
    common = {"arm": arm, "which": "teacher", "checkpoint_step": plan["step"],
              "checkpoint_sha256": pin["checkpoint_sha256"], "code_sha": pin["extraction_code_sha"]}
    for key, value in common.items():
        require(prov.get(key) == value and receipt.get(key) == value, "extraction identity mismatch: " + key)
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True, "incomplete extraction receipt")
    require(receipt.get("postload_recipe_passed") is True, "missing effective model recipe verification")
    output = receipt.get("outputs", {}).get(split, {})
    require(output.get("sha256") == pin["sha256"] and output.get("provenance_sha256") == pin["provenance_sha256"], "receipt output digest mismatch")
    selection = plan["references"][split]
    for key, value in {"split": split, "stats_sha256": plan["stats_sha256"], "cache": pin["path"],
                       "cache_sha256": pin["sha256"], "cache_bytes": pin["bytes"],
                       **{k: selection[k] for k in SELECTION[split]}}.items():
        require(prov.get(key) == value, "cache provenance mismatch: " + key)
    expected_model = {"dim": 384, "k_latents": 64, "patch_ms": 25., "pose_head": True,
                      "window_ms": None if arm == "T25-L" else 100.,
                      "tokenizer_activation": "gelu" if arm == "T100-G" else "identity"}
    require(all(prov.get("model_config", {}).get(k) == v for k, v in expected_model.items()), "tokenizer model configuration mismatch")
    n = selection["n_windows"]
    expected = {"lat": ([n, 160, 384], "<f2"), "target": ([n, 160, 20], "<f4"),
                "mask": ([n, 160, 20], "|b1"), "std": ([n, 20], "<f4"), "pose_var": ([n], "<f4")}
    headers = T.headers(pin["path"])
    require(all(headers.get(k) == {"shape": shape, "dtype": dtype} for k, (shape, dtype) in expected.items()), "cache geometry/precision mismatch")
    with np.load(pin["path"], allow_pickle=False) as z:
        config = json.loads(str(z["config_json"]))
        require(int(z["dim"]) == 384 and bool(z["has_head"]), "cache model dimension/head mismatch")
    require(config == prov.get("config"), "embedded extraction config mismatch")
    for key, value in {"which": "teacher", "split": split, "code_sha": pin["extraction_code_sha"],
                       "pose_stats_sha256": plan["stats_sha256"], "checkpoint_pose_stats_sha256": plan["stats_sha256"],
                       "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}.items():
        require(config.get(key) == value, "extraction recipe mismatch: " + key)
    data = read_targets(pin["path"])
    users = check_selection(data, selection)
    require(users == reference["users"] and target_signature(data) == reference["targets"], "historical metadata/target/mask/std/pose_var mismatch")
    require(states == {p: T.file_state(p) for p in states}, "artifact changed during validation")
    return {"users": users, "targets": target_signature(data), "states": states, "cache_sha256": pin["sha256"],
            "checkpoint_sha256": pin["checkpoint_sha256"], "arm": arm, "split": split}


def read_data(pin):
    data = read_targets(pin["path"])
    before = T.file_state(pin["path"])
    with np.load(pin["path"], allow_pickle=False) as z:
        data["lat"] = z["lat"]
    require(T.file_state(pin["path"]) == before, "cache changed while reading latents")
    require(data["lat"].ndim == 3 and data["lat"].dtype == np.float16 and data["lat"].shape[:2] == data["target"].shape[:2], "expected FP16 mean latents")
    return data


def grouped_stats(train, batch=BATCH):
    users = np.asarray([m["user"] for m in train["meta"]])
    dim, joints = train["lat"].shape[-1], train["target"].shape[-1]
    parts = {}
    for start in range(0, len(users), batch):
        sl = slice(start, start+batch)
        x = T.normalized_inputs(train["lat"][sl], train["lengths"][sl])[:, OFFSET:-OFFSET]
        y, mask = train["target"][sl, OFFSET:-OFFSET], train["central_mask"][sl]
        require(x.shape[:2] == y.shape[:2] == mask.shape[:2], "crop6 feature/target support mismatch")
        for user in sorted(set(users[sl])):
            pick = users[sl] == user
            stats = E.sufficient(x[pick].reshape(-1, dim), y[pick].reshape(-1, joints), mask[pick].reshape(-1, joints))
            parts[user] = E.merge_stats([parts[user], stats], joints) if user in parts else stats
    return parts, dim, joints


def fit_reader(train, expected_folds=None):
    started = perf_counter()
    parts, dim, joints = grouped_stats(train)
    stat_seconds = perf_counter()-started
    folds = [sorted(parts)[::2], sorted(parts)[1::2]]
    require(expected_folds is None or folds == expected_folds, "training fold users differ")
    started = perf_counter()
    cv = E.cross_validate(parts, dim, joints)
    require(not cv["boundary_unresolved"], "alpha boundary unresolved; no scientific report")
    weights, bias = E.prepare_solver(E.merge_stats(list(parts.values()), joints), dim, joints)(cv["alpha"])
    return weights, bias, {"selection": cv, "n_features": dim, "n_users": len(parts),
                           "statistics_seconds": stat_seconds, "selection_refit_seconds": perf_counter()-started,
                           "fit_support_sha256": arrays_hash(train["central_mask"]), "coefficients_sha256": arrays_hash(weights, bias)}


def predict(lat, lengths, weights, bias, batch=BATCH):
    # No target or pose-validity information is accepted by the feature path.
    result = np.empty((len(lat), lat.shape[1]-2*OFFSET, len(bias)), dtype=np.float64)
    for start in range(0, len(lat), batch):
        sl = slice(start, start+batch)
        x = T.normalized_inputs(lat[sl], lengths[sl])[:, OFFSET:-OFFSET]
        result[sl] = x.astype(np.float64) @ weights + bias
    require(np.isfinite(result).all(), "nonfinite prediction")
    return result


def summarize(data, prediction):
    users = np.asarray([m["user"] for m in data["meta"]])
    pv, mask = data["pose_var"], data["central_mask"]
    require(prediction.shape == mask.shape, "prediction/support shape mismatch")
    selected = {"all": np.ones(len(pv), bool), "dynamic": np.isfinite(pv) & (pv >= THRESHOLD),
                "static": np.isfinite(pv) & (pv < THRESHOLD)}
    accum, maes = {g: {} for g in selected}, {g: {} for g in selected}
    for start in range(0, len(users), BATCH):
        sl = slice(start, start+BATCH)
        y, p, m = data["target"][sl, OFFSET:-OFFSET], prediction[sl], mask[sl]
        moments = D.window_joint_moments(p, y, m)
        count = m.sum(axis=(1, 2))
        errors = (np.where(m, np.abs(p-y), 0)*data["std"][sl, None, :]).sum(axis=(1, 2))
        for group, selection in selected.items():
            D.accumulate(accum[group], moments, users[sl], selection[sl], 2, D.CENTERED_FIELDS)
            for user in sorted(set(users[sl][selection[sl] & (count > 0)])):
                ix = selection[sl] & (users[sl] == user) & (count > 0)
                row = maes[group].setdefault(user, {"error_sum_rad": 0., "n_valid_elements": 0})
                row["error_sum_rad"] += float(errors[ix].sum())
                row["n_valid_elements"] += int(count[ix].sum())
    groups = {}
    for group in selected:
        count = sum(v["n_valid_elements"] for v in maes[group].values())
        error = sum(v["error_sum_rad"] for v in maes[group].values())
        groups[group] = {"centered": D.pooled_summary(accum[group], D.CENTERED_FIELDS),
                         "mae_rad": error/count if count else None, "mae_per_user": maes[group],
                         "n_selected_windows": int(selected[group].sum())}
    return {"groups": groups, "support_sha256": arrays_hash(mask), "targets_sha256": target_signature(data),
            "dynamic_threshold": THRESHOLD, "prediction_sha256": arrays_hash(prediction)}


def paired_comparison(cells, step, n_boot=N_BOOT, seed=SEED):
    require(set(cells) == set(ARMS), "need all three cells; no intersection comparison")
    reference = cells[ARMS[0]]["analysis"]
    for arm, cell in cells.items():
        require(cell.get("complete") is True and cell.get("arm") == arm and cell.get("checkpoint_step") == step and
                cell.get("which") == "teacher", "incomplete/mixed-dose cell")
        require(cell["analysis"]["support_sha256"] == reference["support_sha256"] and
                cell["analysis"]["targets_sha256"] == reference["targets_sha256"], "paired target/support differs")
    contrasts = (("T100-L", "T25-L"), ("T100-G", "T100-L"), ("T100-G", "T25-L"))
    output = {}
    for group in ("dynamic", "all", "static"):
        rows = {a: cells[a]["analysis"]["groups"][group]["centered"]["per_user"] for a in ARMS}
        users = sorted(rows[ARMS[0]])
        require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), "paired users missing")
        base = rows[ARMS[0]]
        for values in rows.values():
            for user in users:
                require(all(values[user][k] == base[user][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")), "paired support counts differ")
                require(values[user]["sums"]["vary"] == base[user]["sums"]["vary"], "paired denominator differs")
        denominator = np.asarray([base[u]["sums"]["vary"] for u in users])
        draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
        den = denominator[draws].sum(1)
        require(np.isfinite(den).all() and (den > 0).all(), "invalid bootstrap denominator; no replicas discarded")
        errors = {a: np.asarray([rows[a][u]["sums"]["centered"] for u in users]) for a in ARMS}
        result = {"users": users, "n_bootstrap": n_boot, "seed": seed, "draws_sha256": arrays_hash(draws), "contrasts": {}, "cell_ratio_ci95": {}}
        for arm in ARMS:
            ratios = errors[arm][draws].sum(1)/den
            require(np.isfinite(ratios).all(), "nonfinite bootstrap ratio")
            result["cell_ratio_ci95"][arm] = np.percentile(ratios, [2.5, 97.5]).tolist()
        for candidate, control in contrasts:
            numerator = errors[candidate]-errors[control]
            values = numerator[draws].sum(1)/den
            require(np.isfinite(values).all(), "nonfinite bootstrap contrast")
            ci = np.percentile(values, [2.5, 97.5]).tolist()
            relative = float(-numerator.sum()/errors[control].sum()) if errors[control].sum() > 0 else None
            result["contrasts"][candidate+"_minus_"+control] = {
                "candidate": candidate, "control": control, "delta_ratio": float(numerator.sum()/denominator.sum()),
                "ci95": ci, "relative_error_reduction": relative,
                "signal_2percent_and_ci_below_zero": relative is not None and relative >= .02 and ci[1] < 0,
                "per_user": {u: {"delta_centered_sse": float(v), "target_variance_sum": float(d)}
                             for u, v, d in zip(users, numerator, denominator)}}
        output[group] = result
    return output


def benchmark_train(train):
    started = perf_counter()
    parts, dim, joints = grouped_stats(train)
    statistics_seconds = perf_counter()-started
    users = sorted(parts)
    started = perf_counter()
    for subset in (users[::2], users[1::2], users):
        E.prepare_solver(E.merge_stats([parts[u] for u in subset], joints), dim, joints)(1e5)
    return {"statistics_seconds": statistics_seconds, "three_eigensystems_seconds": perf_counter()-started,
            "n_features": dim, "n_users": len(users), "valid_fit_elements": int(train["central_mask"].sum()),
            "note": "fixed alpha 1e5; train only; no validation or CV score"}


def run(plan, out_dir):
    cells = {}
    for arm in ARMS:
        # Fit completes before opening this arm's validation payload.
        train = read_data(plan["caches"][arm]["train"])
        weights, bias, fit = fit_reader(train, plan["fold_users"])
        train_users = {m["user"] for m in train["meta"]}
        del train
        val = read_data(plan["caches"][arm]["val"])
        require(train_users.isdisjoint({m["user"] for m in val["meta"]}), "train/val user leakage")
        started = perf_counter()
        prediction = predict(val["lat"], val["lengths"], weights, bias)
        predict_seconds = perf_counter()-started
        coeff_path, pred_path = out_dir/(arm+".coefficients.npz"), out_dir/(arm+".predictions.npy")
        with coeff_path.open("xb") as stream:
            np.savez_compressed(stream, weights=weights, bias=bias)
        with pred_path.open("xb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        cells[arm] = {"complete": True, "arm": arm, "which": "teacher", "checkpoint_step": plan["step"],
                      "checkpoint_sha256": plan["caches"][arm]["train"]["checkpoint_sha256"],
                      "fit": fit, "prediction_seconds": predict_seconds, "analysis": summarize(val, prediction),
                      "coefficients": {"path": str(coeff_path), "sha256": D.file_sha256(coeff_path)},
                      "predictions": {"path": str(pred_path), "sha256": D.file_sha256(pred_path)}}
        del val, prediction
    return {"status": "completed", "complete": True, "cells": cells,
            "paired": paired_comparison(cells, plan["step"]),
            "scope": "Single-seed, reused development validation; no general transfer or between-training uncertainty claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "benchmark", "ridge", "compare"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", required=True, type=Path, help="new external output directory")
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0], help="benchmark arm only")
    parser.add_argument("--cells", type=Path, help="completed ridge report, compare only")
    parser.add_argument("--cells-sha256", help="immutable report pin, compare only")
    args = parser.parse_args(argv)
    D.fresh_destination(args.out)
    plan_bytes = args.plan.read_bytes()
    plan, plan_sha = read_json(args.plan), hashlib.sha256(plan_bytes).hexdigest()
    validate_plan(plan)
    identity = runtime_identity(args.expected_code_sha)
    states, inspections = {}, {}
    if args.stage == "compare":
        require(args.cells is not None and _digest(args.cells_sha256), "compare needs pinned completed cells report")
        states.update(verify_file(args.cells, args.cells_sha256))
        prior = read_json(args.cells)
        require(prior.get("complete") is True and prior.get("status") == "completed" and prior.get("protocol") == PROTOCOL and
                prior.get("plan_sha256") == plan_sha and prior.get("step") == plan["step"] and prior.get("recipe") == recipe(), "incomplete or incompatible ridge report")
        require(prior.get("input_pins") == plan["caches"], "report cache pins mismatch")
        for arm in ARMS:
            cell = prior.get("cells", {}).get(arm, {})
            require(cell.get("checkpoint_sha256") == plan["caches"][arm]["train"]["checkpoint_sha256"], "cell checkpoint mismatch")
            require(cell.get("fit", {}).get("selection", {}).get("boundary_unresolved") is False, "incomplete ridge selection")
            for key in ("coefficients", "predictions"):
                require(key in cell, "missing scientific artifact")
                states.update(verify_file(cell[key]["path"], cell[key]["sha256"]))
        result = {"status": "compared", "complete": True, "paired": paired_comparison(prior["cells"], plan["step"]),
                  "cells_report": {"path": str(args.cells), "sha256": args.cells_sha256}}
    else:
        require(args.cells is None and args.cells_sha256 is None, "cells input belongs to compare only")
        arms = (args.arm,) if args.stage == "benchmark" else ARMS
        splits = ("train",) if args.stage == "benchmark" else ("train", "val")
        references = {split: validate_reference(plan["references"][split]) for split in splits}
        require(references["train"]["users"] == sorted(sum(plan["fold_users"], [])), "reference train folds differ")
        if "val" in references:
            require(set(references["train"]["users"]).isdisjoint(references["val"]["users"]), "train/val user leakage")
        for reference in references.values():
            states.update(reference["states"])
        for arm in arms:
            for split in splits:
                inspection = validate_artifact(plan["caches"][arm][split], arm, split, plan, references[split])
                inspections[arm+"/"+split] = inspection
                states.update(inspection["states"])
        if args.stage == "validate":
            result = {"status": "validated", "complete": True, "scientific_metrics_computed": False}
        elif args.stage == "benchmark":
            result = {"status": "train_only_timing", "complete": True, "arm": args.arm,
                      "timings": benchmark_train(read_data(plan["caches"][args.arm]["train"]))}
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            result = run(plan, args.out)
    require(args.plan.read_bytes() == plan_bytes and runtime_identity(args.expected_code_sha) == identity, "plan/source/runtime changed during execution")
    require(states == {p: T.file_state(p) for p in states}, "input artifact changed during execution")
    result.update(protocol=PROTOCOL, step=plan["step"], recipe=recipe(), runtime=identity, plan_sha256=plan_sha,
                  input_pins=plan["caches"], reference_pins=plan["references"], inspections=inspections,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if args.stage != "ridge":
        args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
