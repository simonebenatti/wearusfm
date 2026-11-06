"""Two-size, three-view teacher pose readout at the 19,618-step endpoint.

Derived from scaling_pose_readout.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py; the 30M control is the reused tokenizer-campaign extraction.

Natural 30M, natural 300M and projected 300M use the same crop, targets, train
users, alpha selection and subject bootstrap. Projection precedes LayerNorm;
the 30M matched control is the unchanged natural cell, not a fourth fit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import scaling_evaluation19618 as V
import scaling_projection as PR
import scaling_train as ST
import tokenizer_pose_readout as Q

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "scaling-pose-readout-19618-20261106-v1"
ARMS, STEP = V.ARMS, V.STEP
CELLS = {"T100-G-30M-natural": (ARMS[0], "natural"),
         "T100-G-300M-natural": (ARMS[1], "natural"),
         "T100-G-300M-matched": (ARMS[1], "matched")}
CONTRASTS = {"natural_300m_minus_30m": ("T100-G-300M-natural", "T100-G-30M-natural"),
             "matched_300m_minus_30m": ("T100-G-300M-matched", "T100-G-30M-natural"),
             "natural_300m_minus_matched_300m": ("T100-G-300M-natural", "T100-G-300M-matched")}
OFFSET, BATCH, N_BOOT, SEED = Q.OFFSET, Q.BATCH, Q.N_BOOT, Q.SEED
require, read_json, verify_file = Q.require, Q.read_json, Q.verify_file


def recipe():
    r = Q.recipe()
    del r["dim"]
    return {**r, "dimensions": dict(V.DIMENSIONS), "cells": {k: list(v) for k, v in CELLS.items()},
            "matched": {"input_dim": 896, "output_dim": 384, "projection": PR.recipe(),
                        "order": "FP16 cache to FP32; fixed backbone projection; stateless LayerNorm FP32",
                        "control": "30M natural reused without transformation or refit"}}


def source_hashes():
    paths = (Path(__file__), Path(PR.__file__), Path(V.__file__), Path(ST.__file__))
    return {**Q.source_hashes(), **{str(p.relative_to(ROOT)): Q.D.file_sha256(p) for p in paths}}


def make_plan(caches, references, fold_users, projection, training_source_contract):
    plan = copy.deepcopy({"protocol": PROTOCOL, "step": STEP, "recipe": recipe(), "stats_sha256": Q.E.STATS_SHA,
                          "fold_users": fold_users, "sources": source_hashes(), "references": references,
                          "caches": caches, "projection": projection, "training_source_contract": training_source_contract})
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL and type(plan.get("step")) is int and plan["step"] == STEP,
            "wrong scaling pose protocol/dose")
    require(plan.get("recipe") == recipe() and plan.get("stats_sha256") == Q.E.STATS_SHA, "scaling pose recipe/stats mismatch")
    require(plan.get("sources") == source_hashes(), "readout source pins mismatch")
    folds = plan.get("fold_users")
    require(isinstance(folds, list) and len(folds) == 2 and all(isinstance(f, list) for f in folds), "invalid train folds")
    users = sorted(sum(folds, []))
    require(len(users) == len(set(users)) == 40 and all(isinstance(u, str) and u for u in users) and
            folds == [users[::2], users[1::2]], "expected alternating folds of 40 train users")
    require(set(plan.get("references", {})) == {"train", "val"}, "missing reference split")
    paths = []
    for split, pin in plan["references"].items():
        require(set(pin) == Q.REFERENCE_FIELDS and all(pin[k] == v for k, v in Q.SELECTION[split].items()), "reference selection differs")
        require(Path(pin["path"]).is_absolute() and Q._digest(pin["sha256"]) and
                type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid reference pin")
        paths.append(str(Path(pin["path"]).resolve()))
    require(set(plan.get("caches", {})) == set(ARMS), "need exactly two scaling arms")
    for arm in ARMS:
        require(set(plan["caches"][arm]) == {"train", "val"}, "missing cache split")
        for pin in plan["caches"][arm].values():
            require(set(pin) == Q.CACHE_FIELDS, "cache pin fields mismatch")
            require(all(Path(pin[k]).is_absolute() for k in ("path", "provenance_path", "receipt_path")), "cache paths must be absolute")
            require(all(Q._digest(pin[k]) for k in ("sha256", "provenance_sha256", "receipt_sha256", "checkpoint_sha256")) and
                    Q._digest(pin["extraction_code_sha"], 40), "invalid cache digest")
            require(type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid cache size")
            paths.append(str(Path(pin["path"]).resolve()))
        require(plan["caches"][arm]["train"]["checkpoint_sha256"] == plan["caches"][arm]["val"]["checkpoint_sha256"],
                "train/val checkpoint mismatch")
    require(len({plan["caches"][a]["train"]["checkpoint_sha256"] for a in ARMS}) == 2, "duplicate scaling checkpoints")
    # References may be the historical 30M cache itself: targets only are read.
    cache_paths = [str(Path(pin["path"]).resolve()) for arm in ARMS for pin in plan["caches"][arm].values()]
    require(len(set(cache_paths)) == 4 and len(set(paths[:2])) == 2, "duplicate cache splits/arms or reference splits")
    for key in ("projection", "training_source_contract"):
        ST.pin_schema(plan[key])
    require(Path(plan["projection"]["path"]).is_absolute(), "projection manifest must be absolute")


def validate_artifact(pin, arm, split, plan, reference):
    states = verify_file(pin["path"], pin["sha256"], pin["bytes"])
    states.update(verify_file(pin["provenance_path"], pin["provenance_sha256"]))
    states.update(verify_file(pin["receipt_path"], pin["receipt_sha256"]))
    prov, receipt = read_json(pin["provenance_path"]), read_json(pin["receipt_path"])
    legacy = arm == ARMS[0]  # the 19618 control is always the reused tokenizer-campaign extraction
    protocol, source_arm = (V.E.PROTOCOL, "T100-G") if legacy else (V.PROTOCOL, arm)
    common = {"protocol": protocol, "arm": source_arm, "which": "teacher", "checkpoint_step": STEP,
              "checkpoint_sha256": pin["checkpoint_sha256"], "code_sha": pin["extraction_code_sha"]}
    for key, value in common.items():
        require(prov.get(key) == value and receipt.get(key) == value, "extraction identity mismatch: " + key)
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True and
            receipt.get("postload_recipe_passed") is True and receipt.get("stage") == "extract-pose", "incomplete extraction receipt")
    if not legacy:
        for record in (prov, receipt):
            require(record.get("training_source_contract_sha256") == plan["training_source_contract"]["sha256"],
                    "extraction training source contract differs")
    output = receipt.get("outputs", {}).get(split, {})
    require(output.get("path") == pin["path"] and output.get("sha256") == pin["sha256"] and
            output.get("bytes") == pin["bytes"] and output.get("provenance_path") == pin["provenance_path"] and
            output.get("provenance_sha256") == pin["provenance_sha256"], "receipt output pin mismatch")
    selection, dim = plan["references"][split], V.dimension(arm)
    for key, value in {"split": split, "stats_sha256": plan["stats_sha256"], "cache": pin["path"],
                       "cache_sha256": pin["sha256"], "cache_bytes": pin["bytes"],
                       **{k: selection[k] for k in Q.SELECTION[split]}}.items():
        require(prov.get(key) == value, "cache provenance mismatch: " + key)
    V.model_recipe(prov.get("model_config", {}), arm)
    n = selection["n_windows"]
    expected = {"lat": ([n, 160, dim], "<f2"), "target": ([n, 160, 20], "<f4"),
                "mask": ([n, 160, 20], "|b1"), "std": ([n, 20], "<f4"), "pose_var": ([n], "<f4")}
    headers = Q.T.headers(pin["path"])
    require(all(headers.get(k) == {"shape": shape, "dtype": dtype} for k, (shape, dtype) in expected.items()), "cache geometry/precision mismatch")
    with np.load(pin["path"], allow_pickle=False) as z:
        config = json.loads(str(z["config_json"]))
        require(int(z["dim"]) == dim and bool(z["has_head"]), "cache dimension/head mismatch")
    require(config == prov.get("config"), "embedded extraction config mismatch")
    for key, value in {"which": "teacher", "split": split, "code_sha": pin["extraction_code_sha"],
                       "pose_stats_sha256": plan["stats_sha256"], "checkpoint_pose_stats_sha256": plan["stats_sha256"],
                       "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}.items():
        require(config.get(key) == value, "extraction recipe differs: " + key)
    data = Q.read_targets(pin["path"])
    users = Q.check_selection(data, selection)
    require(users == reference["users"] and Q.target_signature(data) == reference["targets"], "historical target/support mismatch")
    require(states == {p: Q.T.file_state(p) for p in states}, "artifact changed during validation")
    return {"users": users, "targets": Q.target_signature(data), "states": states, "cache_sha256": pin["sha256"],
            "checkpoint_sha256": pin["checkpoint_sha256"], "arm": arm, "source_arm": source_arm, "split": split}


def normalized_inputs(lat, lengths, arm, view, projection):
    require(arm in ARMS and view in ("natural", "matched"), "unknown readout arm/view")
    require(lat.ndim == 3 and lat.shape[-1] == V.dimension(arm), "latent width differs from arm")
    if view == "natural" or arm == ARMS[0]:
        return Q.T.normalized_inputs(lat, lengths)
    lengths = np.asarray(lengths)
    require(lengths.shape == (len(lat),) and lengths.dtype.kind in "iu" and
            ((lengths > 2 * OFFSET) & (lengths <= lat.shape[1])).all(), "invalid real patch lengths")
    available = np.arange(lat.shape[1])[None, :] < lengths[:, None]
    require(np.isfinite(lat[available]).all(), "nonfinite real input context")
    safe = np.where(available[..., None], lat, 0).astype(np.float32)
    return Q.T.normalized_inputs(PR.project_pose(safe, projection), lengths)


def grouped_stats(train, arm, view, projection=None, batch=BATCH):
    users = np.asarray([m["user"] for m in train["meta"]])
    dim = 384 if view == "matched" else V.dimension(arm)
    joints, parts = train["target"].shape[-1], {}
    for start in range(0, len(users), batch):
        sl = slice(start, start + batch)
        x = normalized_inputs(train["lat"][sl], train["lengths"][sl], arm, view, projection)[:, OFFSET:-OFFSET]
        y, mask = train["target"][sl, OFFSET:-OFFSET], train["central_mask"][sl]
        require(x.shape[:2] == y.shape[:2] == mask.shape[:2], "crop6 feature/target support mismatch")
        for user in sorted(set(users[sl])):
            pick = users[sl] == user
            stats = Q.E.sufficient(x[pick].reshape(-1, dim), y[pick].reshape(-1, joints), mask[pick].reshape(-1, joints))
            parts[user] = Q.E.merge_stats([parts[user], stats], joints) if user in parts else stats
    return parts, dim, joints


def fit_reader(train, arm, view, projection=None, expected_folds=None):
    started = perf_counter()
    parts, dim, joints = grouped_stats(train, arm, view, projection)
    statistics_seconds = perf_counter() - started
    folds = [sorted(parts)[::2], sorted(parts)[1::2]]
    require(expected_folds is None or folds == expected_folds, "training fold users differ")
    started = perf_counter()
    cv = Q.E.cross_validate(parts, dim, joints)
    require(not cv["boundary_unresolved"], "alpha boundary unresolved; no scientific report")
    weights, bias = Q.E.prepare_solver(Q.E.merge_stats(list(parts.values()), joints), dim, joints)(cv["alpha"])
    return weights, bias, {"selection": cv, "n_features": dim, "n_users": len(parts),
                          "statistics_seconds": statistics_seconds, "selection_refit_seconds": perf_counter() - started,
                          "fit_support_sha256": Q.arrays_hash(train["central_mask"]),
                          "coefficients_sha256": Q.arrays_hash(weights, bias)}


def predict(lat, lengths, weights, bias, arm, view, projection=None, batch=BATCH):
    result = np.empty((len(lat), lat.shape[1] - 2 * OFFSET, len(bias)), dtype=np.float64)
    for start in range(0, len(lat), batch):
        sl = slice(start, start + batch)
        x = normalized_inputs(lat[sl], lengths[sl], arm, view, projection)[:, OFFSET:-OFFSET]
        result[sl] = x.astype(np.float64) @ weights + bias
    require(np.isfinite(result).all(), "nonfinite prediction")
    return result


def benchmark_train(train, arm, view, projection=None):
    started = perf_counter()
    parts, dim, joints = grouped_stats(train, arm, view, projection)
    statistics_seconds = perf_counter() - started
    users = sorted(parts)
    started = perf_counter()
    for subset in (users[::2], users[1::2], users):
        Q.E.prepare_solver(Q.E.merge_stats([parts[u] for u in subset], joints), dim, joints)(1e5)
    return {"statistics_seconds": statistics_seconds, "three_eigensystems_seconds": perf_counter() - started,
            "n_features": dim, "n_users": len(users), "valid_fit_elements": int(train["central_mask"].sum()),
            "note": "fixed alpha 1e5; train only; no validation or CV score"}


def paired_comparison(cells, n_boot=N_BOOT, seed=SEED):
    require(set(cells) == set(CELLS), "need three complete scaling cells; no intersection")
    reference = cells["T100-G-30M-natural"]["analysis"]
    for name, (arm, view) in CELLS.items():
        cell = cells[name]
        require(cell.get("complete") is True and cell.get("arm") == arm and cell.get("view") == view and
                cell.get("checkpoint_step") == STEP and cell.get("which") == "teacher", "incomplete/mixed-dose cell")
        require(Q._digest(cell.get("checkpoint_sha256")) and cell.get("fit", {}).get("n_features") ==
                (384 if view == "matched" else V.dimension(arm)) and
                cell.get("fit", {}).get("selection", {}).get("boundary_unresolved") is False,
                "invalid checkpoint/feature dimension or unresolved solver")
        require(Q._digest(cell.get("projection_manifest_sha256")) if view == "matched" else
                cell.get("projection_manifest_sha256") is None, "wrong projection identity for cell")
        require(cell["analysis"]["support_sha256"] == reference["support_sha256"] and
                cell["analysis"]["targets_sha256"] == reference["targets_sha256"], "paired target/support differs")
    require(cells["T100-G-300M-natural"]["checkpoint_sha256"] == cells["T100-G-300M-matched"]["checkpoint_sha256"] !=
            cells["T100-G-30M-natural"]["checkpoint_sha256"], "natural/matched checkpoint pairing differs")
    output = {}
    for group in ("dynamic", "all", "static"):
        rows = {name: cell["analysis"]["groups"][group]["centered"]["per_user"] for name, cell in cells.items()}
        users = sorted(rows["T100-G-30M-natural"])
        require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), "paired users missing")
        base = rows["T100-G-30M-natural"]
        for values in rows.values():
            for user in users:
                require(all(values[user][k] == base[user][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")) and
                        values[user]["sums"]["vary"] == base[user]["sums"]["vary"], "paired support counts/denominator differ")
        denominator = np.asarray([base[u]["sums"]["vary"] for u in users])
        draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
        den = denominator[draws].sum(1)
        require(np.isfinite(den).all() and (den > 0).all(), "invalid bootstrap denominator; no replicas discarded")
        errors = {name: np.asarray([r[u]["sums"]["centered"] for u in users]) for name, r in rows.items()}
        result = {"users": users, "n_bootstrap": n_boot, "seed": seed, "draws_sha256": Q.arrays_hash(draws),
                  "contrasts": {}, "cell_ratio_ci95": {}, "cell_ratio": {}}
        for name, error in errors.items():
            ratios = error[draws].sum(1) / den
            require(np.isfinite(ratios).all(), "nonfinite bootstrap ratio")
            result["cell_ratio_ci95"][name] = np.percentile(ratios, [2.5, 97.5]).tolist()
            result["cell_ratio"][name] = float(error.sum() / denominator.sum())
        for name, (candidate, control) in CONTRASTS.items():
            numerator = errors[candidate] - errors[control]
            values = numerator[draws].sum(1) / den
            require(np.isfinite(values).all(), "nonfinite bootstrap contrast")
            ci = np.percentile(values, [2.5, 97.5]).tolist()
            relative = float(-numerator.sum() / errors[control].sum()) if errors[control].sum() > 0 else None
            result["contrasts"][name] = {"candidate": candidate, "control": control,
                "delta_ratio": float(numerator.sum() / denominator.sum()), "ci95": ci,
                "relative_error_reduction": relative,
                "signal_2percent_and_ci_below_zero": relative is not None and relative >= .02 and ci[1] < 0,
                "per_user": {u: {"delta_centered_sse": float(v), "target_variance_sum": float(d)}
                             for u, v, d in zip(users, numerator, denominator)}}
        output[group] = result
    return output


def run(plan, out_dir, projection):
    cells = {}
    for name, (arm, view) in CELLS.items():
        train = Q.read_data(plan["caches"][arm]["train"])
        weights, bias, fit = fit_reader(train, arm, view, projection, plan["fold_users"])
        train_users = {m["user"] for m in train["meta"]}
        del train
        val = Q.read_data(plan["caches"][arm]["val"])
        require(train_users.isdisjoint({m["user"] for m in val["meta"]}), "train/val user leakage")
        started = perf_counter()
        prediction = predict(val["lat"], val["lengths"], weights, bias, arm, view, projection)
        predict_seconds = perf_counter() - started
        coeff_path, pred_path = out_dir / (name + ".coefficients.npz"), out_dir / (name + ".predictions.npy")
        with coeff_path.open("xb") as stream:
            np.savez_compressed(stream, weights=weights, bias=bias)
        with pred_path.open("xb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        cells[name] = {"complete": True, "arm": arm, "view": view, "which": "teacher", "checkpoint_step": STEP,
            "checkpoint_sha256": plan["caches"][arm]["train"]["checkpoint_sha256"],
            "projection_manifest_sha256": projection.pin["sha256"] if view == "matched" else None,
            "fit": fit, "prediction_seconds": predict_seconds, "analysis": Q.summarize(val, prediction),
            "coefficients": {"path": str(coeff_path), "sha256": Q.D.file_sha256(coeff_path)},
            "predictions": {"path": str(pred_path), "sha256": Q.D.file_sha256(pred_path)}}
        del val, prediction
    return {"status": "completed", "complete": True, "cells": cells, "paired": paired_comparison(cells),
            "scope": "Single pretraining seed, reused development validation. Natural widths primary; fixed projection secondary."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "benchmark", "ridge"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", choices=tuple(CELLS), default="T100-G-300M-natural", help="benchmark cell only")
    args = parser.parse_args(argv)
    Q.D.fresh_destination(args.out)
    plan_bytes = args.plan.read_bytes()
    plan, plan_sha = read_json(args.plan), hashlib.sha256(plan_bytes).hexdigest()
    validate_plan(plan)
    identity = Q.runtime_identity(args.expected_code_sha)
    identity["sources"] = source_hashes()
    projection = PR.load(plan["projection"])
    states, inspections = dict(projection.states), {}
    source = ST.resolve_pin(plan["training_source_contract"])
    checked = ST.check_source_contract(source["path"])
    states[source["path"]] = Q.T.file_state(source["path"])
    for record in checked["states"].values():
        states[record["path"]] = Q.T.file_state(record["path"])
    require(plan["caches"][ARMS[0]]["train"]["checkpoint_sha256"] == V.CONTROL_CHECKPOINT_SHA256,
            "wrong registered control checkpoint")
    arms = (CELLS[args.cell][0],) if args.stage == "benchmark" else ARMS
    splits = ("train",) if args.stage == "benchmark" else ("train", "val")
    references = {split: Q.validate_reference(plan["references"][split]) for split in splits}
    require(references["train"]["users"] == sorted(sum(plan["fold_users"], [])), "reference train folds differ")
    if "val" in references:
        require(set(references["train"]["users"]).isdisjoint(references["val"]["users"]), "train/val user leakage")
    for reference in references.values():
        states.update(reference["states"])
    for arm in arms:
        for split in splits:
            inspection = validate_artifact(plan["caches"][arm][split], arm, split, plan, references[split])
            inspections[arm + "/" + split] = inspection
            states.update(inspection["states"])
    if args.stage == "validate":
        result = {"status": "validated_input_metadata", "complete": True, "scientific_metrics_computed": False,
                  "note": "latent finiteness is checked on every real patch during benchmark/fit/predict"}
    elif args.stage == "benchmark":
        arm, view = CELLS[args.cell]
        result = {"status": "train_only_timing", "complete": True, "cell": args.cell,
                  "timings": benchmark_train(Q.read_data(plan["caches"][arm]["train"]), arm, view, projection)}
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        result = run(plan, args.out, projection)
    after = Q.runtime_identity(args.expected_code_sha)
    after["sources"] = source_hashes()
    require(args.plan.read_bytes() == plan_bytes and after == identity, "plan/source/runtime changed")
    require(states == {p: Q.T.file_state(p) for p in states}, "input artifact changed during execution")
    result.update(protocol=PROTOCOL, step=STEP, recipe=recipe(), runtime=identity, plan_sha256=plan_sha,
                  input_pins=plan["caches"], reference_pins=plan["references"], projection_pin=plan["projection"],
                  training_source_contract=plan["training_source_contract"], inspections=inspections,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if args.stage != "ridge":
        args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
