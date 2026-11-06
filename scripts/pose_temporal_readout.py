"""Frozen pose readout: instantaneous versus [-4, 0, +4] patch ridge.

No encoder is instantiated. Validation reads hashes, headers and metadata only;
benchmarking opens train caches only. The 15 evaluation users never select a
hyperparameter. Inputs retain the existing bidirectional encoder and mean-K
pooling, so this diagnostic makes no causal/online or full-slot claim.
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
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_info

import pose_dynamics as D
import pose_e3_b as E

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/infra/pilot_posa/b19618_20261029/final"
PROTOCOL = "pose-temporal-readout-20261102-v1"
MODELS, READERS = ("B", "R3c"), ("instantaneous", "temporal")
OFFSET, EPS, SEED, N_BOOT = 4, 1e-5, 20261102, 2000
THRESHOLD = 0.2574231028556824
BATCH = 64
require = D.require


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def source_hashes():
    files = [Path(__file__), Path(E.__file__), Path(D.__file__), ARCHIVE / "reports/pose_comparison.json"]
    launcher = ROOT / "scripts/slurm/pose_temporal_readout.sbatch"
    if launcher.exists():
        files.append(launcher)
    for model in MODELS:
        base = ARCHIVE / "extraction_metadata" / model / "pose"
        files.extend(base / name for name in ("train_subset.npz.provenance.json", "val.npz.provenance.json", "extraction_receipt.json"))
    return {str(p.relative_to(ROOT)): D.file_sha256(p) for p in files}


def recipe():
    return {"lags_patches": [-OFFSET, 0, OFFSET], "patch_ms": 25.0,
            "features": "FP32 stateless LayerNorm independently at each real input patch; no PCA/scaler",
            "normalization_eps": EPS, "common_centers": "4 <= t and t+4 < n_patches",
            "target_support": "center target mask only; neighboring target masks never define input validity",
            "cv": "alternating sorted train users, two folds; masked pooled standardized SSE; E3 alpha grid/extension",
            "fit": "float64 sufficient statistics; unpenalized intercept; refit all 40 train users",
            "dynamic_threshold": THRESHOLD, "dynamic_selection": "original full-window cached pose_var >= frozen median",
            "bootstrap": {"unit": "user", "replicates": N_BOOT, "seed": SEED}, "batch_windows": BATCH}


def make_plan(cache_root=None):
    """Create immutable input/source pins from archived evidence, without fitting."""
    caches = {}
    users = None
    for model in MODELS:
        caches[model] = {}
        for split, name in (("train", "train_subset.npz"), ("val", "val.npz")):
            base = ARCHIVE / "extraction_metadata" / model / "pose"
            pp = base / (name + ".provenance.json")
            p = json.loads(pp.read_text())
            if model == "B" and split == "train":
                users = sorted({r["user"] for r in p["records"]})
            caches[model][split] = {
                "path": str(Path(cache_root) / model / "pose" / name) if cache_root else p["cache"],
                "original_path": p["cache"], "sha256": p["cache_sha256"], "bytes": p["cache_bytes"],
                "provenance_sha256": D.file_sha256(pp), "receipt_sha256": D.file_sha256(base / "extraction_receipt.json"),
                "checkpoint_sha256": p["checkpoint_sha256"], "n_windows": p["n_windows"],
                "n_users": p["n_users"], "n_recordings": p["n_recordings"]}
    return {"protocol": PROTOCOL, "recipe": recipe(), "stats_sha256": E.STATS_SHA,
            "dynamic_threshold": THRESHOLD, "fold_users": [users[::2], users[1::2]],
            "sources": source_hashes(), "caches": caches}


def validate_plan(plan):
    reference = make_plan()
    for key in ("protocol", "recipe", "stats_sha256", "dynamic_threshold", "fold_users", "sources"):
        require(plan.get(key) == reference[key], "plan differs from frozen evidence: " + key)
    require(set(plan["caches"]) == set(MODELS), "wrong models")
    for model in MODELS:
        require(set(plan["caches"][model]) == {"train", "val"}, "wrong splits")
        for split in ("train", "val"):
            pin, ref = plan["caches"][model][split], reference["caches"][model][split]
            require(set(pin) == set(ref), "wrong cache pin fields")
            require(Path(pin["path"]).is_absolute(), "cache path must be absolute")
            require(all(pin[k] == v for k, v in ref.items() if k != "path"), "cache pin differs from frozen evidence")


def runtime_identity(expected_code_sha):
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(code == expected_code_sha and len(code) == 40, "execution code SHA mismatch")
    return {"code_sha": code, "sources": source_hashes(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "torch_threads": torch.get_num_threads(),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer")}
                            for p in threadpool_info()]}


def file_state(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            require(item.filename.endswith(".npy") and "/" not in item.filename, "invalid NPZ member")
            name = item.filename[:-4]
            require(name not in result, "duplicate NPZ member")
            with archive.open(item) as f:
                version = np.lib.format.read_magic(f)
                require(version in ((1, 0), (2, 0)), "unsupported NPY version")
                shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                   else np.lib.format.read_array_header_2_0(f))
                require(not dtype.hasobject, "object arrays prohibited")
                require(f.tell() + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize == item.file_size,
                        "NPY payload/header mismatch")
            result[name] = {"shape": list(shape), "dtype": dtype.str}
    return result


def validate_artifact(pin, model, split):
    """Hash full bytes; inspect shapes/config/users without reading scientific arrays."""
    path = Path(pin["path"])
    pp, rp = Path(str(path) + ".provenance.json"), path.parent / "extraction_receipt.json"
    states = {str(p): file_state(p) for p in (path, pp, rp)}
    require(path.stat().st_size == pin["bytes"] and D.file_sha256(path) == pin["sha256"], "cache SHA/size mismatch")
    require(D.file_sha256(pp) == pin["provenance_sha256"], "provenance SHA mismatch")
    require(D.file_sha256(rp) == pin["receipt_sha256"], "receipt SHA mismatch")
    prov, receipt = json.loads(pp.read_text()), json.loads(rp.read_text())
    for key, value in {"model": model, "which": "student", "split": split, "checkpoint_step": 19618,
                       "checkpoint_sha256": pin["checkpoint_sha256"], "stats_sha256": E.STATS_SHA,
                       "code_sha": E.CODE_SHA, "cache": pin["original_path"], "cache_sha256": pin["sha256"],
                       "cache_bytes": pin["bytes"], "same_metadata_target_mask_std_pose_var_as_B5000": True}.items():
        require(prov.get(key) == value, "cache identity mismatch: " + key)
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True, "incomplete extraction")
    require(receipt["outputs"][split]["sha256"] == pin["sha256"] and
            receipt["outputs"][split]["provenance_sha256"] == pin["provenance_sha256"], "receipt output mismatch")
    for key in ("model", "checkpoint_sha256", "checkpoint_step", "code_sha"):
        require(receipt[key] == prov[key], "receipt identity mismatch")
    shape = headers(path)
    n = pin["n_windows"]
    expected = {"lat": ([n, 160, 384], "<f2"), "target": ([n, 160, 20], "<f4"),
                "mask": ([n, 160, 20], "|b1"), "pred": ([n, 160, 20], "<f4"),
                "std": ([n, 20], "<f4"), "pose_var": ([n], "<f4")}
    for key, (size, dtype) in expected.items():
        require(shape.get(key) == {"shape": size, "dtype": dtype}, "unexpected cache array: " + key)
    with np.load(path, allow_pickle=False) as z:
        config, meta = json.loads(str(z["config_json"])), json.loads(str(z["meta_json"]))
        require(int(z["dim"]) == 384 and int(z["n_recordings"]) == pin["n_recordings"], "dimension/recording mismatch")
        require(bool(z["has_head"]) == (model == "B"), "head identity mismatch")
    require(config == prov["config"], "embedded config mismatch")
    for key, value in {"which": "student", "split": split, "code_sha": E.CODE_SHA,
                       "pose_stats_sha256": E.STATS_SHA, "window_s": 4.0, "seed": 20261026,
                       "checkpoint_pose_stats_sha256": E.STATS_SHA if model == "B" else None}.items():
        require(config.get(key) == value, "config identity mismatch: " + key)
    require(len(meta) == n, "metadata count mismatch")
    require(all(isinstance(m.get("user"), str) and m["user"] and m.get("n_patches") == 160 for m in meta), "invalid window metadata")
    users = sorted({m["user"] for m in meta})
    require(len(users) == pin["n_users"], "user count mismatch")
    require(states == {p: file_state(p) for p in states}, "artifact changed during validation")
    return {"model": model, "split": split, "cache_sha256": pin["sha256"], "headers": shape,
            "metadata_sha256": json_hash(meta), "users": users, "states": states,
            "scope": "hash/header/metadata only; payload finiteness checked by run"}


def lengths_for(data):
    values = [m["n_patches"] for m in data["meta"]]
    require(all(type(v) is int and 0 < v <= data["lat"].shape[1] for v in values), "invalid real patch lengths")
    return np.asarray(values, dtype=np.int64)


def central_mask(mask, lengths, offset=OFFSET):
    require(mask.ndim == 3 and mask.dtype == bool and mask.shape[1] > 2 * offset, "invalid target mask")
    require(len(lengths) == len(mask), "length/mask mismatch")
    t = np.arange(offset, mask.shape[1] - offset)
    return mask[:, offset:-offset].copy() & (t[None, :] + offset < lengths[:, None])[..., None]


def normalized_inputs(lat, lengths):
    """Only real EMG-patch availability defines input validity, never pose masks."""
    require(lat.ndim == 3 and lat.dtype.kind == "f" and len(lengths) == len(lat), "invalid latents")
    available = np.arange(lat.shape[1])[None, :] < np.asarray(lengths)[:, None]
    require(np.isfinite(lat[available]).all(), "nonfinite real input context")
    safe = np.where(available[..., None], lat, 0).astype(np.float32)
    with torch.no_grad():
        return torch.nn.functional.layer_norm(torch.from_numpy(safe), (lat.shape[-1],), eps=EPS).numpy()


def lag_features(normalized, reader, offset=OFFSET):
    require(reader in READERS and normalized.ndim == 3 and normalized.shape[1] > 2 * offset, "invalid temporal view")
    center = normalized[:, offset:-offset]
    return center if reader == "instantaneous" else np.concatenate((normalized[:, :-2*offset], center, normalized[:, 2*offset:]), axis=2)


def read_data(pin):
    path = Path(pin["path"])
    before = file_state(path)
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in ("lat", "target", "mask", "std", "pose_var")}
        data["meta"] = json.loads(str(z["meta_json"]))
    require(file_state(path) == before, "cache changed during reading")
    lat, y, mask = (data[k] for k in ("lat", "target", "mask"))
    require(lat.ndim == y.ndim == 3 and lat.shape[:2] == y.shape[:2] and y.shape == mask.shape and mask.dtype == bool,
            "array shape/mask mismatch")
    lengths = lengths_for(data)
    require(not np.any(mask & (np.arange(mask.shape[1])[None, :] >= lengths[:, None])[..., None]), "target mask includes padding")
    require(np.isfinite(y[mask]).all(), "nonfinite supported target")
    require(data["std"].shape == (len(y), y.shape[-1]) and np.isfinite(data["std"]).all() and (data["std"] > 0).all(), "invalid target scale")
    pv = data["pose_var"]
    require(pv.shape == (len(y),) and not np.isinf(pv).any() and (pv[np.isfinite(pv)] >= 0).all(), "invalid pose variance")
    data["lengths"], data["central_mask"] = lengths, central_mask(mask, lengths)
    for start in range(0, len(lat), BATCH):
        normalized_inputs(lat[start:start+BATCH], lengths[start:start+BATCH])
    return data


def target_signature(data):
    return {"arrays": arrays_hash(*(data[k] for k in ("target", "mask", "std", "pose_var"))),
            "metadata": json_hash(data["meta"]), "central_mask": arrays_hash(data["central_mask"])}


def grouped_stats(train, reader, batch=BATCH):
    users = np.asarray([m["user"] for m in train["meta"]])
    dim = train["lat"].shape[-1] * (3 if reader == "temporal" else 1)
    joints = train["target"].shape[-1]
    parts = {}
    for start in range(0, len(users), batch):
        sl = slice(start, start + batch)
        x = lag_features(normalized_inputs(train["lat"][sl], train["lengths"][sl]), reader)
        for user in sorted(set(users[sl])):
            pick = users[sl] == user
            stats = E.sufficient(x[pick].reshape(-1, dim), train["target"][sl][pick, OFFSET:-OFFSET].reshape(-1, joints),
                                 train["central_mask"][sl][pick].reshape(-1, joints))
            parts[user] = E.merge_stats([parts[user], stats], joints) if user in parts else stats
    return parts, dim, joints


def fit_reader(train, reader, expected_folds=None):
    started = perf_counter()
    parts, dim, joints = grouped_stats(train, reader)
    stat_seconds = perf_counter() - started
    actual_folds = [sorted(parts)[::2], sorted(parts)[1::2]]
    require(expected_folds is None or actual_folds == expected_folds, "training fold users differ")
    started = perf_counter()
    cv = E.cross_validate(parts, dim, joints)
    require(not cv["boundary_unresolved"], "alpha boundary unresolved; no scientific report")
    weights, bias = E.prepare_solver(E.merge_stats(list(parts.values()), joints), dim, joints)(cv["alpha"])
    return weights, bias, {"selection": cv, "n_features": dim, "n_users": len(parts),
                           "statistics_seconds": stat_seconds, "selection_refit_seconds": perf_counter() - started,
                           "fit_support_sha256": arrays_hash(train["central_mask"]),
                           "coefficients_sha256": arrays_hash(weights, bias)}


def predict(lat, lengths, reader, weights, bias, batch=BATCH):
    # Deliberately accepts no target, target mask, std or pose variance.
    result = np.empty((len(lat), lat.shape[1] - 2*OFFSET, len(bias)), dtype=np.float64)
    for start in range(0, len(lat), batch):
        sl = slice(start, start + batch)
        x = lag_features(normalized_inputs(lat[sl], lengths[sl]), reader)
        result[sl] = x.astype(np.float64) @ weights + bias
    require(np.isfinite(result).all(), "nonfinite prediction")
    return result


def summarize(data, prediction):
    """Original dynamic labels; moments recomputed on the shared central mask."""
    users = np.asarray([m["user"] for m in data["meta"]])
    pv, mask = data["pose_var"], data["central_mask"]
    require(prediction.shape == mask.shape, "prediction/support shape mismatch")
    selections = {"all": np.ones(len(pv), bool), "dynamic": np.isfinite(pv) & (pv >= THRESHOLD),
                  "static": np.isfinite(pv) & (pv < THRESHOLD)}
    accumulators, maes = {g: {} for g in selections}, {g: {} for g in selections}
    for start in range(0, len(users), BATCH):
        sl = slice(start, start + BATCH)
        y, p, m = data["target"][sl, OFFSET:-OFFSET], prediction[sl], mask[sl]
        moments = D.window_joint_moments(p, y, m)
        count = m.sum(axis=(1, 2))
        errors = (np.where(m, np.abs(p-y), 0) * data["std"][sl, None, :]).sum(axis=(1, 2))
        for group, selected in selections.items():
            D.accumulate(accumulators[group], moments, users[sl], selected[sl], 2, D.CENTERED_FIELDS)
            for user in sorted(set(users[sl][selected[sl] & (count > 0)])):
                ix = selected[sl] & (users[sl] == user) & (count > 0)
                row = maes[group].setdefault(user, {"error_sum_rad": 0., "n_valid_elements": 0})
                row["error_sum_rad"] += float(errors[ix].sum())
                row["n_valid_elements"] += int(count[ix].sum())
    result = {}
    for group in selections:
        summary = D.pooled_summary(accumulators[group], D.CENTERED_FIELDS)
        error = sum(v["error_sum_rad"] for v in maes[group].values())
        count = sum(v["n_valid_elements"] for v in maes[group].values())
        result[group] = {"centered": summary, "mae_rad": error/count if count else None,
                         "mae_per_user": maes[group], "n_selected_windows": int(selections[group].sum())}
    return {"groups": result, "support_sha256": arrays_hash(mask), "targets_sha256": target_signature(data),
            "dynamic_threshold": THRESHOLD, "prediction_sha256": arrays_hash(prediction)}


def paired_comparison(cells, n_boot=N_BOOT, seed=SEED):
    require(set(cells) == {f"{m}/{r}" for m in MODELS for r in READERS}, "need all four cells")
    reference = cells["B/instantaneous"]["analysis"]
    for cell in cells.values():
        require(cell["analysis"]["support_sha256"] == reference["support_sha256"] and
                cell["analysis"]["targets_sha256"] == reference["targets_sha256"], "paired metric support differs")
    contrasts = {"B_temporal_minus_instantaneous": {"B/temporal": 1., "B/instantaneous": -1.},
                 "R3c_temporal_minus_instantaneous": {"R3c/temporal": 1., "R3c/instantaneous": -1.},
                 "difference_of_differences": {"B/temporal": 1., "B/instantaneous": -1.,
                                                "R3c/temporal": -1., "R3c/instantaneous": 1.}}
    output = {}
    for group in ("dynamic", "all", "static"):
        rows = {key: value["analysis"]["groups"][group]["centered"]["per_user"] for key, value in cells.items()}
        users = sorted(rows["B/instantaneous"])
        require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), "paired users missing")
        base = rows["B/instantaneous"]
        denominator = np.asarray([base[u]["sums"]["vary"] for u in users])
        for values in rows.values():
            for user in users:
                require(all(values[user][key] == base[user][key] for key in ("n_windows", "n_window_joints", "n_valid_elements")),
                        "paired per-user support differs")
                require(values[user]["sums"]["vary"] == base[user]["sums"]["vary"], "paired target variance differs")
        draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
        den = denominator[draws].sum(1)
        require(np.isfinite(den).all() and (den > 0).all(), "invalid bootstrap denominator; no replicas discarded")
        group_result = {"users": users, "n_bootstrap": n_boot, "seed": seed, "contrasts": {}, "cell_ratio_ci95": {}}
        errors = {key: np.asarray([values[u]["sums"]["centered"] for u in users]) for key, values in rows.items()}
        for key, error in errors.items():
            group_result["cell_ratio_ci95"][key] = np.percentile(error[draws].sum(1)/den, [2.5, 97.5]).tolist()
        for label, coefficients in contrasts.items():
            numerator = sum(coef * errors[key] for key, coef in coefficients.items())
            values = numerator[draws].sum(1)/den
            require(np.isfinite(values).all(), "nonfinite bootstrap contrast")
            group_result["contrasts"][label] = {"delta_ratio": float(numerator.sum()/denominator.sum()),
                "ci95": np.percentile(values, [2.5, 97.5]).tolist(), "coefficients": coefficients,
                "per_user": {u: {"delta_centered_sse": float(v), "target_variance_sum": float(d)}
                             for u, v, d in zip(users, numerator, denominator)}}
        output[group] = group_result
    return output


def benchmark_train(train):
    """All train sufficient statistics/three solves, fixed alpha, no val cache."""
    result = {}
    for reader in READERS:
        started = perf_counter()
        parts, dim, joints = grouped_stats(train, reader)
        stat_seconds = perf_counter() - started
        users = sorted(parts)
        started = perf_counter()
        for selected in (users[::2], users[1::2], users):
            E.prepare_solver(E.merge_stats([parts[u] for u in selected], joints), dim, joints)(1e5)
        result[reader] = {"statistics_seconds": stat_seconds, "three_eigensystems_seconds": perf_counter()-started,
                          "n_features": dim, "n_users": len(users), "valid_fit_elements": int(train["central_mask"].sum()),
                          "one_gram_bytes": 8*dim*dim,
                          "note": "timing only; alpha fixed 1e5, no validation inference, no CV score"}
    return result


def run(plan):
    cells, coefficients, identities, previous = {}, {}, {}, None
    for model in MODELS:
        train = read_data(plan["caches"][model]["train"])
        val = read_data(plan["caches"][model]["val"])
        train_users, val_users = [{m["user"] for m in d["meta"]} for d in (train, val)]
        require(train_users.isdisjoint(val_users), "train/val user leakage")
        require(float(np.nanmedian(val["pose_var"])) == THRESHOLD, "original dynamic threshold changed")
        signatures = {"train": target_signature(train), "val": target_signature(val)}
        require(previous is None or signatures == previous, "B/R3c target/support mismatch")
        previous, identities[model] = signatures, signatures
        for reader in READERS:
            weights, bias, fit = fit_reader(train, reader, plan["fold_users"])
            start = perf_counter()
            prediction = predict(val["lat"], val["lengths"], reader, weights, bias)
            predict_seconds = perf_counter() - start
            key = f"{model}/{reader}"
            cells[key] = {"fit": fit, "prediction_seconds": predict_seconds, "analysis": summarize(val, prediction)}
            coefficients[f"{model}_{reader}_weights"], coefficients[f"{model}_{reader}_bias"] = weights, bias
            del prediction
        del train, val
    return {"status": "completed", "cells": cells, "paired": paired_comparison(cells), "target_identities": identities,
            "scope": "Exploratory reused evaluation users; fixed encoders and folds; no online claim, endpoint gate or automatic FM continuation."}, coefficients


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "benchmark", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--model", choices=MODELS, default="B", help="benchmark model only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    D.fresh_destination(args.out)
    coefficient_path = args.out.with_suffix(".coefficients.npz")
    if args.stage == "run":
        D.fresh_destination(coefficient_path)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    identity = runtime_identity(args.expected_code_sha)
    require(identity["sources"] == plan["sources"], "source pins mismatch")
    inspections = {}
    models = (args.model,) if args.stage == "benchmark" else MODELS
    splits = ("train",) if args.stage == "benchmark" else ("train", "val")
    for model in models:
        for split in splits:
            inspections[f"{model}/{split}"] = validate_artifact(plan["caches"][model][split], model, split)
    for model in models:
        require(inspections[f"{model}/train"]["users"] == sorted(sum(plan["fold_users"], [])), "train user identity mismatch")
        if "val" in splits:
            require(set(inspections[f"{model}/train"]["users"]).isdisjoint(inspections[f"{model}/val"]["users"]), "train/val leakage")
    if args.stage == "validate":
        result = {"status": "validated_hashes_headers_metadata", "scientific_metrics_computed": False}
    elif args.stage == "benchmark":
        result = {"status": "train_only_timing", "model": args.model,
                  "timings": benchmark_train(read_data(plan["caches"][args.model]["train"]))}
    else:
        result, coefficients = run(plan)
    require(args.plan.read_bytes() == plan_bytes and runtime_identity(args.expected_code_sha) == identity,
            "source/plan/numerical environment changed during execution")
    for item in inspections.values():
        require(item["states"] == {p: file_state(p) for p in item["states"]}, "input artifact changed during execution")
    result.update(protocol=PROTOCOL, recipe=recipe(), runtime=identity, plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                  inspections=inspections, input_pins=plan["caches"], slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        with coefficient_path.open("xb") as stream:
            np.savez_compressed(stream, **coefficients)
        result["coefficients"] = {"path": str(coefficient_path), "sha256": D.file_sha256(coefficient_path)}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.out.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
