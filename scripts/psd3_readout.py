#!/usr/bin/env python3
"""Prepare or evaluate PSD3 frozen readers. No scheduler, GPU, or FM training.

The committed plan is a preparation artifact. Execution follows the probe audit
and the measured resource plan in docs/psd3_readers_20261029.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness.downstream_ab import fingerprint
from wearusfm.harness.psd3_inputs import inspect_cache, load_cache_view, validate_roles
from wearusfm.harness.psd3_readers import (
    FeatureConfig, LogisticConfig, MLPConfig, prepare_features, fit_logistic, fit_mlp,
)

PROTOCOL = "psd3-frozen-readers-20261029-v1"
MODELS = ("B", "R3c")
DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6", "epn612", "uci_emg")
VIEWS = ("pool", "dec")


def recipe(feature_config=FeatureConfig(), logistic_config=LogisticConfig(), mlp_config=MLPConfig()):
    return json.loads(json.dumps({
        "protocol": PROTOCOL, "features": asdict(feature_config),
        "logistic": asdict(logistic_config), "mlp": asdict(mlp_config),
        "preprocessing": "train-only StandardScaler + randomized PCA; no whitening or second scaler; float64 fit/float32 coordinates",
        "selection": "validation subjects only; refit logistic on train only; retain selected MLP candidates",
        "unseen_classes": "drop val/held labels absent from fold train, identically for every cell",
        "metric": "OOF balanced accuracy; mean of seed-wise MLP scores, not ensemble",
        "bootstrap": {"unit": "subject", "replicates": 2000, "seed": 20261029},
        "views": list(VIEWS), "models": list(MODELS), "datasets": list(DATASETS),
    }))


def write_new(path, obj):
    """Never overwrite a result, including an incomplete/failed earlier run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)


def make_plan(archive):
    archive = Path(archive)
    entries = []
    for model in MODELS:
        for task in ("p1", "p2"):
            report = json.loads((archive / "reports" / f"{model}_{task}_pool.json").read_text())
            for dataset, item in report["datasets"].items():
                sidecar = archive / "extraction_metadata" / model / task / f"{dataset}.npz.provenance.json"
                prov = json.loads(sidecar.read_text())
                roles = [{k: fold[k] for k in ("train", "val", "held")} for fold in item["folds"]]
                validate_roles(roles, item["subjects"])
                pins = {k: prov[k] for k in ("checkpoint_sha256", "input_sha256", "splits_sha256",
                        "extraction_recipe", "code_sha", "n_windows", "n_subjects")}
                pins.update(bytes=prov["cache_bytes"], sha256=prov["cache_sha256"],
                            encoder_dim=384, channels=item["readout"]["channels"],
                            protocol="readout_per_canale_v1", feature_set="channel_readout", task=task)
                entries.append({"model": model, "dataset": dataset, "task": task,
                    "path": prov["cache"], "pins": pins, "roles": roles,
                    "subjects": item["subjects"], "roles_sha256": fingerprint(roles),
                    "expected_oof_rows_sha256": item["oof_rows_sha256"],
                    "expected_oof_windows": item["n_oof_windows"],
                    "historical_train_windows": [f["n_train"] for f in item["folds"]],
                    "source_sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()})
    plan = {"protocol": PROTOCOL, "status": "prepared_not_launched", "recipe": recipe(),
            "evidence": "archived B19618 sidecars/reports; cache hashes not remeasured by plan generation",
            "entries": entries, "shards": 20, "transforms": 80,
            "logistic_fits": 560, "mlp_fits": 480, "gpu_hours": 0}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get("protocol") != PROTOCOL or plan.get("recipe") != recipe():
        raise ValueError("plan recipe differs from the frozen defaults")
    entries = plan["entries"]
    keys = [(e["model"], e["dataset"]) for e in entries]
    if len(keys) != 10 or set(keys) != {(m, d) for m in MODELS for d in DATASETS}:
        raise ValueError("plan must contain exactly ten distinct caches")
    for e in entries:
        validate_roles(e["roles"], e["subjects"])
        if len(e["roles"]) != 4 or fingerprint(e["roles"]) != e["roles_sha256"]:
            raise ValueError("invalid frozen fold roles")
    for dataset in DATASETS:
        a, b = [e for e in entries if e["dataset"] == dataset]
        for key in ("input_sha256", "splits_sha256", "extraction_recipe", "n_windows", "n_subjects"):
            if a["pins"][key] != b["pins"][key]:
                raise ValueError(f"unpaired plan for {dataset}: {key}")
        for key in ("roles_sha256", "subjects", "expected_oof_rows_sha256", "expected_oof_windows"):
            if a[key] != b[key]:
                raise ValueError(f"unpaired plan for {dataset}: {key}")


def source_identity():
    files = [Path(__file__), *(ROOT / "src/wearusfm/harness" / name for name in
              ("psd3_inputs.py", "psd3_readers.py", "psd3_compare.py", "downstream_ab.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    import sklearn
    import scipy
    import torch
    from threadpoolctl import threadpool_info
    return {"source_sha256": fingerprint(hashes), "files": hashes,
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "sklearn": sklearn.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "torch_threads": torch.get_num_threads(),
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads",
                                                  "architecture", "threading_layer")} for p in threadpool_info()]}


def numerical_identity(identity):
    return {k: identity.get(k) for k in ("source_sha256", "versions", "threads", "torch_threads", "threadpools")}


def inventory(plan, verify_sha=False):
    caches = []
    for entry in plan["entries"]:
        info = inspect_cache(entry["path"], compute_sha=verify_sha)
        pins = entry["pins"]
        if info["bytes"] != pins["bytes"] or (verify_sha and info["sha256"] != pins["sha256"]):
            raise ValueError("inventory byte/hash mismatch")
        for key in info["metadata"].keys() & pins.keys():
            if info["metadata"][key] != pins[key]:
                raise ValueError(f"inventory metadata mismatch: {key}")
        for key in ("checkpoint_sha256", "input_sha256", "splits_sha256", "code_sha", "extraction_recipe"):
            if info["metadata"].get(key) != pins[key]:
                raise ValueError(f"inventory missing/different identity: {key}")
        for view, width, dtype in (("pool", 2*pins["encoder_dim"], "<f4"),
                                  ("dec", pins["channels"]*pins["encoder_dim"], "<f2")):
            arr = info["arrays"].get("x_"+view, {})
            if arr.get("shape") != [pins["n_windows"], width] or arr.get("dtype") != dtype:
                raise ValueError(f"inventory unexpected {view} shape/dtype")
        info.update(model=entry["model"], dataset=entry["dataset"])
        caches.append(info)
    return {"status": "hash_and_header_verified" if verify_sha else "header_only_pins_verified_no_hash",
            "feature_payload_finiteness_checked": False, "caches": caches}


def fold_indices(y, s, roles):
    """Fixed subject roles plus the historical train-class eligibility rule."""
    result = []
    for role in roles:
        train, val, held = [np.flatnonzero(np.isin(s, role[k])) for k in ("train", "val", "held")]
        classes = np.unique(y[train])
        keep_v, keep_h = np.isin(y[val], classes), np.isin(y[held], classes)
        if min(len(train), int(keep_v.sum()), int(keep_h.sum())) == 0 or len(classes) < 2:
            raise ValueError("empty role after train-class eligibility filtering")
        result.append((train, val[keep_v], held[keep_h],
                       {"val": int((~keep_v).sum()), "held": int((~keep_h).sum())}))
    return result


def run_shard(entry, view, *, feature_config=FeatureConfig(), logistic_config=LogisticConfig(),
              mlp_config=MLPConfig(), progress=None):
    """Configs are injectable for unit tests; CLI exposes only the fixed recipe."""
    identity = source_identity()
    actual_recipe = recipe(feature_config, logistic_config, mlp_config)
    x, y, s, provenance = load_cache_view(entry["path"], entry["pins"], view)
    validate_roles(entry["roles"], s.tolist())
    folds = fold_indices(y, s, entry["roles"])
    rows = np.concatenate([f[2] for f in folds]).astype(np.int64)
    if len(np.unique(rows)) != len(rows) or len(rows) != entry["expected_oof_windows"]:
        raise ValueError("unexpected OOF row support")
    if fingerprint(rows) != entry["expected_oof_rows_sha256"]:
        raise ValueError("OOF row order/support differs from the historical reference")
    provenance["oof_data_sha256"] = fingerprint(rows.tolist(), y[rows].tolist(), s[rows].tolist())
    output = {"logistic": {}, "mlp": {}}
    diagnostics = []
    for i, (train, val, held, dropped) in enumerate(folds):
        started = perf_counter()
        feat = prepare_features(x[train], x[val], x[held], config=feature_config)
        diagnostic = {"fold": i, "roles": entry["roles"][i], "dropped_windows": dropped,
                      "preprocessing": feat.metadata, "preprocessing_seconds": perf_counter() - started}
        for name, fit, cfg in (("logistic", fit_logistic, logistic_config), ("mlp", fit_mlp, mlp_config)):
            result = fit(feat.train, y[train], feat.val, y[val], feat.held, config=cfg)
            predictions = result.pop("predictions")
            for seed, prediction in predictions.items():
                output[name].setdefault(seed, []).extend(prediction)
            diagnostic[name] = result
            if progress:
                progress({"fold": i, "reader": name, "result": result})
        diagnostics.append(diagnostic)
    if source_identity() != identity:
        raise RuntimeError("source or numerical environment changed during shard")
    # Bind the comparison recipe to actual source contents and library versions.
    return {"protocol": PROTOCOL, "recipe": actual_recipe,
            "recipe_sha256": fingerprint(actual_recipe, numerical_identity(identity)),
            "model": entry["model"], "dataset": entry["dataset"], "task": entry["task"], "view": view,
            "input_sha256": entry["pins"]["input_sha256"], "roles_sha256": entry["roles_sha256"],
            "checkpoint_sha256": entry["pins"]["checkpoint_sha256"], "provenance": provenance,
            "runtime": identity, "rows": rows.tolist(), "y": y[rows].tolist(), "subjects": s[rows].tolist(),
            "predictions": output, "folds": diagnostics, "status": "completed"}


def benchmark(entry, view):
    """Timing only: first train fold, no validation/held rows or scientific scores.

    Full train PCA, two C=1 fits, one 5-epoch MLP fit. Dummy evaluation rows are
    copied from train and scores are discarded. Projection is not a safe bound:
    C=100 may take longer than C=1 and resource assignment needs a margin.
    """
    identity = source_identity()
    x, y, s, provenance = load_cache_view(entry["path"], entry["pins"], view)
    idx = np.flatnonzero(np.isin(s, entry["roles"][0]["train"]))
    xt, yt = x[idx], y[idx]
    started = perf_counter()
    feat = prepare_features(xt, xt[:1], xt[:1])
    pca_seconds = perf_counter() - started
    log = fit_logistic(feat.train, yt, feat.train, yt, feat.held,
                       config=replace(LogisticConfig(), c_grid=(1.0,)))
    mlp = fit_mlp(feat.train, yt, feat.train, yt, feat.held,
                  config=replace(MLPConfig(), epochs=5, lambdas=(1e-3,), seeds=(0,)))
    logistic_seconds = sum(r["duration_s"] for r in log["audit"])
    mlp_seconds = sum(r["duration_s"] for r in mlp["audit"])
    if source_identity() != identity:
        raise RuntimeError("source or numerical environment changed during benchmark")
    return {"protocol": PROTOCOL, "status": "timing_only_not_scientific_result", "model": entry["model"],
            "dataset": entry["dataset"], "view": view, "fold": 0, "n_train": len(idx),
            "provenance": provenance, "runtime": identity, "pca_seconds": pca_seconds,
            "two_C1_fits_seconds": logistic_seconds, "one_5epoch_mlp_seconds": mlp_seconds,
            "rough_full_fold_seconds": pca_seconds + 3.5 * logistic_seconds + 72 * mlp_seconds,
            "caveat": "projection, not upper bound; difficult C values and folds may be slower",
            "fit_warnings": [a.get("warnings", []) for a in log["audit"] + mlp["audit"]]}


def validate_report(report, entry, plan_sha):
    """Require the complete frozen experiment, not merely plausible predictions."""
    json.dumps(report, allow_nan=False)
    if report.get("status") != "completed" or report.get("protocol") != PROTOCOL:
        raise ValueError("incomplete shard")
    if report.get("recipe") != recipe() or report.get("plan_sha256") != plan_sha:
        raise ValueError("shard recipe/plan mismatch")
    runtime = report["runtime"]
    if runtime["source_sha256"] != fingerprint(runtime["files"]):
        raise ValueError("source fingerprint mismatch")
    if report["recipe_sha256"] != fingerprint(report["recipe"], numerical_identity(runtime)):
        raise ValueError("numerical recipe fingerprint mismatch")
    for key in ("input_sha256", "checkpoint_sha256"):
        if report[key] != entry["pins"][key]:
            raise ValueError(f"shard pin mismatch: {key}")
    if report["roles_sha256"] != entry["roles_sha256"] or report["task"] != entry["task"]:
        raise ValueError("shard roles/task mismatch")
    provenance = report["provenance"]
    view = report["view"]
    if view not in VIEWS or provenance["view"] != view:
        raise ValueError("view/provenance mismatch")
    pins = entry["pins"]
    if "encoder_dim" in pins and "channels" in pins:
        width = 2*pins["encoder_dim"] if view == "pool" else pins["channels"]*pins["encoder_dim"]
        dtype = "<f4" if view == "pool" else "<f2"
        if provenance["shape"] != [pins["n_windows"], width] or provenance["dtype"] != dtype:
            raise ValueError("view shape/dtype mismatch")
    for key in ("sha256", "bytes"):
        if provenance[key] != entry["pins"][key]:
            raise ValueError("cache provenance mismatch")
    for key in ("checkpoint_sha256", "input_sha256", "splits_sha256", "code_sha", "extraction_recipe"):
        if provenance["metadata"][key] != entry["pins"][key]:
            raise ValueError("cache metadata mismatch")
    rows = np.asarray(report["rows"], dtype=np.int64)
    if (len(rows) != entry["expected_oof_windows"] or
            fingerprint(rows) != entry["expected_oof_rows_sha256"]):
        raise ValueError("report OOF support differs from frozen reference")
    subjects = np.asarray(report["subjects"])
    validate_roles(entry["roles"], report["subjects"])
    if len(report["y"]) != len(rows) or len(subjects) != len(rows):
        raise ValueError("OOF labels/subjects length mismatch")
    if provenance["oof_data_sha256"] != fingerprint(report["rows"], report["y"], report["subjects"]):
        raise ValueError("OOF labels/subjects changed after cache loading")
    if len(rows) == provenance["shape"][0]:
        order = np.argsort(rows)
        if (not np.array_equal(rows[order], np.arange(len(rows))) or
                provenance["canonical_rows_sha256"] != fingerprint(rows[order].tolist(),
                    np.asarray(report["y"])[order].tolist(), subjects[order].tolist())):
            raise ValueError("complete OOF data differs from loaded cache")
    if len(report["folds"]) != 4:
        raise ValueError("need four completed folds")
    for i, fold in enumerate(report["folds"]):
        if fold["fold"] != i or fold["roles"] != entry["roles"][i]:
            raise ValueError("fold role mismatch")
        prep = fold["preprocessing"]
        if prep["config"] != recipe()["features"] or prep["fit_role"] != "train" or prep["second_scaler"]:
            raise ValueError("unexpected preprocessing")
        held = np.isin(subjects, fold["roles"]["held"])
        if int(held.sum()) != prep["input_shapes"][2][0]:
            raise ValueError("held subject support differs from preprocessing")
        train_fingerprints = []
        for reader, expected_fits, seeds in (("logistic", 7, {"0"}), ("mlp", 6, {"0", "1"})):
            result = fold[reader]
            if (result.get("status") != "completed" or result["config"] != recipe()[reader] or
                    len(result["audit"]) != expected_fits or set(report["predictions"][reader]) != seeds):
                raise ValueError("incomplete or different reader configuration")
            for audit in result["audit"]:
                if audit["status"] != "completed" or audit["n_features"] != 128:
                    raise ValueError("incomplete fit audit")
                train_fingerprints.append(audit["train_fingerprint"])
            if reader == "logistic":
                candidates = [a["C"] for a in result["audit"] if a["phase"] == "candidate"]
                refits = [a["C"] for a in result["audit"] if a["phase"] == "refit"]
                if candidates != recipe()[reader]["c_grid"] or refits != [result["selected_hyperparameter"]]:
                    raise ValueError("unexpected logistic selection/refit audit")
            else:
                expected = {(v, seed) for v in recipe()[reader]["lambdas"] for seed in (0, 1)}
                actual = {(a["lambda"], a["seed"]) for a in result["audit"]}
                if actual != expected or any(len(a["epochs"]) != 60 for a in result["audit"]):
                    raise ValueError("unexpected MLP candidate/epoch audit")
            for predictions in report["predictions"][reader].values():
                pred = np.asarray(predictions)
                if len(pred) != len(subjects) or not np.isin(pred[held], result["classes"]).all():
                    raise ValueError("prediction outside the actual fold train classes")
        if len(set(train_fingerprints)) != 1:
            raise ValueError("readers did not use identical train coordinates/labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "inventory", "benchmark", "run", "compare"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "results/infra/pilot_posa/b19618_20261029/final")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--reports", type=Path, nargs="+")
    parser.add_argument("--verify-sha", action="store_true", help="inventory only: also read/hash complete cache bytes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".progress.jsonl").exists():
        raise FileExistsError(args.out)
    if args.stage == "plan":
        result = make_plan(args.archive)
    elif args.stage == "compare":
        from wearusfm.harness.psd3_compare import compare_shards
        if not args.reports or not args.plan:
            parser.error("compare requires --plan and --reports (all twenty shards)")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        shards = [json.loads(p.read_text()) for p in args.reports]
        for shard in shards:
            matches = [e for e in plan["entries"] if (e["model"], e["dataset"]) == (shard["model"], shard["dataset"])]
            if len(matches) != 1:
                raise ValueError("unknown shard identity")
            validate_report(shard, matches[0], plan_sha)
        result = compare_shards(shards)
        result["plan_sha256"] = plan_sha
        result["report_files_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports}
    else:
        if not args.plan:
            parser.error("this stage requires --plan")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        if args.stage == "inventory":
            result = inventory(plan, args.verify_sha)
            result["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        else:
            if not all((args.model, args.dataset, args.view)):
                parser.error("benchmark/run require --model --dataset --view")
            entry = next(e for e in plan["entries"] if e["model"] == args.model and e["dataset"] == args.dataset)
            if args.stage == "benchmark":
                result = benchmark(entry, args.view)
            else:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.with_suffix(".progress.jsonl").open("x") as stream:
                    def progress(row):
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                    try:
                        result = run_shard(entry, args.view, progress=progress)
                    except Exception as exc:
                        progress({"status": "failed", "exception": repr(exc), "audit": getattr(exc, "psd3_audit", [])})
                        raise
            result["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    write_new(args.out, result)


if __name__ == "__main__":
    main()
