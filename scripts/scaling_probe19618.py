"""19618-step 30M/300M classification probes; no job submission.

Derived from scaling_probe.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py; the 30M control caches are the reused tokenizer-campaign extraction.

Numerical folds use tokenizer_probe.fit_fold unchanged. Three unique cells:
30M-natural, 300M-natural and 300M-matched; 30M-matched is an explicit alias.
Train-only benchmarks do not select hyperparameters or inspect held-out scores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tokenizer_probe as T
import scaling_evaluation19618 as E
import scaling_projection as SP
import scaling_train as ST

PROTOCOL = "scaling-probe-19618-20261106-v1"
RESOURCE_PROTOCOL = "scaling-probe-resources-19618-20261106-v1"
STEP = E.STEP
ARMS, TASKS, DATASETS, WEIGHTS = E.ARMS, T.TASKS, T.DATASETS, T.WEIGHTS
CELLS = {"T100-G-30M-natural": (ARMS[0], "natural"),
         "T100-G-300M-natural": (ARMS[1], "natural"),
         "T100-G-300M-matched": (ARMS[1], "matched")}
ALIASES = {"T100-G-30M-matched": "T100-G-30M-natural"}
require, sha, jhash, labels = T.require, T.sha, T.jhash, T.labels
fit_fold, AuditLog, audit_summary = T.fit_fold, T.AuditLog, T.audit_summary
C_GRID, MAX_ITER = T.C_GRID, T.MAX_ITER


def width(cell):
    require(cell in CELLS, "unregistered cell; 30M matched is an alias, not a new fit")
    arm, readout = CELLS[cell]
    return 768 if readout == "matched" else 2 * E.dimension(arm)


def recipe():
    value = T.recipe()
    value.pop("width")
    return {**value, "checkpoint_step": STEP, "widths": {cell: width(cell) for cell in CELLS},
            "cells": {c: {"arm": a, "readout": r} for c, (a, r) in CELLS.items()}, "aliases": ALIASES,
            "primary": "natural", "secondary": "fixed independent backbone/local QR projections before train-only StandardScaler"}


def source_hashes():
    paths = {Path(__file__), Path(T.__file__), Path(E.__file__), Path(SP.__file__), Path(ST.__file__),
             Path(ST.S.__file__), Path(T.CP.__file__), Path(T.CR.__file__), Path(T.DA.__file__), Path(T.P1.__file__),
             Path(E.E.__file__), Path(E.Q.__file__), Path(E.I.__file__)}
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def make_plan(caches, datasets, references, splits, source_contract, projection):
    plan = {"protocol": PROTOCOL, "step": STEP, "recipe": recipe(), "sources": source_hashes(),
            "caches": caches, "datasets": datasets, "references": references, "splits": splits,
            "training_source_contract": source_contract, "projection": projection}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL and plan.get("step") == STEP, "wrong protocol/endpoint")
    require(plan.get("recipe") == recipe() and plan.get("sources") == source_hashes(), "recipe/source pins differ")
    contract_state = ST.resolve_pin(plan["training_source_contract"])
    context = ST.check_source_contract(contract_state["path"])
    ST.resolve_pin(plan["splits"])
    projection = SP.load(plan["projection"])
    require(set(plan.get("datasets", {})) == set(DATASETS) and set(plan.get("references", {})) == set(DATASETS), "need all five datasets/references")
    require(set(plan.get("caches", {})) == set(ARMS), "need both arms")
    splits = json.loads(Path(plan["splits"]["path"]).read_text())
    for ds, entry in plan["datasets"].items():
        subs = entry.get("subjects", [])
        require(isinstance(subs, list) and len(subs) >= 4 and all(isinstance(s, str) and s for s in subs)
                and subs == sorted(set(subs)), "invalid subjects")
        task = next(t for t in TASKS if ds in TASKS[t])
        require(entry.get("roles_sha256") == jhash(T.CP.fold_roles(subs)), "fold roles mismatch")
        require(entry.get("task") == task and entry.get("extraction_recipe") == T.CR.recipe(task), "dataset recipe/task mismatch")
        require(ST.digest(entry.get("input_sha256")), "invalid dataset input pin")
        expected_splits = plan["splits"]["sha256"] if task == "p1" else T.CP.P2_SPLITS_SHA256
        require(entry.get("splits_sha256") == expected_splits, "split identity differs")
        if task == "p1":
            require(T.CP.pretraining_subjects(splits, ds) == subs, "official P1 test subjects not excluded")
        ST.pin_schema(plan["references"][ds])
    paths, checkpoints = [], []
    for arm in ARMS:
        require(set(plan["caches"][arm]) == set(DATASETS), "incomplete cache grid")
        cps = set()
        for ds, pin in plan["caches"][arm].items():
            require(all(Path(pin.get(k, "")).is_absolute() for k in ("path", "receipt_path", "provenance_path", "checkpoint_path")), "absolute input paths required")
            require(all(ST.digest(pin.get(k)) for k in ("sha256", "receipt_sha256", "provenance_sha256", "checkpoint_sha256", "extraction_plan_sha256")), "invalid cache pins")
            require(ST.digest(pin.get("extraction_code_sha"), 40), "invalid extraction code SHA")
            for k in ("input_sha256", "splits_sha256", "extraction_recipe"):
                require(pin.get(k) == plan["datasets"][ds][k], "cache dataset identity differs")
            paths.append(str(Path(pin["path"]).resolve()))
            cps.add(pin["checkpoint_sha256"])
        require(len(cps) == 1, "mixed checkpoints within arm")
        if arm == ARMS[0]:
            require(cps == {E.CONTROL_CHECKPOINT_SHA256}, "wrong registered 30M 19618 control checkpoint")
        checkpoints.extend(cps)
    require(len(set(paths)) == len(paths) and len(set(checkpoints)) == 2, "duplicate cache/checkpoint across arms")
    return context, projection


def load_cache(plan, arm, ds, features=True, context=None):
    require(arm in ARMS and ds in DATASETS, "invalid cache arm/dataset")
    context = context or ST.check_source_contract(ST.resolve_pin(plan["training_source_contract"])["path"])
    pin, expected = plan["caches"][arm][ds], plan["datasets"][ds]
    states = {k: ST.resolve_pin({"path": pin[p], "sha256": pin[k]}) for p, k in
              (("path", "sha256"), ("receipt_path", "receipt_sha256"), ("provenance_path", "provenance_sha256"))}
    prov = json.loads(Path(pin["provenance_path"]).read_text())
    receipt = json.loads(Path(pin["receipt_path"]).read_text())
    common = {"protocol": E.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": STEP,
              "checkpoint_sha256": pin["checkpoint_sha256"], "postload_recipe_passed": True,
              "code_sha": pin["extraction_code_sha"], "plan_sha256": pin["extraction_plan_sha256"],
              "training_source_contract_sha256": plan["training_source_contract"]["sha256"]}
    if arm == ARMS[0]:  # reused tokenizer-campaign extraction of the control: its own identity, no scaling contract
        common = {**{k: v for k, v in common.items() if k != "training_source_contract_sha256"},
                  "protocol": E.E.PROTOCOL, "arm": "T100-G"}
    for k, v in common.items():
        require(receipt.get(k) == v and prov.get(k) == v, "teacher/provenance identity mismatch: " + k)
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True and
            receipt.get("stage") == "extract-classification", "incomplete/wrong extraction receipt")
    expected_model = ST.R.config_to_dict(context["config_30m" if arm == ARMS[0] else "config_300m"])["model"]
    for value in (receipt, prov):
        E.model_recipe(value.get("model_config", {}), arm)
        require(value["model_config"] == expected_model, "extraction model config differs")
    output = receipt.get("outputs", {}).get(ds, {})
    require(output.get("path") == pin["path"] and output.get("sha256") == pin["sha256"] and
            output.get("provenance_path") == pin["provenance_path"] and output.get("provenance_sha256") == pin["provenance_sha256"], "receipt output mismatch")
    require(prov.get("cache") == pin["path"] and prov.get("cache_sha256") == pin["sha256"], "provenance cache mismatch")
    details = E.verify_classification_cache(pin["path"], plan["references"][ds], ds, pin["checkpoint_path"],
        pin["checkpoint_sha256"], pin["extraction_code_sha"], plan["splits"], arm)
    require(all(prov.get(k) == v for k, v in details.items()), "extraction validation details differ")
    for key in ("input_sha256", "splits_sha256", "extraction_recipe"):
        require(prov[key] == pin[key], "provenance dataset differs")
    with np.load(pin["path"], allow_pickle=False) as z:
        y, subjects = labels(z["y"], "targets"), labels(z["s"], "subjects").astype(str)
        require(sorted(set(subjects)) == expected["subjects"], "subject coverage mismatch")
        x = z["x_pool"] if features else None
    if features:
        require(x.shape == (len(y), 2 * E.dimension(arm)) and x.dtype == np.float32 and np.isfinite(x).all(), "invalid pool features")
    E.P.check_unchanged(states)
    return x, y, subjects


def cell_features(plan, cell, ds, context=None, projection=None):
    width(cell)
    arm, readout = CELLS[cell]
    x, y, subjects = load_cache(plan, arm, ds, context=context)
    if readout == "matched":
        x = SP.project_pool(x, projection or SP.load(plan["projection"]))
    require(x.dtype == np.float32 and x.shape == (len(y), width(cell)) and np.isfinite(x).all(), "transformed feature width/precision invalid")
    return x, y, subjects


def completion_identity(plan, identity):
    validate_plan(plan)
    require(ST.digest(identity.get("code_sha"), 40) and ST.digest(identity.get("plan_sha256")), "invalid execution identity")
    if "plan_path" in identity:
        require(sha(identity["plan_path"]) == identity["plan_sha256"], "plan file changed")
        require(E.P.code_identity(identity["code_sha"]) == identity["code_sha"], "execution code changed")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def benchmark(plan, cell, ds, out, identity):
    context, projection = validate_plan(plan)
    arm, _ = CELLS[cell]
    T.fresh(out)
    out = Path(out)
    out.mkdir(parents=True)
    audit = AuditLog(out / "audit_progress.jsonl")
    numeric, started = T.numeric_context(), time.perf_counter()
    try:
        x, y, subjects = cell_features(plan, cell, ds, context, projection)
        roles = T.CP.fold_roles(plan["datasets"][ds]["subjects"])
        train = np.isin(subjects, roles[0]["train"])
        # No validation/held labels, features, scores or C selection enter fits.
        scaler = T.StandardScaler().fit(x[train])
        xt, yt = scaler.transform(x[train]), y[train]
        classes, counts = np.unique(yt, return_counts=True)
        for c in C_GRID:
            model = T.LogisticRegression(C=c, max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4)
            record = {"cell": cell, "arm": arm, "dataset": ds, "fold": 0, "fit_phase": "benchmark_train_only",
                      "C": c, "n_train": len(yt), "n_features": width(cell), "dtype": str(xt.dtype),
                      "classes": classes.tolist(), "class_counts": counts.tolist(), "solver_params": model.get_params(),
                      "roles_sha256": jhash(roles[0]), "max_iter": MAX_ITER, "selected": None,
                      "scaler_sha256": T.DA.fingerprint(scaler.mean_, scaler.scale_)}
            audit.append(record)
            began = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model.fit(xt, yt)
                except BaseException as exc:
                    record.update(exception=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter()-began,
                                  warnings=T.described(caught))
                    audit.persist(record)
                    raise
            count = sum(w.category.__name__ == "ConvergenceWarning" for w in caught)
            n = int(np.max(model.n_iter_))
            record.update(exception=None, elapsed_s=time.perf_counter()-began, warnings=T.described(caught), n_iter=n,
                          convergence_warnings=count, reached_limit=n >= MAX_ITER,
                          converged_by_criterion=count == 0 and n < MAX_ITER)
            audit.persist(record)
            T.reemit(caught)
        load_cache(plan, arm, ds, features=False, context=context)
        require(numeric == T.numeric_context(), "numeric context changed")
        completion_identity(plan, identity)
        result = {"protocol": PROTOCOL, "kind": "benchmark", "complete": True, "cell": cell, "arm": arm, "dataset": ds,
                  "step": STEP, "recipe": recipe(), "plan_content_sha256": jhash(plan), **identity,
                  "train_only": True, "selection_performed": False, "fold": 0, "roles_sha256": jhash(roles[0]),
                  "n_train": len(yt), "fold_train_counts": [int(np.isin(subjects, r["train"]).sum()) for r in roles],
                  "n_features": width(cell), "fits": list(audit), "audit_summary": audit_summary(audit),
                  "elapsed_seconds": time.perf_counter()-started, "peak_rss_bytes": peak_rss_bytes(),
                  "numeric_context": numeric, "numeric_sha256": jhash(numeric)}
        T.write_json(out / "benchmark.json", result)
        return result
    except BaseException as exc:
        T.write_json(out / "failed.json", {"complete": False, "exception": f"{type(exc).__name__}: {exc}", **identity})
        raise


def validate_resources(plan, cell, task, datasets, folds, pin, identity):
    state = ST.resolve_pin(pin)
    budget = json.loads(Path(state["path"]).read_text())
    require(budget.get("protocol") == RESOURCE_PROTOCOL and budget.get("complete") is True and
            budget.get("plan_content_sha256") == jhash(plan), "resource plan mismatch/incomplete")
    require(budget.get("code_sha") == identity["code_sha"], "resource code differs")
    wanted = {"cell": cell, "task": task, "datasets": datasets, "folds": folds}
    selected = [s for s in budget.get("shards", []) if all(s.get(k) == v for k, v in wanted.items())]
    require(len(selected) == 1, "shard missing/duplicate in resource plan")
    allocation = selected[0]
    caps = budget.get("caps", {})
    require(type(caps.get("cpus")) is int and caps["cpus"] > 0 and
            type(caps.get("walltime_seconds")) is int and caps["walltime_seconds"] > 0 and
            isinstance(caps.get("memory_gib"), (int, float)) and np.isfinite(caps["memory_gib"]) and caps["memory_gib"] > 0,
            "resource plan needs positive published CPU/RAM/time caps")
    require(type(allocation.get("cpus")) is int and 0 < allocation["cpus"] <= caps["cpus"] and
            type(allocation.get("walltime_seconds")) is int and 0 < allocation["walltime_seconds"] <= caps["walltime_seconds"] and
            isinstance(allocation.get("memory_gib"), (int, float)) and 0 < allocation["memory_gib"] <= caps["memory_gib"],
            "resource allocation exceeds published caps")
    needed, memory = 120., 0
    for ds in datasets:
        bp = budget.get("benchmarks", {}).get(cell, {}).get(ds)
        st = ST.resolve_pin(bp)
        b = json.loads(Path(st["path"]).read_text())
        require(b.get("protocol") == PROTOCOL and b.get("kind") == "benchmark" and b.get("complete") is True and
                b.get("plan_content_sha256") == jhash(plan) and b.get("recipe") == recipe() and b.get("step") == STEP and
                b.get("code_sha") == identity["code_sha"] and b.get("plan_sha256") == identity["plan_sha256"] and
                b.get("cell") == cell and b.get("dataset") == ds and b.get("train_only") is True and
                b.get("selection_performed") is False and b.get("fold") == 0 and b.get("n_features") == width(cell), "benchmark identity differs")
        require(b["numeric_context"] == T.numeric_context() and b["numeric_sha256"] == jhash(b["numeric_context"]), "benchmark numeric context differs")
        fits = b["fits"]
        require(len(fits) == len(C_GRID) and [f["C"] for f in fits] == list(C_GRID) and
                audit_summary(fits) == b["audit_summary"] and all(f["fit_phase"] == "benchmark_train_only" and
                f.get("selected") is None and np.isfinite(f["elapsed_s"]) and f["elapsed_s"] > 0 for f in fits), "benchmark fit coverage differs")
        roles = T.CP.fold_roles(plan["datasets"][ds]["subjects"])
        _, targets, subjects = load_cache(plan, CELLS[cell][0], ds, features=False)
        train = np.isin(subjects, roles[0]["train"])
        classes, class_counts = np.unique(targets[train], return_counts=True)
        for f in fits:
            expected_solver = T.LogisticRegression(C=f["C"], max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4).get_params()
            require(f.get("solver_params") == expected_solver and f.get("max_iter") == MAX_ITER and
                    f.get("cell") == cell and f.get("arm") == CELLS[cell][0] and f.get("dataset") == ds and
                    f.get("fold") == 0 and f.get("roles_sha256") == jhash(roles[0]) and f.get("n_features") == width(cell) and
                    f.get("dtype") == "float32" and f.get("n_train") == int(train.sum()) and
                    f.get("classes") == classes.tolist() and f.get("class_counts") == class_counts.tolist(),
                    "benchmark solver/train identity differs")
        counts = [int(np.isin(subjects, r["train"]).sum()) for r in roles]
        require(b["fold_train_counts"] == counts and b["n_train"] == counts[0] and b["roles_sha256"] == jhash(roles[0]), "benchmark train support differs")
        per_fold = sum(f["elapsed_s"] for f in fits) + max(f["elapsed_s"] for f in fits)
        needed += 1.5 * per_fold * sum(max(1., counts[f]/counts[0]) for f in folds)
        require(type(b.get("peak_rss_bytes")) is int and b["peak_rss_bytes"] > 0, "invalid benchmark memory")
        memory = max(memory, b["peak_rss_bytes"])
    require(allocation["walltime_seconds"] >= needed and allocation["memory_gib"] * 2**30 >= memory * 1.3,
            "measured benchmark exceeds published shard budget")
    return {"resource_plan": state, "allocation": allocation, "estimated_seconds_with_margin": needed,
            "observed_peak_rss_bytes": memory}


def shard_selection(cell, task, datasets, folds):
    width(cell)
    require(task in TASKS and datasets and len(set(datasets)) == len(datasets) and set(datasets) <= set(TASKS[task]), "invalid shard datasets")
    require(folds and folds == sorted(set(folds)) and all(type(f) is int and f in range(4) for f in folds), "invalid shard folds")


def run_shard(plan, cell, task, datasets, folds, out, identity, resource_plan):
    context, projection = validate_plan(plan)
    shard_selection(cell, task, datasets, folds)
    budget = validate_resources(plan, cell, task, datasets, folds, resource_plan, identity)
    arm, readout = CELLS[cell]
    T.fresh(out)
    out = Path(out)
    out.mkdir(parents=True)
    audit, arrays, reports = AuditLog(out / "audit_progress.jsonl"), {}, {}
    numeric = T.numeric_context()
    try:
        for ds in datasets:
            x, y, subjects = cell_features(plan, cell, ds, context, projection)
            roles = T.CP.fold_roles(plan["datasets"][ds]["subjects"])
            fit_context = {"cell": cell, "readout": readout, "arm": arm, "dataset": ds, "task": task,
                "checkpoint_sha256": plan["caches"][arm][ds]["checkpoint_sha256"],
                "cache_sha256": plan["caches"][arm][ds]["sha256"], "projection_sha256": plan["projection"]["sha256"], **identity}
            results = [fit_fold(x, y, subjects, roles, f, audit, fit_context) for f in folds]
            for key in ("row", "y", "pred", "s", "fold"):
                arrays[f"{ds}_{key}"] = np.concatenate([r[key] for r in results])
            reports[ds] = {"folds": [r["report"] for r in results]}
            del x
            load_cache(plan, arm, ds, features=False, context=context)
        require(numeric == T.numeric_context() and len(audit) == 7 * len(folds) * len(datasets), "numeric context/fit count changed")
        summary = audit_summary(audit)
        completion_identity(plan, identity)
        require(E.P.file_identity(budget["resource_plan"]["path"]) == budget["resource_plan"], "resource plan changed")
        np.savez(out / "shard.oof.npz", **arrays)
        with (out / "audit.jsonl").open("x") as f:
            for record in audit:
                f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        result = {"protocol": PROTOCOL, "kind": "shard", "complete": True, "cell": cell, "arm": arm, "readout": readout,
                  "task": task, "step": STEP, "recipe": recipe(), "plan_content_sha256": jhash(plan), **identity,
                  "projection_sha256": plan["projection"]["sha256"], "resource_verification": budget,
                  "numeric_context": numeric, "numeric_sha256": jhash(numeric), "datasets": reports,
                  "audit_summary": summary, "audit": str(out / "audit.jsonl"), "audit_sha256": sha(out / "audit.jsonl"),
                  "oof": str(out / "shard.oof.npz"), "oof_sha256": sha(out / "shard.oof.npz")}
        T.write_json(out / "shard.json", result)
        return result
    except BaseException as exc:
        T.write_json(out / "failed.json", {"complete": False, "exception": f"{type(exc).__name__}: {exc}", "completed_datasets": list(reports), **identity})
        raise


def read_completed(path, kind, plan):
    r = json.loads(Path(path).read_text())
    require(r.get("complete") is True and r.get("kind") == kind and r.get("protocol") == PROTOCOL, "incomplete/wrong report")
    require(r.get("plan_content_sha256") == jhash(plan) and r.get("recipe") == recipe() and r.get("step") == STEP, "report recipe/plan mismatch")
    require(r.get("cell") in CELLS and (r.get("arm"), r.get("readout")) == CELLS[r["cell"]] and r.get("task") in TASKS,
            "report cell identity differs")
    require(r.get("projection_sha256") == plan["projection"]["sha256"], "report projection differs")
    require(sha(r["oof"]) == r["oof_sha256"] and sha(r["audit"]) == r["audit_sha256"], "report output hash mismatch")
    require(jhash(r["numeric_context"]) == r["numeric_sha256"], "numeric signature mismatch")
    records = [json.loads(line) for line in Path(r["audit"]).read_text().splitlines()]
    require(audit_summary(records) == r["audit_summary"], "audit summary mismatch")
    with np.load(r["oof"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require(set(arrays) == {f"{ds}_{k}" for ds in r["datasets"] for k in ("row", "y", "pred", "s", "fold")}, "unexpected/missing OOF arrays")
    if kind == "aggregate":
        validate_aggregate_report(plan, r, arrays, records)
    return r, arrays, records


def validate_fold(plan, cell, ds, f, fr, arrays, records, identity, original_y, original_s):
    arm, readout = CELLS[cell]
    task = plan["datasets"][ds]["task"]
    roles = T.CP.fold_roles(plan["datasets"][ds]["subjects"])
    role = roles[f]
    mt, mv, mh = (np.isin(original_s, role[k]) for k in ("train", "val", "held"))
    kv, kh = T.P1.keep_train_classes(original_y[mt], original_y[mv], original_y[mh])
    expected_rows = np.flatnonzero(mh)[kh]
    order = np.argsort(arrays["row"])
    require(np.array_equal(arrays["row"][order], expected_rows) and
            np.array_equal(arrays["y"][order], original_y[expected_rows]) and
            np.array_equal(arrays["s"][order].astype(str), original_s[expected_rows]), "OOF rows/targets/subjects do not match fold")
    require(all(fr[k] == role[k] for k in role) and fr["roles_sha256"] == jhash(role), "reported roles differ")
    require(fr["n_train"] == int(mt.sum()) and fr["n_val"] == int(kv.sum()) and fr["n_held"] == len(expected_rows) and
            fr["dropped_val_windows"] == int((~kv).sum()) and fr["dropped_held_windows"] == int((~kh).sum()), "fold counts mismatch")
    fits = [r for r in records if r["dataset"] == ds and r["fold"] == f]
    require(len(fits) == 7 and sorted(r["C"] for r in fits if r["fit_phase"] == "candidate") == list(C_GRID) and
            len([r for r in fits if r["fit_phase"] == "refit"]) == 1, "audit candidate/refit coverage differs")
    classes, counts = np.unique(original_y[mt], return_counts=True)
    for r in fits:
        expected_solver = T.LogisticRegression(C=r["C"], max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4).get_params()
        require(r["solver_params"] == expected_solver and r["n_features"] == width(cell) and r["dtype"] == "float32" and
                r["max_iter"] == MAX_ITER and r["cell"] == cell and r["readout"] == readout and r["arm"] == arm and
                r["roles_sha256"] == jhash(role) and r["scaler_sha256"] == fr["scaler_sha256"] and r["n_train"] == fr["n_train"] and
                r["classes"] == classes.tolist() and r["class_counts"] == counts.tolist() and
                r["checkpoint_sha256"] == plan["caches"][arm][ds]["checkpoint_sha256"] and
                r["cache_sha256"] == plan["caches"][arm][ds]["sha256"] and r["task"] == task and
                r["projection_sha256"] == plan["projection"]["sha256"] and r["code_sha"] == identity["code_sha"] and
                r["plan_sha256"] == identity["plan_sha256"], "solver/feature identity differs")
        require(type(r["n_iter"]) is int and 0 < r["n_iter"] <= MAX_ITER and
                r["reached_limit"] == (r["n_iter"] >= MAX_ITER) and
                r["convergence_warnings"] == sum(w["category"] == "ConvergenceWarning" for w in r["warnings"]) and
                r["converged_by_criterion"] == (not r["reached_limit"] and r["convergence_warnings"] == 0), "fit convergence audit inconsistent")
    candidates = {r["C"]: r for r in fits if r["fit_phase"] == "candidate"}
    require(all(np.isfinite(r["val_bacc"]) and 0 <= r["val_bacc"] <= 1 for r in candidates.values()), "invalid validation score")
    chosen = max(C_GRID, key=lambda c: (candidates[c]["val_bacc"], -c))
    require(fr["c"] == chosen and fr["val_bacc"] == candidates[chosen]["val_bacc"] and
            fr["val_scores"] == {str(c): candidates[c]["val_bacc"] for c in C_GRID} and
            all(r["selected"] == (r["C"] == chosen) for r in candidates.values()) and
            next(r["C"] for r in fits if r["fit_phase"] == "refit") == chosen, "C selection/refit differs")
    require(fr["held_bacc"] == T.scored(arrays["y"], arrays["pred"])[0], "held score differs from OOF")



def validate_aggregate_report(plan, report, arrays, records):
    """Recheck completed aggregates before downstream paired interpretation."""
    cell, arm, task = report["cell"], report["arm"], report["task"]
    require(set(report["datasets"]) == set(TASKS[task]) and report.get("weights") == dict(zip(TASKS[task], WEIGHTS[task])) and
            report.get("aliases") == ALIASES, "aggregate dataset/weight/alias identity differs")
    require(len(records) == 28 * len(TASKS[task]), "aggregate audit coverage differs")
    require(report.get("reading_status") == ("solver_converged_by_criterion" if report["audit_summary"]["converged_by_criterion"] else "solver_unresolved"),
            "aggregate solver interpretation differs")
    identity = {"code_sha": report["code_sha"], "plan_sha256": report["plan_sha256"]}
    for ds in TASKS[task]:
        a = {k: arrays[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")}
        T.validate_oof(*(a[k] for k in ("row", "y", "pred", "s", "fold")))
        _, original_y, original_s = load_cache(plan, arm, ds, features=False)
        detail = report["datasets"][ds]
        folds = detail["folds"]
        require(len(folds) == 4 and sorted(fr["fold"] for fr in folds) == list(range(4)) and
                set(a["fold"].tolist()) == set(range(4)), "aggregate fold coverage differs")
        for fr in folds:
            f = fr["fold"]
            pick = a["fold"] == f
            validate_fold(plan, cell, ds, f, fr, {k: v[pick] for k, v in a.items()}, records, identity, original_y, original_s)
        require(detail["balanced_accuracy"] == T.scored(a["y"], a["pred"])[0] and
                detail["n_oof_windows"] == len(a["row"]) and detail["n_oof_subjects"] == len(set(a["s"].tolist())) and
                detail["identity_sha256"] == T.DA.fingerprint(*(a[k] for k in ("row", "y", "s", "fold"))), "aggregate metrics/identity differ")
    expected = float(np.dot(WEIGHTS[task], [report["datasets"][d]["balanced_accuracy"] for d in TASKS[task]]))
    require(report["balanced_accuracy"] == expected, "aggregate task score differs")


def aggregate(plan, cell, task, shards, out, identity):
    validate_plan(plan)
    width(cell)
    require(task in TASKS and shards and len(set(map(str, shards))) == len(shards), "invalid aggregation")
    arm, readout = CELLS[cell]
    out = Path(out)
    for p in (out, out.with_suffix(".oof.npz"), out.with_suffix(".audit.jsonl")):
        T.fresh(p)
    loaded = [read_completed(p, "shard", plan) for p in shards]
    require(all(r["cell"] == cell and r["task"] == task and r["code_sha"] == identity["code_sha"] and
                r["plan_sha256"] == identity["plan_sha256"] for r, _, _ in loaded), "shard identity mismatch")
    require(len({r["numeric_sha256"] for r, _, _ in loaded}) == 1, "mixed numeric fit contexts")
    records = sum((audit for _, _, audit in loaded), [])
    dataset_reports, arrays = {}, {}
    for ds in TASKS[task]:
        cells = {}
        for report, values, _ in loaded:
            require(set(report["datasets"]) <= set(TASKS[task]), "foreign dataset in shard")
            if ds not in report["datasets"]:
                continue
            T.validate_oof(*(values[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")))
            fold = values[f"{ds}_fold"]
            for fr in report["datasets"][ds]["folds"]:
                f = fr["fold"]
                require(type(f) is int and f in range(4) and f not in cells, "duplicate/invalid fold")
                pick = fold == f
                require(pick.any(), "missing fold predictions")
                cells[f] = fr, {k: values[f"{ds}_{k}"][pick] for k in ("row", "y", "pred", "s", "fold")}
            require(set(fold.tolist()) == {fr["fold"] for fr in report["datasets"][ds]["folds"]}, "unreported OOF fold")
        require(set(cells) == set(range(4)), "missing dataset/fold")
        _, original_y, original_s = load_cache(plan, arm, ds, features=False)
        for f, (fr, values) in sorted(cells.items()):
            validate_fold(plan, cell, ds, f, fr, values, records, identity, original_y, original_s)
        joined = {k: np.concatenate([cells[f][1][k] for f in range(4)]) for k in ("row", "y", "pred", "s", "fold")}
        T.validate_oof(*(joined[k] for k in ("row", "y", "pred", "s", "fold")))
        arrays.update({f"{ds}_{k}": v for k, v in joined.items()})
        score, metric_warnings = T.scored(joined["y"], joined["pred"])
        dataset_reports[ds] = {"balanced_accuracy": score, "n_oof_windows": len(joined["row"]),
                              "n_oof_subjects": len(set(joined["s"].tolist())), "folds": [cells[f][0] for f in range(4)],
                              "identity_sha256": T.DA.fingerprint(*(joined[k] for k in ("row", "y", "s", "fold"))),
                              "scoring_warnings": metric_warnings}
    require(len(records) == 28 * len(TASKS[task]), "extra audit fits")
    completion_identity(plan, identity)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".oof.npz"), **arrays)
    with out.with_suffix(".audit.jsonl").open("x") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary = audit_summary(records)
    result = {"protocol": PROTOCOL, "kind": "aggregate", "complete": True, "cell": cell, "arm": arm, "readout": readout,
              "task": task, "step": STEP, "recipe": recipe(), "plan_content_sha256": jhash(plan), **identity,
              "projection_sha256": plan["projection"]["sha256"], "aliases": ALIASES,
              "numeric_context": loaded[0][0]["numeric_context"], "numeric_sha256": loaded[0][0]["numeric_sha256"],
              "datasets": dataset_reports, "weights": dict(zip(TASKS[task], WEIGHTS[task])),
              "balanced_accuracy": float(np.dot(WEIGHTS[task], [dataset_reports[d]["balanced_accuracy"] for d in TASKS[task]])),
              "audit_summary": summary, "reading_status": "solver_converged_by_criterion" if summary["converged_by_criterion"] else "solver_unresolved",
              "audit": str(out.with_suffix(".audit.jsonl")), "audit_sha256": sha(out.with_suffix(".audit.jsonl")),
              "oof": str(out.with_suffix(".oof.npz")), "oof_sha256": sha(out.with_suffix(".oof.npz")),
              "shards": [{"path": str(p), "sha256": sha(p)} for p in shards]}
    T.write_json(out, result)
    return result



CONTRASTS = {"natural_300m_minus_30m": (-1., 1., 0.),
             "matched_300m_minus_30m": (-1., 0., 1.),
             "natural_300m_minus_matched_300m": (0., 1., -1.)}


def _summary(points, draws):
    return {"cells": {cell: {"balanced_accuracy": float(points[i]),
                     "ci95_percent": (100*np.quantile(draws[:, i], [.025, .975])).tolist()}
                     for i, cell in enumerate(CELLS)},
            "contrasts": {name: {"delta_pp": float(100*(points @ np.asarray(co))),
                          "ci95_pp": (100*np.quantile(draws @ np.asarray(co), [.025, .975])).tolist()}
                          for name, co in CONTRASTS.items()}}


def compare_arrays(cells, n_boot=T.N_BOOT, seed=T.SEED):
    """Shared subject draws, native three-cell identities, no historical aliasing."""
    require(set(cells) == set(CELLS) and all(set(c) == set(DATASETS) for c in cells.values()),
            "need complete three-cell/five-dataset grid")
    require(type(n_boot) is int and n_boot >= 2 and type(seed) is int, "invalid bootstrap count/seed")
    rng, datasets, points, samples = np.random.default_rng(seed), {}, {}, {}
    for ds in sorted(DATASETS):
        aligned = []
        for cell in CELLS:
            a = cells[cell][ds]
            T.validate_oof(*(a[k] for k in ("row", "y", "pred", "s", "fold")))
            ix = np.argsort(a["row"])
            aligned.append({k: a[k][ix] for k in ("row", "y", "pred", "s", "fold")})
        base = aligned[0]
        require(all(all(np.array_equal(a[k], base[k]) for k in ("row", "y", "s", "fold")) for a in aligned), "OOF pairing differs")
        stats = [T.DA.subject_counts(a["y"], a["pred"], a["s"]) for a in aligned]
        users, denominator, _ = T.DA._arrays(stats[0])
        counts = [T.DA._arrays(s)[2] for s in stats]
        draw = rng.integers(0, len(users), size=(n_boot, len(users)))
        den = denominator[draw].sum(1)
        present = den > 0
        classes_present = present.sum(1)
        require((classes_present > 0).all(), "empty bootstrap targets")
        values = np.column_stack([np.divide(hits[draw].sum(1), den, out=np.zeros_like(den), where=present).sum(1)/classes_present for hits in counts])
        total = denominator.sum(0)
        point = np.asarray([np.mean(h.sum(0)[total > 0]/total[total > 0]) for h in counts])
        require(np.isfinite(values).all(), "invalid bootstrap metric")
        points[ds], samples[ds] = point, values
        datasets[ds] = {**_summary(point, values), "subjects": users, "classes": stats[0]["classes"],
                        "draws_sha256": T.DA.fingerprint(draw), "paired_counts": dict(zip(CELLS, stats)),
                        "replicates_missing_classes": int((classes_present < len(total)).sum()),
                        "single_class_replicates": int((classes_present == 1).sum()),
                        "missing_class_rule": "macro recall over target classes present in each draw; all draws retained",
                        "role": "primary_classification" if ds == "epn612" else "secondary_dataset"}
    tasks = {task: _summary(np.sum([w*points[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0),
                           np.sum([w*samples[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0)) for task in TASKS}
    return {"datasets": datasets, "tasks": tasks, "aliases": ALIASES, "n_bootstrap": n_boot, "seed": seed,
            "uncertainty": "paired subject bootstrap conditional on one pretraining seed, fixed checkpoints and fitted probes; development validation; no multiplicity adjustment"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("benchmark", "fit-shard", "aggregate"))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--expected-plan-sha256", required=True)
    p.add_argument("--expected-code-sha", required=True)
    p.add_argument("--cell", choices=CELLS, required=True)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--shards", nargs="+", type=Path)
    p.add_argument("--resource-plan", type=Path)
    p.add_argument("--expected-resource-plan-sha256")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    require(sha(a.plan) == a.expected_plan_sha256, "plan file hash mismatch")
    E.P.code_identity(a.expected_code_sha)
    plan = json.loads(a.plan.read_text())
    identity = {"code_sha": a.expected_code_sha, "plan_sha256": a.expected_plan_sha256, "plan_path": str(a.plan.resolve())}
    if a.stage == "benchmark":
        require(a.dataset, "benchmark requires dataset")
        result = benchmark(plan, a.cell, a.dataset, a.out, identity)
    elif a.stage == "fit-shard":
        require(a.task and a.datasets and a.folds is not None and a.resource_plan and a.expected_resource_plan_sha256,
                "fit requires task/datasets/folds and measured resource plan")
        result = run_shard(plan, a.cell, a.task, a.datasets, a.folds, a.out, identity,
                          {"path": str(a.resource_plan.resolve()), "sha256": a.expected_resource_plan_sha256})
    else:
        require(a.task and a.shards, "aggregate requires task/shards")
        result = aggregate(plan, a.cell, a.task, a.shards, a.out, identity)
    print(json.dumps({"complete": result["complete"], "kind": result["kind"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
