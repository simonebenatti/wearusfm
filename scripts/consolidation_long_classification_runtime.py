"""Frozen execution adapters for the reviewed long-dose classification matrix.

This module never submits jobs. Every stage writes a fresh external directory;
the controller supplies a reviewed context and immutable artifact manifests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

import consolidation_long_common as C
import consolidation_long_classification as M

BENCHMARK_POLICY = {
    "protocol": "consolidation-classification-resources-20261105-v1",
    "reference_step": 19618, "population": "complete fold0 train only",
    "C_grid": list(M.C_GRID), "max_iter": M.MAX_ITER,
    "validation_or_held_scored": False, "requires_all_15_benchmarks": True,
    "projection": "measured six converged full-train fits plus slowest candidate refit; fold rows/classes scaling; historical completed pool-fold costs scaled by measured view/pool ratio as a floor",
    "safety_factor": 3.0, "data_and_prediction_multiplier": 1.5,
    "fixed_seconds_per_job": 600.0, "memory_multiplier": 3.0,
    "max_projected_seconds": 13800.0, "walltime_seconds": 14400,
    "cpus": 8, "memory_gib": 30, "memory_fraction": .85, "blas_threads": 8,
    "fixed_folds_by_dataset": {ds: [[0, 1], [2, 3]] if ds == "ninapro_db2" else [[0, 1, 2, 3]] for ds in M.DATASETS},
    "max_projected_seconds_by_dataset": {ds: 13800.0 for ds in M.DATASETS},
    "walltime_seconds_by_dataset": {ds: 14400 for ds in M.DATASETS},
    "new_fit_cells": 40, "physical_fit_jobs": 48,
    "limitation": "empirical planning estimate, not a runtime guarantee; later checkpoint conditioning can change solver cost; no automatic shard, retry, or budget extension",
}


def _policy(ctx):
    C.require(ctx["plan"]["classification_benchmark_policy"] == BENCHMARK_POLICY,
              "classification benchmark resource policy differs")
    return BENCHMARK_POLICY


def _threads(expected=8):
    C.require(expected in (1, 8) and all(os.environ.get(k) == str(expected) for k in
                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
              "classification BLAS/OpenMP threads differ from the historical phase")
    numeric = M.TP.numeric_context()
    C.require(numeric["threadpools"] and all(p["num_threads"] == expected for p in numeric["threadpools"]),
              "active numeric thread count differs")
    return numeric


def _historical(ctx):
    plan = C.read_pin(ctx["plan"]["historical_classification_plan"])
    M.TP.validate_plan(plan)
    pins = {ds: {**plan["caches"]["T100-G"][ds], **plan["datasets"][ds]}
            for ds in M.DATASETS}
    return plan, pins


def _historical_reports(ctx, task):
    pins = ctx["plan"]["historical_classification_reports"][task]
    return pins if isinstance(pins, list) else [pins]


def _historical_report(ctx, pin, task, historical):
    """Verify a pinned report header without opening its held predictions."""
    r = C.read_pin(pin)
    C.require(r.get("complete") is True and r.get("protocol") == M.TP.PROTOCOL
              and r.get("kind") in ("aggregate", "shard") and r.get("step") == 19618
              and r.get("arm") == "T100-G" and r.get("task") == task
              and r.get("recipe") == M.TP.recipe()
              and r.get("plan_content_sha256") == M.jhash(historical), "historical timing source differs")
    C.require(M.jhash(r["numeric_context"]) == r["numeric_sha256"], "historical numeric identity differs")
    return r


def _historical_numeric(ctx, historical):
    reports, signatures, numeric = {}, set(), None
    for task in M.TASKS:
        pins = _historical_reports(ctx, task)
        C.require(pins, "missing historical numeric reference")
        reports[task] = pins
        for pin in pins:
            r = _historical_report(ctx, pin, task, historical)
            signatures.add(r["numeric_sha256"])
            numeric = r["numeric_context"]
    C.require(len(signatures) == 1, "mixed historical P1/P2 numeric environments")
    return {"numeric_sha256": next(iter(signatures)), "numeric_context": numeric, "reports": reports}


def _numeric_match(numeric, historical_numeric):
    C.require(M.jhash(historical_numeric["numeric_context"]) == historical_numeric["numeric_sha256"],
              "historical numeric identity differs")
    return {**M.numeric_compatibility(numeric, historical_numeric["numeric_context"]),
            "historical_reports": historical_numeric["reports"]}


def _historical_timing(ctx, dataset, historical):
    """Read archived fit costs, never evaluate historical held/validation scores."""
    task, records, sources, signatures = M._task(dataset), [], [], set()
    for pin in _historical_reports(ctx, task):
        r = _historical_report(ctx, pin, task, historical)
        signatures.add(r["numeric_sha256"])
        audit = C.pinned({"path": r["audit"], "sha256": r["audit_sha256"]})
        entries = [json.loads(line) for line in audit.read_text().splitlines()]
        C.require(r["audit_summary"] == M.TP.audit_summary(entries), "historical timing audit differs")
        C.require(all(v["dataset"] in r["datasets"] and v["task"] == task and v["arm"] == "T100-G"
                      and v["code_sha"] == r["code_sha"] and v["plan_sha256"] == r["plan_sha256"]
                      for v in entries), "historical timing audit source differs")
        records.extend(v for v in entries if v["dataset"] == dataset)
        sources.append(pin)
    C.require(len(records) == 28, "historical timing lacks complete four-fold grid")
    C.require(len(signatures) == 1, "mixed historical timing numeric environments")
    folds = []
    for f in range(4):
        values = [r for r in records if r["fold"] == f]
        candidates = [r for r in values if r["fit_phase"] == "candidate"]
        refits = [r for r in values if r["fit_phase"] == "refit"]
        C.require(len(values) == 7 and len(candidates) == 6 and {r["C"] for r in candidates} == set(M.C_GRID)
                  and len(refits) == 1 and all(r["exception"] is None and r["converged_by_criterion"] is True
                  and r["max_iter"] == M.MAX_ITER and np.isfinite(r["elapsed_s"]) and r["elapsed_s"] > 0
                  and r["cache_sha256"] == historical["caches"]["T100-G"][dataset]["sha256"]
                  and r["checkpoint_sha256"] == historical["caches"]["T100-G"][dataset]["checkpoint_sha256"]
                  for r in values), "invalid historical complete-fit timings")
        C.require(len({r["n_train"] for r in values}) == 1, "historical training counts differ")
        folds.append({"fold": f, "fit_seconds": sum(r["elapsed_s"] for r in values),
                      "train_rows": values[0]["n_train"], "classes": len(values[0]["classes"])})
    return {"dataset": dataset, "folds": folds, "reports": sources,
            "numeric_sha256": next(iter(signatures)),
            "scope": "archived completed fitting elapsed times only; no validation or held scoring"}


def _cache(ctx, step, dataset):
    entry = ctx["caches"]["classification"][str(step)][dataset]
    C.verified_extraction(ctx, entry, step, "extract-classification")
    historical, old = _historical(ctx)
    expected = {**entry["cache"], "dataset": dataset, "task": M._task(dataset), "step": step,
                "checkpoint_sha256": ctx["plan"]["endpoints"][str(step)]["checkpoint"]["sha256"],
                "subjects": historical["datasets"][dataset]["subjects"]}
    for key in ("input_sha256", "splits_sha256", "roles_sha256"):
        expected[key] = historical["datasets"][dataset][key]
    return entry, expected, historical, old


def _check_benchmark(bench, dataset, view):
    C.require(bench["protocol"] == M.PROTOCOL and bench["kind"] == "benchmark"
              and bench["step"] == 19618 and bench["dataset"] == dataset and bench["view"] == view
              and bench["timing_only"] is True and bench["validation_or_held_scored"] is False
              and bench["full_fold0_train"] is True and bench["sample_rows"] == bench["full_train_rows"]
              and bench["sample_classes"] == bench["full_train_classes"]
              and bench["max_iter"] == M.MAX_ITER and len(bench["fits"]) == 6
              and bench["width"] == M.probe_recipe(view, bench["encoder_dim"])["width"]
              and bench["subjects"] == bench["fold0_train_subjects"]
              and {r["C"] for r in bench["fits"]} == set(M.C_GRID), "incomplete full-train benchmark")
    C.require(all(r["fit_phase"] == "benchmark_train_only" and r["exception"] is None
                  and r["max_iter"] == M.MAX_ITER and r["n_train"] == bench["full_train_rows"]
                  and np.isfinite(r["elapsed_s"]) and r["elapsed_s"] > 0 and r["converged_by_criterion"] is True
                  and r["n_iter"] < M.MAX_ITER and r["convergence_warnings"] == 0 for r in bench["fits"]),
              "benchmark solver is unresolved")
    C.require(bench["fold_train_rows"][0] == bench["full_train_rows"]
              and bench["fold_train_classes"][0] == bench["full_train_classes"]
              and len(bench["fold_train_rows"]) == len(bench["fold_train_classes"]) == 4
              and min(bench["fold_train_rows"]) > 0 and min(bench["fold_train_classes"]) >= 2,
              "invalid fold planning support")
    C.require(all(np.isfinite(bench[k]) and bench[k] >= 0 for k in ("load_s", "scale_s")), "invalid benchmark overhead")


def shard_key(step, view, dataset, folds):
    return f"{step}/{view}/{dataset}/folds" + "-".join(map(str, folds))


def resource_gate(reports, policy=BENCHMARK_POLICY):
    """Global40-cell/48-fixed-job gate from15 complete train-only timings."""
    C.require(policy == BENCHMARK_POLICY and set(reports) == set(M.DATASETS)
              and all(set(reports[d]) == set(M.VIEWS) for d in reports), "all fifteen benchmark cells required")
    cells, shards = {}, {}
    numeric, historical_signatures = set(), set()
    for ds in M.DATASETS:
        references = reports[ds]
        for view in M.VIEWS:
            item, bench = references[view], references[view]["benchmark"]
            _check_benchmark(bench, ds, view)
            signature = M.jhash(bench["numeric_context"])
            numeric.add(signature)
            evidence = item["historical_numeric_match"]
            historical_signatures.add(item["historical_timing"]["numeric_sha256"])
            expected_match = _numeric_match(bench["numeric_context"], {
                "numeric_sha256": item["historical_timing"]["numeric_sha256"],
                "numeric_context": evidence["historical_numeric_context"],
                "reports": evidence["historical_reports"]})
            C.require(evidence == expected_match,
                      "benchmark/historical classification numeric compatibility evidence differs")
            C.require(np.isfinite(item["peak_rss_bytes"]) and item["peak_rss_bytes"] > 0,
                      "invalid benchmark memory measurement")
            C.require(item["historical_timing"]["dataset"] == ds
                      and [f["fold"] for f in item["historical_timing"]["folds"]] == list(range(4)),
                      "historical planning support differs")
        base = references["pool"]["benchmark"]
        C.require(len({references[v]["benchmark"]["cache_sha256"] for v in M.VIEWS}) == 1,
                  "benchmark views use different extraction caches")
        for view in M.VIEWS:
            item, bench = references[view], references[view]["benchmark"]
            for key in ("input_sha256", "splits_sha256", "roles_sha256", "ordered_rows_sha256", "checkpoint_sha256",
                        "encoder_dim", "fold_train_rows", "fold_train_classes"):
                C.require(bench[key] == base[key], "benchmark views differ in source/support: " + key)
            pool_work = sum(r["elapsed_s"] for r in base["fits"]) + max(r["elapsed_s"] for r in base["fits"])
            work = sum(r["elapsed_s"] for r in bench["fits"]) + max(r["elapsed_s"] for r in bench["fits"])
            fold_costs = []
            for f, old in enumerate(item["historical_timing"]["folds"]):
                C.require(old["train_rows"] == bench["fold_train_rows"][f]
                          and old["classes"] == bench["fold_train_classes"][f]
                          and np.isfinite(old["fit_seconds"]) and old["fit_seconds"] > 0,
                          "historical/full-train benchmark fold population differs")
                scaled = work * bench["fold_train_rows"][f] / bench["full_train_rows"]
                scaled *= bench["fold_train_classes"][f] / bench["full_train_classes"]
                fold_costs.append(max(scaled, old["fit_seconds"]*work/pool_work)
                                  * policy["safety_factor"] * policy["data_and_prediction_multiplier"])
            total = sum(fold_costs) + policy["fixed_seconds_per_job"] + 2*(bench["load_s"] + bench["scale_s"])
            memory = item["peak_rss_bytes"] * policy["memory_multiplier"]
            memory_ok = memory <= policy["memory_gib"]*1024**3*policy["memory_fraction"]
            for step in M.DOSES:
                if step == 19618 and view == "pool":
                    continue  # The historical scientific pool/OOF is reused.
                keys = []
                for indices in policy["fixed_folds_by_dataset"][ds]:
                    key = shard_key(step, view, ds, indices)
                    projection = sum(fold_costs[f] for f in indices) + policy["fixed_seconds_per_job"] + 2*(bench["load_s"] + bench["scale_s"])
                    shards[key] = {"step": step, "view": view, "dataset": ds, "folds": indices,
                        "projected_seconds": projection, "projected_peak_rss_bytes": memory,
                        "max_projected_seconds": policy["max_projected_seconds_by_dataset"][ds],
                        "walltime_seconds": policy["walltime_seconds_by_dataset"][ds],
                        "passed": bool(memory_ok and projection <= policy["max_projected_seconds_by_dataset"][ds])}
                    keys.append(key)
                cells[f"{step}/{view}/{ds}"] = {"step": step, "view": view, "dataset": ds,
                    "folds": list(range(4)), "projected_fit_seconds_by_fold": fold_costs,
                    "full_cell_projected_seconds": total, "projected_peak_rss_bytes": memory,
                    "unsharded_passed": bool(memory_ok and total <= policy["max_projected_seconds"]),
                    "projected_seconds": sum(shards[k]["projected_seconds"] for k in keys),
                    "shards": keys, "passed": all(shards[k]["passed"] for k in keys), "reference_step": 19618}
    C.require(len(numeric) == 1, "mixed numeric environments across benchmark cells")
    C.require(len(historical_signatures) == 1, "mixed historical numeric environments across benchmark cells")
    C.require(len(cells) == 40 and len(shards) == 48, "registered physical/logical matrix differs")
    return {"policy": policy, "cells": cells, "shards": shards, "all_passed": all(c["passed"] for c in shards.values()),
            "physical_fit_jobs": len(shards),
            "new_fit_cells": len(cells), "projected_seconds": sum(c["projected_seconds"] for c in cells.values()),
            "projected_core_hours": policy["cpus"]*sum(c["projected_seconds"] for c in cells.values())/3600,
            "numeric_sha256": next(iter(numeric))}


def _benchmarks(ctx):
    manifest = C.read_pin(ctx["plan"]["benchmark_receipts"]["classification"])
    C.require(set(manifest["reports"]) == set(M.DATASETS), "benchmark manifest lacks datasets")
    reports = {}
    historical, _ = _historical(ctx)
    reference = _historical_numeric(ctx, historical)
    for ds in M.DATASETS:
        C.require(set(manifest["reports"][ds]) == set(M.VIEWS), "benchmark manifest lacks views")
        reports[ds] = {}
        _, expected, _, _ = _cache(ctx, 19618, ds)
        for view in M.VIEWS:
            report = C.read_pin(manifest["reports"][ds][view])
            _receipt(ctx, report, "benchmark-classification")
            C.require(report["policy"] == _policy(ctx), "benchmark receipt policy differs")
            C.require(report["historical_numeric_match"] == _numeric_match(report["benchmark"]["numeric_context"], reference),
                      "benchmark historical numeric evidence differs")
            C.require(report["benchmark"]["cache_sha256"] == expected["sha256"]
                      and report["benchmark"]["checkpoint_sha256"] == expected["checkpoint_sha256"],
                      "benchmark receipt cache/checkpoint differs")
            reports[ds][view] = report
    return reports


def _receipt(ctx, report, kind):
    C.require(report["protocol"] == C.PROTOCOL and report["kind"] == kind and report["complete"] is True
              and report["source_sha256"] == ctx["source_sha256"]
              and report["recipe_sha256"] == ctx["plan"]["recipe_sha256"], "stage receipt differs")


def execute(stage, ctx, out, step=None, task=None, dataset=None, view=None, folds=None):
    """Execute exactly one registered stage, without submission or mutation of inputs."""
    C.require(stage in ("extract-classification", "benchmark-classification", "fit-classification", "compare-classification"),
              "unknown classification execution stage")
    _policy(ctx)
    out = Path(out)
    if stage == "extract-classification":
        C.require(step in M.DOSES and task in M.TASKS and dataset is None and view is None and folds is None,
                  "extraction requires one task and dose")
        _threads(1)
        historical, old = _historical(ctx)
        split_pin = ctx["legacy"]["inputs"]["splits"]
        splits = C.read_pin(split_pin)
        model, teacher = C.load_teacher(ctx, step, device="cuda")
        report = C.begin_output(ctx, out, stage, task=task, **teacher)
        artifacts = {}
        for ds in M.TASKS[task]:
            args = ctx["legacy"]["paths"]
            extraction = M.extract_dataset(model, ds, roots=args["roots"], epn_roots=args["epn_roots"],
                uci_root=args["uci_root"], splits=splits, splits_sha256=split_pin["sha256"],
                batch=ctx["plan"]["extraction_batch"]["classification"], device="cuda")
            identity = {"step": step, "checkpoint": teacher["checkpoint"],
                        "checkpoint_sha256": teacher["checkpoint_sha256"], "code_sha": ctx["code_sha"],
                        "plan_sha256": ctx["plan_pin"]["sha256"]}
            M.save_cache(out / (ds+".npz"), extraction, identity, old[ds])
            artifacts[ds] = C.file_pin(out / (ds+".npz"))
        C.pinned(ctx["plan"]["endpoints"][str(step)]["checkpoint"])
        del model
        return C.finish_output(ctx, out, {**report, "artifacts": artifacts})
    if stage == "benchmark-classification":
        C.require(step == 19618 and dataset in M.DATASETS and view in M.VIEWS and task is None and folds is None,
                  "benchmark requires one dataset/view at19618")
        numeric = _threads()
        entry, expected, historical, _ = _cache(ctx, step, dataset)
        match = _numeric_match(numeric, _historical_numeric(ctx, historical))
        timing = _historical_timing(ctx, dataset, historical)
        report = C.begin_output(ctx, out, stage, dataset=dataset, view=view, checkpoint_step=step,
                                checkpoint_sha256=expected["checkpoint_sha256"])
        start = time.perf_counter()
        bench = M.benchmark_cache(expected["path"], view, expected=expected)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak*1024)
        C.require(M.jhash(_threads()) == match["numeric_sha256"]
                  and M.jhash(bench["numeric_context"]) == match["numeric_sha256"],
                  "numeric environment changed during benchmark")
        return C.finish_output(ctx, out, {**report, "policy": BENCHMARK_POLICY, "benchmark": bench,
            "historical_numeric_match": match,
            "historical_timing": timing, "extraction_receipt": entry["receipt"],
            "peak_rss_bytes": peak_bytes, "elapsed_seconds": time.perf_counter()-start,
            "artifacts": {}})
    if stage == "fit-classification":
        C.require(step in M.DOSES and dataset in M.DATASETS and view in M.VIEWS and task is None
                  and not (step == 19618 and view == "pool"), "fit requires one of40 new logical cells")
        registered = BENCHMARK_POLICY["fixed_folds_by_dataset"][dataset]
        C.require(folds is not None or len(registered) == 1, "DB2 requires an explicit registered fold pair")
        chosen = registered[0] if folds is None else folds
        C.require(chosen in registered and all(type(f) is int for f in chosen), "unregistered fit fold shard")
        numeric = _threads()
        historical, _ = _historical(ctx)
        match = _numeric_match(numeric, _historical_numeric(ctx, historical))
        gate = resource_gate(_benchmarks(ctx), _policy(ctx))
        C.require(gate["all_passed"], "classification matrix resource gate failed; reviewed resource amendment required")
        C.require(M.jhash(numeric) == gate["numeric_sha256"], "fit/benchmark numeric environment differs")
        entry, expected, _, _ = _cache(ctx, step, dataset)
        report = C.begin_output(ctx, out, stage, dataset=dataset, view=view, fold_indices=chosen, checkpoint_step=step,
                                checkpoint_sha256=expected["checkpoint_sha256"])
        probe = M.fit_cache(expected["path"], view, out / "probe.json", expected=expected,
                    identity={"code_sha": ctx["code_sha"], "plan_sha256": ctx["plan_pin"]["sha256"]}, folds=chosen)
        C.require(M.jhash(_threads()) == gate["numeric_sha256"] == probe["numeric_sha256"],
                  "numeric environment changed during fit")
        artifacts = {"probe": C.file_pin(out/"probe.json"), "oof": C.file_pin(out/"probe.oof.npz"),
                     "audit": C.file_pin(out/"probe.audit.jsonl")}
        return C.finish_output(ctx, out, {**report, "artifacts": artifacts, "extraction_receipt": entry["receipt"],
            "historical_numeric_match": match,
            "benchmark_manifest": ctx["plan"]["benchmark_receipts"]["classification"],
            "resource_gate": gate["shards"][shard_key(step, view, dataset, chosen)], "global_resource_gate_passed": True})
    C.require(all(value is None for value in (step, task, dataset, view, folds)), "comparison takes only pinned manifests")
    _threads()
    historical, old = _historical(ctx)
    reference = _historical_numeric(ctx, historical)
    manifest = C.read_pin(ctx["plan"]["classification_fit_manifest"])
    probes, numeric = [], set()
    for pin in manifest["reports"]:
        r = C.read_pin(pin)
        _receipt(ctx, r, "fit-classification")
        for artifact in r["artifacts"].values():
            C.pinned(artifact)
        probe = C.read_pin(r["artifacts"]["probe"])
        C.require(r["historical_numeric_match"] == _numeric_match(probe["numeric_context"], reference),
                  "fit historical numeric evidence differs")
        C.require(probe["dataset"] in M.DATASETS and probe["view"] in M.VIEWS
                  and probe["fold_indices"] in BENCHMARK_POLICY["fixed_folds_by_dataset"][probe["dataset"]]
                  and not (probe["step"] == 19618 and probe["view"] == "pool"), "unregistered fit receipt cell/shard")
        C.require(probe["step"] == r["checkpoint_step"] and probe["view"] == r["view"]
                  and probe["dataset"] == r["dataset"] and probe["checkpoint_sha256"] == r["checkpoint_sha256"]
                  and probe["fold_indices"] == r["fold_indices"]
                  and probe["code_sha"] == r["code_sha"] and probe["plan_sha256"] == r["plan_sha256"]
                  and all(r["artifacts"][key]["path"] == probe[key]
                          and r["artifacts"][key]["sha256"] == probe[key+"_sha256"] for key in ("oof", "audit"))
                  and r["benchmark_manifest"] == ctx["plan"]["benchmark_receipts"]["classification"]
                  and r["resource_gate"]["folds"] == probe["fold_indices"] and r["resource_gate"]["passed"] is True
                  and r["global_resource_gate_passed"] is True, "probe/stage receipt identity differs")
        probes.append(r["artifacts"]["probe"])
        numeric.add(probe["numeric_sha256"])
    C.require(len(probes) == 48, "complete forty-eight-job fit manifest required")
    C.require(len(numeric) == 1, "mixed numeric environments across new classification fits")
    by_dose = {s: {ds: _cache(ctx, s, ds)[1] for ds in M.DATASETS} for s in M.DOSES}
    cache_pins = {M.cell_name(s, v): by_dose[s]
                  for s in M.DOSES for v in M.VIEWS}
    reports = [_historical_reports(ctx, t) for t in M.TASKS]
    report = C.begin_output(ctx, out, stage)
    result = M.compare_reports(probes, cache_pins=cache_pins, historical_report_pins=reports,
        historical_cache_pins=old, historical_plan=historical,
        historical_baseline_policy=ctx["plan"]["classification_baseline_policy"], out=out/"comparison.json",
        identity={"code_sha": ctx["code_sha"], "plan_sha256": ctx["plan_pin"]["sha256"]})
    return C.finish_output(ctx, out, {**report, "artifacts": {"comparison": C.file_pin(out/"comparison.json")},
                                    "historical_numeric_match": result["numeric_comparison"],
                                    "reading": result["reading"]})
