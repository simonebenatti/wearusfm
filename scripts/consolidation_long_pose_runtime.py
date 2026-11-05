"""Artifact-pinned execution for the frozen long-dose pose matrix.

Only extraction loads an FM. Ridge fits finish before the held payload is
opened; the trained pose head is read directly from its extraction cache.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_info

import consolidation_long_common as C
import consolidation_long_pose as P

STAGES = ("extract-pose-train", "extract-pose-val", "benchmark-pose", "fit-pose", "compare-pose")
FIT_RESOURCES = {"gpus": 0, "cpus": 8, "memory_gib": 30, "seconds": 3600}
BENCHMARK_POLICY = {
    "reference_step": 19618, "safety_factor": 2., "fixed_seconds": 120.,
    "max_seconds_per_dose": 3300., "memory_fraction": .85, "memory_bytes": 30*1024**3,
    "train_windows": 1907, "held_windows": 18548,
    "fit_floor": "project_fit_seconds(max_depth_statistics_and_eigensystems, n_readers=2, safety_factor=2, fixed_seconds=0)",
    "other_work": "2 * (read_validate * (1+r) + fixed_alpha_fit + predict * r + summary * r + E2_pairing * r**2 + bootstrap_diagnostic * r + artifact_write * r) + 120; r=18548/1907",
    "memory_projection": "max_RSS_bytes * (18548/1907)",
    "data_policy": "train payload only; held cache/reference payload is never opened, hashed or scored",
}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "threadpools": [{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer")}
                            for p in threadpool_info()]}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def _manifest(ctx):
    manifest = C.read_pin(ctx["plan"]["cache_manifest"])
    C.require(manifest == ctx["caches"] and manifest["recipe_sha256"] == ctx["plan"]["recipe_sha256"]
              and manifest["source_sha256"] == ctx["source_sha256"], "pose cache manifest recipe/source differs")
    C.require(set(manifest["pose"]) == set(map(str, P.DOSES)), "pose cache manifest doses differ")
    C.require(all(set(v) == {"train", "val"} for v in manifest["pose"].values()), "pose cache split missing")
    return manifest


def _entry(ctx, step, split):
    entry = _manifest(ctx)["pose"][str(step)][split]
    C.verified_extraction(ctx, entry, step, "extract-pose-"+split)
    return entry


def _reference(ctx, split):
    pin = ctx["legacy"]["pose_references"][split]
    reference = P.R.read_targets(C.pinned(pin))
    C.pinned(pin)
    P.R.check_selection(reference, pin)
    C.require(reference["target"].shape[1:] == (160, 20), "historical pose reference geometry differs")
    return reference, pin


def _read(ctx, step, split):
    entry = _entry(ctx, step, split)
    reference, selection = _reference(ctx, split)
    data = P.read_cache(C.pinned(entry["cache"]), reference, selection)
    C.pinned(entry["cache"])
    expected = {"which": "teacher", "split": split, "checkpoint_step": step,
                "checkpoint": ctx["plan"]["endpoints"][str(step)]["checkpoint"]["path"],
                "checkpoint_sha256": ctx["plan"]["endpoints"][str(step)]["checkpoint"]["sha256"],
                "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"],
                "pose_stats_sha256": ctx["legacy"]["inputs"]["pose_stats"]["sha256"],
                "checkpoint_pose_stats_sha256": ctx["legacy"]["inputs"]["pose_stats"]["sha256"],
                "n_windows": 10, "window_s": 4., "seed": P.PE.SEED_WINDOWS, "per_side": False}
    C.require(all(data["config"].get(k) == v for k, v in expected.items()), "pose cache embedded identity/recipe differs")
    C.require(data["dim"] == 384 and data["target"].shape[1:] == (160, 20), "pose cache geometry differs")
    return data, entry


def _extract(ctx, out, stage, step):
    split = stage.removeprefix("extract-pose-")
    reference, selection = _reference(ctx, split)
    records, sources = C.E.pose_sources(ctx["legacy"], split)
    C.require(len(records) == selection["n_recordings"], "historical recording selection differs")
    model, guard = C.load_teacher(ctx, step, device="cuda")
    loader = C.T.R.sanity_config().loader
    original = C.canonical(asdict(loader))
    C.require(all(guard["config"]["loader"][k] == original[k] for k in C.E.PREPROCESSING),
              "pose preprocessing differs from historical loader")
    stats = C.pinned(ctx["legacy"]["inputs"]["pose_stats"])
    C.require(stats == Path(ctx["legacy"]["paths"]["pose_root"]) / "pose_stats.json"
              and str(stats.parent) == guard["config"]["loader"]["pose_root"], "pose root/statistics mismatch")
    ev = P.PE.EvalConfig(batch=ctx["plan"]["extraction_batch"]["pose"], device="cuda", loader=loader)
    store = C.E.PT.PoseStore(Path(ctx["legacy"]["paths"]["pose_root"]))
    data = P.extract_pose(model, records, store, ev, reference=reference, selection=selection)
    data["config"].update({"checkpoint": guard["checkpoint"], "checkpoint_step": step,
                          "checkpoint_sha256": guard["checkpoint_sha256"], "which": "teacher", "split": split,
                          "code_sha": ctx["code_sha"], "source_sha256": ctx["source_sha256"],
                          "recipe_sha256": ctx["plan"]["recipe_sha256"],
                          "checkpoint_pose_stats_sha256": guard["checkpoint_pose_stats_sha256"]})
    C.require(C.E.pose_sources(ctx["legacy"], split) == (records, sources), "pose source selection/inventory changed")
    C.require(C.E.source_states([Path(p) for p in sources]) == sources, "pose sources changed during extraction")
    for pin in ctx["legacy"]["inputs"].values():
        C.pinned(pin)
    C.pinned(ctx["plan"]["endpoints"][str(step)]["checkpoint"])
    C.require(not (Path(guard["checkpoint"]).parent / "STOP").exists(), "STOP present after extraction")
    path = Path(out) / (split+".npz")
    P.save_cache(path, data)
    details = P.validate_cache(P.read_cache(path, reference, selection), reference, selection)
    del data, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"which": "teacher", "checkpoint_step": step, "checkpoint_sha256": guard["checkpoint_sha256"],
            "checkpoint_guard": guard, "checkpoint_unchanged": True, "split": split, "details": details,
            "raw_inputs": {"file_states": sources, "unchanged": True}, "artifacts": {"cache": C.file_pin(path)}}


def project_benchmark(timing, peak_rss):
    """Fixed admission arithmetic. Neither timing nor projection reads held data."""
    names = ("read_validate", "fixed_alpha_fit", "predict", "summary", "E2_pairing", "bootstrap_diagnostic", "artifact_write")
    C.require(all(np.isfinite(timing[k]) and timing[k] >= 0 for k in names)
              and np.isfinite(peak_rss) and peak_rss > 0, "invalid pose benchmark measurements")
    C.require(set(timing["ridge_by_depth"]) == set(P.DEPTHS)
              and all(np.isfinite(d[k]) and d[k] >= 0 for d in timing["ridge_by_depth"].values()
                      for k in ("statistics_seconds", "three_eigensystems_seconds")), "invalid depth benchmark measurements")
    stats = {k: max(d[k] for d in timing["ridge_by_depth"].values())
             for k in ("statistics_seconds", "three_eigensystems_seconds")}
    floor = P.project_fit_seconds(stats, n_readers=2, safety_factor=2., fixed_seconds=0.)
    r = BENCHMARK_POLICY["held_windows"] / BENCHMARK_POLICY["train_windows"]
    other = (timing["read_validate"]*(1+r) + timing["fixed_alpha_fit"] + timing["predict"]*r
             + timing["summary"]*r + timing["E2_pairing"]*r*r + timing["bootstrap_diagnostic"]*r + timing["artifact_write"]*r)
    seconds = floor["fit_seconds"] + 2*other + 120.
    memory = float(peak_rss*r)
    return {"seconds_per_dose": seconds, "fit_floor": floor, "projected_peak_rss_bytes": memory,
            "admitted": seconds <= BENCHMARK_POLICY["max_seconds_per_dose"]
                        and memory <= BENCHMARK_POLICY["memory_fraction"]*BENCHMARK_POLICY["memory_bytes"]}


def _benchmark(ctx, out):
    started = perf_counter()
    train, entry = _read(ctx, 19618, "train")
    C.require(len(train["meta"]) == BENCHMARK_POLICY["train_windows"], "benchmark training selection differs")
    timing = {"read_validate": perf_counter()-started, "ridge_by_depth": {}, "fixed_alpha_fit": 0.,
              "predict": 0., "summary": 0., "artifact_write": 0.}
    artifacts = {}
    started = perf_counter()
    support = P.e2_support(train)
    timing["E2_pairing"] = perf_counter()-started
    analyses = {}
    for depth in P.DEPTHS:
        timing["ridge_by_depth"][depth] = P.benchmark_train(train, depth)
        started = perf_counter()
        parts, dim, joints = P.R.grouped_stats(P.view_data(train, depth))
        weights, bias = P.R.E.prepare_solver(P.R.E.merge_stats(list(parts.values()), joints), dim, joints)(1e5)
        timing["fixed_alpha_fit"] += perf_counter()-started
        del parts
        started = perf_counter()
        prediction = P.predict_reader(train, depth, weights, bias)
        timing["predict"] += perf_counter()-started
        started = perf_counter()
        artifacts[depth+"_coefficients"] = _save_array(Path(out)/(depth+".timing.coefficients.npz"), weights=weights, bias=bias)
        artifacts[depth+"_predictions"] = _save_array(Path(out)/(depth+".timing.predictions.npz"), prediction=prediction)
        timing["artifact_write"] += perf_counter()-started
        started = perf_counter()
        analyses[depth] = P.summarize(train, prediction, support=support)
        timing["summary"] += perf_counter()-started
        del prediction, weights, bias
    started = perf_counter()
    analyses["trained_head"] = P.summarize(train, train["pred"][:, P.OFFSET:-P.OFFSET], support=support)
    timing["summary"] += perf_counter()-started
    started = perf_counter()
    artifacts["trained_head_predictions"] = _save_array(Path(out)/"head.timing.predictions.npz",
                                                       prediction=train["pred"][:, P.OFFSET:-P.OFFSET].astype(np.float64))
    # Exercise report serialization and a same-size write without persisting
    # training metrics as evaluation evidence.
    n_report_bytes = len(json.dumps(analyses, sort_keys=True, allow_nan=False).encode())
    surrogate = Path(out)/"report_io_surrogate.json"
    C.write_json(surrogate, {"timing_only": True, "score_free_padding": "0"*n_report_bytes})
    artifacts["report_io_surrogate"] = C.file_pin(surrogate)
    timing["artifact_write"] += perf_counter()-started
    started = perf_counter()
    for group in ("dynamic", "all", "static"):
        P._bootstrap_group(analyses, group, P.N_BOOT, P.SEED)
    P._bootstrap_e2(analyses, P.N_BOOT, P.SEED)
    timing["bootstrap_diagnostic"] = perf_counter()-started
    peak = peak_rss_bytes()
    return {"checkpoint_step": 19618, "checkpoint_sha256": ctx["plan"]["endpoints"]["19618"]["checkpoint"]["sha256"],
            "benchmark_policy": BENCHMARK_POLICY, "fit_resources": FIT_RESOURCES, "timings_seconds": timing,
            "peak_rss_bytes": peak, "projection": project_benchmark(timing, peak), "train_cache": entry,
            "cache_manifest": ctx["plan"]["cache_manifest"], "held_payload_opened": False, "held_scored": False,
            "training_scores_persisted": False, "timing_artifacts_only": True,
            "artifacts": artifacts, "environment": environment()}


def _benchmark_gate(ctx, *, check_environment=True):
    pin = ctx["plan"]["benchmark_receipts"]["pose"]
    report = C.read_pin(pin)
    C.require(report.get("complete") is True and report.get("protocol") == C.PROTOCOL
              and report.get("kind") == "benchmark-pose" and report.get("source_sha256") == ctx["source_sha256"]
              and report.get("recipe_sha256") == ctx["plan"]["recipe_sha256"], "pose benchmark provenance differs")
    C.require(report.get("checkpoint_step") == 19618
              and report.get("checkpoint_sha256") == ctx["plan"]["endpoints"]["19618"]["checkpoint"]["sha256"]
              and report.get("cache_manifest") == ctx["plan"]["cache_manifest"]
              and report.get("train_cache") == _manifest(ctx)["pose"]["19618"]["train"], "pose benchmark data identity differs")
    C.require(report.get("benchmark_policy") == BENCHMARK_POLICY and report.get("fit_resources") == FIT_RESOURCES,
              "pose benchmark policy/resources differ")
    C.require(report.get("held_payload_opened") is False and report.get("held_scored") is False
              and report.get("training_scores_persisted") is False and report.get("timing_artifacts_only") is True,
              "pose benchmark used scoring payload")
    C.require(set(report.get("artifacts", {})) == {"pre_coefficients", "post_coefficients", "pre_predictions",
                                                    "post_predictions", "trained_head_predictions", "report_io_surrogate"},
              "pose benchmark timing artifact set differs")
    for artifact in report["artifacts"].values():
        C.pinned(artifact)
    C.require(not check_environment or report.get("environment") == environment(), "pose benchmark numeric environment differs")
    projection = project_benchmark(report["timings_seconds"], report["peak_rss_bytes"])
    C.require(report.get("projection") == projection and projection["admitted"] is True,
              "pose fit exceeds benchmark time/memory admission")
    return pin


def _save_array(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return C.file_pin(path)


def _fit(ctx, out, step, benchmark):
    train, train_entry = _read(ctx, step, "train")
    train_users = {m["user"] for m in train["meta"]}
    fitted, artifacts = {}, {}
    for depth in P.DEPTHS:
        weights, bias, fit = P.fit_reader(train, depth, ctx["pose_reader"]["fold_users"])
        reader = depth+"_ridge"
        coeff = _save_array(Path(out)/(reader+".coefficients.npz"), weights=weights, bias=bias)
        fitted[reader] = (depth, weights, bias, fit, coeff)
        artifacts[reader+"_coefficients"] = coeff
    del train
    gc.collect()
    # No held payload is opened until both training selections/refits complete.
    held, held_entry = _read(ctx, step, "val")
    C.require(train_users.isdisjoint({m["user"] for m in held["meta"]}), "pose train/held user leakage")
    support = P.e2_support(held)
    cells = {}
    for reader in P.READERS:
        started = perf_counter()
        if reader == "trained_head":
            prediction = held["pred"][:, P.OFFSET:-P.OFFSET].astype(np.float64)
            fit = {"kind": "historical trained post head", "supervised_updates": 0}
            coeff = None
        else:
            depth, weights, bias, fit, coeff = fitted[reader]
            prediction = P.predict_reader(held, depth, weights, bias)
        prediction_seconds = perf_counter()-started
        pred = _save_array(Path(out)/(reader+".predictions.npz"), prediction=prediction)
        artifacts[reader+"_predictions"] = pred
        cells[reader] = {"complete": True, "which": "teacher", "checkpoint_step": step,
                         "checkpoint_sha256": ctx["plan"]["endpoints"][str(step)]["checkpoint"]["sha256"],
                         "reader": reader, "fit": fit, "prediction_seconds": prediction_seconds,
                         "analysis": P.summarize(held, prediction, support=support),
                         "artifacts": {"prediction": pred, **({"coefficients": coeff} if coeff is not None else {})}}
        del prediction
    return {"checkpoint_step": step, "checkpoint_sha256": ctx["plan"]["endpoints"][str(step)]["checkpoint"]["sha256"],
            "which": "teacher", "cells": cells, "artifacts": artifacts, "caches": {"train": train_entry, "val": held_entry},
            "cache_manifest": ctx["plan"]["cache_manifest"], "benchmark_receipt": benchmark,
            "environment": environment(), "held_payload_opened_after_both_ridge_fits": True,
            "support_policy": "crop6 common support; E2 historical full-window partner choices frozen before crop6 intersection; distinct from historical full-support E2"}


def _compare(ctx):
    _manifest(ctx)
    benchmark_pin = _benchmark_gate(ctx, check_environment=False)
    benchmark_environment = C.read_pin(benchmark_pin)["environment"]
    fits = C.read_pin(ctx["plan"]["pose_fit_manifest"])["fits"]
    C.require(set(fits) == set(map(str, P.DOSES)), "pose fit manifest doses differ")
    cells, reports = {}, {}
    for step in P.DOSES:
        report = C.read_pin(fits[str(step)]["report"])
        C.require(report.get("complete") is True and report.get("protocol") == C.PROTOCOL and report.get("kind") == "fit-pose"
                  and report.get("source_sha256") == ctx["source_sha256"]
                  and report.get("recipe_sha256") == ctx["plan"]["recipe_sha256"]
                  and report.get("checkpoint_step") == step and report.get("which") == "teacher"
                  and report.get("checkpoint_sha256") == ctx["plan"]["endpoints"][str(step)]["checkpoint"]["sha256"],
                  "pose fit receipt provenance/dose differs")
        C.require(report.get("cache_manifest") == ctx["plan"]["cache_manifest"]
                  and report.get("benchmark_receipt") == ctx["plan"]["benchmark_receipts"]["pose"]
                  and report.get("held_payload_opened_after_both_ridge_fits") is True, "pose fit input/control provenance differs")
        C.require(report.get("environment") == benchmark_environment, "pose fit numeric environment differs from benchmark")
        C.require(report.get("caches") == {split: _entry(ctx, step, split) for split in ("train", "val")},
                  "pose fit cache identities differ")
        C.require(set(report["cells"]) == set(P.READERS), "pose fit readers missing")
        expected_artifacts = {r+"_predictions" for r in P.READERS} | {d+"_ridge_coefficients" for d in P.DEPTHS}
        C.require(set(report["artifacts"]) == expected_artifacts, "pose fit artifact set differs")
        for pin in report["artifacts"].values():
            C.pinned(pin)
        for reader, cell in report["cells"].items():
            expected = {"prediction": report["artifacts"][reader+"_predictions"]}
            if reader.endswith("_ridge"):
                expected["coefficients"] = report["artifacts"][reader+"_coefficients"]
            C.require(cell.get("artifacts") == expected, "pose cell artifact identity differs")
            C.require(cell.get("checkpoint_sha256") == report["checkpoint_sha256"], "pose cell checkpoint differs")
            cells[f"{step}/{reader}"] = cell
        reports[str(step)] = fits[str(step)]["report"]
    return {"comparison": P.aggregate(cells), "fit_reports": reports,
            "cache_manifest": ctx["plan"]["cache_manifest"], "artifacts": {}}


def execute(stage, ctx, out, step=None):
    C.require(stage in STAGES, "unregistered pose stage")
    if stage in ("extract-pose-train", "extract-pose-val", "fit-pose"):
        C.require(type(step) is int and step in P.DOSES, "pose stage requires registered dose")
    else:
        C.require(step is None, "pose benchmark/compare does not accept a selected dose")
    benchmark = _benchmark_gate(ctx) if stage == "fit-pose" else None
    report = C.begin_output(ctx, out, stage)
    started = perf_counter()
    if stage.startswith("extract-pose-"):
        details = _extract(ctx, out, stage, step)
    elif stage == "benchmark-pose":
        details = _benchmark(ctx, out)
    elif stage == "fit-pose":
        details = _fit(ctx, out, step, benchmark)
    else:
        details = _compare(ctx)
    return C.finish_output(ctx, out, {**report, **details, "elapsed_seconds": perf_counter()-started})
