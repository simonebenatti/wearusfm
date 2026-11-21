"""Measured, fail-closed resource gate for the frozen pose-reader experiment.

Values are seconds/bytes, never forecasts substituted for missing measurements.
The scheduler allowance is included once in every requested allocation.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

HEADS = ("L0", "L13", "MLP", "TCN_post", "TCN_pre")
CV_MULTIPLICITY = {"L0": 3, "L13": 1, "MLP": 3, "TCN_post": 3, "TCN_pre": 1}
GIB = 1024**3
OVERHEAD_SECONDS = 180
MAX_LRS, MAX_EPOCHS = 6, 300
MAX_EVALS = MAX_EPOCHS + 4  # epoch zero and three first-epoch fractions


class ResourceEvidenceError(ValueError):
    """Evidence is missing/inconsistent; it must not become an optimistic gate."""


def _require(condition, message):
    if not condition:
        raise ResourceEvidenceError(message)


def _number(value, name, *, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0 and (not positive or value > 0),
             f"missing or invalid measured {name}")
    return float(value)


def percentile95(samples):
    values = np.asarray(samples, dtype=float)
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
             and (values >= 0).all(), "invalid timing samples")
    return float(np.quantile(values, .95, method="linear"))


def matrix_counts(fold_windows, full_windows, *, batch_size=32):
    _require(len(fold_windows) == 2 and all(type(n) is int and n > 0 for n in fold_windows),
             "two measured nonempty fitting populations required")
    _require(type(full_windows) is int and full_windows == sum(fold_windows),
             "folds must partition full train")
    _require(batch_size == 32, "frozen batch size is 32")
    steps = [math.ceil(n / batch_size) for n in fold_windows]
    full_steps = math.ceil(full_windows / batch_size)
    return {"cv_trajectories": 132, "refit_trajectories": 15, "trajectories": 147,
            "maximum_lrs": MAX_LRS, "maximum_epochs": MAX_EPOCHS,
            "cv_train_epochs": 132 * MAX_EPOCHS,
            "refit_train_epochs": 15 * MAX_EPOCHS,
            "held_evaluations": 132 * MAX_EVALS,
            "fitting_evaluations": 37, "fold_steps": steps, "full_steps": full_steps,
            "updates": 11 * MAX_LRS * MAX_EPOCHS * sum(steps) + 15 * MAX_EPOCHS * full_steps}


def _receipt(receipt, kind):
    _require(receipt.get("kind") == kind and receipt.get("schema") == "pose-temponet-cost-v1",
             f"wrong {kind} benchmark schema")
    _require(receipt.get("train_only") is True and receipt.get("measured") is True
             and receipt.get("test_only") is False, "only real train-only measurements admit production")
    _require(isinstance(receipt.get("hardware"), Mapping) and receipt["hardware"], "hardware missing")
    _require(isinstance(receipt.get("numeric_profile"), Mapping), "numeric profile missing")
    profile = receipt["numeric_profile"]
    _require(profile.get("threads") == 8 and profile.get("interop_threads") == 1,
             "benchmark threads differ from frozen fits")
    _require(receipt.get("input_identity"), "benchmark input identity missing")


def _cell(cell):
    result = {}
    for metric in ("train", "eval", "save", "load_state", "artifact_write", "artifact_read"):
        samples = cell.get(metric + "_seconds")
        _require(isinstance(samples, list) and len(samples) == 10,
                 f"ten measured {metric} samples after warmup required")
        result[metric] = percentile95(samples)
        _require(result[metric] > 0, f"zero {metric} timing is not evidence")
    _require(cell.get("warmup_epochs") == 1 and cell.get("timed_epochs") == 10,
             "wrong benchmark warmup/horizon")
    for key in ("train_windows", "eval_windows"):
        _require(type(cell.get(key)) is int and cell[key] > 0, f"missing {key}")
        result[key] = cell[key]
    result["startup"] = _number(cell.get("startup_seconds"), "startup", positive=True)
    result["artifact_write_bytes"] = _number(cell.get("artifact_write_bytes"), "state artifact bytes", positive=True)
    return result


def _job(name, phase, raw_seconds, *, gpus=0, cpus=8, margin=2., requested_cap=None):
    _number(raw_seconds, name, positive=True)
    # SLURM stores a minute-granularity TimeLimit. Publish that exact value,
    # including its cost, rather than relying on an implicit scheduler round-up.
    requested = 60 * math.ceil(max(1, math.ceil(margin * raw_seconds)) / 60)
    allocation = requested + OVERHEAD_SECONDS
    return {"name": name, "phase": phase, "requested_seconds": requested,
            "overhead_seconds": OVERHEAD_SECONDS, "allocation_seconds": allocation,
            "gpus": gpus, "cpus": cpus, "host_ram_gib": 32 if gpus else 24, "gpu_hours": allocation * gpus / 3600,
            "core_hours": allocation * cpus / 3600,
            "within_job_limit": requested_cap is None or requested <= requested_cap}


def gate_and_pack(cpu_receipt, gpu_receipt, *, work_free_bytes, pending_write_bytes,
                  persistent_output_bytes, benchmark_gpu_seconds=None, benchmark_cpu_seconds=None,
                  checkout_seconds=720, collection_seconds=720, budget_caps=None):
    """Return all independent gates and exact packs; never silently shrink a matrix.

    Two CPU reference jobs, one per outer fold, preserve the signed 03:57 limit.
    Eleven paired-fold searches and five three-seed refits remain separate jobs;
    the operator adds dependency IDs, never changes their scientific contents.
    """
    from .budget_amendment import ORIGINAL_CAPS, CAPS
    limits = ORIGINAL_CAPS if budget_caps is None else budget_caps
    _require(limits in (ORIGINAL_CAPS, CAPS), "unauthorized budget ceilings")
    _receipt(cpu_receipt, "cpu")
    _receipt(gpu_receipt, "gpu")
    _require(cpu_receipt["input_identity"] == gpu_receipt["input_identity"], "input identities differ")
    cpu_numeric, gpu_numeric = cpu_receipt["numeric_profile"], gpu_receipt["numeric_profile"]
    _require(cpu_numeric == gpu_numeric, "CPU/GPU benchmark numerical profiles differ")
    _require(gpu_numeric.get("precision") == "float32_no_autocast"
             and gpu_numeric.get("deterministic_algorithms") is True
             and gpu_numeric.get("cudnn_deterministic") is True
             and gpu_numeric.get("cudnn_benchmark") is False
             and gpu_numeric.get("cudnn_tf32") is False
             and gpu_numeric.get("matmul_tf32") is False
             and gpu_numeric.get("cublas_workspace") == ":4096:8", "unfrozen numerical settings")
    shape = gpu_receipt["population"]
    counts = matrix_counts(shape["fit_fold_windows"], shape["fit_train_windows"])
    _require(len(shape["fold_windows"]) == 2 and all(shape["fit_fold_windows"][f] <= shape["fold_windows"][f] for f in (0,1))
             and shape["fit_train_windows"] <= shape["train_windows"]
             and shape["train_windows"] == sum(shape["fold_windows"])
             and shape.get("excluded_fitting_windows") == [shape["fold_windows"][f]-shape["fit_fold_windows"][f] for f in (0,1)],
             "fitting support counts differ from complete evaluation populations")
    _require(shape.get("validation_windows") == 18548, "validation-size projection must be 18,548")
    _require(cpu_receipt.get("population") == shape, "CPU/GPU populations differ")
    io = gpu_receipt["io"]
    load = _number(io.get("input_load_seconds"), "input load", positive=True)
    input_bytes = _number(io.get("input_bytes"), "input bytes", positive=True)
    write = _number(io.get("output_write_seconds"), "prediction write", positive=True)
    write_bytes = _number(io.get("output_write_bytes"), "prediction bytes", positive=True)
    references = _number(io.get("reference_read_seconds"), "reference read", positive=True)
    _number(io.get("reference_read_bytes"), "reference bytes", positive=True)
    jobs, training_jobs = [], []
    cells = {}
    for head in HEADS:
        cells[head] = {part: _cell(gpu_receipt["cells"][head][part]) for part in ("fold0", "fold1", "full")}
        for fold in (0, 1):
            c = cells[head][f"fold{fold}"]
            _require(c["train_windows"] == shape["fit_fold_windows"][fold]
                     and c["eval_windows"] == shape["fold_windows"][1-fold], "measured fold geometry differs")
        _require(cells[head]["full"]["train_windows"] == shape["fit_train_windows"]
                 and cells[head]["full"]["eval_windows"] == shape["train_windows"], "full measurement must use train only")
        for group in range(CV_MULTIPLICITY[head]):
            raw = load
            for fold in (0, 1):
                c = cells[head][f"fold{fold}"]
                # Current/best snapshots are in-memory operations. The final
                # artifact retains two states per fold/LR (24 per search),
                # rather than performing a filesystem write at every epoch.
                raw += MAX_LRS * (MAX_EPOCHS*c["train"] + MAX_EVALS*c["eval"]
                                 + 2*MAX_EVALS*c["save"] + 2*c["load_state"] + c["startup"]
                                 + 2*c["artifact_write"])
                raw += c["eval"] * shape["fold_windows"][fold] / c["eval_windows"]
            name = f"select-{head}-" + ("real" if group == 0 else ("linear" if group == 1 else "temporal"))
            job = _job(name, "neural", raw, gpus=1, requested_cap=24*3600-180)
            job.update(stage="selection" if group == 0 else "controls", head=head,
                       world=None if group == 0 else ("linear" if group == 1 else "temporal"))
            training_jobs.append(job)
        c = cells[head]["full"]
        raw = load + 3*(MAX_EPOCHS*c["train"] + c["eval"] + c["save"]
                       + c["load_state"] + c["startup"] + c["artifact_write"])
        job = _job(f"refit-{head}", "neural", raw, gpus=1, requested_cap=24*3600-180)
        job.update(stage="refit", head=head, world=None)
        training_jobs.append(job)
    jobs.extend(training_jobs)
    # Stream final predictions: one readout at a time; scale by both bytes and
    # population, so a compressible benchmark cannot under-price larger data.
    val_bytes = _number(gpu_receipt["memory"].get("validation_view_bytes"), "validation view", positive=True)
    input_scale = max(1., shape["validation_windows"]/shape["train_windows"], val_bytes/input_bytes)
    final_raw = 2*load*input_scale + references
    for head in HEADS:
        c = cells[head]["full"]
        batches = gpu_receipt["cells"][head]["full"].get("inference_batches")
        _require(isinstance(batches, Mapping), "full and final partial inference batch measurements missing")
        full_batch = percentile95(batches.get("full_seconds", []))
        last_batch = percentile95(batches.get("last_seconds", []))
        _require(len(batches.get("full_seconds", [])) == 10 and len(batches.get("last_seconds", [])) == 10
                 and batches.get("last_size") == 18548 % 32 and full_batch > 0 and last_batch > 0,
                 "ten full/actual-last batch timings required")
        final_raw += 3*(18548//32 * full_batch + last_batch + c["artifact_read"]
                       + c["load_state"] + c["startup"])
    predicted_bytes = 15*18548*160*20*4
    final_raw += write * max(1., predicted_bytes/write_bytes)
    infer_job = _job("final-inference", "inference", final_raw, gpus=1, requested_cap=3600-180)
    infer_job["projected_required_seconds"] = infer_job["requested_seconds"]
    infer_job.update(requested_seconds=3420, allocation_seconds=3600, gpu_hours=1., core_hours=8.)
    jobs.append(infer_job)
    folds = cpu_receipt.get("fold_projection_seconds")
    _require(isinstance(folds, list) and len(folds) == 2, "two measured/projected CPU folds required")
    _require(cpu_receipt.get("all_patterns_measured") is True
             and cpu_receipt.get("all_alphas_measured") is True
             and cpu_receipt.get("grid_combinations_measured") == [1936, 1936],
             "CPU measurement lacks complete designs/patterns/alpha/grid")
    for f, raw in enumerate(folds):
        job = _job(f"references-fold{f}", "references", raw, requested_cap=4*3600-180)
        job.update(requested_seconds=14220, allocation_seconds=14400, core_hours=32.)
        jobs.append(job)
    report_job = _job("report", "report", _number(cpu_receipt.get("report_projection_seconds"), "report", positive=True), requested_cap=3600-180)
    report_job.update(requested_seconds=3420, allocation_seconds=3600, core_hours=8.)
    jobs.append(report_job)
    # Measured benchmark allocations, including one grace allowance each.
    gpu_bench = gpu_receipt.get("elapsed_seconds") if benchmark_gpu_seconds is None else benchmark_gpu_seconds
    cpu_bench = cpu_receipt.get("elapsed_seconds") if benchmark_cpu_seconds is None else benchmark_cpu_seconds
    gpu_job = _job("benchmark-gpu", "benchmark_gpu", gpu_bench, gpus=1, margin=1., requested_cap=1800-180)
    gpu_job.update(requested_seconds=1620, allocation_seconds=1800, gpu_hours=.5, core_hours=4.)
    jobs.append(gpu_job)
    cpu_job = _job("benchmark-cpu", "benchmark_cpu", cpu_bench, margin=1., requested_cap=1800-180)
    cpu_job.update(requested_seconds=1620, allocation_seconds=1800, core_hours=4.)
    jobs.append(cpu_job)
    jobs.append(_job("checkout", "checkout", checkout_seconds, cpus=2, margin=1., requested_cap=900-180))
    jobs.append(_job("collection", "collection", collection_seconds, cpus=2, margin=1., requested_cap=900-180))
    def phase(name, key):
        return sum(job[key] for job in jobs if job["phase"] == name)
    gpu_hours = sum(job["gpu_hours"] for job in jobs)
    cpu_hours = sum(job["core_hours"] for job in jobs if not job["gpus"])
    total_core_hours = sum(job["core_hours"] for job in jobs)
    gpu_jobs, cpu_jobs = sum(j["gpus"] > 0 for j in jobs), sum(j["gpus"] == 0 for j in jobs)
    gm, cm = gpu_receipt["memory"], cpu_receipt["memory"]
    gpu_host = max(_number(gm.get("peak_rss_bytes"), "GPU RSS", positive=True),
                   1.25*sum(_number(gm.get(k), k) for k in ("fixed_host_bytes", "validation_view_bytes", "batch_bytes")))
    cpu_host = max(_number(cm.get("peak_rss_bytes"), "CPU RSS", positive=True),
                   1.25*_number(cm.get("projected_peak_bytes"), "CPU projected RSS", positive=True))
    vram = _number(gm.get("peak_vram_bytes"), "VRAM", positive=True)
    available = _number(work_free_bytes, "WORK free")
    pending = _number(pending_write_bytes, "pending writes")
    output = _number(persistent_output_bytes, "output projection", positive=True)
    gates = {"neural_gpu_hours": phase("neural", "gpu_hours") <= limits["neural_gpu_hours"],
             "inference_gpu_hours": phase("inference", "gpu_hours") <= 1 and infer_job["within_job_limit"],
             "references_core_hours": phase("references", "core_hours") <= 64 and all(j["within_job_limit"] for j in jobs if j["phase"] == "references"),
             "report_core_hours": phase("report", "core_hours") <= 8 and report_job["within_job_limit"],
             "benchmark_gpu_hours": phase("benchmark_gpu", "gpu_hours") <= .5 and gpu_job["within_job_limit"],
             "benchmark_cpu_core_hours": phase("benchmark_cpu", "core_hours") <= 4 and cpu_job["within_job_limit"],
             "gpu_total": gpu_hours <= limits["gpu_total_hours"],
             "cpu_only_total": cpu_hours <= limits["cpu_only_core_hours"],
             "core_total": total_core_hours <= limits["total_core_hours"],
             "job_counts": gpu_jobs <= 40 and cpu_jobs <= 7,
             "job_limits": all(j["within_job_limit"] for j in jobs),
             "cpu_host_memory": cpu_host <= 24*GIB, "gpu_host_memory": gpu_host <= 32*GIB,
             "vram": vram <= 48*GIB,
             "quota": available >= 12*GIB+pending,
             "persistent_output": predicted_bytes <= output <= 8*GIB}
    return {"all_passed": all(gates.values()), "gates": gates, "matrix": counts, "jobs": jobs,
            "projection": {"gpu_hours": gpu_hours, "cpu_only_core_hours": cpu_hours,
                           "allocated_core_hours": total_core_hours, "gpu_jobs": gpu_jobs, "cpu_jobs": cpu_jobs,
                           "gpu_host_bytes": gpu_host, "cpu_host_bytes": cpu_host,
                           "vram_bytes": vram, "prediction_bytes": predicted_bytes},
            "policy": {"factor": 2., "overhead_seconds_per_job": 180, "walltime_quantum_seconds": 60,
                       "packing": "paired-selection-and-three-refit-seeds",
                       "maximum_lrs": 6, "maximum_epochs": 300, "readout_only": True}}
