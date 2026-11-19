"""Train-only, instrumented cost measurements; no reader is selected or released.

A benchmark receipt is evidence about the actual input, hardware and file
placement. Missing measurements raise; there are no reference hardware speeds.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
from typing import Callable

import numpy as np
import torch

from . import controls, data as data_api, heads, linear, metrics, resources, train, workflow, provenance as P

SCHEMA = "pose-temponet-cost-v1"


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _hardware():
    from threadpoolctl import threadpool_info
    result = {"hostname": platform.node(), "system": platform.platform(),
              "processor": platform.processor(), "python": platform.python_version(),
              "numpy": np.__version__, "torch": str(torch.__version__),
              "cpu_count": os.cpu_count(), "threadpools": threadpool_info(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["cuda"] = {"name": props.name, "total_memory": props.total_memory,
                          "capability": [props.major, props.minor],
                          "runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    return result


def _timed(operation, *, cuda=False):
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    if cuda:
        torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _array_bytes(data):
    return sum(v.nbytes for v in data.values() if isinstance(v, np.ndarray))


def _validate_train(data, allowed_train_users, input_identity, folds=None, *, _expected_users=40):
    if not input_identity:
        raise ValueError("pinned input identity required")
    config = data.get("config", {})
    if config.get("split") != "train":
        raise ValueError("benchmark requires config.split=train, never development data")
    if config.get("which") != "teacher" or config.get("checkpoint_step") != 49046:
        raise ValueError("benchmark requires the frozen teacher at checkpoint 49046")
    labels = [row["user"] for row in data["meta"]]
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("training users must be normalized nonempty strings")
    users = set(labels)
    if len(users) != _expected_users or users != set(allowed_train_users):
        raise ValueError("benchmark population must contain exactly the declared training users")
    if data.get("split") in ("val", "validation", "development", "test", "held"):
        raise ValueError("benchmark cannot consume development data")
    if np.asarray(data["lat"]).shape != (len(data["meta"]), 160, 384):
        raise ValueError("benchmark requires real 160 by 384 cache geometry")
    if np.asarray(data["target"]).shape != (len(data["meta"]), 160, 20):
        raise ValueError("benchmark target geometry differs")
    if folds is None:
        users = sorted(users)
        selected = [set(users[::2]), set(users[1::2])]
        folds = [np.array([i for i, row in enumerate(data["meta"]) if str(row["user"]) in u], dtype=int)
                 for u in selected]
    if len(folds) != 2:
        raise ValueError("two fitting populations required")
    controls.validate_outer_partition(data["meta"], folds[0], folds[1], allowed_train_users)
    if any(len({data["meta"][int(i)]["user"] for i in fold}) != _expected_users//2 for fold in folds):
        raise ValueError("benchmark requires equal 20-user fitting folds")
    if set(map(int, np.concatenate(folds))) != set(range(len(data["meta"]))):
        raise ValueError("benchmark folds do not cover train")
    support = data_api.central_support(data["mask"], data_api.lengths_for(data), data.get("key_time_valid"))
    supported = support.any(axis=(1, 2))
    fitted = [int(supported[ix].sum()) for ix in folds]
    if min(fitted) == 0:
        raise ValueError("each fitting fold needs supported windows")
    population = {"train_windows": len(data["meta"]), "fold_windows": [len(x) for x in folds],
                  "fit_train_windows": int(supported.sum()), "fit_fold_windows": fitted,
                  "excluded_fitting_windows": [len(ix)-int(supported[ix].sum()) for ix in folds],
                  "validation_windows": 18548}
    return tuple(np.asarray(x, dtype=int) for x in folds), population


def _base_receipt(kind, population, input_identity, numeric):
    return {"schema": SCHEMA, "kind": kind, "train_only": True, "measured": True,
            "test_only": False, "input_identity": input_identity, "population": population,
            "numeric_profile": numeric, "hardware": _hardware()}


def _load_measurement(loader: Callable, input_identity):
    if not callable(loader):
        raise ValueError("actual train-cache loader required to measure decompression and placement")
    loaded, elapsed = _timed(loader)
    if not isinstance(loaded, dict) or not _array_bytes(loaded):
        raise ValueError("loader must return the actual decompressed cache mapping")
    # Caller pins the source files. Check a stable population fingerprint here,
    # rather than timing a synthetic in-memory stand-in.
    identity = hashlib.sha256(controls.canonical(loaded["meta"]).encode()).hexdigest()
    return loaded, {"input_load_seconds": elapsed, "input_bytes": _array_bytes(loaded),
                    "population_sha256": identity, "source_identity": input_identity,
                    "load_scope": "first_application_load", "os_page_cache_controlled": False}


def _input_io(data, loader, input_identity, input_measurement):
    """Reuse the first runtime load; never time a silent second warm reload."""
    if input_measurement is None:
        loaded, evidence = _load_measurement(loader, input_identity)
        if controls.canonical(loaded["meta"]) != controls.canonical(data["meta"]):
            raise ValueError("timed loader population differs")
        # Direct callers supplied an already loaded cache; this is not cold IO.
        evidence["load_scope"] = "direct_caller_reload"
        return evidence
    evidence = dict(input_measurement)
    expected = hashlib.sha256(controls.canonical(data["meta"]).encode()).hexdigest()
    if (evidence.get("population_sha256") != expected or evidence.get("source_identity") != input_identity or
            evidence.get("input_bytes") != _array_bytes(data) or evidence.get("input_load_seconds", 0) <= 0 or
            evidence.get("load_scope") != "first_application_load"):
        raise ValueError("first-load evidence differs from measured cache")
    return evidence


def _scoring_bundle(data, folds):
    """Only a cost adapter; evaluations run the exact production implementation."""
    full = np.arange(len(data["meta"]))
    populations = (*folds, full)
    target = np.asarray(data["target"])
    variance = metrics.dynamic_variance(target, data["mask"])
    thresholds = tuple(metrics.training_threshold(target[ix], data["mask"][ix]) for ix in populations)
    support = data_api.central_support(data["mask"], data_api.lengths_for(data), data.get("key_time_valid"))
    return workflow.FoldBundle(data, "real", tuple(populations), (folds[1], folds[0], full),
                               (target,)*3, (target,)*3, (variance,)*3, thresholds, support, {})


def _production_score(model, bundle, fold, held, depth, device="cuda", *, include_clean=False):
    return workflow._score_model(model, bundle, fold, held, depth, device, include_clean=include_clean)


def _prediction_batch(model, data, depth, rows, device="cuda"):
    """Same input/cast/crop/finite path as the locked production inference."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            x = torch.from_numpy(data_api.view_inputs(data, depth, rows)).to(device)
            values = model(x)[:, data_api.CROP:-data_api.CROP].detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("nonfinite cost prediction")
            return values
    finally:
        model.train(was_training)


def _read_state_artifact(pin):
    check = P.verify_pin(pin)
    value = torch.load(pin["path"], map_location="cpu", weights_only=False)
    P.unchanged({pin["path"]: check["state"]})
    return value


def _dataset(data, indices, depth):
    if depth not in ("pre", "post"):
        raise ValueError("invalid depth")
    source = data["pre" if depth == "pre" else "lat"]
    view = np.asarray(source)[indices].copy()
    if view.dtype != np.float16:
        raise ValueError("production cache storage must remain FP16 until the batch cast")
    lengths = data_api.lengths_for(data)[indices]
    available = np.arange(view.shape[1])[None, :] < lengths[:, None]
    if not np.isfinite(view[available]).all():
        raise ValueError("nonfinite real latent context")
    view[~available] = 0
    mask = data_api.central_support(data["mask"], data_api.lengths_for(data), data.get("key_time_valid"))
    identifiers = [controls.canonical(data["meta"][int(i)]) for i in indices]
    return train.TensorTrainingSet(torch.from_numpy(view),
                                   torch.from_numpy(np.asarray(data["target"])[indices].astype(np.float32)),
                                   torch.from_numpy(mask[indices]), identifiers)


def _eval_model(model, dataset, panel, threshold, *, device="cuda"):
    model.eval()
    n = len(dataset.features)
    if n != len(panel["meta"]):
        raise ValueError("evaluation storage and panel differ")
    output = np.empty((n, 160, 20), dtype=np.float32)
    records = []
    # Evaluation covers all windows in physical order. Empty support affects
    # metric accumulators, never the population or timing of this forward.
    with torch.no_grad():
        for start in range(0, n, 32):
            stop = min(start+32, n)
            x = dataset.features[start:stop].to(device=device, dtype=torch.float32)
            prediction = model(x).detach().cpu().numpy()
            output[start:stop] = prediction
            rows = np.arange(start, stop)
            records.append(metrics.selection_statistics(prediction, data_api.subset(panel, rows), threshold))
    return output, controls.pool_statistics(records)


def _reference_read(paths):
    if not paths:
        raise ValueError("pinned reference files required for inference I/O cost")
    total = 0
    started = time.perf_counter()
    for raw in paths:
        path = Path(raw)
        with path.open("rb") as handle:
            while chunk := handle.read(8*1024**2):
                total += len(chunk)
    if not total:
        raise ValueError("reference I/O measurement is empty")
    return {"reference_read_seconds": time.perf_counter()-started, "reference_read_bytes": total}



def _reference_proxy(train_path, declared_reference_bytes):
    if type(declared_reference_bytes) is not int or declared_reference_bytes <= 0:
        raise ValueError("positive pinned reference byte count required")
    measured = _reference_read([train_path])
    scale = max(1., declared_reference_bytes/measured["reference_read_bytes"])
    return {"reference_read_seconds": measured["reference_read_seconds"]*scale,
            "reference_read_bytes": declared_reference_bytes,
            "reference_io_is_projection": True,
            "reference_proxy_path": str(train_path),
            "reference_proxy_bytes": measured["reference_read_bytes"],
            "reference_proxy_seconds": measured["reference_read_seconds"],
            "reference_projection_multiplier": scale,
            "development_reference_files_opened": False}

def _validation_surrogate(data, n_validation=18548):
    """Touch one FP32 feature view and streaming target/prediction/mask buffers.

    Values only repeat train. The dimensions reproduce the declared downstream
    view and are not evidence from development labels.
    """
    n = len(data["lat"])
    source = np.asarray(data["lat"])
    arrays = (np.empty((n_validation, 160, 384), np.float32),
              np.empty((n_validation, 160, 20), np.float32),
              np.empty((n_validation, 160, 20), np.float32),
              np.empty((n_validation, 160, 20), bool))
    for start in range(0, n_validation, 32):
        ids = np.arange(start, min(start+32, n_validation)) % n
        arrays[0][start:start+len(ids)] = source[ids]
        arrays[1][start:start+len(ids)] = np.asarray(data["target"])[ids]
        arrays[2][start:start+len(ids)] = np.asarray(data["target"])[ids]
        arrays[3][start:start+len(ids)] = np.asarray(data["mask"])[ids]
    size, peak = sum(a.nbytes for a in arrays), _rss()
    del arrays
    gc.collect()
    return size, peak


def benchmark_gpu(data, *, input_loader, output_dir, reference_byte_count, reference_proxy_path, allowed_train_users,
                  input_identity, folds=None, input_measurement=None):
    """Measure all 5 heads × 2 folds/full, 1 excluded warmup + 10 epochs.

    Requires the actual CUDA device and production data; CPU fixtures cannot
    accidentally become production receipts. Artifacts are cost-only states.
    """
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("a real CUDA benchmark is required; no CPU timing fallback")
    numeric = train.configure_numerics()
    folds, population = _validate_train(data, allowed_train_users, input_identity, folds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    io = _input_io(data, input_loader, input_identity, input_measurement)
    gc.collect()
    fixed_host = _rss()
    validation_bytes, surrogate_peak = _validation_surrogate(data)
    torch.cuda.reset_peak_memory_stats()
    cells = {}
    full = np.arange(population["train_windows"])
    bundle = _scoring_bundle(data, folds)
    last_prediction = None
    for name in heads.HEADS:
        cells[name] = {}
        depth = "pre" if name == "TCN_pre" else "post"
        for part, fit, held, fold in (("fold0", folds[0], folds[1], 0),
                                      ("fold1", folds[1], folds[0], 1), ("full", full, full, 2)):
            dataset_started = time.perf_counter()
            fitting = _dataset(data, fit, depth)
            # The first state hashes the immutable dataset once; charge that
            # startup cost once rather than multiplying it by every checkpoint.
            fitting.identity
            dataset_seconds = time.perf_counter()-dataset_started
            # Factory, optimizer and initial host-to-device placement are timed.
            trajectory, startup = _timed(lambda: train.Trajectory(
                lambda seed, f: heads.make_head(name, seed), fitting,
                lr=1e-3, seed=0, fold=fold, device="cuda"), cuda=True)
            trajectory.train_to_step(trajectory.steps_per_epoch)
            _production_score(trajectory.model, bundle, fold, held, depth, include_clean=True)
            warm_state = trajectory.state()
            trajectory.restore(warm_state)
            warm_pin = P.atomic_torch(output_dir/f"cost-{name}-{part}-warm.pt", warm_state, root=output_dir)
            _read_state_artifact(warm_pin)
            torch.cuda.synchronize()
            record = {"startup_seconds": startup+dataset_seconds, "warmup_epochs": 1, "timed_epochs": 10,
                      "train_windows": fitting.size, "eval_windows": len(held),
                      "evaluation_path": "workflow._score_model", "evaluation_includes_clean_control_score": True,
                      "dataset_identity_warmed": True,
                      "save_scope": "in_memory_state_capture", "load_state_scope": "in_memory_restore",
                      "excluded_fitting_windows": fitting.excluded_windows,
                      "steps_per_epoch": trajectory.steps_per_epoch,
                      **{key+"_seconds": [] for key in ("train", "eval", "save", "load_state", "artifact_write", "artifact_read")}}
            artifact_sizes = []
            for epoch in range(10):
                _, seconds = _timed(lambda: trajectory.train_to_step((epoch+2)*trajectory.steps_per_epoch), cuda=True)
                record["train_seconds"].append(seconds)
                _, seconds = _timed(
                    lambda: _production_score(trajectory.model, bundle, fold, held, depth, include_clean=True), cuda=True)
                record["eval_seconds"].append(seconds)
                state, seconds = _timed(trajectory.state, cuda=True)
                record["save_seconds"].append(seconds)
                _, seconds = _timed(lambda: trajectory.restore(state), cuda=True)
                record["load_state_seconds"].append(seconds)
                checkpoint = output_dir/f"cost-{name}-{part}-{epoch}.pt"
                pin, seconds = _timed(lambda: P.atomic_torch(checkpoint, state, root=output_dir))
                record["artifact_write_seconds"].append(seconds)
                artifact_sizes.append(pin["bytes"])
                _, seconds = _timed(lambda: _read_state_artifact(pin))
                record["artifact_read_seconds"].append(seconds)
            if part == "full":
                batches = {"full_seconds": [], "last_seconds": [], "last_size": 18548 % 32}
                for _ in range(10):
                    for size, key in ((32, "full_seconds"), (18548 % 32, "last_seconds")):
                        rows = np.arange(size) % len(data["meta"])
                        _, seconds = _timed(lambda: _prediction_batch(trajectory.model, data, depth, rows), cuda=True)
                        batches[key].append(seconds)
                record["inference_batches"] = batches
            record["checkpoint_file_bytes"] = max(artifact_sizes)
            record["artifact_write_bytes"] = max(artifact_sizes)
            record["curve_record_bytes"] = len(json.dumps(trajectory.epoch_losses[-1], sort_keys=True).encode())
            record["statistics"] = {key: {"median": float(np.median(record[key+"_seconds"])),
                                                 "p95": resources.percentile95(record[key+"_seconds"])}
                                    for key in ("train", "eval", "save", "load_state", "artifact_write", "artifact_read")}
            cells[name][part] = record
            if name == heads.HEADS[-1] and part == "full":
                last_prediction = np.concatenate([_prediction_batch(trajectory.model, data, depth, full[start:start+32])
                                                  for start in range(0, len(full), 32)])
            del fitting, trajectory, warm_state, state

            gc.collect()
    prediction_path = output_dir/"cost-prediction.npz"
    _, write_seconds = _timed(lambda: np.savez_compressed(prediction_path, prediction=last_prediction))
    # Scale by uncompressed payload too: downstream predictions may compress less.
    io.update(output_write_seconds=write_seconds, output_write_bytes=min(int(last_prediction.nbytes), prediction_path.stat().st_size))
    io.update(_reference_proxy(reference_proxy_path, reference_byte_count))
    checkpoint_bytes = max(c["checkpoint_file_bytes"] for parts in cells.values() for c in parts.values())
    curve_record_bytes = max(c["curve_record_bytes"] for parts in cells.values() for c in parts.values())
    storage = {"prediction_payload_bytes": 15*18548*160*20*4,
               "checkpoint_bytes_max_measured": checkpoint_bytes,
               "retained_checkpoint_count": 2*147,
               "curve_record_bytes_measured": curve_record_bytes,
               "curve_records_bound": 40128,
               "control_payload_bytes": 2*3*population["train_windows"]*160*20*8}
    # Two current/selected states per maximal trajectory; twice the serialized
    # curve record and control metadata allowance, plus the actual cost files.
    storage["projected_bytes"] = (storage["prediction_payload_bytes"] + 294*checkpoint_bytes
                                  + 2*40128*curve_record_bytes + 2*storage["control_payload_bytes"]
                                  + sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file()))
    result = _base_receipt("gpu", population, input_identity, numeric)
    result.update(cells=cells, io=io, storage=storage, persistent_output_bytes=storage["projected_bytes"],
                  memory={"peak_rss_bytes": max(_rss(), surrogate_peak), "fixed_host_bytes": fixed_host,
                          "validation_view_bytes": validation_bytes,
                          "batch_bytes": 32*160*(384*4+20*4+20*4+20),
                          "peak_vram_bytes": int(torch.cuda.max_memory_reserved()),
                          "surrogate_uses_train_only": True},
                  elapsed_seconds=time.perf_counter()-started)
    return result



def _design_memory(support, indices, *, dimension, basis_count=4, block_windows=32):
    groups = linear.mask_groups(support[np.asarray(indices)])
    factors = [8*(8*(dimension+1)**2 + 4*(dimension+1)*len(g)*basis_count) for g in groups]
    scratch = max(3*(block_windows*support.shape[1]+dimension+1)
                  *(dimension+1+len(g)*basis_count)*8 for g in groups)
    return {"dimension": dimension, "patterns": len(groups), "factor_bytes": factors,
            "resident_factor_bytes": sum(factors), "max_scratch_bytes": scratch,
            "projected_working_bytes": sum(factors)+scratch}


def _memory_projection(data, support, folds, *, baseline_rss, block_windows=32):
    """Mirror the factors retained concurrently by the actual reference code."""
    dimension = np.asarray(data["lat"]).shape[-1]
    full_target64 = np.asarray(data["target"]).size*8
    # L/T/noise, observed/clean worlds, predictions/RHS copies and normalization
    # buffers; moments are K<=3 quadratic arrays for all alpha paths.
    target_buffers = 10*full_target64
    moment_buffers = len(data["meta"])*14*3*3*3*8
    plans = []
    for f in (0, 1):
        fit = folds[f]
        outer = [_design_memory(support, fit, dimension=dimension*k, block_windows=block_windows) for k in (1,13)]
        inner = [_design_memory(support, inner_fit, dimension=dimension, block_windows=block_windows)
                 for inner_fit, _ in controls.inner_partitions(data["meta"], fit)]
        # evaluate_outer_references keeps L0 while creating/using L13;
        # calibration retains both inner factors until the grid is chosen.
        simultaneous = {name: sum(d["resident_factor_bytes"] for d in designs)
                         +max(d["max_scratch_bytes"] for d in designs)
                        for name, designs in (("outer",outer),("inner",inner))}
        plans.append({"outer_fold":f,"outer_designs":outer,"inner_designs":inner,
                      "simultaneous_working_bytes":simultaneous})
    working = max(max(p["simultaneous_working_bytes"].values()) for p in plans)
    projected = baseline_rss+working+target_buffers+moment_buffers
    runtime_guard = all(d["resident_factor_bytes"]<=24*resources.GIB
                        for p in plans for d in p["outer_designs"]+p["inner_designs"])
    return {"folds":plans,"baseline_rss_bytes":baseline_rss,"target_buffer_bytes":target_buffers,
            "quadratic_buffer_bytes":moment_buffers,"projected_peak_bytes":projected,
            "projected_with_margin_bytes":1.25*projected,"runtime_factor_bound_passed":runtime_guard,
            "within_host_cap":runtime_guard and 1.25*projected<=24*resources.GIB}

def _measure_design(features, basis, support, fit, held, *, context, block_windows=32):
    """Measure full-width QR/SVD, projecting only the number of row blocks.

    At least one update after R attains its full row dimension is timed before
    extrapolation. If the complete fitting design has fewer rows, all blocks
    are measured. Rank is never inferred from a smaller feature space.
    """
    dimension = features.shape[-1]*(2*context+1)
    n_joints = support.shape[-1]
    names = tuple(basis)
    groups = linear.mask_groups(support[np.asarray(fit)])
    records, coefficients = [], {alpha: {name: np.empty((dimension+1, n_joints), np.float64) for name in names}
                                  for alpha in (0., *(10.**k for k in range(-5, 8)))}
    total_projection, memory_projection = 0., 0
    design_memory = _design_memory(support, fit, dimension=dimension, basis_count=len(names), block_windows=block_windows)
    if design_memory["resident_factor_bytes"] > 24*resources.GIB:
        raise resources.ResourceEvidenceError("all retained mask factors exceed the runtime24GiB bound")
    for joints in groups:
        factor = linear.WeightedQR(dimension, len(joints)*len(names))
        rows = []
        saturated_update_measured = False
        blocks = [fit[start:start+block_windows] for start in range(0, len(fit), block_windows)]
        total_blocks = len(blocks)
        max_supported_ids = max(blocks, key=lambda ids: int(support[ids, :, joints[0]].sum()))
        max_supported_rows = int(support[max_supported_ids, :, joints[0]].sum())
        def build(ids):
            x = linear.design_block(features[ids], context=context)
            q = support[ids].sum(axis=(1, 2))
            valid = support[ids, :, joints[0]]
            weights = np.divide(valid, q[:, None], out=np.zeros(valid.shape, np.float64), where=q[:, None] > 0)
            y = np.concatenate([np.asarray(basis[key])[ids, 6:-6][:, :, joints] for key in names], axis=-1)
            return x.reshape(-1, dimension), y.reshape(-1, len(joints)*len(names)), weights.ravel()
        for ids in blocks:
            (x, y, weights), build_seconds = _timed(lambda: build(ids))
            before_saturated = len(factor.r) == dimension+1
            _, qr_seconds = _timed(lambda: factor.add(x, y, weights))
            memory_projection = max(memory_projection, factor.projected_factor_bytes
                                    + 3*(len(x)+len(factor.r))*(dimension+1+factor.n_rhs)*8)
            rows.append({"windows": len(ids), "rows": len(x), "supported_rows": int((weights > 0).sum()),
                         "design_seconds": build_seconds, "qr_seconds": qr_seconds,
                         "r_rows": len(factor.r), "already_full_r": before_saturated})
            del x, y, weights
            if before_saturated and len(ids) == block_windows:
                saturated_update_measured = True
                break
        if len(rows) < total_blocks and not saturated_update_measured:
            raise ValueError("cannot project TSQR before a full-width saturated update is measured")
        # Masks can make the first saturated block cheaper than a later one.
        # Time the actual maximum-row block against a saturated R as well;
        # this duplicate is cost-only and its factor is never released.
        if len(rows) < total_blocks and max(r["supported_rows"] for r in rows if r["already_full_r"]) < max_supported_rows:
            (x, y, weights), build_seconds = _timed(lambda: build(max_supported_ids))
            _, qr_seconds = _timed(lambda: factor.add(x, y, weights))
            memory_projection = max(memory_projection, factor.projected_factor_bytes
                                    +3*(len(x)+len(factor.r))*(dimension+1+factor.n_rhs)*8)
            rows.append({"windows": len(max_supported_ids), "rows": len(x),
                         "supported_rows": int((weights > 0).sum()), "design_seconds": build_seconds,
                         "qr_seconds": qr_seconds, "r_rows": len(factor.r), "already_full_r": True,
                         "projection_probe_only": True})
            del x, y, weights
        solve_seconds, diagnostics = {}, {}
        for alpha in coefficients:
            solution, seconds = _timed(lambda: factor.solve(alpha))
            solve_seconds[str(alpha)] = seconds
            diagnostics[str(alpha)] = {"rank": solution.rank,
                                      "normal_residual_max": float(solution.normal_residual_ratio.max())}
            for j, name in enumerate(names):
                coefficients[alpha][name][:, joints] = solution.coefficients[:, j*len(joints):(j+1)*len(joints)]
        # Predictor cost is measured on a full block (both forward design and
        # all RHS). No held outcomes or minima are retained by this measurement.
        ids = np.asarray(held)[:block_windows]
        if not len(ids):
            raise ValueError("prediction timing needs an internal train-held block")
        prediction_seconds = []
        for alpha in coefficients:
            def predict():
                x = linear.design_block(features[ids], context=context)
                return [x@coefficients[alpha][key][1:, joints] + coefficients[alpha][key][0, joints] for key in names]
            _, seconds = _timed(predict)
            prediction_seconds.append(seconds)
        remaining = total_blocks-sum(not row.get("projection_probe_only", False) for row in rows)
        build_max = max(row["design_seconds"] for row in rows)
        update_max = max(row["qr_seconds"] for row in rows)
        measured_build_qr = sum(row["design_seconds"]+row["qr_seconds"] for row in rows)
        # Thirteen ridge alphas + WLS are counted even for the outer L13
        # diagnostic; this is deliberately more work than its final demand.
        prediction_blocks = math.ceil((len(fit)+len(held))/len(ids))
        projection = measured_build_qr + remaining*(build_max+update_max) + sum(solve_seconds.values())
        projection += prediction_blocks*sum(prediction_seconds)
        records.append({"joints": list(joints), "dimension": dimension, "fit_windows": len(fit),
                        "held_windows": len(held), "measured_blocks": rows, "total_blocks": total_blocks,
                        "saturated_update_measured": saturated_update_measured,
                        "all_blocks_measured": remaining == 0, "maximum_supported_rows": max_supported_rows, "solve_seconds": solve_seconds,
                        "solution_diagnostics": diagnostics, "prediction_block_seconds": prediction_seconds,
                        "prediction_block_windows": len(ids), "projected_seconds": projection,
                        "projection_rule": "measured blocks plus remaining times maximum measured full-width block"})
        total_projection += projection
        del factor
    return {"context": context, "patterns": records, "projected_seconds": total_projection,
            "projected_working_bytes": max(memory_projection, design_memory["projected_working_bytes"]),
            "retained_factor_memory": design_memory}, coefficients


def _cost_inner_panel(data, components, support, inner_fit, inner_held, coefficients, *, block_windows=32):
    """True quadratic/alpha/grid workload, bounded windows, no calibration result."""
    # Bound the prediction payload, then scale window-linear moment work using
    # the actual number of windows; the 1,936/13-alpha scan is never subsampled.
    ids = np.asarray(inner_held)[:block_windows]
    actual = [np.asarray(components[k])[ids, 6:-6] for k in ("L", "T", "epsilon")]
    zero = np.zeros_like(actual[0])
    started = time.perf_counter()
    x = linear.design_block(np.asarray(data["lat"])[ids])
    ridge_panels = {}
    clean_panel = None
    for alpha, coeff in coefficients.items():
        pred = {key: x@value[1:]+value[0] for key, value in coeff.items()}
        if alpha == 0:
            clean_panel = controls._quadratic_panel([pred["L"]-actual[0], pred["T"]-actual[1], -actual[2]], actual, support[ids])
        else:
            ridge_panels[alpha] = controls._quadratic_panel([pred[key]-actual[j] for j, key in enumerate(("L", "T", "epsilon"))], actual, support[ids])
    oracle = controls._quadratic_panel([zero, zero, -actual[2]], actual, support[ids])
    fit_ids = np.asarray(inner_fit)[:block_windows]
    fit_gram = controls._pose_variance_gram([components[k][fit_ids] for k in ("L", "T")], np.asarray(data["mask"])[fit_ids])
    held_gram = controls._pose_variance_gram([components[k][ids] for k in ("L", "T")], np.asarray(data["mask"])[ids])
    aggregate = {}
    for a in (0., *controls.A_GRID):
        fit_variance = controls._quad(fit_gram, [1., a])
        threshold = controls._threshold(fit_variance)
        held_variance = controls._quad(held_gram, [1., a])
        dynamic = np.isfinite(held_variance) & (held_variance >= threshold)
        aggregate[a] = {"ridge": {alpha: controls.aggregate_quadratic_panel(panel, dynamic) for alpha, panel in ridge_panels.items()},
                        "clean_ls": controls.aggregate_quadratic_panel(clean_panel, dynamic),
                        "oracle": controls.aggregate_quadratic_panel(oracle, dynamic)}
    seconds = time.perf_counter()-started
    scale = max(1., len(inner_fit)/len(fit_ids), len(inner_held)/len(ids))
    return aggregate, {"measured_seconds": seconds, "projection_multiplier": scale,
                       "projected_seconds": seconds*scale, "sample_fit_windows": len(fit_ids),
                       "sample_held_windows": len(ids), "selection_used": False}


def _scan_full_grid(panels):
    """Evaluate every allowed alpha for every pair, even a noncalibrable pair."""
    started = time.perf_counter()
    combinations, alpha_evaluations = 0, 0
    checksum = 0.
    for a in (0., *controls.A_GRID):
        for sigma in controls.SIGMA_GRID:
            weights = [1., a, sigma]
            curve = {}
            for alpha in (10.**k for k in range(-5, 8)):
                stats = controls.pool_statistics([controls.score_aggregate(p[a]["ridge"][alpha], weights) for p in panels])
                curve[alpha] = controls.statistic_values(stats)["MSE"]
                alpha_evaluations += 1
            choice = linear.select_alpha(lambda alpha: curve[alpha])
            checksum += choice["mse"]
            for kind in ("clean_ls", "oracle"):
                stats = controls.pool_statistics([controls.score_aggregate(p[a][kind], weights) for p in panels])
                # Arithmetic is measured on true, train-derived moments; values
                # are not published as qualification/scientific scores.
                controls.statistic_values(stats)
            combinations += 1
    return {"seconds": time.perf_counter()-started, "combinations": combinations,
            "alpha_evaluations": alpha_evaluations, "finite_checksum": bool(np.isfinite(checksum)),
            "cost_only": True, "choice_published": False}


def _report_measurement(data, *, output_dir, block_windows=32, input_load_seconds=0.):
    """Measure historical moment accumulation + the real 2,000-draw aggregator.

    Exactly fifteen *training* users form the cost surrogate. No new held
    labels, model selection, or synthetic hardware timings enter this path.
    """
    from . import report
    selected_users = sorted({row["user"] for row in data["meta"]})[:15]
    if len(selected_users) != 15:
        raise ValueError("report cost measurement needs fifteen distinct training users")
    # One window per training user is enough for per-user bootstrap dimensions;
    # accumulation is separately scaled to the actual final population.
    ids = [next(i for i, row in enumerate(data["meta"]) if row["user"] == user) for user in selected_users]
    full_prediction = np.asarray(data["target"], dtype=np.float32)*.75
    _, accumulation = _timed(lambda: metrics.evaluate(full_prediction, data, dynamic_threshold=metrics.THRESHOLD, per_joint=True, batch_windows=32))
    # R1 re-evaluates both historical predictions (no per-joint breakdown).
    # The synthetic scores are not parity evidence; only this exact cost path is used.
    _, r1_seconds = _timed(lambda: [metrics.evaluate(full_prediction[:, 6:-6].astype(np.float64), data)
                                    for _ in range(2)])
    # Actual hash-verified compressed prediction reads, projected to 15+2 files.
    proxy = Path(output_dir)/"cost-metrics-input.npz"
    proxy_pin = P.atomic_npz(proxy, {"prediction": full_prediction[:, 6:-6]}, root=Path(output_dir))
    def read_prediction_proxy():
        check = P.verify_pin(proxy_pin)
        with np.load(proxy, allow_pickle=False) as archive:
            result = archive["prediction"]
        P.unchanged({str(proxy): check["state"]})
        return result
    _, prediction_read_seconds = _timed(read_prediction_proxy)
    del full_prediction
    # Both dynamic and static groups must be populated in an ordinary report;
    # use two actual windows per user if present, and repeat the same train
    # window otherwise solely for this shape/cost surrogate.
    ids = [index for index in ids for _ in range(2)]
    # subset rejects duplicates intentionally; create a bounded local panel.
    panel = {k: np.asarray(v)[ids] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(data["meta"]) else v
             for k, v in data.items()}
    panel["meta"] = [dict(data["meta"][i], cost_repeat=j%2) for j, i in enumerate(ids)]
    panel["pose_var"] = np.tile([0., metrics.THRESHOLD*2], 15)
    prediction = np.asarray(panel["target"], dtype=np.float32)*.75
    analysis, surrogate_seconds = _timed(lambda: metrics.evaluate(prediction, panel, dynamic_threshold=metrics.THRESHOLD, per_joint=True, batch_windows=32))
    cell = {"complete": True, "analysis": analysis}
    cells = {head: {str(seed): cell for seed in (0, 1, 2)} for head in heads.HEADS}
    references = {name: cell for name in report.REFERENCES}
    assessments = {head: {"optimizer_alarms": [], "objective_tradeoff": {"flag": False}} for head in heads.HEADS}
    result, aggregate_seconds = _timed(lambda: report.aggregate(cells, references, assessments=assessments,
                                                               common_gates={name: True for name in report.COMMON_GATES}))
    if result.get("complete") is not True:
        raise ValueError("cost-only report did not exercise complete bootstrap: " + str(result.get("reason")))
    path = Path(output_dir)/"cost-report.json"
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    _, serialization = _timed(lambda: path.write_text(encoded))
    # Seventeen reports: fifteen new readouts and two historical references.
    ratio = 18548/len(data["meta"])
    input_io_projection = max(1., ratio)*input_load_seconds + ratio*17*prediction_read_seconds
    projected = ratio*(accumulation*17 + r1_seconds) + aggregate_seconds + serialization + surrogate_seconds + input_io_projection
    return {"moment_seconds": accumulation, "moment_windows": len(data["meta"]), "report_cells": 17,
            "batch_windows": 32, "historical_R1_evaluations": 2, "historical_R1_seconds": r1_seconds,
            "prediction_read_seconds": prediction_read_seconds, "prediction_read_bytes": proxy_pin["bytes"],
            "input_io_projection_seconds": input_io_projection, "input_io_is_train_only_projection": True,
            "bootstrap_seconds": aggregate_seconds, "bootstrap_replicates": 2000,
            "serialization_seconds": serialization, "serialization_bytes": path.stat().st_size,
            "projected_seconds": projected, "train_only_surrogate": True}


def benchmark_cpu(data, *, input_loader, output_dir, allowed_train_users, input_identity,
                  folds=None, block_windows=32, input_measurement=None):
    """Measure all actual mask patterns, full feature widths, 2×1,936 scans.

    The full-width TSQR sample is not a scientific reference. Factors are
    discarded; the subsequent gated reference phase reruns the complete data.
    """
    started = time.perf_counter()
    if block_windows != 32:
        raise ValueError("production cost projection uses fixed blocks of 32 windows")
    numeric = train.configure_numerics()
    folds, population = _validate_train(data, allowed_train_users, input_identity, folds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    io = _input_io(data, input_loader, input_identity, input_measurement)
    gc.collect()
    baseline_rss = _rss()
    features, real, mask, support = controls._data_arrays(data)
    layout = _memory_projection(data, support, folds, baseline_rss=baseline_rss, block_windows=block_windows)
    if not layout["within_host_cap"]:
        result = _base_receipt("cpu", population, input_identity, numeric)
        result.update(measured=False, status="resource_memory_limit", memory=layout,
                      elapsed_seconds=time.perf_counter()-started)
        return result
    records, fold_projection, grid_counts = [], [], []
    working_bytes = 0
    for fold in (0, 1):
        fit, held = folds[fold], folds[1-fold]
        components, generator_seconds = _timed(lambda: controls.generate_components(
            features, data["meta"], support, fit, held, allowed_train_users,
            outer_fold=fold, n_joints=20, block_windows=32))
        basis = {"real": real, **{key: components[key] for key in ("L", "T", "epsilon")}}
        designs, projected = [], io["input_load_seconds"] + generator_seconds
        for context in (0, 6):
            measured, coeff = _measure_design(features, basis, support, fit, held, context=context)
            designs.append({"kind": "outer", **measured})
            projected += measured["projected_seconds"]
            working_bytes = max(working_bytes, measured["projected_working_bytes"])
            del coeff
        panels, moments = [], []
        for inner_fit, inner_held in controls.inner_partitions(data["meta"], fit):
            measured, coeff = _measure_design(features, basis, support, inner_fit, inner_held, context=0)
            designs.append({"kind": "inner", **measured})
            projected += measured["projected_seconds"]
            working_bytes = max(working_bytes, measured["projected_working_bytes"])
            panel, moment = _cost_inner_panel(data, components, support, inner_fit, inner_held, coeff)
            projected += moment["projected_seconds"]
            panels.append(panel)
            moments.append(moment)
            del coeff
        scan = _scan_full_grid(panels)
        if scan["combinations"] != 1936 or scan["alpha_evaluations"] != 1936*13 or not scan["finite_checksum"]:
            raise ValueError("incomplete CPU cost scan")
        projected += scan["seconds"]
        records.append({"outer_fold": fold, "generator_seconds": generator_seconds,
                        "designs": designs, "moments": moments, "scan": scan})
        grid_counts.append(scan["combinations"])
        fold_projection.append(projected)
        del components, basis, panels
        gc.collect()
    report_cost = _report_measurement(data, output_dir=output_dir, input_load_seconds=io["input_load_seconds"])
    result = _base_receipt("cpu", population, input_identity, numeric)
    result.update(io=io, folds=records, fold_projection_seconds=fold_projection,
                  persistent_cost_bytes=sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file()),
                  report=report_cost, report_projection_seconds=report_cost["projected_seconds"],
                  all_patterns_measured=True, all_alphas_measured=True,
                  grid_combinations_measured=grid_counts,
                  memory={"peak_rss_bytes": _rss(), "projected_peak_bytes": layout["projected_peak_bytes"],
                          "retained_factor_layout": layout},
                  elapsed_seconds=time.perf_counter()-started)
    return result
