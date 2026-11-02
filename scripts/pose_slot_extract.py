"""Stream fixed backbone-slot contrasts without storing full slot tensors.

The historical pose extraction's record order, scale RNG, windows and FP32
encoder path are preserved. Target arrays are used only for identity checks;
they never determine the features. A receipt is written only after all checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pose_temporal_readout as T
from wearusfm.data import pose_targets as PT
from wearusfm.data import pretraining_loader as L
from wearusfm.harness import pose_eval as PE

require = T.require
PROTOCOL = "pose-slot-readout-20261102-v1"
WIDTH = 336
MEAN_GATE = {"normalized_patch_rms_max": 1e-3, "normalized_max_abs": 1e-2,
             "denominator_relative_max": 1e-3, "eps": 1e-5,
             "support": "every real patch; never filtered by target mask"}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def resource_usage(device):
    usage = {"process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
             * (1 if sys.platform == "darwin" else 1024)}
    if str(device).startswith("cuda"):
        usage.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     gpu_name=torch.cuda.get_device_name(device))
    return usage


def record_input_states(record, pose_store):
    directory = Path(record["path"])
    pose = PT.pose_dir(pose_store.root, directory)
    paths = [directory / "metadata.json", directory / "data_int16.npy",
             pose / PT.POSE_FILE, pose / PT.VALID_FILE, pose / PT.META_FILE]
    result = {}
    for path in paths:
        require(path.is_file() and not path.is_symlink(), "missing or symlinked recording input: " + str(path))
        item = {"state": T.file_state(path)}
        if path.suffix == ".json":
            item["sha256"] = T.D.file_sha256(path)
        result[str(path)] = item
    return result


def verify_input_states(states):
    for path, expected in states.items():
        require(T.file_state(path) == expected["state"], "recording input changed: " + path)
        if "sha256" in expected:
            require(T.D.file_sha256(path) == expected["sha256"], "recording metadata changed: " + path)


def model_inputs(xs, view, ev):
    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors

    layout, code = CC.layout_from_montage(view.montage), CC.anatomy_codes(view.montage)
    sets = CC.attention_sets(layout, ev.loader.k_neighbors)
    n, c, device = len(xs), layout.n_channels, ev.device
    return ModelInputs([torch.from_numpy(x).to(device) for x in xs], [view.fs] * n,
                       {k: v.to(device) for k, v in codes_to_tensors(CC.pack_codes([code] * n)).items()},
                       {k: v.to(device) for k, v in sets_to_tensors(CC.pack_attention_sets([sets] * n)).items()},
                       [c] * n, torch.as_tensor(layout.qc_valid).repeat(n).to(device))


def window_metadata(record, ri, view, window):
    return {"user": record["user"], "session": record["session"], "stage": record["stage"],
            "side": record["side"], "stem": record["stem"], "recording": ri,
            "start": int(window.start), "fs": float(view.fs), "n_patches": int(window.n_patches)}


def compare_mean(actual, expected):
    """Preregistered parity gate on all real patches, after FP16 then FP32 LN."""
    require(actual.shape == expected.shape and expected.dtype == np.float16, "mean shape/dtype mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "nonfinite mean feature")
    rounded = actual.astype(np.float16)
    require(np.isfinite(rounded).all(), "mean overflows historical FP16 precision")
    difference = np.abs(rounded.astype(np.float32) - expected.astype(np.float32))
    with torch.no_grad():
        new, old = torch.from_numpy(rounded.astype(np.float32)), torch.from_numpy(expected.astype(np.float32))
        norm_new = torch.nn.functional.layer_norm(new, (384,), eps=MEAN_GATE["eps"])
        norm_old = torch.nn.functional.layer_norm(old, (384,), eps=MEAN_GATE["eps"])
        norm_diff = norm_new - norm_old
        denominator_new = torch.sqrt(new.var(dim=-1, unbiased=False) + MEAN_GATE["eps"])
        denominator_old = torch.sqrt(old.var(dim=-1, unbiased=False) + MEAN_GATE["eps"])
        patch_rms = torch.sqrt(torch.mean(norm_diff.square(), dim=-1))
        denom_relative = (denominator_new - denominator_old).abs() / denominator_old
    # Monotonic half-float bit ordering, with +0/-0 identified. ULP is audit only.
    def ordered_half(value):
        bits = value.view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)

    result = {"n_elements": int(expected.size), "n_unequal_fp16": int(np.count_nonzero(rounded != expected)),
              "max_absolute_fp16_difference": float(difference.max(initial=0)),
              "raw_squared_difference_sum": float(np.square(difference.astype(np.float64)).sum()),
              "max_ulp_fp16_difference": int(np.abs(ordered_half(rounded) - ordered_half(expected)).max(initial=0)),
              "max_absolute_fp32_to_reference": float(np.abs(actual.astype(np.float32) - expected.astype(np.float32)).max(initial=0)),
              "normalized_patch_rms_max": float(patch_rms.max()),
              "normalized_max_abs": float(norm_diff.abs().max()),
              "denominator_relative_max": float(denom_relative.max())}
    require(all(np.isfinite(value) for value in result.values()), "nonfinite mean parity statistic")
    require(all(result[key] <= MEAN_GATE[key] for key in
                ("normalized_patch_rms_max", "normalized_max_abs", "denominator_relative_max")),
            "historical mean parity failed: " + json.dumps(result))
    return result


def selected_reference_rows(reference, recordings, selected_recording_indices):
    selected = list(range(len(recordings))) if selected_recording_indices is None else list(selected_recording_indices)
    require(selected and all(type(i) is int and 0 <= i < len(recordings) for i in selected)
            and selected == sorted(set(selected)), "invalid original recording indices")
    meta = reference["meta"]
    require(len(meta) == len(reference["lat"]) and all(type(m.get("recording")) is int
            and 0 <= m["recording"] < len(recordings) for m in meta), "invalid reference recording identity")
    recording_ids = [m["recording"] for m in meta]
    require(recording_ids == sorted(recording_ids), "reference records are not ordered")
    rows = [i for i, m in enumerate(meta) if m["recording"] in set(selected)]
    require(rows, "selected records contain no reference windows")
    return selected, rows


def extract_stream(model, recordings, pose_store, ev, reference, out_dir, *, model_name, split,
                   projection_record, identity=None, selected_recording_indices=None,
                   final_guard=None, log=print):
    """Write one model/split's raw FP32 residual features and an attested receipt.

    ``reference`` is the historical pose cache loaded by T.read_data. Selecting
    recording indices retains their original ri for session-scale randomness.
    ``final_guard`` verifies caller-owned checkpoint/code/cache pins before the
    receipt becomes complete. Failed output directories are never overwritten.
    """
    import pose_slot_features as F

    started = perf_counter()
    require(projection_record == F.recipe(), "projection record differs from actual projection")
    require(model_name in T.MODELS and split in ("train", "val"), "invalid model/split")
    require(selected_recording_indices is None or split == "train", "preflight may read train only")
    require(reference["lat"].ndim == 3 and reference["lat"].shape[1:] == (160, 384)
            and reference["lat"].dtype == np.float16, "historical latent geometry changed")
    require(model.cfg.dim == 384 and model.cfg.k_latents == 64, "unexpected backbone geometry")
    require(ev.seed == 20261026 and ev.n_windows == 10 and ev.window_s == 4.0 and ev.batch == 32,
            "historical extraction recipe changed")
    selected, row_indices = selected_reference_rows(reference, recordings, selected_recording_indices)
    out_dir = Path(out_dir)
    T.D.fresh_destination(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    residual_path = out_dir / "residual.npy"
    residual = np.lib.format.open_memmap(residual_path, mode="w+", dtype=np.float32,
                                         shape=(len(row_indices), 160, WIDTH))
    model.eval()
    if str(ev.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(ev.device)
    parity = {"n_elements": 0, "n_unequal_fp16": 0, "max_absolute_fp16_difference": 0.,
              "max_absolute_fp32_to_reference": 0., "max_ulp_fp16_difference": 0,
              "raw_squared_difference_sum": 0., "normalized_patch_rms_max": 0.,
              "normalized_max_abs": 0., "denominator_relative_max": 0.}
    raw_states, observed_meta, record_audit = {}, [], []
    cursor = 0
    for ri in selected:
        record = recordings[ri]
        states = record_input_states(record, pose_store)
        raw_states.update(states)
        view = L.SessionView.open(Path(record["path"]), ev.loader.split_at_gaps)
        rng = np.random.default_rng([ev.seed, ri])
        expected_rows = [i for i in row_indices if reference["meta"][i]["recording"] == ri]
        try:
            scale = L.estimate_session_scale(view, ev.loader, rng)
        except ValueError:
            require(not expected_rows, "historical supported recording now has no scale")
            record_audit.append({"recording": ri, "n_windows": 0, "skip": "scale unavailable"})
            verify_input_states(states)
            continue
        require(np.isfinite(scale) and scale > 0, "invalid session scale")
        windows = PE.fixed_windows(view, ev.loader, ev.n_windows, ev.window_s, ev.seed)
        require(len(windows) == len(expected_rows), "historical window count changed")
        record_audit.append({"recording": ri, "n_windows": len(windows), "scale": float(scale)})
        for start in range(0, len(windows), ev.batch):
            chunk = windows[start:start + ev.batch]
            xs = [L.read_window(view, w.trial, w.start, w.stop, w.span, ev.loader).astype(np.float32)
                  / np.float32(scale) for w in chunk]
            require(all(np.isfinite(x).all() for x in xs), "nonfinite EMG input")
            with torch.no_grad():
                encoded = model.encode(model_inputs(xs, view, ev), None)
                z = encoded.z
                require(z.shape == (len(chunk), 160, 64, 384) and z.dtype == torch.float32,
                        "backbone geometry or precision changed")
                require(bool(torch.isfinite(z).all()), "nonfinite backbone slot")
                mean = z.mean(dim=2).float().cpu().numpy()
                projected = F.projected_residuals(z)
                require(projected.shape == (len(chunk), 160, WIDTH) and projected.dtype == torch.float32,
                        "projected residual shape/dtype mismatch")
                require(bool(torch.isfinite(projected).all()), "nonfinite slot projection")
                values = projected.cpu().numpy()
            for j, window in enumerate(chunk):
                global_row = expected_rows[start + j]
                require(cursor < len(row_indices) and row_indices[cursor] == global_row, "window ordering changed")
                meta = window_metadata(record, ri, view, window)
                require(meta == reference["meta"][global_row], "historical window metadata changed")
                require(window.n_patches == 160, "unexpected patch count")
                target, mask, std = pose_store.window({"dataset": PE.DATASET}, view, window, ev.loader.patch_ms)
                target, std = target.astype(np.float32), std.astype(np.float32)
                valid_rows = mask.any(axis=1)
                pose_var = np.float32(target[valid_rows].var(axis=0).mean() if valid_rows.sum() > 1 else np.nan)
                for name, value in (("target", target), ("mask", mask), ("std", std), ("pose_var", pose_var)):
                    expected = reference[name][global_row]
                    require(np.asarray(value).dtype == np.asarray(expected).dtype
                            and np.array_equal(value, expected, equal_nan=True), "historical target identity changed: " + name)
                check = compare_mean(mean[j], reference["lat"][global_row])
                for key, value in check.items():
                    parity[key] = parity[key] + value if key.startswith("n_") or key.endswith("_sum") else max(parity[key], value)
                residual[cursor] = values[j]
                observed_meta.append(meta)
                cursor += 1
            del encoded, z, projected, values, mean
        verify_input_states(states)
        log(f"{model_name}/{split} recording {ri}: {len(windows)} windows; {cursor}/{len(row_indices)}", flush=True)
    require(cursor == len(row_indices), "incomplete extraction")
    loop_seconds = perf_counter() - started
    residual.flush()
    del residual
    residual_info = {"path": str(residual_path), "sha256": T.D.file_sha256(residual_path),
                     "bytes": residual_path.stat().st_size, "shape": [len(row_indices), 160, WIDTH], "dtype": "<f4"}
    verify_input_states(raw_states)
    if final_guard is not None:
        final_guard()
    parity.update(passed=True, raw_rmse=float(np.sqrt(parity["raw_squared_difference_sum"] / parity["n_elements"])))
    receipt = {**(identity or {}), "protocol": PROTOCOL, "complete": True, "model": model_name, "split": split,
               "which": "student", "checkpoint_step": 19618, "projection_record": projection_record,
               "reference_row_indices": row_indices, "selected_original_recording_indices": selected,
               "n_windows": len(row_indices), "metadata_sha256": T.json_hash(observed_meta),
               "mean_parity_gate": MEAN_GATE, "mean_parity": parity, "raw_inputs_states": raw_states, "record_audit": record_audit,
               "loop_seconds": loop_seconds, "elapsed_seconds": perf_counter() - started,
               "resources": resource_usage(ev.device),
               "residual": residual_info}
    write_json(out_dir / "receipt.json", receipt)
    return receipt


def load_records(plan, model, split):
    path = Path(plan["caches"][model][split]["path"])
    sidecar = json.loads(Path(str(path) + ".provenance.json").read_text())
    records = PE.recordings_for(Path(plan["paths"]["csv"]), Path(plan["paths"]["processed_root"]), split,
                                n_users=40 if split == "train" else None,
                                n_recordings=200 if split == "train" else None, seed=20261026)
    require(records == sidecar["records"], "original ordered record list changed")
    return records


def global_input_states(plan):
    paths = {"csv": Path(plan["paths"]["csv"]), "manifest": Path(plan["paths"]["manifest"]),
             "stats": Path(plan["paths"]["pose_root"]) / PT.STATS_FILE}
    result = {}
    for key, path in paths.items():
        require(T.D.file_sha256(path) == plan["input_hashes"][key], "global input SHA mismatch: " + key)
        result[str(path)] = {"state": T.file_state(path), "sha256": plan["input_hashes"][key]}
    return result


def main(argv=None):
    import pose_slot_features as F
    import pose_slot_plan as P
    from wearusfm.harness.fm_features import load_model
    from wearusfm.training import run as R

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "extract"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=T.MODELS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require((args.stage == "extract") == (args.model is not None), "extract requires model; preflight runs both models")
    plan_sha = T.D.file_sha256(args.plan)
    plan = json.loads(args.plan.read_text())
    P.validate_plan(plan)
    require(plan["mean_parity_gate"] == MEAN_GATE, "mean parity gate differs from frozen plan")
    runtime = P.runtime_identity(args.expected_code_sha)
    root = Path(plan["output_root"]) / ("preflight" if args.stage == "preflight" else args.model)
    require(args.out == root, "output differs from frozen campaign path")
    preflight = P.validate_preflight(plan, args.expected_code_sha, plan_sha) if args.stage == "extract" else None
    T.D.fresh_destination(args.out)
    globals_before = global_input_states(plan)
    args.out.mkdir(parents=True, exist_ok=False)
    started, outputs, inspections, checkpoints = perf_counter(), {}, {}, {}
    for model_name in T.MODELS if args.stage == "preflight" else (args.model,):
        checkpoints[model_name] = P.validate_checkpoint(plan, model_name)
        checkpoint = Path(plan["models"][model_name]["checkpoint"])
        network = load_model(str(checkpoint), device=args.device, which="student")
        store = PT.PoseStore(plan["paths"]["pose_root"])
        ev = PE.EvalConfig(loader=R.sanity_config().loader, device=args.device)
        for split in ("train",) if args.stage == "preflight" else ("train", "val"):
            inspection = P.validate_reference(plan, model_name, split)
            inspections[f"{model_name}/{split}"] = inspection
            reference = T.read_data(plan["caches"][model_name][split])
            records = load_records(plan, model_name, split)
            destination = args.out / model_name / split if args.stage == "preflight" else args.out / split

            def guard():
                require(P.runtime_identity(args.expected_code_sha) == runtime, "runtime/source changed during extraction")
                require(T.D.file_sha256(args.plan) == plan_sha, "plan changed during extraction")
                if preflight is not None:
                    require(P.validate_preflight(plan, args.expected_code_sha, plan_sha) == preflight,
                            "preflight changed during extraction")
                verify_input_states(globals_before)
                for path, state in inspection["states"].items():
                    require(T.file_state(path) == state, "reference input changed during extraction")
                require(P.validate_checkpoint(plan, model_name) == checkpoints[model_name], "checkpoint identity changed")

            outputs[f"{model_name}/{split}"] = extract_stream(
                network, records, store, ev, reference, destination, model_name=model_name, split=split,
                projection_record=F.recipe(), selected_recording_indices=plan["preflight_recording_indices"] if args.stage == "preflight" else None,
                identity={"code_sha": args.expected_code_sha, "plan_sha256": plan_sha,
                          "checkpoint_sha256": plan["models"][model_name]["checkpoint_sha256"],
                          "reference_cache_sha256": plan["caches"][model_name][split]["sha256"]}, final_guard=guard)
            del reference
        del network, store
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    require(P.runtime_identity(args.expected_code_sha) == runtime, "final runtime/source changed")
    require(T.D.file_sha256(args.plan) == plan_sha, "final plan changed")
    if preflight is not None:
        require(P.validate_preflight(plan, args.expected_code_sha, plan_sha) == preflight, "final preflight changed")
    verify_input_states(globals_before)
    receipt = {"protocol": PROTOCOL, "complete": True, "stage": args.stage, "code_sha": args.expected_code_sha,
               "plan_sha256": plan_sha, "runtime": runtime, "global_inputs": globals_before,
               "checkpoints": checkpoints, "reference_inspections": inspections, "preflight": preflight,
               "outputs": {key: {"receipt": str(Path(value["residual"]["path"]).parent / "receipt.json"),
                                 "receipt_sha256": T.D.file_sha256(Path(value["residual"]["path"]).parent / "receipt.json"),
                                 "n_windows": value["n_windows"], "elapsed_seconds": value["elapsed_seconds"],
                                 "loop_seconds": value["loop_seconds"],
                                 "resources": value["resources"], "mean_parity": value["mean_parity"]}
                           for key, value in outputs.items()},
               "elapsed_seconds": perf_counter() - started, "resources": resource_usage(args.device)}
    if args.stage == "preflight":
        receipt["full_extraction_window_scaled_seconds"] = {
            model: outputs[f"{model}/train"]["loop_seconds"] / outputs[f"{model}/train"]["n_windows"]
            * sum(plan["caches"][model][s]["n_windows"] for s in ("train", "val")) for model in T.MODELS}
        receipt["estimate_note"] = ("Window-scaled loop timing including session loading/scaling, inference and memmap writes; "
                                    "excludes full output hashing and final guards. Not a full run or bound; raw I/O and session lengths vary.")
    write_json(args.out / "receipt.json", receipt)
    print(json.dumps({"complete": True, "stage": args.stage, "elapsed_seconds": receipt["elapsed_seconds"],
                      "receipt": str(args.out / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
