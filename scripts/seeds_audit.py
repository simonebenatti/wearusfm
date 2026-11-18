#!/usr/bin/env python3
"""Read-only structural audit of the external SEEDS MATLAB payload.

This does not fit models, select targets, filter signals, or establish physiological
alignment. A sample audit can never pass the full 5,850-trial/25-calibration gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

# Reuse the frozen manifest/isolation contract without depending on the caller's
# sys.path (both direct CLI execution and import-by-path are supported).
_spec = importlib.util.spec_from_file_location("_seeds_download_contract", Path(__file__).with_name("seeds_download.py"))
_download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_download)

TRIAL = re.compile(r"detop_exp01_subj(?P<subject>\d{2})_Sess(?P<session>\d)_(?P<movement>\d{1,2})_(?P<repetition>\d{1,2})\.mat")
CALIBRATION = re.compile(r"subj(?P<subject>\d{2})_calibration_values\.mat")
EXPECTED_TRIALS = set(itertools.product(range(1, 26), range(1, 4), range(1, 14), range(1, 7)))
EXPECTED_SUBJECTS = set(range(1, 26))
ALIGNMENT = {
    "evidence": "published_helper_convention_only_not_physiological_validation",
    "author_matlab_helper": "resample_glove.m: interp1(1:N_glove, glove', linspace(1,N_glove,N_emg))'",
    "author_matlab_orig_fs_argument_used": False,
    "author_python_helper": "functions.py: scipy.signal.resample to int(new_fs/orig_fs)*N_glove, then truncate to N_emg",
    "author_python_time_argument": "linspace(0,N_glove/orig_fs,N_glove,endpoint=True); returned times discarded",
    "author_helpers_equivalent": False,
    "native_glove_hz_from_publication": 60,
    "distributed_glove_hz": 256,
    "distributed_emg_hz": 2048,
    "strict_duration_difference_limit_glove_samples": 1,
    "warning": "Length agreement alone does not prove synchronization. MATLAB endpoint interpolation and Python Fourier resampling/truncation differ; downstream convention needs an explicit choice and oracle.",
}


def _identity(path: str) -> dict:
    name = Path(path).name
    trial, calibration = TRIAL.fullmatch(name), CALIBRATION.fullmatch(name)
    match = trial or calibration
    if match is None:
        raise ValueError("unrecognized MATLAB filename")
    result = {k: int(v) for k, v in match.groupdict().items()}
    result["kind"] = "trial" if trial else "calibration"
    if not 1 <= result["subject"] <= 25:
        raise ValueError("subject outside 01..25")
    if Path(path).parts[0] != f"subj{result['subject']:02d}":
        raise ValueError("filename subject differs from top-level subject directory")
    if trial and tuple(result[k] for k in ("subject", "session", "movement", "repetition")) not in EXPECTED_TRIALS:
        raise ValueError("session/movement/repetition outside published ranges")
    return result


def _text(value, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.kind not in "US" or array.size != 1:
        raise ValueError(f"{name} must be one string")
    return str(array.item()).strip()


def _number(value, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite real number")
    return float(array.item())


def _names(value, count: int, name: str) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count:
        raise ValueError(f"{name} must contain {count} ordered names")
    if array.dtype.kind in "US":
        result = [str(v).strip() for v in array]
    elif array.dtype.kind == "O":
        result = [_text(v, name) for v in array]
    else:
        raise ValueError(f"{name} is not textual")
    if any(not x for x in result) or len(set(result)) != count:
        raise ValueError(f"{name} contains empty or repeated names")
    return result


def _signal(value, channels: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != channels or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape ({channels}, samples>=2)")
    if array.dtype.kind != "f":
        raise ValueError(f"{name} must contain real floating point samples")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite samples")
    return array


def inspect_mat(path: Path, identity: dict) -> dict:
    """Return integrity metadata only; never modify or resample signal arrays."""
    data = loadmat(path, squeeze_me=True, chars_as_strings=True)
    if identity["kind"] == "calibration":
        lo, hi = (np.asarray(data[name]) for name in ("calibration_min", "calibration_max"))
        if any(a.shape != (18,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in (lo, hi)):
            raise ValueError("calibration extrema must be 18 finite real values each")
        spans = hi.astype(np.float64) - lo.astype(np.float64)
        if np.any(spans <= 0):
            raise ValueError("calibration_max must exceed calibration_min for every sensor")
        return {"calibration_sensors": 18, "minimum_calibration_span": float(spans.min()),
                "maximum_calibration_span": float(spans.max()), "physical_angle_units_established": False}

    subject = data["subject"]
    if np.asarray(subject).dtype.kind in "US":
        subject_text = _text(subject, "subject")
        match = re.fullmatch(r"(?:subj)?(\d{1,2})", subject_text)
        if not match or int(match.group(1)) != identity["subject"]:
            raise ValueError("MAT subject differs from filename")
    elif _number(subject, "subject") != identity["subject"]:
        raise ValueError("MAT subject differs from filename")
    if _number(data["session"], "session") != identity["session"]:
        raise ValueError("MAT session differs from filename")
    movement = _text(data["movement"], "movement")
    if not movement:
        raise ValueError("empty movement name")
    speed = _text(data["speed"], "speed")
    if speed not in {"fast", "slow"}:
        raise ValueError("speed is not fast/slow")
    date = _text(data["date"], "date")
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError("date must be YYYYMMDD")
    dt.datetime.strptime(date, "%Y%m%d")
    fs_emg, fs_glove = (_number(data[k], k) for k in ("fs_emg", "fs_glove"))
    if fs_emg != 2048 or fs_glove != 256:
        raise ValueError("sampling frequencies differ from published 2048/256 Hz")
    emg, glove = _signal(data["emg"], 134, "emg"), _signal(data["glove"], 18, "glove")
    emg_names = _names(data["channels_emg"], 134, "channels_emg")
    glove_names = _names(data["channels_glove"], 18, "channels_glove")
    n_emg, n_glove = emg.shape[1], glove.shape[1]
    delta_glove_samples = n_emg * fs_glove / fs_emg - n_glove
    if abs(delta_glove_samples) >= 1:
        raise ValueError(f"duration discrepancy >= one glove sample: {delta_glove_samples}")
    # Out-of-range calibrated glove values are explicitly allowed by Usage Notes.
    below, above = int(np.count_nonzero(glove < 0)), int(np.count_nonzero(glove > 1))
    return {
        "date": date, "movement_name": movement, "speed": speed,
        "fs_emg": fs_emg, "fs_glove": fs_glove,
        "emg_shape": list(emg.shape), "glove_shape": list(glove.shape),
        "emg_dtype": str(emg.dtype), "glove_dtype": str(glove.dtype),
        "duration_emg_seconds": n_emg / fs_emg, "duration_glove_seconds": n_glove / fs_glove,
        "duration_difference_glove_samples": delta_glove_samples,
        "endpoint_span_difference_seconds": (n_emg - 1) / fs_emg - (n_glove - 1) / fs_glove,
        "exact_sample_ratio_8": n_emg == 8 * n_glove,
        "emg_range": [float(emg.min()), float(emg.max())],
        "glove_range": [float(glove.min()), float(glove.max())],
        "glove_values_below_zero": below, "glove_values_above_one": above,
        "glove_values": int(glove.size), "all_values_finite": True,
        "channels_emg": emg_names, "channels_glove": glove_names,
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)), "p05": float(np.quantile(a, .05)),
            "p95": float(np.quantile(a, .95))}


def run(manifest_path: Path, raw_root: Path, *, sample: int | None = None,
        check_duplicates: bool = False) -> dict:
    """Audit all MATLAB entries or the first N available entries in sample mode."""
    if sample is not None and (type(sample) is not int or sample <= 0):
        raise ValueError("sample must be a positive count")
    raw = manifest_path.read_bytes()
    entries = _download.validate_manifest(json.loads(raw))
    root = raw_root.absolute()
    _download._no_symlinks(root / _download.POLICY_NAME)
    if tuple(root.parts[-2:]) != ("external_benchmarks", "seeds"):
        raise ValueError("raw root must end in external_benchmarks/seeds")
    if json.loads((root / _download.POLICY_NAME).read_text()) != _download.POLICY:
        raise ValueError("missing or mismatched external-only policy")

    mats = sorted((e for e in entries if e["path"].lower().endswith(".mat")), key=lambda e: e["path"])
    inventory_errors, identities, trial_keys, calibration_subjects = [], {}, [], []
    for entry in mats:
        try:
            identity = _identity(entry["path"])
            identities[entry["path"]] = identity
            if identity["kind"] == "trial":
                trial_keys.append(tuple(identity[k] for k in ("subject", "session", "movement", "repetition")))
            else:
                calibration_subjects.append(identity["subject"])
        except ValueError as error:
            inventory_errors.append({"path": entry["path"], "error": str(error)})
    duplicate_keys = [list(k) for k, count in Counter(trial_keys).items() if count > 1]
    duplicate_calibrations = [s for s, count in Counter(calibration_subjects).items() if count > 1]
    inventory_complete = (not inventory_errors and not duplicate_keys and not duplicate_calibrations
                          and set(trial_keys) == EXPECTED_TRIALS and set(calibration_subjects) == EXPECTED_SUBJECTS)
    # All paths are checked even when a sample is requested; no symlink traversal.
    targets = {e["path"]: _download._target(root, e) for e in mats}
    available = [e for e in mats if targets[e["path"]].is_file()]
    selected = available[:sample] if sample is not None else mats
    rows, references, movement_names, hashes = [], {}, {}, defaultdict(list)
    for entry in selected:
        name = entry["path"]
        result = {"path": name, "status": "failed", **identities.get(name, {})}
        rows.append(result)
        try:
            if name not in identities:
                raise ValueError("unrecognized filename; see manifest_inventory.errors")
            target = targets[name]
            actual = _download._hashes(target)
            _download._assert_hashes(actual, entry)
            result.update(actual)
            metadata = inspect_mat(target, identities[name])
            for channel_field in ("channels_emg", "channels_glove"):
                if channel_field not in metadata:
                    continue
                names = metadata.pop(channel_field)
                if channel_field in references and names != references[channel_field]:
                    raise ValueError(f"{channel_field} order/names differ from first verified trial")
                references.setdefault(channel_field, names)
                result[channel_field + "_sha256"] = hashlib.sha256(json.dumps(names).encode()).hexdigest()
            if result["kind"] == "trial":
                key = result["movement"]
                label = metadata["movement_name"]
                if key in movement_names and label != movement_names[key]:
                    raise ValueError("movement ID maps to inconsistent names")
                if label in movement_names.values() and movement_names.get(key) != label:
                    raise ValueError("movement name maps to multiple IDs")
                movement_names[key] = label
                hashes[actual["sha256"]].append(name)
            result.update(metadata, status="verified")
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"

    good_trials = [r for r in rows if r["status"] == "verified" and r["kind"] == "trial"]
    errors = [r for r in rows if r["status"] != "verified"]
    duplicates = [{"sha256": digest, "paths": names} for digest, names in sorted(hashes.items()) if len(names) > 1] if check_duplicates else []
    warnings = []
    if sample is not None:
        warnings.append("Partial sample audit: never a PASS for the full dataset.")
    if any(r["glove_values_below_zero"] or r["glove_values_above_one"] for r in good_trials):
        warnings.append("Glove values outside [0,1] are retained; the authors describe this as possible after calibration.")
    if duplicates:
        warnings.append("Byte-identical trial files detected; investigate provenance before fitting.")
    warnings.append(ALIGNMENT["warning"])
    structural_ok = bool(rows) and not errors and not inventory_errors and not duplicate_keys and not duplicate_calibrations
    full_pass = sample is None and inventory_complete and structural_ok and not duplicates
    status = "passed" if full_pass else ("partial" if sample is not None and structural_ok and not duplicates else "failed")
    return {
        "schema_version": 1, "dataset": "SEEDS", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status, "full_dataset_pass": full_pass,
        "audit_mode": "sample" if sample is not None else "full", "requested_sample_files": sample,
        "sample_selection": "first N locally available MATLAB paths in lexicographic order" if sample is not None else None,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "raw_root": str(root),
        "pretraining_allowed": False, "alignment": ALIGNMENT,
        "manifest_inventory": {
            "manifest_files": len(entries), "mat_files": len(mats), "trial_files": len(trial_keys),
            "calibration_files": len(calibration_subjects), "expected_trials": 5850, "expected_calibrations": 25,
            "trial_counts_by_subject": {f"{s:02d}": sum(k[0] == s for k in trial_keys) for s in range(1, 26)},
            "missing_trial_keys": [list(k) for k in sorted(EXPECTED_TRIALS - set(trial_keys))],
            "missing_calibration_subjects": sorted(EXPECTED_SUBJECTS - set(calibration_subjects)),
            "duplicate_trial_keys": duplicate_keys, "duplicate_calibration_subjects": duplicate_calibrations,
            "errors": inventory_errors,
        },
        "gates": {"complete_published_manifest": inventory_complete, "all_mat_files_audited": len(selected) == len(mats) and bool(mats),
                  "all_selected_files_pass_structure_and_source_hashes": structural_ok,
                  "not_sample_mode": sample is None, "duplicate_trial_hash_check_requested": check_duplicates,
                  "no_duplicate_trial_hashes": not duplicates if check_duplicates else None},
        "summary": {
            "locally_available_mat_files": len(available), "audited_files": len(rows), "failed_files": len(errors),
            "verified_trial_files": len(good_trials), "verified_calibration_files": sum(r["status"] == "verified" and r.get("kind") == "calibration" for r in rows),
            "verified_trial_counts_by_subject": dict(sorted(Counter(f"{r['subject']:02d}" for r in good_trials).items())),
            "speed_counts": dict(Counter(r["speed"] for r in good_trials)),
            "duration_emg_seconds": _distribution([r["duration_emg_seconds"] for r in good_trials]),
            "duration_difference_glove_samples": _distribution([r["duration_difference_glove_samples"] for r in good_trials]),
            "endpoint_span_difference_seconds": _distribution([r["endpoint_span_difference_seconds"] for r in good_trials]),
            "non_exact_ratio_8_trial_files": sum(not r["exact_sample_ratio_8"] for r in good_trials),
            "glove_values_below_zero": sum(r["glove_values_below_zero"] for r in good_trials),
            "glove_values_above_one": sum(r["glove_values_above_one"] for r in good_trials),
        },
        "channel_reference": references, "movement_names": {str(k): v for k, v in sorted(movement_names.items())},
        "duplicate_trial_hashes": duplicates, "warnings": warnings, "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sample", nargs="?", const=1, type=int, help="Audit at most N locally present MATs; always partial, default N=1")
    parser.add_argument("--check-duplicates", action="store_true")
    args = parser.parse_args()
    _download._no_symlinks(args.report.absolute())
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")
    if args.report.absolute().is_relative_to(args.raw_root.absolute()):
        raise ValueError("report must be outside the read-only raw root")
    report = run(args.manifest, args.raw_root, sample=args.sample, check_duplicates=args.check_duplicates)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "full_dataset_pass": report["full_dataset_pass"], **report["summary"]}))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
