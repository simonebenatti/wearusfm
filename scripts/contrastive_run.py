"""Pinned entry point for reviewed contrastive calibration/benchmark/training.

No submission. A recipe check is explicitly distinct from a complete release dry run.
Scientific training remains blocked until a measured, reviewed resource receipt exists.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import (config_from_dict, build_review_configs, BASELINE, BASELINE_SHA256,
    source_digest, _read, _pin, authenticate_calibration, calibrated_configs, _coverage, validate_input_bindings)
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_recipe(raw, stage, calibration=None, calibration_run_dir=None):
    if stage not in ("calibration", "benchmark", "training"):
        raise ValueError("unknown comparison stage")
    if sha(BASELINE) != BASELINE_SHA256:
        raise ValueError("historical baseline changed")
    recipes = build_review_configs(_read(BASELINE))
    arm = raw.get("contrastive", {}).get("arm")
    if stage == "calibration":
        expected = recipes["calibration"]
    else:
        if arm not in ("C0", "CN", "CR"):
            raise ValueError("benchmark/training requires a calibrated comparison arm")
        endpoint = raw.get("max_steps")
        if type(endpoint) is not int or (stage == "benchmark" and endpoint not in (544, 576)):
            raise ValueError("technical benchmark requires full 576 or frozen split at 544")
        if stage == "training" and endpoint not in (5000, 19618):
            raise ValueError("scientific dose must be 5000 or 19618")
        expected = calibrated_configs(calibration, calibration_run_dir, endpoint)[arm]
    cfg = config_from_dict(raw)
    ContrastiveConfig(**cfg.contrastive)
    if R.config_to_dict(cfg) != expected:
        raise ValueError("configuration differs from the frozen comparison recipe")
    return cfg


def validate_cohort(path, calibration_path, calibration_run_dir, config_path):
    """A single immutable bundle carries the receipt for every matched arm."""
    cohort = _read(path)
    receipt = _read(calibration_path)
    authenticate_calibration(receipt, calibration_run_dir)
    if cohort.get("schema") != 1 or cohort.get("training_approved") is not False:
        raise ValueError("invalid calibrated cohort bundle")
    if cohort.get("calibration_sha256") != sha(calibration_path) or cohort.get("source_sha256") != source_digest():
        raise ValueError("cohort calibration/source pins differ")
    if cohort.get("initial_state_sha256") != receipt["initial_state_sha256"]:
        raise ValueError("cohort initial state differs from calibration")
    endpoint = cohort.get("endpoint")
    expected = calibrated_configs(receipt, calibration_run_dir, endpoint)
    pins = cohort.get("configs", {})
    if set(pins) != set(expected):
        raise ValueError("cohort must bind exactly all three arms")
    found = False
    for arm, cfg in expected.items():
        pin = pins[arm]
        candidate = Path(path).parent / pin["path"]
        if candidate.resolve().parent != Path(path).resolve().parent:
            raise ValueError("cohort config must be in the bundle directory")
        if _pin(candidate, Path(path).parent) != pin or _read(candidate) != cfg:
            raise ValueError("cohort candidate pin/recipe differs")
        found |= candidate.resolve() == Path(config_path).resolve()
    # Technical split and scientific endpoints change only dose. Every endpoint
    # remains bound to the SAME original three-arm cohort/receipt identity.
    raw = _read(config_path)
    split_expected = copy.deepcopy(expected.get(raw.get("contrastive", {}).get("arm")))
    if not found and endpoint == 576 and type(raw.get("max_steps")) is int and raw["max_steps"] in (544, 5000, 19618) and split_expected:
        split_expected["max_steps"] = raw["max_steps"]
        found = raw == split_expected
    if not found:
        raise ValueError("config is not a member of the common calibrated cohort")
    return cohort


def validate_audit(path, inputs, calibration_config):
    report = _read(path)
    expected = {name: sha(value) for name, value in inputs.items()}
    if report.get("source_sha256") != source_digest() or report.get("inputs") != expected:
        raise ValueError("audit provenance/data pins differ")
    if (report.get("steps") != 1024 or report.get("world_size") != 4 or report.get("workers_per_rank") != 6
            or report.get("config_sha256") != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
        raise ValueError("audit stream/config differs from the calibration recipe")
    panel = report.get("calibration_panel")
    if not isinstance(panel, dict):
        raise ValueError("audit lacks the frozen calibration panel")
    checked = _coverage({"expected_datasets": report.get("expected_datasets"), "coverage_completed_blocks": [],
                         "coverage_partial_block": panel, "config": calibration_config["contrastive"]})
    if report.get("passed") is not True or checked["passed"] is not True:
        raise ValueError("audit/calibration coverage gate failed")
    return report


def validate_measured_budget(path, release, cohort, benchmark_paths):
    """Fail closed until all three measured benchmarks and their budget review exist."""
    if path is None or benchmark_paths is None or set(benchmark_paths) != {"C0", "CN", "CR"}:
        raise ValueError("training requires measured three-arm benchmark and resource receipt")
    from contrastive_benchmark_readout import authenticate_preflight
    receipt = authenticate_preflight(path)
    if receipt.get("status") != "PASSED_FOR_REVIEW" or receipt.get("failure_reasons") != []:
        raise ValueError("measured resource/technical gates did not all pass")
    for key in ("source_sha256", "calibration_sha256", "initial_state_sha256"):
        if receipt.get(key) != cohort[key]:
            raise ValueError("measured resource receipt belongs to a different cohort")
    if receipt.get("cohort_sha256") != release.get("cohort_sha256"):
        raise ValueError("measured resource cohort pin differs")
    if receipt.get("benchmark_summaries") != {arm: sha(p) for arm, p in benchmark_paths.items()}:
        raise ValueError("measured benchmark pins differ")
    for arm, path_ in benchmark_paths.items():
        summary = _read(path_)
        c = summary.get("contrastive", {})
        if (summary.get("steps") != 576 or summary.get("stopped") != "max_steps" or
            summary.get("world_size") != 4 or c.get("config", {}).get("arm") != arm or
            c.get("initial_sha256") != cohort["initial_state_sha256"] or
            c.get("provenance", {}).get("source_sha256") != cohort["source_sha256"] or
            c.get("provenance", {}).get("calibration_sha256") != cohort["calibration_sha256"] or
            c.get("provenance", {}).get("cohort_sha256") != release["cohort_sha256"]):
            raise ValueError("incomplete/unmatched measured benchmark: " + arm)
    allocations = receipt.get("training_allocations_gpu_hours")
    if not isinstance(allocations, dict) or set(allocations) != {"C0", "CN", "CR"}:
        raise ValueError("resource receipt must cover all three training allocations")
    values = list(allocations.values())
    cap = release.get("approved_training_gpu_hours")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
            or type(cap) not in (int, float) or not math.isfinite(cap) or sum(values) > cap):
        raise ValueError("three-arm allocation exceeds reviewed measured budget")
    by_phase = receipt.get("training_allocations_gpu_hours_by_phase")
    if not isinstance(by_phase, dict) or set(by_phase) != set(allocations):
        raise ValueError("resource receipt must separately bound both scientific phases")
    for arm, phases in by_phase.items():
        if (not isinstance(phases, dict) or set(phases) != {"5000", "19618"} or
                any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in phases.values()) or
                not math.isclose(sum(phases.values()), allocations[arm], rel_tol=1e-12)):
            raise ValueError("phase allocations do not sum to the reviewed arm allocation")
    arm = release.get("arm")
    endpoint = "5000" if release.get("phase") == "initial" else "19618"
    if arm not in allocations or 4 * release["time_limit_s"] / 3600 > by_phase[arm][endpoint]:
        raise ValueError("requested allocation exceeds measured per-arm budget")
    if not release.get("scientific_decision_pin") or not release.get("budget_decision_pin"):
        raise ValueError("training requires pinned scientific and budget decisions")
    return receipt


def validate_release(release, stage, config_path, input_paths, calibration_path=None, time_limit_s=None,
                     *, audit_path=None, cohort_path=None, calibration_run_dir=None,
                     budget_path=None, benchmark_paths=None):
    required = "APPROVED_FOR_TRAINING" if stage == "training" else "APPROVED_FOR_PREPARATION"
    if release.get("status") != required or release.get("stage") != stage:
        raise ValueError("review-only plan is not an execution release")
    if release.get("source_sha256") != source_digest() or release.get("config_sha256") != sha(config_path):
        raise ValueError("release source/config pins differ")
    if type(release.get("world_size")) is not int or release["world_size"] != 4 or type(release.get("workers_per_rank")) is not int or release["workers_per_rank"] != 6:
        raise ValueError("release must preserve four ranks and six workers per rank")
    pins = release.get("inputs", {})
    if set(pins) != set(input_paths) or any(pins[k] != sha(p) for k, p in input_paths.items()):
        raise ValueError("release data pins differ")
    validate_input_bindings(pins)
    if not release.get("review_pin"):
        raise ValueError("release must identify the converged review")
    limit = release.get("time_limit_s")
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit or time_limit_s != limit:
        raise ValueError("missing/mismatched bounded time allocation")
    if audit_path is None or release.get("audit_sha256") != sha(audit_path):
        raise ValueError("CPU audit receipt missing or changed")
    raw = _read(config_path)
    validate_audit(audit_path, input_paths, build_review_configs(_read(BASELINE))["calibration"])
    cohort = None
    if stage != "calibration":
        if calibration_path is None or release.get("calibration_sha256") != sha(calibration_path):
            raise ValueError("calibration receipt missing or changed")
        if cohort_path is None or release.get("cohort_sha256") != sha(cohort_path):
            raise ValueError("shared calibrated cohort missing or changed")
        cohort = validate_cohort(cohort_path, calibration_path, calibration_run_dir, config_path)
        if _read(calibration_path).get("input_sha256") != pins:
            raise ValueError("execution inputs differ from the shared calibration evidence")
        if release.get("arm") != raw["contrastive"]["arm"]:
            raise ValueError("release arm differs from common cohort member")
    if stage == "benchmark":
        phase = release.get("phase")
        if phase not in ("full", "split", "resume") or raw["max_steps"] != (544 if phase == "split" else 576):
            raise ValueError("benchmark phase must be full576, split544 or resume576")
    if stage == "training":
        if release.get("phase") not in ("initial", "resume") or raw["max_steps"] != (5000 if release["phase"] == "initial" else 19618):
            raise ValueError("training phase must be initial5000 or explicit resume19618")
        if budget_path is None or release.get("measured_budget_sha256") != sha(budget_path):
            raise ValueError("training requires measured three-arm benchmark and resource receipt")
        validate_measured_budget(budget_path, release, cohort, benchmark_paths)
    return cohort


def validate_resume(release, out_dir, cohort):
    """An explicit pinned continuation, never an implicit retry."""
    checkpoint, summary_path = Path(out_dir)/"checkpoint.pt", Path(out_dir)/"summary.json"
    if (release.get("resume_checkpoint_sha256") != sha(checkpoint) or
            release.get("resume_summary_sha256") != sha(summary_path)):
        raise ValueError("resume checkpoint/summary pins differ")
    summary = _read(summary_path)
    c = summary.get("contrastive", {})
    prior_step = 544 if release["stage"] == "benchmark" else 5000
    if (type(summary.get("steps")) is not int or summary["steps"] != prior_step or summary.get("stopped") != "max_steps" or
            c.get("initial_sha256") != cohort["initial_state_sha256"] or
            c.get("provenance", {}).get("calibration_sha256") != cohort["calibration_sha256"] or
            c.get("config", {}).get("arm") != release["arm"] or (Path(out_dir)/"STOP").exists()):
        raise ValueError(f"resume requires the successful matching {prior_step}-update split")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (type(state.get("step")) is not int or state["step"] != prior_step or state.get("world_size") != 4 or
            state.get("contrastive_initial_sha256") != cohort["initial_state_sha256"] or
            state.get("contrastive_provenance") != c.get("provenance") or
            state.get("contrastive_state", {}).get("queue", {}).get("last_step") != prior_step - 1 or
            len(state.get("rank_rng", [])) != 4):
        raise ValueError("resume checkpoint state disagrees with the successful four-rank split")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=("calibration", "benchmark", "training"), required=True)
    ap.add_argument("--calibration-report", type=Path)
    ap.add_argument("--calibration-run-dir", type=Path)
    ap.add_argument("--cohort", type=Path)
    check = ap.add_mutually_exclusive_group()
    check.add_argument("--validate-only", action="store_true", help="validate the complete release and all evidence")
    check.add_argument("--recipe-only", action="store_true", help="check recipe only; never validates a release")
    ap.add_argument("--release", type=Path)
    ap.add_argument("--audit-report", type=Path)
    ap.add_argument("--measured-budget", type=Path)
    ap.add_argument("--benchmark-summary", action="append", help="ARM=summary.json, each of C0/CN/CR")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--scales", type=Path)
    ap.add_argument("--window-seconds", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--time-limit-s", type=float)
    a = ap.parse_args()
    raw = _read(a.config)
    calibration = _read(a.calibration_report) if a.calibration_report else None
    cfg = validate_recipe(raw, a.stage, calibration, a.calibration_run_dir)
    if a.recipe_only:
        if a.release is not None:
            ap.error("--recipe-only cannot accept a release; use --validate-only")
        print(json.dumps({"recipe_valid": True, "release_valid": False, "source_sha256": source_digest(), "config_sha256": sha(a.config)}))
        return
    if any(v is None for v in (a.release, a.manifest, a.root, a.scales, a.window_seconds, a.out_dir, a.time_limit_s)):
        ap.error("execution/release validation requires reviewed release, all pinned inputs and bounded output/time")
    release = _read(a.release)
    input_paths = {"manifest": a.manifest, "scales": a.scales, "window_seconds": a.window_seconds,
                   "pose_stats": Path(cfg.loader.pose_root)/"pose_stats.json"}
    benchmark_paths = None
    if a.benchmark_summary:
        pairs = [value.split("=", 1) for value in a.benchmark_summary]
        if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
            ap.error("benchmark summaries require distinct ARM=path arguments")
        benchmark_paths = {arm: Path(path) for arm, path in pairs}
    cohort = validate_release(release, a.stage, a.config, input_paths, a.calibration_report, a.time_limit_s,
        audit_path=a.audit_report, cohort_path=a.cohort, calibration_run_dir=a.calibration_run_dir,
        budget_path=a.measured_budget, benchmark_paths=benchmark_paths)
    R.check_input_files(a.manifest, a.window_seconds, cfg.nested, release["inputs"]["manifest"])
    if any(not root.is_dir() for root in a.root):
        raise ValueError("missing data root")
    if release.get("phase") == "resume":
        validate_resume(release, a.out_dir, cohort)
    elif a.out_dir.exists():
        raise ValueError("new phase requires a new output directory; continuation must be explicit")
    if a.validate_only:
        print(json.dumps({"recipe_valid": True, "release_valid": True, "execution_started": False,
                          "source_sha256": source_digest(), "config_sha256": sha(a.config)}))
        return
    ctx = R.dist_from_env()
    if ctx.world_size != 4 or not torch.cuda.is_available():
        raise ValueError("reviewed campaign requires four CUDA ranks")
    torch.cuda.set_device(ctx.local_rank)
    torch.distributed.init_process_group("nccl")
    try:
        provenance = {"source_sha256": source_digest(), "calibration_sha256": sha(a.calibration_report) if a.calibration_report else None,
                      "cohort_sha256": sha(a.cohort) if a.cohort else None, "input_sha256": release["inputs"]}
        if provenance["source_sha256"] != release["source_sha256"]:
            raise ValueError("execution sources changed after release validation")
        summary = R.train(cfg, a.manifest, a.root, a.out_dir, scales=_read(a.scales)["scales"],
            window_s=_read(a.window_seconds)["window_s"], device=f"cuda:{ctx.local_rank}",
            num_workers=6, time_limit_s=a.time_limit_s, dist=ctx, log=lambda m: print(m, flush=True),
            expected_contrastive_initial_sha256=cohort["initial_state_sha256"] if cohort else None,
            contrastive_provenance=provenance)
        if summary.get("steps") != cfg.max_steps or summary.get("stopped") != "max_steps":
            raise RuntimeError("phase incomplete: inspect summary/STOP, do not silently retry")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
