"""Pinned entry point for reviewed contrastive calibration/benchmark/training.

No SLURM submission. The committed review plan cannot execute this entry point:
a later reviewed release must pin this code, the exact config and input files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

# Same import order as production train_jepa: Torch before training/NumPy.
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import config_from_dict, build_review_configs, BASELINE, BASELINE_SHA256
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    paths = sorted((ROOT/"src/wearusfm").rglob("*.py"))
    paths += [ROOT/"scripts"/n for n in ("contrastive_run.py", "contrastive_prepare.py", "train_jepa.py")]
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode()+b"\0"+bytes.fromhex(sha(p)))
    return h.hexdigest()


def validate_recipe(raw, stage, calibration=None):
    if sha(BASELINE) != BASELINE_SHA256:
        raise ValueError("historical baseline changed")
    recipes = build_review_configs(json.loads(BASELINE.read_text()))
    arm = raw.get("contrastive", {}).get("arm")
    if stage == "calibration":
        expected = recipes["calibration"]
    else:
        if arm not in ("C0", "CN", "CR") or not isinstance(calibration, dict) or calibration.get("status") != "PASSED_FOR_REVIEW":
            raise ValueError("benchmark/training requires a complete calibrated candidate and receipt")
        expected = recipes[arm]
        weights = calibration.get("weights", {})
        for key in ("nce", "rnc"):
            value = weights.get(key)
            if type(value) not in (float, int) or not 1e-4 <= value <= 1:
                raise ValueError("invalid calibrated contrastive weight")
            expected["contrastive"][key+"_weight"] = value
        endpoint = raw.get("max_steps")
        if stage == "benchmark":
            if endpoint != 576:
                raise ValueError("technical benchmark must cover 64 full-weight updates after 512")
        elif endpoint not in (5000, 19618):
            raise ValueError("scientific dose must be 5000 or 19618")
        expected["max_steps"] = endpoint
    cfg = config_from_dict(raw)
    ContrastiveConfig(**cfg.contrastive)
    if R.config_to_dict(cfg) != expected:
        raise ValueError("configuration differs from the frozen comparison recipe")
    return cfg


def validate_release(release, stage, config_path, input_paths, calibration_path=None, time_limit_s=None):
    if stage == "training":
        raise ValueError("training release unavailable in review package: measured three-arm benchmark "
                         "and resource validator must be implemented and reviewed before deployment")
    required = "APPROVED_FOR_TRAINING" if stage == "training" else "APPROVED_FOR_PREPARATION"
    if release.get("status") != required or release.get("stage") != stage:
        raise ValueError("review-only plan is not an execution release")
    if release.get("source_sha256") != source_digest() or release.get("config_sha256") != sha(config_path):
        raise ValueError("release source/config pins differ")
    if release.get("world_size") != 4 or release.get("workers_per_rank") != 6:
        raise ValueError("release must preserve four ranks and six workers per rank")
    pins = release.get("inputs", {})
    if set(pins) != set(input_paths) or any(pins[k] != sha(p) for k, p in input_paths.items()):
        raise ValueError("release data pins differ")
    if stage != "calibration" and (calibration_path is None or release.get("calibration_sha256") != sha(calibration_path)):
        raise ValueError("calibration receipt missing or changed")
    if not release.get("review_pin"):
        raise ValueError("release must identify the converged review")
    limit = release.get("time_limit_s")
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit or time_limit_s != limit:
        raise ValueError("missing/mismatched measured time allocation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=("calibration", "benchmark", "training"), required=True)
    ap.add_argument("--calibration-report", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--release", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--scales", type=Path)
    ap.add_argument("--window-seconds", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--time-limit-s", type=float)
    a = ap.parse_args()
    raw = json.loads(a.config.read_text())
    calibration = json.loads(a.calibration_report.read_text()) if a.calibration_report else None
    cfg = validate_recipe(raw, a.stage, calibration)
    if a.validate_only:
        print(json.dumps({"recipe_valid": True, "executable_without_release": False,
                          "source_sha256": source_digest(), "config_sha256": sha(a.config)}))
        return
    if any(v is None for v in (a.release, a.manifest, a.root, a.scales, a.window_seconds, a.out_dir, a.time_limit_s)):
        ap.error("execution requires reviewed release, all pinned inputs and bounded output/time")
    release = json.loads(a.release.read_text())
    input_paths = {"manifest": a.manifest, "scales": a.scales, "window_seconds": a.window_seconds,
                   "pose_stats": Path(cfg.loader.pose_root)/"pose_stats.json"}
    validate_release(release, a.stage, a.config, input_paths, a.calibration_report, a.time_limit_s)
    if a.stage != "training" and a.out_dir.exists():
        raise ValueError("preparatory run requires a new output directory")
    ctx = R.dist_from_env()
    if ctx.world_size != 4 or not torch.cuda.is_available():
        raise ValueError("reviewed campaign requires four CUDA ranks")
    torch.cuda.set_device(ctx.local_rank)
    torch.distributed.init_process_group("nccl")
    try:
        R.check_input_files(a.manifest, a.window_seconds, cfg.nested, release["inputs"]["manifest"])
        summary = R.train(cfg, a.manifest, a.root, a.out_dir, scales=json.loads(a.scales.read_text())["scales"],
            window_s=json.loads(a.window_seconds.read_text())["window_s"], device=f"cuda:{ctx.local_rank}",
            num_workers=6, time_limit_s=a.time_limit_s, dist=ctx, log=lambda m: print(m, flush=True))
        if summary.get("steps") != cfg.max_steps or summary.get("stopped") != "max_steps":
            raise RuntimeError("phase incomplete: inspect summary/STOP, do not silently retry")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
