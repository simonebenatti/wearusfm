"""Rebuild a released reload command and validate its common fork before submission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from contrastive_job import build_command
from contrastive_prepare import _read
from contrastive_release import checked_pin, make_job

PACKAGE = Path("results/infra/contrastive_targets_20261118")
BINDING = PACKAGE / "releases/audit_20261120/review_binding.json"
RELEASE = PACKAGE / "releases/reloads_20261121"
FULL_COHORT = PACKAGE / "releases/benchmark_20261121/cohort"
FULL_ARCHIVE = PACKAGE / "benchmark_execution_20261121"
FULL_SHA = "746112dd711903e5c92ade2efd90834264b2676a"
CELLS = ("CN_A", "CN_B", "CR_A", "CR_B")


def validate(cell):
    arm, repeat = cell.split("_")
    root = Path(os.environ["WORK"]) / "wearusfm_runs/results/contrastive_targets_20261118"
    prefix_dir = root / "benchmark" / arm
    output = root / "reloads" / cell
    logical_id = f"CONTRAST-RELOAD-{arm}-{repeat}-targets-v3-20261118"
    job_path = RELEASE / f"reload_{cell}.job.json"
    evidence = _read(RELEASE / f"reload_{cell}.evidence.json")
    expected_job = make_job(_read(BINDING), logical_id, evidence, output)
    if (json.dumps(expected_job, indent=2, allow_nan=False) + "\n").encode() != job_path.read_bytes():
        raise ValueError("Released job differs from the regenerated cell")
    job = _read(job_path)
    config_path, release_path, cohort_path = [checked_pin(job[key]) for key in ("config", "release", "cohort")]
    if (config_path != (FULL_COHORT / f"{arm}.config.json").resolve()
            or release_path != (RELEASE / f"reload_{cell}.release.json").resolve()
            or cohort_path != (FULL_COHORT / "cohort.json").resolve()):
        raise ValueError("Wrong reload config, release or common cohort")
    raw, release, cohort = map(_read, (config_path, release_path, cohort_path))
    cfg = raw["contrastive"]
    if (job.get("mode") != "resume" or job.get("repeat") != repeat.lower()
            or release.get("stage") != "benchmark" or release.get("phase") != "resume_" + repeat.lower()
            or any(payload.get("arm") != arm for payload in (job, release, cfg))
            or cohort.get("training_approved") is not False or cohort.get("endpoint") != 576
            or raw.get("batch_size") != 32 or raw.get("max_steps") != 576 or cfg["calibration_only"]
            or raw["loader"]["rms_window_ms"] != 100 or raw["loader"]["ms_fast_window_ms"] != 125
            or output.exists() or output.is_symlink() or output.parent.is_symlink()):
        raise ValueError("Reload dose, arm, repeat, targets, or destination differ")
    calibration = checked_pin(job["calibration_report"])
    receipt = _read(calibration)
    if (calibration != (root / "calibration_report.json").resolve()
            or checked_pin(job["calibration_summary"]) != (root / "calibration/summary.json").resolve()
            or checked_pin(job["cuda_report"]) != (root / "cuda.json").resolve()
            or cfg["nce_weight"] != receipt["weights"]["nce"] or cfg["rnc_weight"] != receipt["weights"]["rnc"]):
        raise ValueError("Wrong calibration, coefficients or CUDA receipt")

    # Bind to the one original step-544 file of this arm; build_command and
    # the runner below independently hash the physical file and inspect its payload.
    fork = job["fork_checkpoint"]
    fork_name = "checkpoint_step0000544.pt"
    rel_fork = f"{arm}/{fork_name}"
    fingerprint_lines = (FULL_ARCHIVE / "checkpoints_remote.txt").read_text().splitlines()
    if (fork != release.get("fork_checkpoint") or Path(fork["path"]) != prefix_dir / fork_name
            or f"{fork['sha256']}  {rel_fork}" not in fingerprint_lines
            or f"{fork['bytes']} bytes {rel_fork}" not in fingerprint_lines):
        raise ValueError("Fork differs from the archived checkpoint identity or its original path")
    summary_path = checked_pin(job["continuous_summary"])
    if summary_path != (prefix_dir / "summary.json").resolve() or job["continuous_summary"] != release.get("continuous_summary"):
        raise ValueError("Wrong continuous benchmark summary")
    summary = _read(summary_path)
    if (summary["code"]["sha"] != FULL_SHA or summary.get("steps") != 576 or summary.get("stopped") != "max_steps"
            or summary["contrastive"]["config"]["arm"] != arm or (prefix_dir / "STOP").exists()):
        raise ValueError("Full run is not the completed matching continuous arm")

    command = build_command(job, "gpu", 1800)
    runner = str(Path("scripts/contrastive_run.py").resolve())
    worker_prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", runner]
    if command[:len(worker_prefix)] != worker_prefix:
        raise ValueError("Unexpected reload worker")
    subprocess.run([sys.executable, runner, "--validate-only", *command[len(worker_prefix):]], check=True)
    print(json.dumps({"id": logical_id, "job_arm": job["arm"], "config_arm": cfg["arm"], "release_arm": release["arm"],
                      "phase": release["phase"], "start_checkpoint_step": 544, "end_step": 576,
                      "fork_sha256": fork["sha256"], "fork_bytes": fork["bytes"], "output": str(output),
                      "full_summary_sha256": job["continuous_summary"]["sha256"], "batch_size": raw["batch_size"],
                      "calibrated_coefficients": receipt["weights"], "execution_argv": command,
                      "execution_started": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", choices=CELLS)
    validate(parser.parse_args().cell)
