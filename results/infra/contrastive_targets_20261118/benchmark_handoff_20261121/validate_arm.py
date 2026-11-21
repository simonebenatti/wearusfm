"""Replay the released cell and its actual runner validation, without starting a rank."""
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
RELEASE = PACKAGE / "releases/benchmark_20261121"


def validate(arm):
    root = Path(os.environ["WORK"]) / "wearusfm_runs/results/contrastive_targets_20261118"
    output = root / "benchmark" / arm
    logical_id = f"CONTRAST-BENCH-{arm}-targets-v3-20261118"
    job_path = RELEASE / f"benchmark_{arm}.job.json"
    evidence = _read(RELEASE / f"benchmark_{arm}.evidence.json")
    expected_job = make_job(_read(BINDING), logical_id, evidence, output)
    if (json.dumps(expected_job, indent=2, allow_nan=False) + "\n").encode() != job_path.read_bytes():
        raise ValueError("Published cell differs from regenerated job manifest")
    job = _read(job_path)
    config_path = checked_pin(job["config"])
    release_path = checked_pin(job["release"])
    cohort_path = checked_pin(job["cohort"])
    raw = _read(config_path)
    release = _read(release_path)
    cohort = _read(cohort_path)
    cfg = raw["contrastive"]
    receipt = _read(checked_pin(job["calibration_report"]))
    if (config_path != (RELEASE / "cohort" / f"{arm}.config.json").resolve()
            or release_path != (RELEASE / f"benchmark_{arm}.release.json").resolve()
            or cohort_path != (RELEASE / "cohort/cohort.json").resolve()
            or any(payload.get("arm") != arm for payload in (job, release, cfg))
            or job.get("mode") != "full" or job.get("repeat") is not None
            or release.get("stage") != "benchmark" or release.get("phase") != "full"
            or cohort.get("training_approved") is not False or cohort.get("endpoint") != 576
            or raw.get("batch_size") != 32 or raw.get("max_steps") != 576 or cfg["calibration_only"]
            or cfg["nce_weight"] != receipt["weights"]["nce"] or cfg["rnc_weight"] != receipt["weights"]["rnc"]
            or raw["loader"]["rms_window_ms"] != 100 or raw["loader"]["ms_fast_window_ms"] != 125
            or checked_pin(job["calibration_report"]) != (root / "calibration_report.json").resolve()
            or checked_pin(job["calibration_summary"]) != (root / "calibration/summary.json").resolve()
            or checked_pin(job["cuda_report"]) != (root / "cuda.json").resolve()
            or output.exists() or output.is_symlink()):
        raise ValueError(f"Cell, config, release, cohort, inputs or output differ for {arm}")

    command = build_command(job, "gpu", 1800)
    runner = str(Path("scripts/contrastive_run.py").resolve())
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", runner]
    if command[:len(prefix)] != prefix:
        raise ValueError("Unexpected full-benchmark worker")
    # The actual worker parses/validates the release and checks data, CUDA and rank-exit receipts.
    # Only its command-line validation branch runs on the login node.
    subprocess.run([sys.executable, runner, "--validate-only", *command[len(prefix):]], check=True)
    print(json.dumps({"id": logical_id, "job_arm": job["arm"], "config_arm": cfg["arm"], "release_arm": release["arm"],
                      "phase": release["phase"], "dose": raw["max_steps"], "batch_size": raw["batch_size"],
                      "rms_window_ms": raw["loader"]["rms_window_ms"], "fast_window_ms": raw["loader"]["ms_fast_window_ms"],
                      "calibrated_coefficients": receipt["weights"], "execution_argv": command, "execution_started": False},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("C0", "CN", "CR"))
    validate(parser.parse_args().arm)
