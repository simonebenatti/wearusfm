"""Read the completed CUDA reports on Leonardo. Never submits or starts another phase."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from contrastive_job import rank_exit_report
from contrastive_prepare import _read, authenticate_calibration
from contrastive_release import checked_pin, file_sha, pin, validate_cuda_prerequisite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-sha", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != args.execution_sha:
        raise ValueError("Use the execution checkout for the CUDA readout")
    job = _read(HERE.parent / "releases/cuda_20261121/cuda.job.json")
    calibration = _read(checked_pin(job["calibration_report"]))
    folder = checked_pin(job["calibration_summary"]).parent
    authenticate_calibration(calibration, folder)
    cuda_path = Path(job["out"])
    result = _read(cuda_path)
    if result.get("rank_exit_report_path") != str(rank_exit_report(cuda_path)):
        raise ValueError("Rank-exit path differs from this CUDA cell")
    if result.get("environment", {}).get("code_sha") != actual:
        raise ValueError("CUDA report was written by a different execution commit")
    if result.get("inputs", {}).get("config") != job["config"]["sha256"]:
        raise ValueError("CUDA report used a different config file")
    if result.get("inputs", {}).get("calibration_checkpoint") != job["calibration_checkpoint"]["sha256"]:
        raise ValueError("CUDA report used a different checkpoint")
    # The frozen validator checks all ranks, thresholds, the physical checkpoint and rank-exit evidence.
    result = validate_cuda_prerequisite(pin(cuda_path), calibration, calibration["input_sha256"], folder)
    print(json.dumps({
        "job": job["id"], "passed": result["passed"], "report_sha256": file_sha(cuda_path),
        "rank_exit_sha256": result["rank_exit_report_sha256"], "world_size": result["world_size"],
        "execution_sha": actual, "checkpoint_sha256": result["inputs"]["calibration_checkpoint"],
        "calibration_report_sha256": job["calibration_report"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
