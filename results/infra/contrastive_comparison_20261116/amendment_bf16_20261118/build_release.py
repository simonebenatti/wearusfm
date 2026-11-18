"""Build the bf16-diagnostic-amendment deployment artifacts locally; never submit a job.

Copy of GPT's compatibility_20261118/build_release.py (mmap release) by Claude, 18/11/2026,
with only the new review, cohort, release and remote-output references and the consumed
resources updated after CUDA 64730941. Existing evidence, configurations and releases are
read-only; new destinations must not exist. Frozen Git SHA and exact submission commands
are published separately by the line-3 operator.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "results/infra/contrastive_comparison_20261116"
MANIFEST = BASE / "compatibility_20261118/manifest.json"
REVIEW = ROOT / "docs/reviews/claude_contrastive_bf16_amendment_20261118.md"
OLD_COHORT = BASE / "calibrated_cohort_20261117"
NEW_COHORT = BASE / "calibrated_cohort_20261118_bf16"
OLD_RELEASES = BASE / "releases/benchmarks_20261117"
NEW_RELEASES = BASE / "releases/bf16_20261118"
CALIBRATION = BASE / "calibration_completed_20261117/calibration_report.json"
REMOTE = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261118_bf16"
HISTORICAL_SOURCE = "ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b"
CELLS = (("full", "C0"), ("full", "CN"), ("full", "CR"),
         ("split", "CN"), ("split", "CR"))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, base=ROOT):
    path = Path(path)
    return {"path": str(path.relative_to(base)), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from contrastive_prepare import source_digest

    if NEW_COHORT.exists() or NEW_RELEASES.exists():
        raise ValueError("new release destinations already exist; never overwrite a release")
    manifest = read(MANIFEST)
    source = source_digest()
    if (manifest.get("historical_source_sha256") != HISTORICAL_SOURCE or
            manifest.get("current_source_sha256") != source or source == HISTORICAL_SOURCE):
        raise ValueError("compatibility manifest is not final for the current source")
    review_pin, compatibility_pin = pin(REVIEW), pin(MANIFEST)
    original_cohort = read(OLD_COHORT / "cohort.json")
    receipt = read(CALIBRATION)
    if (receipt.get("source_sha256") != HISTORICAL_SOURCE or
            receipt.get("status") != "PASSED_FOR_REVIEW" or receipt.get("failure_reasons") != [] or
            original_cohort.get("source_sha256") != HISTORICAL_SOURCE or
            original_cohort.get("calibration_sha256") != sha(CALIBRATION) or
            original_cohort.get("training_approved") is not False):
        raise ValueError("original calibration/cohort evidence changed")
    configs = sorted(OLD_COHORT.glob("*.config.json"))
    expected_names = {f"{arm}{suffix}.config.json" for arm in ("C0", "CN", "CR")
                      for suffix in ("", ".split", ".5000", ".19618")}
    if {path.name for path in configs} != expected_names:
        raise ValueError("unexpected historical config set")
    # Read all historical job/release inputs before creating any output.
    old_cuda = read(BASE / "releases/cuda.job.json")
    old_cells = {(mode, arm): (read(OLD_RELEASES / f"{mode}_{arm}.job.json"),
                              read(OLD_RELEASES / f"{mode}_{arm}.release.json"))
                 for mode, arm in CELLS}
    for job, release in old_cells.values():
        if (job["source_sha256"] != HISTORICAL_SOURCE or
                release["source_sha256"] != HISTORICAL_SOURCE or
                job["calibration_report"] != pin(CALIBRATION) or
                release["calibration_sha256"] != sha(CALIBRATION)):
            raise ValueError("historical benchmark bindings changed")
    if old_cuda["source_sha256"] != HISTORICAL_SOURCE:
        raise ValueError("historical CUDA source changed")

    NEW_COHORT.mkdir()
    NEW_RELEASES.mkdir()
    for old in configs:
        new = NEW_COHORT / old.name
        with new.open("xb") as handle:
            handle.write(old.read_bytes())
        assert new.read_bytes() == old.read_bytes()
    cohort = copy.deepcopy(original_cohort)
    cohort["source_sha256"] = source
    cohort["configs"] = {arm: pin(NEW_COHORT / f"{arm}.config.json", NEW_COHORT)
                         for arm in ("C0", "CN", "CR")}
    write(NEW_COHORT / "cohort.json", cohort)
    cohort_pin = pin(NEW_COHORT / "cohort.json")

    cuda = copy.deepcopy(old_cuda)
    cuda.update(source_sha256=source, review=review_pin,
                compatibility_manifest=compatibility_pin, out=REMOTE + "/cuda.json")
    write(NEW_RELEASES / "cuda.job.json", cuda)
    job_pins = {"CUDA": pin(NEW_RELEASES / "cuda.job.json")}
    release_pins = {}
    cuda_gate = None

    for mode, arm in CELLS:
        old_job, old_release = old_cells[(mode, arm)]
        job, release = copy.deepcopy(old_job), copy.deepcopy(old_release)
        config = NEW_COHORT / (f"{arm}.split.config.json" if mode == "split"
                               else f"{arm}.config.json")
        release.update(source_sha256=source, review_pin=review_pin,
                       compatibility_manifest=compatibility_pin,
                       cohort_sha256=cohort_pin["sha256"], config_sha256=sha(config))
        release["cuda_gate"]["report"] = REMOTE + "/cuda.json"
        cuda_gate = copy.deepcopy(release["cuda_gate"])
        release_path = NEW_RELEASES / f"{mode}_{arm}.release.json"
        write(release_path, release)
        release_pins[f"{mode}_{arm}"] = pin(release_path)
        job.update(source_sha256=source, review=review_pin,
                   compatibility_manifest=compatibility_pin, config=pin(config),
                   cohort=cohort_pin, release=pin(release_path),
                   out=REMOTE + ("/resume/" if mode == "split" else "/benchmark/") + arm)
        if mode == "split":
            job["resume_config"] = pin(NEW_COHORT / f"{arm}.config.json")
        # All actual evidence and numerical recipe identities remain unchanged.
        for field in ("inputs", "roots", "audit", "calibration_report", "calibration_summary"):
            assert job[field] == old_job[field]
        for field in ("inputs", "audit_sha256", "calibration_sha256", "time_limit_s",
                      "world_size", "workers_per_rank", "phase", "arm"):
            assert release[field] == old_release[field]
        assert release["config_sha256"] == old_release["config_sha256"]
        if mode == "split":
            assert release["paired_resume"] == old_release["paired_resume"]
        job_path = NEW_RELEASES / f"{mode}_{arm}.job.json"
        write(job_path, job)
        job_pins[f"{mode}_{arm}"] = pin(job_path)

    # Calibration 0.86 GPU-h; CUDA 64706211 (67 s) with checkout 64705989 (67 s);
    # CUDA 64730941 (127 s, 4 GPU, 32 CPU) with checkout 64730738 (36 s, 4 CPU).
    completed_gpu = 0.86 + 4 * 67 / 3600 + 4 * 127 / 3600
    completed_cpu = 15.45888888888889 + (4 + 32) * 67 / 3600 + (4 * 36 + 32 * 127) / 3600
    resources = {
        "completed_gpu_hours": completed_gpu,
        "completed_core_hours": completed_cpu,
        "remaining_gpu_cells": 6, "gpus_per_cell": 4,
        "cpus_per_gpu_cell": 32, "seconds_per_gpu_cell": 1800,
        "memory_gib_per_gpu_cell": 240,
        "remaining_checkouts": 2, "checkout_cpus": 4, "checkout_seconds": 240,
        "final_cpu_readout_core_hours_cap": 8,
        "actual_plus_remaining_max_gpu_hours": completed_gpu + 6 * 2,
        "actual_plus_remaining_max_core_hours": completed_cpu + 6 * 16 + 2 * 4 * 240 / 3600 + 8,
        "campaign_gpu_hours_cap": 14, "campaign_core_hours_cap": 129.33333333333334,
    }
    assert resources["actual_plus_remaining_max_gpu_hours"] <= resources["campaign_gpu_hours_cap"]
    assert resources["actual_plus_remaining_max_core_hours"] <= resources["campaign_core_hours_cap"]
    assert source_digest() == source
    assert pin(MANIFEST) == compatibility_pin and pin(REVIEW) == review_pin
    assert cuda["calibration_checkpoint"] == old_cuda["calibration_checkpoint"]
    assert cuda["inputs"] == old_cuda["inputs"] and cuda["config"] == old_cuda["config"]
    package = {
        "status": "CUDA_RELEASED_BENCHMARKS_CONDITIONAL_ON_CUDA",
        "review": review_pin, "compatibility_manifest": compatibility_pin,
        "historical_source_sha256": HISTORICAL_SOURCE, "source_sha256": source,
        "calibration_report": pin(CALIBRATION), "cohort": cohort_pin,
        "historical_cohort": pin(OLD_COHORT / "cohort.json"),
        "job_files": job_pins, "release_files": release_pins,
        "cuda_gate": cuda_gate, "resources": resources,
        "build_validation": {
            "all_12_configs_byte_identical": True,
            "original_calibration_receipt_unchanged": True,
            "audit_checkpoint_inputs_unchanged": True,
            "source_matches_compatibility_manifest": True,
            "compatibility_manifest_pin_in_all_6_jobs_and_5_releases": True,
            "runtime_and_external_review_required_before_submission": True,
        },
        "submitted": False, "training_approved": False,
    }
    write(NEW_RELEASES / "release.json", package)
    print(json.dumps({"release": str(NEW_RELEASES.relative_to(ROOT) / "release.json"),
                      "source_sha256": source, "configs_byte_identical": len(configs),
                      "job_count": len(job_pins), "resources": resources}, indent=2))


if __name__ == "__main__":
    main()
