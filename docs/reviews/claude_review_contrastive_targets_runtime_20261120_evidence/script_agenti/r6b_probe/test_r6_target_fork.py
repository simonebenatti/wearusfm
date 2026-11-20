"""r6 probe: contrastive_run.validate_target_fork (no repository test references it)."""
import json, shutil
from pathlib import Path
import pytest
import torch
import contrastive_run as runner
import contrastive_release as REL


def setup(tmp_path):
    full = tmp_path / "full_CN"; full.mkdir()
    cfg = {"max_steps": 576, "schedule": {"keep_checkpoints": [544, 5000, 19618]}, "contrastive": {"arm": "CN"}}
    cohort = {"initial_state_sha256": "i" * 64, "calibration_sha256": "c" * 64}
    inputs = {"manifest": "m" * 64}
    provenance = {"source_sha256": runner.source_digest(), "calibration_sha256": cohort["calibration_sha256"],
                  "cohort_sha256": "b" * 64, "input_sha256": inputs}
    state = {"step": 544, "world_size": 4, "config": cfg, "contrastive_initial_sha256": cohort["initial_state_sha256"],
             "contrastive_provenance": provenance, "contrastive_state": {"queue": {"last_step": 543}},
             "rank_rng": [{"torch_rng": torch.zeros(3, dtype=torch.uint8), "cuda_rng": torch.zeros(3, dtype=torch.uint8)} for _ in range(4)]}
    torch.save(state, full / "checkpoint_step0000544.pt")
    summary = {"steps": 576, "stopped": "max_steps", "world_size": 4,
               "contrastive": {"config": {"arm": "CN"}, "initial_sha256": cohort["initial_state_sha256"], "provenance": provenance}}
    (full / "summary.json").write_text(json.dumps(summary))
    release = {"fork_checkpoint": REL.pin(full / "checkpoint_step0000544.pt"), "continuous_summary": REL.pin(full / "summary.json"),
               "arm": "CN", "cohort_sha256": "b" * 64, "inputs": inputs}
    return full, cfg, cohort, release, summary


def test_accepts_matching_fork(tmp_path):
    full, cfg, cohort, release, _ = setup(tmp_path)
    assert runner.validate_target_fork(release, cfg, cohort).name == "checkpoint_step0000544.pt"


@pytest.mark.parametrize("mutation", ["stop", "steps", "arm_release", "other_dir", "renamed", "provenance"])
def test_rejects(tmp_path, mutation):
    full, cfg, cohort, release, summary = setup(tmp_path)
    if mutation == "stop":
        (full / "STOP").write_text("x")
    elif mutation == "steps":
        summary["steps"] = 575; (full / "summary.json").write_text(json.dumps(summary))
        release["continuous_summary"] = REL.pin(full / "summary.json")
    elif mutation == "arm_release":
        release["arm"] = "CR"
    elif mutation == "other_dir":
        other = tmp_path / "other"; other.mkdir(); shutil.copy(full / "summary.json", other / "summary.json")
        release["continuous_summary"] = REL.pin(other / "summary.json")
    elif mutation == "renamed":
        shutil.copy(full / "checkpoint_step0000544.pt", full / "checkpoint.pt")
        release["fork_checkpoint"] = REL.pin(full / "checkpoint.pt")
    else:
        release["cohort_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        runner.validate_target_fork(release, cfg, cohort)
