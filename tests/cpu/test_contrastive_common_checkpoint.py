"""Control a live/save/reload split in the real trainer, including actual inputs."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/diagnostics"))
import contrastive_common_checkpoint as check
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as trainer


def _partial(directory, cfg, manifest, root):
    with check.trace_updates(directory / "trace-0.jsonl", 4):
        summary = trainer.train(cfg, manifest, [root], directory, log=lambda _: None)
    assert summary["steps"] == 5


def test_production_checkpoint_branch_and_trace(tmp_path):
    data, manifest, pose = _pose_setup(tmp_path)
    initial_cfg = _config(pose, "CR")
    cfg = replace(initial_cfg, max_steps=2, model=replace(initial_cfg.model, muscle_dropout=.4),
                  schedule=replace(initial_cfg.schedule, total_steps=20, keep_checkpoints=()))
    trainer.train(cfg, manifest, [data], tmp_path / "historical", log=lambda _: None)
    original = tmp_path / "historical/checkpoint.pt"
    before = check.file_sha(original)
    common = tmp_path / "continuous"
    common.mkdir()
    check.diagnostic_seed(original, common / "checkpoint.pt", before, 4)
    cfg = replace(cfg, max_steps=5, schedule=replace(cfg.schedule, keep_checkpoints=(4,)))
    _partial(common, cfg, manifest, data)
    fork = common / "checkpoint_step0000004.pt"
    for phase in check.PHASES[1:]:
        folder = tmp_path / phase
        folder.mkdir()
        shutil.copyfile(fork, folder / "checkpoint.pt")
        torch.multiprocessing.spawn(_spawn_reload, args=(folder, cfg, manifest, data), nprocs=1)
    assert check.file_sha(original) == before
    analysis = check.cpu_analysis(tmp_path, 4, 5, 1)
    assert not analysis["boundary_state_or_rng_mismatches"]
    for row in analysis["pairs"].values():
        assert not row["input_rng_boundary_differences"]
        assert all(row["final_exact"].values())
        assert row["model_relative_to_left_displacement_from_fork"] == {"student": 0., "teacher": 0.}
        assert row["optimizer_moments_relative_to_left"] == {"exp_avg": 0., "exp_avg_sq": 0.}
        assert row["queue_keys_relative_to_left"] == 0.
    trace = tmp_path / "reload_a/trace-0.jsonl"
    record = json.loads(trace.read_text())
    record["rng_before_sha256"] = "wrong"
    trace.write_text(json.dumps(record) + "\n")
    bad = check.cpu_analysis(tmp_path, 4, 5, 1)
    assert bad["boundary_state_or_rng_mismatches"] == [["reload_a", 0, "rng_before_sha256"]]
    assert bad["pairs"]["continuous__reload_a"]["input_rng_boundary_differences"]
    with pytest.raises(ValueError, match="pin"):
        check.diagnostic_seed(original, tmp_path / "bad.pt", "wrong", 4)


def _spawn_reload(rank, folder, cfg, manifest, data):
    torch.set_num_threads(1)
    _partial(folder, cfg, manifest, data)
