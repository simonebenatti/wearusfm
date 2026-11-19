"""Common-fork restart gates, including live trainer instrumentation on CPU."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import contrastive_target_resume as C
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256


def _state(initial=0., final=1.):
    return {
        "student": {"backbone.w": torch.tensor([final]), "b": torch.tensor([0])},
        "teacher": {"backbone.w": torch.tensor([final]), "b": torch.tensor([0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([1.]), "exp_avg_sq": torch.tensor([1.e-6]), "step": torch.tensor(576.)}},
                      "param_groups": [{"lr": .001, "params": [0]}]},
        "contrastive_state": {"queue": {"last_step": 575, "entries": [{"id": "a", "key": torch.tensor([1.])}]}},
        "stream": {"seed": 1}, "config": {}, "data_signature": {}, "pose_stats_sha256": "p",
        "contrastive_provenance": {"s": "a"}, "contrastive_initial_sha256": "i", "alarm_counts": [0, 0],
        "rank_rng": [{"torch_rng": torch.tensor([1]), "cuda_rng": torch.tensor([2])}],
    }


def test_original_denominator_and_fork_distance_are_distinct():
    left, right, fork = _state(), _state(final=1.005), _state(final=.99)
    init = {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}
    result = C.historical_numeric_pair(left, right, init, {"backbone.w"}, fork=fork)
    assert result["passed"]
    assert result["relative_to_left_displacement_from_initial"]["student"] == pytest.approx(.005, rel=1e-4)
    assert result["relative_to_left_displacement_from_544"]["student"] == pytest.approx(.5, rel=1e-4)
    assert result["by_block_relative_to_left_displacement_from_fork"]["teacher"]["backbone"] == pytest.approx(.5, rel=1e-4)


@pytest.mark.parametrize("part", ["student", "teacher"])
def test_historical_one_percent_applies_to_both_models(part):
    left, right = _state(), _state()
    right[part]["backbone.w"].add_(.02)
    result = C.historical_numeric_pair(left, right, {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}, {"backbone.w"})
    assert not result["passed"] and part in result["failure_reasons"][0]


@pytest.mark.parametrize("field", ["exp_avg", "exp_avg_sq"])
def test_single_small_adam_tensor_cannot_hide_in_aggregate(field):
    left, right = _state(), _state()
    right["optimizer"]["state"][0][field].mul_(1.02)
    result = C.historical_numeric_pair(left, right, {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}, {"backbone.w"})
    assert not result["passed"]
    assert f"optimizer.state.0.{field}" in result["floating_state_errors"]


def test_queue_single_tensor_bound_and_exact_metadata():
    left, right = _state(), _state()
    right["contrastive_state"]["queue"]["entries"][0]["key"].mul_(1.02)
    result = C.historical_numeric_pair(left, right, {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}, {"backbone.w"})
    assert not result["passed"] and "contrastive_state.queue.entries.0.key" in result["floating_state_errors"]
    right["contrastive_state"]["queue"]["entries"][0]["id"] = "different"
    with pytest.raises(ValueError, match="metadata differs"):
        C.historical_numeric_pair(left, right, {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}, {"backbone.w"})


@pytest.mark.parametrize("mutate", ["buffer", "counter", "rng_cpu", "rng_cuda"])
def test_buffers_counters_and_each_rng_domain_are_exact(mutate):
    left, right = _state(), _state()
    if mutate == "buffer": right["student"]["b"] += 1
    elif mutate == "counter": right["optimizer"]["state"][0]["step"] += 1
    elif mutate == "rng_cpu": right["rank_rng"][0]["torch_rng"] += 1
    else: right["rank_rng"][0]["cuda_rng"] += 1
    with pytest.raises(ValueError, match="differs exactly"):
        C.historical_numeric_pair(left, right, {"backbone.w": torch.tensor([0.]), "b": torch.tensor([0])}, {"backbone.w"})


def _live_train(folder, cfg, manifest, data, phase, initial_sha, provenance):
    with C.trace_resume_updates(folder / "trace-0.jsonl", phase, fork_step=2, end_step=4):
        summary = R.train(cfg, manifest, [data], folder, log=lambda _: None,
            expected_contrastive_initial_sha256=initial_sha, contrastive_provenance=provenance)
    assert summary["steps"] == 4


def _fresh_reload(rank, folder, cfg, manifest, data, phase, initial_sha, provenance):
    torch.set_num_threads(1)
    _live_train(folder, cfg, manifest, data, phase, initial_sha, provenance)


@pytest.fixture(scope="module")
def common_paths(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("targets_common_fork")
    data, manifest, pose = _pose_setup(tmp)
    base = _config(pose, "CR")
    cfg = replace(base, max_steps=4, model=replace(base.model, muscle_dropout=.4), log_batch_origin=True,
                  schedule=replace(base.schedule, total_steps=20, keep_checkpoints=(2,)))
    calibration = replace(cfg, contrastive={**cfg.contrastive, "calibration_only": True})
    provenance = {"source_sha256": "test", "input_sha256": {}}
    R.train(calibration, manifest, [data], tmp / "calibration", log=lambda _: None,
            contrastive_provenance=provenance)
    initial = C.checkpoint(tmp / "calibration/checkpoint.pt")["student"]
    initial_sha = state_sha256(initial)
    paths = {phase: tmp / phase for phase in C.PHASES}
    _live_train(paths["continuous"], cfg, manifest, data, "continuous", initial_sha, provenance)
    fork = paths["continuous"] / "checkpoint_step0000002.pt"
    fork_sha = C.file_sha(fork)
    for phase in C.PHASES[1:]:
        C.prepare_reload(fork, fork_sha, paths[phase], cfg, provenance, initial_sha,
                         phase=phase, fork_step=2, end_step=4, ranks=1)
        torch.multiprocessing.spawn(_fresh_reload,
            args=(paths[phase], cfg, manifest, data, phase, initial_sha, provenance), nprocs=1)
    return paths, cfg, initial_sha, initial, fork_sha, provenance


def test_live_continuous_saves_once_then_two_fresh_processes(common_paths):
    paths, cfg, initial_sha, initial, fork_sha, provenance = common_paths
    result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    assert result["passed"] and result["status"] == "PASSED"
    assert len(result["pairs"]) == 3 and all(p["passed"] for p in result["pairs"].values())
    assert not result["boundary_mismatches"] and not result["missing_active_support"]
    for pair in result["pairs"].values():
        assert pair["relative_to_left_displacement_from_initial"] == {"student": 0., "teacher": 0.}
        assert not pair["input_rng_counter_order_differences"]
    for phase in C.PHASES[1:]:
        assert C.read(paths[phase] / "start.json")["copied_checkpoint_sha256"] == fork_sha
        assert len((paths[phase] / "metrics.jsonl").read_text().splitlines()) == 2
        assert C.file_sha(paths[phase] / "origin_sessions.json") == C.file_sha(paths["continuous"] / "origin_sessions.json")


def test_wrong_fork_pin_or_config_or_duplicate_path_stops(common_paths, tmp_path):
    paths, cfg, initial_sha, initial, fork_sha, provenance = common_paths
    fork = paths["continuous"] / "checkpoint_step0000002.pt"
    with pytest.raises(ValueError, match="SHA256"):
        C.validate_fork(fork, "wrong", cfg, provenance, initial_sha, fork_step=2, end_step=4, ranks=1)
    with pytest.raises(ValueError, match="state/config/provenance"):
        C.validate_fork(fork, fork_sha, replace(cfg, lr=cfg.lr*2), provenance, initial_sha, fork_step=2, end_step=4, ranks=1)
    with pytest.raises(FileExistsError):
        C.prepare_reload(fork, fork_sha, paths["reload_a"], cfg, provenance, initial_sha,
                         phase="reload_a", fork_step=2, end_step=4, ranks=1)
    with pytest.raises(ValueError, match="distinct"):
        C.compare_three_paths({**paths, "reload_b": paths["reload_a"]}, cfg, initial_sha, initial, fork_sha,
                              ranks=1, fork_step=2, end_step=4)


@pytest.mark.parametrize("field", ["rng_cpu_before_sha256", "rng_cuda_after_sha256", "batch_sha256", "queue_order_before_sha256"])
def test_single_rank_single_step_trace_difference_blocks(common_paths, field):
    paths, cfg, initial_sha, initial, fork_sha, _ = common_paths
    path = paths["reload_b"] / "trace-0.jsonl"
    original = path.read_bytes()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1][field] = "changed"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    try:
        result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
        assert not result["passed"] and result["status"] == "FAILED_REQUIRES_REVIEW"
        assert result["pairs"]["reload_a__reload_b"]["input_rng_counter_order_differences"]
    finally:
        path.write_bytes(original)


def test_missing_support_is_inconclusive_not_success(common_paths):
    paths, cfg, initial_sha, initial, fork_sha, _ = common_paths
    originals = {}
    try:
        for phase, folder in paths.items():
            for name in ("trace-0.jsonl", "metrics.jsonl"):
                path = folder / name
                originals[path] = path.read_bytes()
                records = [json.loads(line) for line in path.read_text().splitlines()]
                for row in records:
                    if row.get("update", row.get("step")) > 2:
                        row["contrast_rnc_pairs"] = 0.
                path.write_text("\n".join(json.dumps(row) for row in records)+"\n")
        result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
        assert not result["passed"] and result["status"] == "INCONCLUSIVE"
    finally:
        for path, data in originals.items(): path.write_bytes(data)


def test_missing_or_duplicate_update_is_rejected(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"update":545}\n{"update":545}\n')
    with pytest.raises(ValueError, match="incomplete, duplicated or unordered"):
        C.rows(path, 544, 546)
