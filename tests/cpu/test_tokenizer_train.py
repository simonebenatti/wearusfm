"""Training orchestration guards; no second trainer implementation."""
from dataclasses import replace
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tokenizer_training_test", ROOT / "scripts/tokenizer_train.py")
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
SIGNATURE = {"test": "data"}


def test_registered_phase_command_changes_only_dose_and_operational_timeout():
    comparison = json.loads((ROOT / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_text())
    for arm in T.P.ARMS:
        original = comparison["arm_configurations"][arm]["validated_trainer_argv"]
        for step in (5000, 19618):
            cmd = T.command_for(comparison, arm, step, 6000)
            assert cmd[:3] == ["torchrun", "--standalone", "--nproc_per_node=4"]
            assert cmd[-2:] == ["--time-limit-s", "5700"]
            recovered = cmd[3:-2]
            assert recovered[recovered.index("--max-steps") + 1] == str(step)
            recovered[recovered.index("--max-steps") + 1] = "5000"
            assert recovered == original
    for arm, step, seconds in (("B", 5000, 6000), ("T25-L", 49046, 6000), ("T25-L", 5000, 300)):
        with pytest.raises(ValueError):
            T.command_for(comparison, arm, step, seconds)


def state(cfg):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=cfg.lr, weight_decay=cfg.weight_decay)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["step"].fill_(5000)
    return {"step": 5000, "config": R.config_to_dict(cfg), "world_size": 4,
            "stream": {"num_workers": 6, "segment": cfg.stream_segment, "world_size": 4},
            "rank_rng": [{"torch_rng": torch.Generator().manual_seed(i).get_state(),
                          "cuda_rng": torch.zeros(16, dtype=torch.uint8)} for i in range(4)],
            "pose_stats_sha256": "757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e",
            "optimizer": optimizer.state_dict(), "data_signature": SIGNATURE, "alarm_counts": (0, 0),
            "student": {"weight": torch.ones(3)}, "teacher": {"weight": torch.zeros(3)}}


def test_checkpoint_identity_accepts_same_training_state_under_distinct_serialization(tmp_path):
    cfg = replace(R.small_config(), max_steps=5000)
    first, last = tmp_path / "named.pt", tmp_path / "checkpoint.pt"
    a = state(cfg)
    torch.save(a, first)
    a["rank_rng"][0]["torch_rng"] = torch.Generator().manual_seed(77).get_state()
    torch.save(a, last)
    x, y = T.checkpoint_identity(first, cfg, 5000, SIGNATURE), T.checkpoint_identity(last, cfg, 5000, SIGNATURE)
    assert x["sha256"] != y["sha256"]
    assert x["trained_state_sha256"] == y["trained_state_sha256"]
    a["teacher"]["weight"][0] = .5
    torch.save(a, last)
    assert x["trained_state_sha256"] != T.checkpoint_identity(last, cfg, 5000, SIGNATURE)["trained_state_sha256"]


@pytest.mark.parametrize("mutation", ["dose", "configuration", "world", "stream", "rng", "pose", "nonfinite",
                                     "cpu_rng", "cuda_missing", "cuda_dtype", "cuda_size", "empty_rank",
                                     "optimizer_nan", "optimizer_empty", "data_missing", "data_wrong", "alarm_missing", "alarm_negative"])
def test_checkpoint_rejects_wrong_resume_identity(tmp_path, mutation):
    cfg = replace(R.small_config(), max_steps=5000)
    s = state(cfg)
    if mutation == "dose": s["step"] = 4999
    if mutation == "configuration": s["config"]["jepa"]["jepa_weight"] = 9
    if mutation == "world": s["world_size"] = 1
    if mutation == "stream": s["stream"]["num_workers"] = 0
    if mutation == "rng": s["rank_rng"] = []
    if mutation == "pose": s["pose_stats_sha256"] = "0" * 64
    if mutation == "nonfinite": s["teacher"]["weight"][0] = float("nan")
    if mutation == "cpu_rng": s["rank_rng"][0]["torch_rng"] = torch.arange(4)
    if mutation == "cuda_missing": s["rank_rng"][0]["cuda_rng"] = None
    if mutation == "cuda_dtype": s["rank_rng"][0]["cuda_rng"] = torch.zeros(16)
    if mutation == "cuda_size": s["rank_rng"][0]["cuda_rng"] = torch.zeros(5, dtype=torch.uint8)
    if mutation == "empty_rank": s["rank_rng"][0] = {}
    if mutation == "optimizer_nan": s["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    if mutation == "optimizer_empty": s["optimizer"] = {"state": {}, "param_groups": []}
    if mutation == "data_missing": del s["data_signature"]
    if mutation == "data_wrong": s["data_signature"] = {"test": "different"}
    if mutation == "alarm_missing": del s["alarm_counts"]
    if mutation == "alarm_negative": s["alarm_counts"] = (0, -1)
    path = tmp_path / "checkpoint.pt"
    torch.save(s, path)
    with pytest.raises(ValueError):
        T.checkpoint_identity(path, cfg, 5000, SIGNATURE)


def test_receipt_pins_immutable_config_and_separate_resume_checkpoint(tmp_path):
    checkpoint, resume, config = [tmp_path / name for name in ("named.pt", "checkpoint.pt", "config.json")]
    checkpoint.write_bytes(b"named")
    resume.write_bytes(b"resume")
    config.write_text("{}")
    receipt = T.receipt_for("T100-G", 5000, checkpoint, resume, config, "a" * 40, "b" * 64, ["torchrun"])
    assert receipt["complete"] and receipt["checkpoint_step"] == 5000
    assert receipt["checkpoint_sha256"] == R.sha256_file(checkpoint)
    assert receipt["resume_checkpoint_sha256"] == R.sha256_file(resume)
    config.write_text('{"changed":true}')
    assert receipt["config_file_sha256"] != R.sha256_file(config)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    """Run the real controller; replace only external training/checkpoint payloads."""
    cfg = replace(R.small_config(), max_steps=5000)
    arm, code = "T100-G", "a"*40
    run_dir = tmp_path/"run"
    outputs = {str(step): tmp_path/("phase_"+str(step)) for step in (5000, 19618)}
    comparison = {"arm_configurations": {arm: {"validated_trainer_argv": [
        "scripts/train_jepa.py", "--out-dir", str(run_dir), "--max-steps", "5000", "--distributed"]}}}
    comparison_path = tmp_path/"comparison.json"
    comparison_path.write_text(json.dumps(comparison))
    source = tmp_path/"preflight.json"
    source.write_text('{}')
    manifest, scales, windows = (tmp_path/name for name in ("manifest.json", "scales.json", "windows.json"))
    manifest.write_text('{}')
    scales.write_text('{"scales": {}}')
    windows.write_text('{"window_s": {}}')
    inputs = {key: T.P.file_identity(path) for key, path in
              (("comparison", comparison_path), ("plan", source), ("manifest", manifest), ("scales", scales), ("windows", windows))}
    plan = {"protocol": T.PROTOCOL, "endpoints": [5000, 19618],
            "preflight_plan": {"path": str(source), "sha256": R.sha256_file(source)},
            "allocations": {arm: {str(s): {"walltime_seconds": 5400 if s == 5000 else 14400} for s in (5000, 19618)}},
            "phase_receipts": {arm: {s: str(p/"receipt.json") for s, p in outputs.items()}}}
    plan_path = tmp_path/"training_plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(T.P, "code_identity", lambda expected: code if expected == code else pytest.fail("wrong execution SHA"))
    monkeypatch.setattr(T.P, "load_plan", lambda path: ({}, {arm: cfg}, copy.deepcopy(inputs)))
    monkeypatch.setattr(T.R, "calibration_signature", lambda *args: SIGNATURE)
    checks, commands = [], []
    def checkpoint(path, expected_cfg, step, data_signature):
        payload = json.loads(Path(path).read_text())
        assert data_signature == SIGNATURE
        T.require(payload["step"] == step and payload["config"] == R.config_to_dict(expected_cfg), "mock checkpoint identity mismatch")
        checks.append((Path(path).name, step))
        return {**T.P.file_identity(path), "trained_state_sha256": payload["trained"]}
    monkeypatch.setattr(T, "checkpoint_identity", checkpoint)
    behavior = {"failure": None}
    def train(command, *, cwd, check):
        assert cwd == T.REPO and check is True
        step = int(command[command.index("--max-steps")+1])
        assert command == T.command_for(comparison, arm, step, plan["allocations"][arm][str(step)]["walltime_seconds"])
        commands.append(list(command))
        if behavior["failure"] == "subprocess":
            raise subprocess.CalledProcessError(1, command)
        run_dir.mkdir(exist_ok=True)
        effective = R.config_to_dict(replace(cfg, max_steps=step))
        (run_dir/"config.json").write_text(json.dumps(effective))
        (run_dir/"summary.json").write_text(json.dumps({"stopped": "time_limit" if behavior["failure"] == "incomplete" else "max_steps", "steps": step}))
        for name in (f"checkpoint_step{step:07d}.pt", "checkpoint.pt"):
            (run_dir/name).write_text(json.dumps({"step": step, "config": effective, "trained": str(step), "serialization": name}))
        if behavior["failure"] == "source":
            manifest.write_text('{"changed": true}')
        if behavior["failure"] == "stop_after":
            (run_dir/"STOP").write_text("numerical failure")
        if behavior["failure"] == "old_checkpoint":
            (run_dir/"checkpoint_step0005000.pt").write_text("changed by continuation")
    monkeypatch.setattr(T.subprocess, "run", train)
    return {"cfg": cfg, "arm": arm, "code": code, "run": run_dir, "out": outputs, "plan": plan_path,
            "commands": commands, "checks": checks, "behavior": behavior}


def execute(fixture, step):
    T.execute(fixture["plan"], fixture["arm"], step, fixture["code"], fixture["out"][str(step)])


def test_execute_fresh_then_own_continuation_writes_immutable_receipts(execution):
    x = execution
    execute(x, 5000)
    first_path = x["out"]["5000"]/"receipt.json"
    first = json.loads(first_path.read_text())
    old = T.P.file_identity(x["run"]/"checkpoint_step0005000.pt")
    config_bytes = (x["out"]["5000"]/"config.json").read_bytes()
    assert first["complete"] is True and first["protocol"] == T.PROTOCOL
    assert first["checkpoint_sha256"] != first["resume_checkpoint_sha256"]
    execute(x, 19618)
    final = json.loads((x["out"]["19618"]/"receipt.json").read_text())
    assert final["complete"] is True and final["checkpoint_step"] == 19618
    assert final["plan_sha256"] == first["plan_sha256"]
    assert T.P.file_identity(x["run"]/"checkpoint_step0005000.pt") == old
    assert (x["out"]["5000"]/"config.json").read_bytes() == config_bytes
    assert json.loads(first_path.read_text()) == first
    assert len(x["commands"]) == 2 and len(x["checks"]) == 6


@pytest.mark.parametrize("failure", ["incomplete", "subprocess", "source", "stop_after"])
def test_execute_failed_fresh_phase_never_publishes_complete_receipt(execution, failure):
    x = execution
    x["behavior"]["failure"] = failure
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        execute(x, 5000)
    assert len(x["commands"]) == 1
    assert (x["out"]["5000"]/"launch.json").exists()
    assert not (x["out"]["5000"]/"receipt.json").exists()


@pytest.mark.parametrize("failure", ["stop", "wrong_arm", "wrong_protocol", "truthy_complete", "checkpoint_sha", "plan_sha"])
def test_execute_continuation_rejects_prior_provenance_before_launch(execution, failure):
    x = execution
    execute(x, 5000)
    receipt_path = x["out"]["5000"]/"receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if failure == "stop": (x["run"]/"STOP").write_text("stop")
    if failure == "wrong_arm": receipt["arm"] = "T25-L"
    if failure == "wrong_protocol": receipt["protocol"] = "other-protocol"
    if failure == "truthy_complete": receipt["complete"] = "false"
    if failure == "checkpoint_sha": receipt["checkpoint_sha256"] = "0"*64
    if failure == "plan_sha": receipt["plan_sha256"] = "0"*64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        execute(x, 19618)
    assert len(x["commands"]) == 1
    assert not x["out"]["19618"].exists()


def test_execute_continuation_preserves_named_screening_checkpoint(execution):
    x = execution
    execute(x, 5000)
    x["behavior"]["failure"] = "old_checkpoint"
    with pytest.raises(ValueError, match="5000 checkpoint was modified"):
        execute(x, 19618)
    assert len(x["commands"]) == 2
    assert not (x["out"]["19618"]/"receipt.json").exists()
