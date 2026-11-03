"""Run one frozen tokenizer training phase and attest its immutable checkpoint.

This is an orchestration wrapper around train_jepa.py, not another trainer.
Only own-run 5000 -> 19618 continuation is accepted; incomplete phases fail.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tokenizer_train_preflight", REPO / "scripts/tokenizer_preflight.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
PROTOCOL = "tokenizer-training-20261103-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_rng(records):
    require(isinstance(records, list) and len(records) == 4, "four rank RNG states required")
    for record in records:
        require(isinstance(record, dict), "invalid rank RNG record")
        for key in ("torch_rng", "cuda_rng"):
            value = record.get(key)
            require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1,
                    "missing or invalid " + key)
        try:
            torch.Generator(device="cpu").set_state(record["torch_rng"].cpu())
            # CUDA Philox states contain seed and offset (legacy/new 16/24 bytes).
            # On the real GPU allocation also ask PyTorch to validate its own format.
            require(record["cuda_rng"].numel() in (16, 24), "invalid CUDA RNG size")
            if torch.cuda.is_available():
                torch.Generator(device="cuda").set_state(record["cuda_rng"].cpu())
        except (RuntimeError, TypeError) as exc:
            raise ValueError("unusable RNG state") from exc


def validate_optimizer(value, cfg, step):
    require(isinstance(value, dict) and isinstance(value.get("state"), dict) and value["state"],
            "missing optimizer state")
    groups = value.get("param_groups")
    require(isinstance(groups, list) and len(groups) == 1, "expected one AdamW parameter group")
    group = groups[0]
    require(isinstance(group, dict) and isinstance(group.get("params"), list) and group["params"],
            "invalid AdamW parameter group")
    ids = group["params"]
    require(all(type(i) is int and i >= 0 for i in ids) and len(ids) == len(set(ids)) and
            set(value["state"]).issubset(ids), "invalid AdamW parameter identities")
    require(tuple(group.get("betas", ())) == (.9, .999) and group.get("eps") == 1e-8 and
            group.get("weight_decay") == cfg.weight_decay and group.get("amsgrad") is False and
            group.get("maximize") is False, "AdamW recipe mismatch")
    lr = group.get("lr")
    require(type(lr) in (float, int) and math.isfinite(lr) and 0 <= lr <= cfg.lr,
            "invalid optimizer learning rate")
    for moments in value["state"].values():
        require(isinstance(moments, dict) and {"step", "exp_avg", "exp_avg_sq"} <= set(moments),
                "incomplete AdamW moments")
        a, b, n = (moments[k] for k in ("exp_avg", "exp_avg_sq", "step"))
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape,
                "invalid AdamW moment shapes")
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and bool((b >= 0).all()),
                "nonfinite or negative AdamW moments")
        require(isinstance(n, torch.Tensor) and n.numel() == 1 and bool(torch.isfinite(n).all()) and
                0 < float(n) <= step and float(n).is_integer(), "invalid AdamW step")


def checkpoint_identity(path, cfg, step, data_signature):
    """Verify resumable state and configuration, without constructing another FM."""
    before = P.file_identity(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state.get("step") == step, "wrong checkpoint step")
    require(state.get("config") == R.config_to_dict(cfg), "checkpoint configuration mismatch")
    require(state.get("world_size") == 4, "checkpoint world size mismatch")
    require(state.get("stream") == {"num_workers": 6, "segment": cfg.stream_segment, "world_size": 4}, "checkpoint stream mismatch")
    validate_rng(state.get("rank_rng"))
    validate_optimizer(state.get("optimizer"), cfg, step)
    require(bool(data_signature) and state.get("data_signature") == data_signature, "data signature mismatch")
    counts = state.get("alarm_counts")
    require(isinstance(counts, (tuple, list)) and len(counts) == 2 and
            all(type(v) is int and v >= 0 for v in counts), "invalid alarm counters")
    require(state.get("pose_stats_sha256") == "757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e", "pose statistics mismatch")
    for name in ("student", "teacher"):
        values = state.get(name)
        require(isinstance(values, dict) and values, "missing model state: " + name)
        require(all(bool(torch.isfinite(v).all()) for v in values.values()), "nonfinite model state: " + name)
    require(P.file_identity(path) == before, "checkpoint changed during verification")
    return {**before, "trained_state_sha256": state_sha256({k: state[k] for k in ("student", "teacher", "optimizer")})}


def command_for(comparison, arm, step, seconds):
    require(arm in P.ARMS and step in (5000, 19618), "unregistered arm or dose")
    require(seconds > 300, "insufficient training walltime")
    argv = list(comparison["arm_configurations"][arm]["validated_trainer_argv"])
    require(argv.count("--max-steps") == 1 and "--time-limit-s" not in argv, "unexpected phase arguments")
    argv[argv.index("--max-steps") + 1] = str(step)
    argv += ["--time-limit-s", str(seconds - 300)]
    return ["torchrun", "--standalone", "--nproc_per_node=4", *argv]


def receipt_for(arm, step, checkpoint, resume_checkpoint, config_file, code_sha, plan_sha, command):
    return {"protocol": PROTOCOL, "complete": True, "arm": arm, "checkpoint_step": step,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": R.sha256_file(checkpoint),
            "config_path": str(config_file.resolve()), "config_file_sha256": R.sha256_file(config_file),
            "resume_checkpoint": str(resume_checkpoint.resolve()), "resume_checkpoint_sha256": R.sha256_file(resume_checkpoint),
            "code_sha": code_sha, "plan_sha256": plan_sha, "command": command,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def execute(plan_path, arm, step, expected_code, out):
    code = P.code_identity(expected_code)
    plan_state = P.file_identity(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    require(plan.get("protocol") == PROTOCOL and plan.get("endpoints") == [5000, 19618], "wrong training plan")
    source = P.pinned_file(plan["preflight_plan"])
    _, configs, inputs = P.load_plan(source["path"])
    comparison = json.loads(Path(inputs["comparison"]["path"]).read_text())
    cfg = replace(configs[arm], max_steps=step)
    scales = json.loads(Path(inputs["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(inputs["windows"]["path"]).read_text())["window_s"]
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs["manifest"]["path"], scales, windows)))
    command = command_for(comparison, arm, step, plan["allocations"][arm][str(step)]["walltime_seconds"])
    run_dir = Path(command[command.index("--out-dir") + 1])
    require(run_dir.is_absolute() and out.is_absolute(), "absolute output paths required")
    roots = (REPO, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"), Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))
    require(not any(run_dir.resolve().is_relative_to(root) or out.resolve().is_relative_to(root) for root in roots), "output inside code")
    require(not out.exists() and not out.is_symlink(), "receipt directory already exists")
    require(str((out / "receipt.json").resolve()) == plan["phase_receipts"][arm][str(step)], "receipt path differs from plan")
    old_checkpoint = None
    if step == 5000:
        require(not run_dir.exists() and not run_dir.is_symlink(), "fresh training requires new run directory")
    else:
        prior = plan["phase_receipts"][arm]["5000"]
        rec = json.loads(Path(prior).read_text())
        named = run_dir / "checkpoint_step0005000.pt"
        require(rec.get("protocol") == PROTOCOL and rec.get("complete") is True and
                rec.get("arm") == arm and rec.get("checkpoint_step") == 5000,
                "missing completed same-arm screening receipt")
        require(rec.get("checkpoint") == str(named) and rec.get("code_sha") == code and
                rec.get("plan_sha256") == plan_state["sha256"], "prior phase provenance mismatch")
        require(not (run_dir / "STOP").exists(), "STOP present")
        old_checkpoint = checkpoint_identity(named, configs[arm], 5000, signature)
        resume_identity = checkpoint_identity(run_dir / "checkpoint.pt", configs[arm], 5000, signature)
        require(old_checkpoint["sha256"] == rec["checkpoint_sha256"] and
                rec.get("resume_checkpoint") == str(run_dir / "checkpoint.pt") and
                resume_identity["sha256"] == rec.get("resume_checkpoint_sha256") and
                old_checkpoint["trained_state_sha256"] == resume_identity["trained_state_sha256"],
                "continuation is not from own verified 5000 state")
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / "launch.json", {"arm": arm, "step": step, "plan_sha256": plan_state["sha256"], "code_sha": code,
                                     "command": command, "previous_checkpoint": old_checkpoint})
    subprocess.run(command, cwd=REPO, check=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    require(summary.get("stopped") == "max_steps" and summary.get("steps") == step, "training phase incomplete")
    require(not (run_dir / "STOP").exists(), "STOP present after phase")
    require(json.loads((run_dir / "config.json").read_text()) == R.config_to_dict(cfg), "final config mismatch")
    named = run_dir / f"checkpoint_step{step:07d}.pt"
    identity = checkpoint_identity(named, cfg, step, signature)
    final_identity = checkpoint_identity(run_dir / "checkpoint.pt", cfg, step, signature)
    require(identity["trained_state_sha256"] == final_identity["trained_state_sha256"], "named checkpoint differs from final trained state")
    if old_checkpoint is not None:
        old_now = P.file_identity(old_checkpoint["path"])
        require(all(old_now[k] == v for k, v in old_checkpoint.items()
                    if k != "trained_state_sha256"), "5000 checkpoint was modified by continuation")
    P.check_unchanged(inputs)
    require(P.file_identity(plan_path) == plan_state and P.code_identity(code) == code, "plan or code changed")
    config_file = out / "config.json"
    P.write_json(config_file, R.config_to_dict(cfg))
    P.write_json(out / "summary.json", summary)
    P.write_json(out / "receipt.json", receipt_for(arm, step, named, run_dir / "checkpoint.pt", config_file, code, plan_state["sha256"], command))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=P.ARMS, required=True)
    parser.add_argument("--step", type=int, choices=(5000, 19618), required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.plan, args.arm, args.step, args.expected_code_sha, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
