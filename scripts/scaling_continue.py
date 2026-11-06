"""Guarded exact continuation of the accepted T100-G-300M run from 5000 to 19618.

The completed 5000 run directory is read-only evidence: its files are pinned in the
plan and hashed before and after training. The new tranche is written in a NEW run
directory, seeded with a byte-identical copy of the verified resume checkpoint and
of the three identity files that the trainer checks at resume. The trainer entry
point (train_jepa.py) and every scientific source stay byte-identical to the frozen
source contract; the recipe differs only in --out-dir and --max-steps, plus the
operational time limit. Student, teacher, AdamW moments, per-rank RNG, data stream
position and the original 49046/1962 schedule are restored by the unchanged trainer.
This is orchestration: not a new trainer and not a submitter.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


M = _module("scaling_continue_train", "scaling_train.py")
T, P, CLI, require = M.T, M.P, M.CLI, M.require
torch = T.torch
PROTOCOL = "scaling-continuation-training-20261106-v1"
RANK_PROTOCOL = "scaling-continuation-rank-20261106-v1"
FREEZE_PROTOCOL = "scaling-continuation19618-budget-freeze-20261105-v2"
ARM = M.ARM
START, END = 5000, 19618
PARAMETERS = 292054847
MARKER = "CONTINUATION.json"
EXECUTION_SOURCES = {"scripts/scaling_continue.py", "scripts/scaling_train.py", "scripts/scaling_preflight.py",
                     "scripts/slurm/scaling_continue.sbatch"}
IDENTITY_FILES = ("fixed_eval_set.json", "pose_panel_set.json", "origin_sessions.json")
SEED_FILES = ("checkpoint.pt",) + IDENTITY_FILES
PREVIOUS_FILES = {"checkpoint.pt", "checkpoint_step0005000.pt", "config.json", "fixed_eval.jsonl", "metrics.jsonl",
                  "pose_panel.jsonl", "pose_provenance.json", "summary.json", *IDENTITY_FILES}
NEW_FILES = {"checkpoint.pt", f"checkpoint_step{END:07d}.pt", "config.json", "fixed_eval.jsonl", "metrics.jsonl",
             "pose_panel.jsonl", "pose_provenance.json", "summary.json", MARKER, *IDENTITY_FILES}
ALLOCATION = {"account": "iscrb_wearusfm", "partition": "boost_usr_prod", "qos": "normal", "nodes": 1, "gpus": 4,
              "cpus": 32, "memory": "480G", "walltime_seconds": 53100}
# Operational only: time left to this wrapper for the checks before and after the trainer.
VERIFICATION_MARGIN_SECONDS = 1800
MEMORY = {"allocated_fraction_limit": .9, "reference_peak_allocated_gib": 37.7518310546875}
SCRATCH_PREFIX = "/leonardo_scratch/"
SCRATCH_RETENTION_DAYS = 40
CONTROL_ARM = "T100-G"
CONTROL_DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6", "epn612", "uci_emg")
CODE_ROOTS = (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"), Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def derive_recipe(context, run_dir):
    """The validated 5000 CLI with only output directory and dose changed, plus the operational time limit."""
    run_dir = Path(run_dir)
    require(run_dir.is_absolute(), "absolute run directory required")
    argv = list(context["comparison"]["arm_configurations"]["T100-G"]["validated_trainer_argv"])
    for flag, value in (("--preset", "300m"), ("--out-dir", str(run_dir)), ("--max-steps", str(END))):
        require(argv.count(flag) == 1, "duplicate/missing scientific argument: " + flag)
        argv[argv.index(flag) + 1] = value
    require(not any(flag in argv for flag in ("--time-limit-s", "--resume", "--init-from", "--branch-from")),
            "unexpected timeout, warm start or branch override")
    cfg = CLI.build_config(CLI.build_parser().parse_args(argv[1:]))
    previous = context["config_300m"]
    old, new = R.config_to_dict(previous), R.config_to_dict(cfg)
    require(new == R.config_to_dict(replace(previous, max_steps=END)) and
            [key for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)] == ["max_steps"],
            "continuation recipe differs from the 5000 recipe beyond the dose")
    schedule = cfg.schedule
    require(previous.max_steps == START and schedule.kind == "wsd" and schedule.total_steps == 49046 and
            cfg.warmup_steps == 1962 and cfg.warmup_steps <= START < END <= schedule.decay_start and
            END in schedule.keep_checkpoints, "continuation leaves the stable phase of the original schedule")
    require(cfg.batch_size == 32 and cfg.accum_steps == 1 and cfg.seed == 0, "wrong batch, accumulation or seed")
    argv += ["--time-limit-s", str(ALLOCATION["walltime_seconds"] - VERIFICATION_MARGIN_SECONDS)]
    return previous, cfg, argv


def control_pins(pose_inventory, probe_inventory):
    """Pins of the existing T100-G 19618 caches, exactly as listed by the reviewed evaluation inventories."""
    require(pose_inventory.get("step") == END and probe_inventory.get("step") == END, "control inventories are not at 19618")
    pins = {}
    for split in ("train", "val"):
        entry = pose_inventory["caches"][CONTROL_ARM][split]
        pins["pose_" + split] = {"path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"]}
        pins["pose_" + split + "_provenance"] = {"path": entry["provenance_path"], "sha256": entry["provenance_sha256"]}
        pins["pose_receipt"] = {"path": entry["receipt_path"], "sha256": entry["receipt_sha256"]}
        reference = pose_inventory["references"][split]
        pins["pose_reference_" + split] = {key: reference[key] for key in ("path", "sha256", "bytes")}
    require(set(probe_inventory["caches"][CONTROL_ARM]) == set(CONTROL_DATASETS), "control classification caches differ")
    for dataset in CONTROL_DATASETS:
        entry = probe_inventory["caches"][CONTROL_ARM][dataset]
        task = Path(entry["path"]).parent.name
        pins[task + "_" + dataset] = {"path": entry["path"], "sha256": entry["sha256"]}
        pins[task + "_" + dataset + "_provenance"] = {"path": entry["provenance_path"], "sha256": entry["provenance_sha256"]}
        pins[task + "_receipt"] = {"path": entry["receipt_path"], "sha256": entry["receipt_sha256"]}
    return pins


def validate_plan(plan, context):
    """Schema, recipe and frozen-budget checks. Remote artifacts are verified at run time."""
    require(plan.get("protocol") == PROTOCOL and plan.get("arm") == ARM and plan.get("start_step") == START and
            plan.get("end_step") == END and plan.get("additional_steps") == END - START,
            "only the registered 300M continuation 5000->19618 is supported")
    require(plan.get("scientific_source_sha256") == context["contract"]["scientific_source_sha256"],
            "plan scientific identity differs")
    sources = plan.get("execution_source_sha256")
    require(isinstance(sources, dict) and set(sources) == EXECUTION_SOURCES and all(M.digest(v) for v in sources.values()),
            "missing exact execution source pins")
    M.pin_schema(plan.get("source_contract"))
    if context["identity"]:
        require(plan["source_contract"]["sha256"] == context["identity"]["sha256"], "source contract pin differs")
    require(plan.get("allocation") == ALLOCATION and plan.get("memory") == MEMORY, "wrong continuation resources")
    M.pin_schema(plan.get("budget_freeze"))
    previous = plan.get("previous")
    require(isinstance(previous, dict) and set(previous) == {"run_dir", "files", "receipt", "receipt_copy"} and
            isinstance(previous["files"], dict) and set(previous["files"]) == PREVIOUS_FILES,
            "missing complete inventory of the 5000 run")
    for pin in previous["files"].values():
        require(isinstance(pin, dict) and set(pin) == {"sha256", "bytes"} and M.digest(pin["sha256"]) and
                type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid 5000 file pin")
    for name in ("receipt", "receipt_copy"):
        M.pin_schema(previous[name])
    require(previous["receipt"]["sha256"] == previous["receipt_copy"]["sha256"] and
            Path(previous["receipt"]["path"]).is_absolute() and not Path(previous["receipt_copy"]["path"]).is_absolute(),
            "remote 5000 receipt and its repository copy must be the same file")
    control = plan.get("control")
    require(isinstance(control, dict) and set(control) == {"checkpoint", "receipt", "pose_inventory",
            "classification_inventory", "artifacts", "required_until_utc"}, "missing verified 30M control at 19618")
    for name in ("checkpoint", "receipt", "pose_inventory", "classification_inventory"):
        M.pin_schema(control[name])
    require(isinstance(control["artifacts"], dict) and control["artifacts"], "missing control cache pins")
    for pin in control["artifacts"].values():
        M.pin_schema(pin)
        require(Path(pin["path"]).is_absolute(), "control caches require absolute paths")
    M.parse_utc(control["required_until_utc"])
    paths = [Path(plan.get(name, "")) for name in ("run_dir", "receipt_dir")]
    old = Path(previous["run_dir"])
    require(all(path.is_absolute() for path in (*paths, old)) and paths[0] != paths[1] and
            not any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a in paths for b in (old,)),
            "new absolute run and receipt directories, distinct from the 5000 run, are required")
    return derive_recipe(context, plan["run_dir"])


def check_budget_freeze(plan, states):
    """The plan must request exactly the single segment and allocation of the frozen v2 budget."""
    states["budget_freeze"] = M.resolve_pin(plan["budget_freeze"])
    freeze = json.loads(Path(states["budget_freeze"]["path"]).read_text())
    training = freeze.get("training", {})
    allocation = dict(training.get("allocation", {}))
    require(freeze.get("protocol") == FREEZE_PROTOCOL and freeze.get("start_step") == START and freeze.get("end_step") == END and
            allocation.pop("walltime", None) == "14:45:00" and allocation.pop("qos_must_be_explicit", None) is True and
            allocation.pop("account", "").lower() == ALLOCATION["account"] and
            allocation == {k: v for k, v in ALLOCATION.items() if k != "account"} and
            training.get("segments") == [{"start_step": START, "end_step": END, "walltime_seconds": ALLOCATION["walltime_seconds"]}] and
            training.get("gpu_hours_cap") == 59.0 and training.get("new_30m_training_included") is False,
            "plan resources differ from the frozen continuation budget")
    require(4 * ALLOCATION["walltime_seconds"] / 3600 <= training["gpu_hours_cap"], "requested GPU hours above the cap")
    frozen = freeze.get("control19618", {})
    control = plan["control"]
    require(frozen.get("historical_arm_name") == CONTROL_ARM and frozen.get("checkpoint_sha256") == control["checkpoint"]["sha256"] and
            frozen.get("pose_inventory", {}).get("sha256") == control["pose_inventory"]["sha256"] and
            frozen.get("classification_inventory", {}).get("sha256") == control["classification_inventory"]["sha256"],
            "control differs from the one named by the frozen budget")
    return freeze


def check_allocation_fields(fields, expected, visible_gpus):
    M.check_allocation_fields(fields, expected, visible_gpus)
    require(fields.get("Partition") == expected["partition"] and fields.get("QOS") == expected["qos"] and
            str(fields.get("Account", "")).lower() == expected["account"], "actual account, partition or QoS differ from plan")


def verify_allocation(plan):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and int(job) > 0, "continuation requires a real SLURM allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(part.split("=", 1) for part in raw.split() if "=" in part)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING", "wrong or nonrunning SLURM allocation")
    check_allocation_fields(fields, plan["allocation"], torch.cuda.device_count())
    return job


def previous_states(previous):
    """Hash every file of the completed 5000 run; the directory must hold exactly the pinned files."""
    directory = Path(previous["run_dir"])
    require(directory.is_dir() and not directory.is_symlink(), "missing 5000 run directory")
    require(sorted(p.name for p in directory.iterdir()) == sorted(previous["files"]),
            "5000 run directory content differs from the pinned inventory")
    states = {}
    for name, pin in sorted(previous["files"].items()):
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "5000 run entry is not a regular file: " + name)
        state = P.file_identity(path)
        require(state["sha256"] == pin["sha256"] and state["bytes"] == pin["bytes"], "5000 run file changed: " + name)
        states["previous:" + name] = state
    return states


def check_previous_receipt(receipt, previous, context, contract_sha):
    directory, files = Path(previous["run_dir"]), previous["files"]
    require(receipt.get("protocol") == M.PROTOCOL and receipt.get("complete") is True and receipt.get("arm") == ARM and
            receipt.get("checkpoint_step") == START and receipt.get("scientific_endpoint") is True,
            "previous receipt is not the completed scientific 5000 phase")
    require(receipt.get("checkpoint") == str(directory / "checkpoint_step0005000.pt") and
            receipt.get("checkpoint_sha256") == files["checkpoint_step0005000.pt"]["sha256"] and
            receipt.get("resume_checkpoint") == str(directory / "checkpoint.pt") and
            receipt.get("resume_checkpoint_sha256") == files["checkpoint.pt"]["sha256"],
            "previous receipt does not attest the pinned checkpoints")
    require(receipt.get("training_source_contract_sha256") == contract_sha and
            receipt.get("scientific_source_sha256") == context["contract"]["scientific_source_sha256"] and
            receipt.get("reference_code_sha") == M.BASE_CODE_SHA, "previous phase used different scientific sources")


def optimizer_mapping(state, cfg, expected_parameters=None):
    """AdamW moments against the trainable parameters of the model built from the same configuration, in optimizer order."""
    student = R.WearUsFM(cfg.model)
    total = sum(p.numel() for p in student.parameters())
    require(expected_parameters is None or total == expected_parameters, "model size differs from the registered arm")
    parameters = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
    saved = state.get("optimizer", {})
    groups = saved.get("param_groups")
    require(isinstance(groups, list) and len(groups) == 1 and groups[0].get("params") == list(range(len(parameters))),
            "AdamW parameter group does not enumerate the trainable parameters in order")
    moments = saved.get("state")
    require(isinstance(moments, dict) and moments and set(moments) <= set(range(len(parameters))), "AdamW moments without parameter")
    steps = {}
    for index, values in moments.items():
        name, parameter = parameters[index]
        require(tuple(values["exp_avg"].shape) == tuple(values["exp_avg_sq"].shape) == tuple(parameter.shape),
                "AdamW moment shape differs from its parameter: " + name)
        steps[name] = int(values["step"])
    reference = {"student": student.state_dict(), "teacher": R.make_teacher(student).state_dict()}
    for part, expected in reference.items():
        values = state.get(part)
        require(isinstance(values, dict) and list(values) == list(expected) and
                all(tuple(values[k].shape) == tuple(expected[k].shape) and values[k].dtype == expected[k].dtype for k in expected),
                "model state keys, order, shapes or types differ: " + part)
    require(max(steps.values()) == state.get("step"), "no AdamW moment reached the checkpoint step")
    return {"parameters": total, "trainable_tensors": len(parameters), "tensors_with_moments": len(steps),
            "tensors_without_moments": sorted(name for name, _ in parameters if name not in steps),
            "adam_step_min": min(steps.values()), "adam_step_max": max(steps.values()), "adam_steps": steps}


def check_optimizer_continuity(before, after, added=END - START):
    """The optimizer was resumed, not reset: every moment kept its history and advanced by at most the new updates."""
    require(before["trainable_tensors"] == after["trainable_tensors"] and before["parameters"] == after["parameters"] and
            set(before["adam_steps"]) <= set(after["adam_steps"]), "optimizer state lost parameters during continuation")
    for name, old in before["adam_steps"].items():
        require(old <= after["adam_steps"][name] <= old + added, "AdamW history was reset or overrun: " + name)
    require(after["adam_step_max"] == before["adam_step_max"] + added, "AdamW did not advance by exactly the new updates")


def check_tranche_logs(run_dir, previous_dir, cfg, start=START, end=END):
    """The new directory holds exactly the records of the new updates, on the original schedule and diagnostics."""
    steps = list(range(start + 1, end + 1))
    metrics = read_lines(Path(run_dir) / "metrics.jsonl")
    require([r.get("step") for r in metrics] == steps, "metrics do not cover exactly the continuation updates")
    require(all(r.get("lr") == R._lr_at(r["step"] - 1, cfg) for r in metrics), "learning rate differs from the original schedule")
    report = {"metrics_records": len(metrics), "first_step": steps[0], "last_step": steps[-1],
              "lr_min": min(r["lr"] for r in metrics), "lr_max": max(r["lr"] for r in metrics)}
    for name, every, key in (("fixed_eval.jsonl", cfg.fixed_eval_every, None), ("pose_panel.jsonl", cfg.pose_panel_every, "panel_sha256")):
        path = Path(run_dir) / name
        if not every:
            require(not path.exists(), "unexpected diagnostic output: " + name)
            continue
        records = read_lines(path)
        require([r.get("step") for r in records] == [s for s in steps if s % every == 0],
                "diagnostic records do not cover exactly the continuation: " + name)
        report[name] = len(records)
        if key:
            old = {r.get(key) for r in read_lines(Path(previous_dir) / name)}
            require(len(old) == 1 and M.digest(next(iter(old))) and {r.get(key) for r in records} == old,
                    "fixed diagnostic set differs from the 5000 run: " + name)
            report[key] = next(iter(old))
    return report


def check_rank_records(out, summary, plan, code, job, plan_sha):
    """Allocated and reserved peak of every rank; the allocated gate is the one already used for production memory."""
    limit, records = plan["memory"]["allocated_fraction_limit"], []
    for rank in range(4):
        record = json.loads((Path(out) / f"rank_{rank}.json").read_text())
        require(record.get("protocol") == RANK_PROTOCOL and record.get("rank") == rank and record.get("local_rank") == rank and
                record.get("world_size") == 4 and record.get("returncode") == 0 and record.get("code_sha") == code and
                record.get("slurm_job_id") == job and record.get("plan_sha256") == plan_sha, "rank record identity differs")
        allocated, reserved, total = (record.get(k) for k in ("peak_allocated_bytes", "peak_reserved_bytes", "device_memory_bytes"))
        require(all(type(v) is int and v > 0 for v in (allocated, reserved, total)), "invalid rank memory record")
        require(allocated <= limit * total, f"allocated memory above the gate on rank {rank}")
        require(allocated <= reserved <= total, f"reserved memory inconsistent on rank {rank}")
        records.append(record)
    peak = max(r["peak_allocated_bytes"] for r in records)
    reported = summary.get("gpu_max_memory_gib")
    require(type(reported) in (int, float) and abs(reported * 2**30 - peak) <= 2**20, "rank peaks do not reconcile with the trainer summary")
    return {"peak_allocated_bytes_by_rank": [r["peak_allocated_bytes"] for r in records],
            "peak_reserved_bytes_by_rank": [r["peak_reserved_bytes"] for r in records],
            "device_memory_bytes_by_rank": [r["device_memory_bytes"] for r in records],
            "allocated_fraction_by_rank": [r["peak_allocated_bytes"] / r["device_memory_bytes"] for r in records],
            "reserved_fraction_by_rank": [r["peak_reserved_bytes"] / r["device_memory_bytes"] for r in records],
            "allocated_fraction_limit": limit, "reference_peak_allocated_gib": plan["memory"]["reference_peak_allocated_gib"],
            "peak_allocated_gib": peak / 2**30}


def check_retention(paths, required_until, now, walltime_seconds):
    """Scratch is purged 40 days after the modification time: every scratch file must outlive the declared horizon."""
    require(required_until > now + timedelta(seconds=walltime_seconds), "declared horizon ends before the training allocation")
    earliest = None
    for path in paths:
        if str(path).startswith(SCRATCH_PREFIX):
            expiry = datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc) + timedelta(days=SCRATCH_RETENTION_DAYS)
            require(expiry >= required_until, "scratch retention ends before the declared horizon: " + str(path))
            earliest = expiry if earliest is None or expiry < earliest else earliest
    return earliest


def check_control(plan, context, signature, states, now=None):
    """Existing T100-G-30M at 19618: checkpoint, training receipt, two pose and five classification caches."""
    control = plan["control"]
    for name in ("receipt", "pose_inventory", "classification_inventory"):
        states["control_" + name] = M.resolve_pin(control[name])
    inventories = [json.loads(Path(states["control_" + name]["path"]).read_text()) for name in ("pose_inventory", "classification_inventory")]
    require(control["artifacts"] == control_pins(*inventories), "control cache pins differ from the reviewed inventories")
    receipt = json.loads(Path(states["control_receipt"]["path"]).read_text())
    require(receipt.get("protocol") == T.PROTOCOL and receipt.get("complete") is True and receipt.get("arm") == CONTROL_ARM and
            receipt.get("checkpoint_step") == END and receipt.get("code_sha") == M.BASE_CODE_SHA and
            receipt.get("checkpoint") == control["checkpoint"]["path"] and
            receipt.get("checkpoint_sha256") == control["checkpoint"]["sha256"], "control is not the completed T100-G 19618")
    T.checkpoint_identity(control["checkpoint"]["path"], replace(context["config_30m"], max_steps=END), END, signature)
    states["control_checkpoint"] = M.resolve_pin(control["checkpoint"])
    for name, pin in sorted(control["artifacts"].items()):
        states["control:" + name] = M.resolve_pin(pin)
    paths = [control["checkpoint"]["path"], *(pin["path"] for pin in control["artifacts"].values())]
    earliest = check_retention(paths, M.parse_utc(control["required_until_utc"]), now or datetime.now(timezone.utc),
                               plan["allocation"]["walltime_seconds"])
    return {"required_until_utc": control["required_until_utc"],
            "earliest_scratch_expiry_utc": None if earliest is None else earliest.strftime("%Y-%m-%dT%H:%M:%SZ")}


def seed_run_dir(previous, run_dir, names=SEED_FILES):
    """New run directory with byte-identical copies of the resume checkpoint and of the identity files."""
    source, run_dir = Path(previous["run_dir"]), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    seeded = {}
    for name in names:
        temporary = run_dir / (name + ".seed")
        shutil.copyfile(source / name, temporary)
        state = P.file_identity(temporary)
        require(state["sha256"] == previous["files"][name]["sha256"] and state["bytes"] == previous["files"][name]["bytes"],
                "seed copy differs from the 5000 run: " + name)
        temporary.replace(run_dir / name)
        seeded[name] = {"sha256": state["sha256"], "bytes": state["bytes"]}
    return seeded


def verify_outputs(plan, run_dir, out, cfg, signature, before, code, job, plan_sha):
    """Complete 19618 endpoint in the new directory, resumed optimizer, original schedule and per-rank memory."""
    run_dir = Path(run_dir)
    require(sorted(p.name for p in run_dir.iterdir()) == sorted(NEW_FILES), "continuation outputs differ from the expected set")
    summary = json.loads((run_dir / "summary.json").read_text())
    require(summary.get("stopped") == "max_steps" and summary.get("steps") == END and summary.get("world_size") == 4 and
            summary.get("parameters") == PARAMETERS and summary.get("accum_steps") == 1 and summary.get("micro_batch") == 8 and
            summary.get("batch_per_process") == 8, "continuation incomplete or wrong recipe")
    require(json.loads((run_dir / "config.json").read_text()) == R.config_to_dict(cfg), "final configuration differs")
    named, final = run_dir / f"checkpoint_step{END:07d}.pt", run_dir / "checkpoint.pt"
    named_state = T.checkpoint_identity(named, cfg, END, signature)
    final_state = T.checkpoint_identity(final, cfg, END, signature)
    require(named_state["trained_state_sha256"] == final_state["trained_state_sha256"], "named and final trained states differ")
    after = optimizer_mapping(torch.load(final, map_location="cpu", weights_only=False), cfg, PARAMETERS)
    check_optimizer_continuity(before, after)
    for name in IDENTITY_FILES:
        require(R.sha256_file(run_dir / name) == plan["previous"]["files"][name]["sha256"], "identity file changed: " + name)
    return {"summary": summary, "named": named, "final": final, "trained_state_sha256": final_state["trained_state_sha256"],
            "optimizer": {k: v for k, v in after.items() if k != "adam_steps"},
            "tranche": check_tranche_logs(run_dir, plan["previous"]["run_dir"], cfg),
            "memory": check_rank_records(out, summary, plan, code, job, plan_sha)}


def load(plan_path, expected_code):
    code = P.code_identity(expected_code)
    plan_state = P.file_identity(plan_path)
    plan = json.loads(Path(plan_state["path"]).read_text())
    contract_state = M.resolve_pin(plan.get("source_contract"))
    context = M.check_source_contract(contract_state["path"])
    previous_cfg, cfg, argv = validate_plan(plan, context)
    return code, plan_state, plan, contract_state, context, previous_cfg, cfg, argv


def worker(plan_path, expected_code, launch_sha):
    """One DDP rank: the unchanged trainer entry point, then this rank's memory peaks."""
    code, plan_state, plan, _, _, _, _, argv = load(plan_path, expected_code)
    rank, local, world = (int(os.environ.get(k, "-1")) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4 and 0 <= rank < 4 and local == rank, "four single-node ranks required")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and int(job) > 0, "real SLURM identity required")
    receipts, run_dir = Path(plan["receipt_dir"]), Path(plan["run_dir"])
    launch = M.resolve_pin({"path": str(receipts / "launch.json"), "sha256": launch_sha})
    record = json.loads(Path(launch["path"]).read_text())
    marker = json.loads((run_dir / MARKER).read_text())
    for value in (record, marker):
        require(value.get("protocol") == PROTOCOL and value.get("plan_sha256") == plan_state["sha256"] and
                value.get("code_sha") == code and value.get("slurm_job_id") == job, "worker launch identity differs")
    require(record.get("trainer_argv") == argv and not (run_dir / "STOP").exists(), "worker recipe differs or run was stopped")
    returncode = CLI.main(argv[1:])  # unchanged entry point: owns DDP, resume, checkpoints and diagnostics
    torch.cuda.synchronize(local)
    result = {"protocol": RANK_PROTOCOL, "rank": rank, "local_rank": local, "world_size": world, "returncode": returncode,
              "code_sha": code, "slurm_job_id": job, "plan_sha256": plan_state["sha256"], "run_dir": str(run_dir),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(local),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(local),
              "device_memory_bytes": torch.cuda.get_device_properties(local).total_memory}
    with (receipts / f"rank_{rank}.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    require(returncode == 0, "trainer failed")


def execute(plan_path, expected_code, out):
    code, plan_state, plan, contract_state, context, previous_cfg, cfg, argv = load(plan_path, expected_code)
    job = verify_allocation(plan)
    out, run_dir, previous = Path(out), Path(plan["run_dir"]), plan["previous"]
    require(out.is_absolute() and str(out) == plan["receipt_dir"], "wrong receipt directory")
    for path in (out, run_dir):
        require(not path.exists() and not path.is_symlink() and not any(path.resolve().is_relative_to(root) for root in CODE_ROOTS),
                "new output outside code required; no overwrite")
    states = dict(context["states"])
    states["execution_plan"] = plan_state
    for path, sha in plan["execution_source_sha256"].items():
        states["execution_source:" + path] = M.resolve_pin({"path": str(M._contained_source(path, ROOT)), "sha256": sha})
    check_budget_freeze(plan, states)
    inputs = context["contract"]["inputs"]
    for name, pin in inputs.items():
        states["input:" + name] = M.resolve_pin(pin)
    scales = json.loads(Path(inputs["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(inputs["windows"]["path"]).read_text())["window_s"]
    R.check_input_files(Path(inputs["manifest"]["path"]), Path(inputs["windows"]["path"]), cfg.nested, inputs["manifest"]["sha256"])

    def signature_of(config):
        return json.loads(json.dumps(R.calibration_signature(config, inputs["manifest"]["path"], scales, windows)))

    signature = signature_of(cfg)
    require(signature == signature_of(previous_cfg), "data signature depends on the dose")
    # The completed 5000 run: exact inventory, receipt, resumable state and optimizer mapping.
    states.update(previous_states(previous))
    for name in ("receipt", "receipt_copy"):
        states["previous_" + name] = M.resolve_pin(previous[name])
    check_previous_receipt(json.loads(Path(states["previous_receipt"]["path"]).read_text()), previous, context, contract_state["sha256"])
    old_dir = Path(previous["run_dir"])
    old_named = T.checkpoint_identity(old_dir / "checkpoint_step0005000.pt", previous_cfg, START, signature)
    old_resume = T.checkpoint_identity(old_dir / "checkpoint.pt", previous_cfg, START, signature)
    require(old_named["trained_state_sha256"] == old_resume["trained_state_sha256"], "named and resume 5000 states differ")
    before = optimizer_mapping(torch.load(old_dir / "checkpoint.pt", map_location="cpu", weights_only=False), previous_cfg, PARAMETERS)
    control = check_control(plan, context, signature_of(context["config_30m"]), states)
    P.check_unchanged(states)
    out.mkdir(parents=True, exist_ok=False)
    seeded = seed_run_dir(previous, run_dir)
    identity = {"protocol": PROTOCOL, "arm": ARM, "start_step": START, "end_step": END, "code_sha": code,
                "plan_sha256": plan_state["sha256"], "slurm_job_id": job, "training_source_contract_sha256": contract_state["sha256"]}
    P.write_json(run_dir / MARKER, {**identity, "previous_run_dir": str(old_dir), "seeded_files": seeded,
                                    "previous_trained_state_sha256": old_resume["trained_state_sha256"]})
    P.write_json(out / "launch.json", {**identity, "trainer_argv": argv, "seeded_files": seeded, "control": control,
        "previous_trained_state_sha256": old_resume["trained_state_sha256"],
        "previous_optimizer": {k: v for k, v in before.items() if k != "adam_steps"}, "verified_inputs": states})
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(ROOT / "scripts/scaling_continue.py"), "worker", "--plan", plan_state["path"],
               "--expected-code-sha", code, "--launch-sha256", R.sha256_file(out / "launch.json")]
    with (out / "training.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "WEARUSFM_CODE_SHA": code, "WEARUSFM_CODE_DIR": str(ROOT)},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    evidence = verify_outputs(plan, run_dir, out, cfg, signature, before, code, job, plan_state["sha256"])
    # Every pinned input, the control and the eleven files of the 5000 run are unchanged after training.
    P.check_unchanged(states)
    require(sorted(p.name for p in old_dir.iterdir()) == sorted(previous["files"]), "5000 run directory gained or lost files")
    P.code_identity(code)
    config_file = out / "config.json"
    P.write_json(config_file, R.config_to_dict(cfg))
    P.write_json(out / "summary.json", evidence["summary"])
    receipt = T.receipt_for(ARM, END, evidence["named"], evidence["final"], config_file, code, plan_state["sha256"], command)
    receipt.update(protocol=PROTOCOL, start_step=START, additional_steps=END - START, trainer_argv=argv,
                   previous_run_dir=str(old_dir), previous_checkpoint_sha256=previous["files"]["checkpoint.pt"]["sha256"],
                   previous_receipt_sha256=previous["receipt"]["sha256"],
                   previous_trained_state_sha256=old_resume["trained_state_sha256"],
                   trained_state_sha256=evidence["trained_state_sha256"],
                   training_source_contract_sha256=contract_state["sha256"],
                   scientific_source_sha256=context["contract"]["scientific_source_sha256"],
                   execution_source_sha256=plan["execution_source_sha256"], reference_code_sha=M.BASE_CODE_SHA,
                   control_checkpoint_sha256=plan["control"]["checkpoint"]["sha256"], control=control,
                   budget_freeze_sha256=plan["budget_freeze"]["sha256"], optimizer=evidence["optimizer"],
                   tranche=evidence["tranche"], memory=evidence["memory"], previous_run_unchanged=True,
                   scientific_endpoint=True, automatic_continuation=False, further_training_authorized=False)
    P.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    offline = commands.add_parser("check-plan", help="local plan, source and recipe checks only; never launches")
    offline.add_argument("--plan", type=Path, required=True)
    for name in ("run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--expected-code-sha", required=True)
    commands.choices["run"].add_argument("--out", type=Path, required=True)
    commands.choices["worker"].add_argument("--launch-sha256", required=True)
    args = parser.parse_args(argv)
    if args.mode == "check-plan":
        plan = json.loads(args.plan.read_text())
        context = M.check_source_contract(M.resolve_pin(plan.get("source_contract"))["path"])
        _, cfg, trainer_argv = validate_plan(plan, context)
        states = {}
        check_budget_freeze(plan, states)
        inventories = [json.loads((ROOT / plan["control"][name]["path"]).read_text()) for name in ("pose_inventory", "classification_inventory")]
        require(plan["control"]["artifacts"] == control_pins(*inventories), "control cache pins differ from the reviewed inventories")
        for path, sha in plan["execution_source_sha256"].items():
            M.resolve_pin({"path": str(M._contained_source(path, ROOT)), "sha256": sha})
        print(json.dumps({"offline_plan_valid": True, "runtime_readiness_checked": False, "max_steps": cfg.max_steps,
                          "trainer_argv": trainer_argv}, indent=2))
    elif args.mode == "run":
        execute(args.plan, args.expected_code_sha, args.out)
    else:
        worker(args.plan, args.expected_code_sha, args.launch_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
