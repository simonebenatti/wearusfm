"""Guarded scientific T100-G300M screening to exactly5000 steps.

Offline contract/plan validation never grants runtime readiness. Execution needs
real pinned technical, production-memory, evaluation-readiness and review reports,
the verified historical30M control, and the unchanged scientific training sources.
This is orchestration around train_jepa.py, not a new trainer or a submitter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


T = _module("scaling_train_tokenizer", "tokenizer_train.py")
S = _module("scaling_train_preflight", "scaling_preflight.py")
CLI = _module("scaling_train_cli", "train_jepa.py")
P, require = T.P, T.require
PROTOCOL = "scaling-training-20261104-v1"
CONTRACT_PROTOCOL = "scaling-training-source-contract-20261104-v1"
BASE_CODE_SHA = "afb644ea10ecc2a716160b4d8a4904e805c366fb"
PREFLIGHT_CODE_SHA = "05d4bf0663481522a56d96c0b07847924cf5bd26"
ARM = "T100-G-300M"
STEP = 5000
SCRIPT_PATHS = {"scripts/train_jepa.py", "scripts/tokenizer_train.py", "scripts/tokenizer_preflight.py",
                "scripts/slurm/tokenizer_train.sbatch", "scripts/slurm/lib_code_dir.sh"}
INPUTS = {"manifest", "scales", "windows", "pose_stats"}
GATES = {"preflight", "production_memory", "evaluation_readiness", "scientific_review"}
REQUIRED_EVALUATORS = {"scripts/scaling_evaluation.py", "scripts/scaling_pose_readout.py", "scripts/scaling_probe.py",
                       "scripts/scaling_compare.py", "scripts/scaling_projection.py", "scripts/scaling_reading.py"}
EXECUTION_SOURCES = {"scripts/scaling_train.py", "scripts/scaling_preflight.py"}


def digest(value, n=64):
    return isinstance(value, str) and len(value) == n and value != "0" * n and all(c in "0123456789abcdef" for c in value)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin_schema(pin):
    require(isinstance(pin, dict) and set(pin) <= {"path", "sha256", "bytes"} and
            isinstance(pin.get("path"), str) and pin["path"] and digest(pin.get("sha256")), "invalid artifact pin")
    require("bytes" not in pin or type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid artifact size")


def resolve_pin(pin, root=ROOT):
    pin_schema(pin)
    path = Path(pin["path"])
    if not path.is_absolute():
        path = Path(root) / path
    state = P.file_identity(path)
    require(state["sha256"] == pin["sha256"] and ("bytes" not in pin or state["bytes"] == pin["bytes"]),
            "artifact SHA/size mismatch: " + str(path))
    return state


def _contained_source(path, root):
    p = Path(path)
    require(not p.is_absolute() and ".." not in p.parts and p.parts, "invalid relative source path")
    full = (Path(root) / p).resolve(strict=True)
    require(full.is_relative_to(Path(root).resolve()), "source escapes checkout")
    return full


def check_source_contract(contract_or_path, root=ROOT):
    """Validate local source bytes and derive both configs; no remote input reads.

    Returns contract, optional identity, states and config_30m/config_300m. A dict
    is accepted for unit/offline callers; execution always uses a pinned file.
    """
    identity = None
    if isinstance(contract_or_path, dict):
        contract = contract_or_path
    else:
        identity = P.file_identity(contract_or_path)
        contract = json.loads(Path(identity["path"]).read_text())
    require(contract.get("protocol") == CONTRACT_PROTOCOL and contract.get("reference_code_sha") == BASE_CODE_SHA,
            "wrong scientific source contract/reference")
    require(contract.get("arm") == ARM and contract.get("checkpoint_step") == STEP and
            contract.get("world_size") == 4 and contract.get("global_batch_size") == 32 and contract.get("num_workers") == 6,
            "wrong scientific arm/dose/stream")
    require(contract.get("allowed_training_cli_changes") == ["--preset", "--out-dir", "--max-steps"] and
            contract.get("schedule") == {"total_steps": 49046, "warmup_steps": 1962, "decay_fraction": .2},
            "source contract changes scientific scope")
    require(set(contract.get("inputs", {})) == INPUTS and digest(contract.get("reference_checkpoint_sha256")),
            "incomplete source input/control pins")
    for pin in contract["inputs"].values():
        pin_schema(pin)
        require(Path(pin["path"]).is_absolute(), "training inputs require absolute paths")
    require(isinstance(contract.get("roots"), list) and len(contract["roots"]) == 2 and
            all(Path(path).is_absolute() for path in contract["roots"]), "invalid processed roots")
    expected_paths = {str(p.relative_to(root)) for p in (Path(root) / "src/wearusfm").rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    expected_paths |= SCRIPT_PATHS
    sources = contract.get("scientific_source_sha256")
    require(isinstance(sources, dict) and set(sources) == expected_paths and all(digest(v) for v in sources.values()),
            "scientific source inventory incomplete or changed")
    states = {}
    for path, sha in sources.items():
        states[path] = resolve_pin({"path": str(_contained_source(path, root)), "sha256": sha}, root)
    for name in ("comparison", "config_30m", "reference_training_receipt"):
        states[name] = resolve_pin(contract[name], root)
    comparison = json.loads(Path(states["comparison"]["path"]).read_text())
    entry = comparison["arm_configurations"]["T100-G"]
    require(entry["sha256"] == contract["config_30m"]["sha256"], "control configuration pin differs")
    argv = entry["validated_trainer_argv"]
    require(argv[0] == "scripts/train_jepa.py", "unexpected training entry point")
    args = CLI.build_parser().parse_args(argv[1:])
    cfg = CLI.build_config(args)
    require(R.config_to_dict(cfg) == json.loads(Path(states["config_30m"]["path"]).read_text()), "base CLI/config mismatch")
    require(args.preset == "sanity" and args.distributed and args.num_workers == 6 and cfg.max_steps == STEP,
            "base recipe differs from historical control")
    require([str(p) for p in args.root] == contract["roots"], "source roots differ from validated CLI")
    for arg, field in (("manifest", "manifest"), ("scales", "scales"), ("window_seconds", "windows")):
        require(str(getattr(args, arg)) == contract["inputs"][field]["path"], "source CLI input differs: " + field)
    require(str(Path(cfg.loader.pose_root) / "pose_stats.json") == contract["inputs"]["pose_stats"]["path"],
            "pose statistics path differs")
    reference = json.loads(Path(states["reference_training_receipt"]["path"]).read_text())
    require(reference.get("protocol") == T.PROTOCOL and reference.get("complete") is True and
            reference.get("arm") == "T100-G" and reference.get("checkpoint_step") == STEP and
            reference.get("code_sha") == BASE_CODE_SHA and reference.get("checkpoint_sha256") == contract["reference_checkpoint_sha256"],
            "historical30M receipt differs")
    if identity:
        states["contract"] = identity
    return {"contract": contract, "identity": identity, "states": states, "comparison": comparison,
            "config_30m": cfg, "config_300m": S.configurations(cfg)[ARM], "reference_receipt": reference}


def derive_recipe(context, run_dir, seconds):
    """Reconstruct CLI; only preset/output/dose differ, plus operational timeout."""
    require(type(seconds) is int and 1200 < seconds <= 21600 and seconds % 900 == 0, "invalid scientific walltime")
    run_dir = Path(run_dir)
    require(run_dir.is_absolute(), "absolute run directory required")
    original = list(context["comparison"]["arm_configurations"]["T100-G"]["validated_trainer_argv"])
    argv = original.copy()
    for flag, value in (("--preset", "300m"), ("--out-dir", str(run_dir)), ("--max-steps", str(STEP))):
        require(argv.count(flag) == 1, "duplicate/missing scientific argument: " + flag)
        argv[argv.index(flag) + 1] = value
    require("--time-limit-s" not in argv and "--resume" not in argv, "unexpected resume/timeout override")
    args = CLI.build_parser().parse_args(argv[1:])
    cfg = CLI.build_config(args)
    require(R.config_to_dict(cfg) == R.config_to_dict(context["config_300m"]), "CLI300M differs from preflight preset")
    argv += ["--time-limit-s", str(seconds - 300)]
    return cfg, ["torchrun", "--standalone", "--nproc_per_node=4", *argv]


def validate_plan(plan, context):
    """Schema/recipe checks only. Actual artifacts must still pass runtime gates."""
    require(plan.get("protocol") == PROTOCOL and plan.get("arm") == ARM and plan.get("checkpoint_step") == STEP,
            "only the registered fresh300M5000 phase is supported")
    require(plan.get("scientific_source_sha256") ==
            context["contract"]["scientific_source_sha256"], "plan scientific identity differs")
    execution_sources = plan.get("execution_source_sha256")
    require(isinstance(execution_sources, dict) and set(execution_sources) == EXECUTION_SOURCES and
            all(digest(v) for v in execution_sources.values()), "missing exact execution source pins")
    pin_schema(plan.get("source_contract"))
    if context["identity"]:
        require(plan["source_contract"]["sha256"] == context["identity"]["sha256"], "source contract pin differs")
    require(set(plan.get("gates", {})) == GATES, "missing concrete runtime readiness gates")
    for pin in plan["gates"].values():
        pin_schema(pin)
    allocation = plan.get("allocation", {})
    require(set(allocation) == {"nodes", "gpus", "cpus", "memory", "walltime_seconds"} and
            all(allocation.get(k) == v for k, v in {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "480G"}.items()),
            "wrong single-node scientific allocation")
    require(isinstance(plan.get("control"), dict) and set(plan["control"]) ==
            {"checkpoint", "receipt", "retention_verified_until_utc", "required_until_utc"},
            "missing verified30M control/retention")
    for name in ("checkpoint", "receipt"):
        pin_schema(plan["control"][name])
    require(plan["control"]["checkpoint"]["sha256"] == context["contract"]["reference_checkpoint_sha256"] and
            plan["control"]["receipt"]["sha256"] == context["contract"]["reference_training_receipt"]["sha256"],
            "control is not the frozen T100-G5000")
    parse_utc(plan["control"]["retention_verified_until_utc"])
    require(parse_utc(plan["control"]["required_until_utc"]) <= parse_utc(plan["control"]["retention_verified_until_utc"]),
            "control retention does not cover the declared comparison horizon")
    for name in ("run_dir", "receipt_dir"):
        require(isinstance(plan.get(name), str) and Path(plan[name]).is_absolute(), "absolute new output required: " + name)
    require(Path(plan["run_dir"]).resolve() != Path(plan["receipt_dir"]).resolve(), "run and receipt outputs must differ")
    return derive_recipe(context, plan["run_dir"], allocation["walltime_seconds"])


def parse_utc(value):
    require(isinstance(value, str) and value.endswith("Z"), "UTC timestamp ending Z required")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid UTC retention timestamp") from exc


def _read_gate(pin, states, name):
    states[name] = resolve_pin(pin)
    return json.loads(Path(states[name]["path"]).read_text())


def _positive_numbers(values, count):
    return isinstance(values, list) and len(values) == count and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)


def check_preflight(report, context):
    require(report.get("protocol") == S.PROTOCOL and report.get("complete") is True and
            report.get("code_sha") == PREFLIGHT_CODE_SHA and report.get("paired_data_masks_queries_targets") is True and
            report.get("one_model_resident") is True and report.get("scientific_checkpoints_written") is False and
            report.get("batches_per_arm") == 128 and report.get("native_warmup_batches") == report.get("peak_stress_batches") == 64 and
            report.get("world_size") == 4 and report.get("windows_per_arm") == 4096,
            "preflight incomplete or wrong protocol/dose/code")
    require(set(report.get("arms", {})) == set(S.ARMS), "preflight arm set differs")
    for name, pin in context["contract"]["inputs"].items():
        require(report.get("inputs", {}).get(name, {}).get("sha256") == pin["sha256"], "preflight input pin differs")
    mean_times, memories = {}, {}
    measured = list(range(8, 64)) + list(range(72, 128))
    for name in S.ARMS:
        arm = report["arms"][name]
        times = arm.get("normal_update_rank_max_seconds")
        require(_positive_numbers(times, 128), "missing128 measured update times")
        require(arm.get("memory_gate_passed") is True and type(arm.get("peak_allocated_bytes")) is int and
                type(arm.get("device_memory_bytes")) is int and 0 < arm["peak_allocated_bytes"] <= .9 * arm["device_memory_bytes"],
                "preflight memory gate failed")
        expected_parameters = 28362013 if name.endswith("30M") and name != ARM else 292054847
        require(arm.get("initial", {}).get("parameters") == expected_parameters, "preflight parameter count differs")
        mean_times[name] = sum(times[i] for i in measured) / len(measured)
        memories[name] = arm["peak_allocated_bytes"]
    ratio = mean_times[ARM] / mean_times[S.ARMS[0]]
    seconds_per_step = max(3883.029760837555 / 5000 * max(1., ratio), mean_times[ARM])
    minimum_seconds = math.ceil((STEP * seconds_per_step * 1.20 + 1200) / 900) * 900
    memory_proxy = memories[ARM] * max(1., 15.473958969116211 * 2**30 / memories[S.ARMS[0]])
    require(memory_proxy <= .9 * report["arms"][ARM]["device_memory_bytes"], "projected production memory exceeds gate")
    return {"measured_steps": 112, "mean_update_seconds": mean_times, "ratio300_to30": ratio,
            "minimum_walltime_seconds": minimum_seconds, "production_memory_proxy_bytes": memory_proxy}


def check_production_memory(report, cfg, contract_sha):
    # Coverage flags and a config with64 updates do not exercise the periodic
    # diagnostics with AdamW/prefetch resident. Verify the pinned500-step evidence.
    guard = _module("scaling_production_memory_guard", "scaling_production_memory.py")
    return guard.validate_report(report, R.config_to_dict(cfg), contract_sha)


def check_evaluation_readiness(report, contract_sha, states):
    require(report.get("protocol") == "scaling-evaluation-readiness-20261104-v1" and report.get("complete") is True and
            report.get("checkpoint_step") == STEP and report.get("training_source_contract_sha256") == contract_sha and
            report.get("dimensions") == {"T100-G-30M": 384, ARM: 896} and
            report.get("pool_widths") == {"T100-G-30M": 768, ARM: 1792},
            "evaluation readiness is missing or for the wrong dimensions/dose")
    sources = report.get("source_files")
    require(isinstance(sources, dict) and REQUIRED_EVALUATORS <= set(sources), "dimension-aware evaluators missing")
    require(report.get("capabilities") == {"dimension_aware_extraction": True, "pose_ridge": True,
            "classification_probe": True, "dimension_matched_secondary": True, "convergence_audit": True,
            "paired_comparison": True},
            "extraction-only readiness is insufficient; readers and secondary control must be ready")
    for path, sha in sources.items():
        states["evaluation_source:" + path] = resolve_pin({"path": str(_contained_source(path, ROOT)), "sha256": sha})
    verification = _read_gate(report.get("verification_report"), states, "evaluation_tests")
    require(verification.get("complete") is True and verification.get("failed") == 0 and
            type(verification.get("passed")) is int and verification["passed"] > 0 and
            verification.get("source_files") == sources, "evaluation tests not bound to these sources or failed")
    resources = _read_gate(report.get("resource_plan"), states, "evaluation_resources")
    require(resources.get("complete") is True and resources.get("checkpoint_step") == STEP and
            resources.get("includes_30m_classification_extraction") is True and
            resources.get("includes_dimension_matched_secondary") is True and
            all(type(resources.get(key)) in (int, float) and math.isfinite(resources[key]) and resources[key] > 0
                for key in ("gpu_hours_cap", "cpu_core_hours_cap")), "evaluation cost/coverage plan missing")
    require(report.get("review_verdict") == "APPROVATO", "evaluation review not approved")
    states["evaluation_review"] = resolve_pin(report.get("review"))
    parse_utc(report.get("control_required_until_utc"))


def check_runtime_gates(plan, context, cfg, states, now=None):
    """Read every pinned gate and verify controls; cannot launch from booleans alone."""
    contract_sha = context["identity"]["sha256"]
    preflight = _read_gate(plan["gates"]["preflight"], states, "preflight")
    estimate = check_preflight(preflight, context)
    require(plan["allocation"]["walltime_seconds"] >= estimate["minimum_walltime_seconds"], "allocation below measured runtime policy")
    memory = _read_gate(plan["gates"]["production_memory"], states, "production_memory")
    check_production_memory(memory, cfg, contract_sha)
    ready = _read_gate(plan["gates"]["evaluation_readiness"], states, "evaluation_readiness")
    check_evaluation_readiness(ready, contract_sha, states)
    review = _read_gate(plan["gates"]["scientific_review"], states, "scientific_review")
    require(review.get("protocol") == "scaling-scientific-review-20261104-v1" and review.get("complete") is True and
            review.get("verdict") == "APPROVATO" and review.get("scope") == "fresh300M5000" and
            review.get("training_source_contract_sha256") == contract_sha and
            review.get("execution_source_sha256") == plan["execution_source_sha256"] and
            review.get("gates_sha256") == {k: plan["gates"][k]["sha256"] for k in GATES - {"scientific_review"}} and
            review.get("allocation") == plan["allocation"], "scientific readiness review not bound to final gates/resources/code")
    states["scientific_review_text"] = resolve_pin(review.get("review"))
    current = now or datetime.now(timezone.utc)
    retain_until = parse_utc(plan["control"]["retention_verified_until_utc"])
    required_until = parse_utc(plan["control"]["required_until_utc"])
    require(ready.get("control_required_until_utc") == plan["control"]["required_until_utc"] and
            required_until > current + timedelta(seconds=plan["allocation"]["walltime_seconds"]) and
            retain_until >= required_until,
            "control retention must cover the published extraction/comparison horizon, not only training")
    checkpoint_path = Path(plan["control"]["checkpoint"]["path"])
    require(checkpoint_path.is_absolute(), "absolute control checkpoint path required")
    if str(checkpoint_path).startswith("/leonardo_scratch/"):
        require(retain_until <= parse_utc(context["contract"]["retention_scratch_not_after_utc"]),
                "scratch retention exceeds declared control deadline; preserve and repin first")
    states["control_receipt"] = resolve_pin(plan["control"]["receipt"])
    control_receipt = json.loads(Path(states["control_receipt"]["path"]).read_text())
    require(control_receipt == context["reference_receipt"], "control receipt changed")
    states["control_checkpoint"] = resolve_pin(plan["control"]["checkpoint"])
    return estimate


def slurm_seconds(value):
    try:
        day, clock = value.split("-", 1) if "-" in value else ("0", value)
        fields = [int(x) for x in clock.split(":")]
        require(len(fields) == 3 and all(v >= 0 for v in fields) and fields[1] < 60 and fields[2] < 60,
                "invalid SLURM clock")
        return int(day) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid SLURM time limit") from exc


def check_allocation_fields(fields, expected, visible_gpus):
    require(fields.get("NumNodes") == "1" and fields.get("NumCPUs") == "32" and visible_gpus == 4,
            "runtime needs one node,32CPUs and4visibleGPUs")
    require(slurm_seconds(fields.get("TimeLimit")) == expected["walltime_seconds"], "actual walltime differs from plan")
    tres = dict(part.split("=", 1) for part in fields.get("AllocTRES", "").split(",") if "=" in part)
    require(tres.get("cpu") == "32" and tres.get("node") == "1" and tres.get("gres/gpu") == "4" and
            tres.get("mem") in ("480G", "491520M"), "actual resources differ from plan")


def verify_allocation(plan):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and int(job) > 0, "scientific run requires a real SLURM allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(part.split("=", 1) for part in raw.split() if "=" in part)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING", "wrong or nonrunning SLURM allocation")
    check_allocation_fields(fields, plan["allocation"], T.torch.cuda.device_count())


def verify_phase_outputs(run_dir, cfg, signature):
    """Require a complete endpoint with production structural-state guards."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    require(summary.get("stopped") == "max_steps" and summary.get("steps") == STEP and
            summary.get("world_size") == 4 and summary.get("parameters") == 292054847 and
            summary.get("accum_steps") == 1 and not (run_dir / "STOP").exists(), "scientific5000 incomplete or wrong recipe")
    require(json.loads((run_dir / "config.json").read_text()) == R.config_to_dict(cfg), "final configuration differs")
    named, final = run_dir / "checkpoint_step0005000.pt", run_dir / "checkpoint.pt"
    named_state = T.checkpoint_identity(named, cfg, STEP, signature)
    final_state = T.checkpoint_identity(final, cfg, STEP, signature)
    require(named_state["trained_state_sha256"] == final_state["trained_state_sha256"], "named and final trained states differ")
    return summary, named, final


def execute(plan_path, expected_code, out):
    code = P.code_identity(expected_code)
    plan_state = P.file_identity(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    contract_state = resolve_pin(plan.get("source_contract"))
    context = check_source_contract(contract_state["path"])
    cfg, command = validate_plan(plan, context)
    verify_allocation(plan)
    out, run_dir = Path(out), Path(plan["run_dir"])
    require(out.is_absolute() and str(out) == plan["receipt_dir"], "wrong receipt directory")
    roots = (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"), Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))
    for path in (out, run_dir):
        require(not path.exists() and not path.is_symlink() and not any(path.resolve().is_relative_to(root) for root in roots),
                "fresh output outside code required; no resume or overwrite")
    states = dict(context["states"])
    states["execution_plan"] = plan_state
    for path, sha in plan["execution_source_sha256"].items():
        states["execution_source:" + path] = resolve_pin({"path": str(_contained_source(path, ROOT)), "sha256": sha})
    estimate = check_runtime_gates(plan, context, cfg, states)
    for name, pin in context["contract"]["inputs"].items():
        states["input:" + name] = resolve_pin(pin)
    inputs = context["contract"]["inputs"]
    scales = json.loads(Path(inputs["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(inputs["windows"]["path"]).read_text())["window_s"]
    R.check_input_files(Path(inputs["manifest"]["path"]), Path(inputs["windows"]["path"]), cfg.nested, inputs["manifest"]["sha256"])
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs["manifest"]["path"], scales, windows)))
    control_signature = json.loads(json.dumps(R.calibration_signature(context["config_30m"], inputs["manifest"]["path"], scales, windows)))
    T.checkpoint_identity(plan["control"]["checkpoint"]["path"], context["config_30m"], STEP, control_signature)
    P.check_unchanged(states)
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / "launch.json", {"protocol": PROTOCOL, "arm": ARM, "checkpoint_step": STEP,
        "code_sha": code, "plan_sha256": plan_state["sha256"], "training_source_contract_sha256": contract_state["sha256"],
        "command": command, "resource_calculation": estimate, "verified_inputs": states})
    subprocess.run(command, cwd=ROOT, check=True)
    summary, named, final = verify_phase_outputs(run_dir, cfg, signature)
    P.check_unchanged(states)
    P.code_identity(code)
    config_file = out / "config.json"
    P.write_json(config_file, R.config_to_dict(cfg))
    P.write_json(out / "summary.json", summary)
    receipt = T.receipt_for(ARM, STEP, named, final, config_file, code, plan_state["sha256"], command)
    receipt.update(protocol=PROTOCOL, training_source_contract_sha256=contract_state["sha256"],
                   scientific_source_sha256=context["contract"]["scientific_source_sha256"],
                   execution_source_sha256=plan["execution_source_sha256"],
                   reference_code_sha=BASE_CODE_SHA, control_checkpoint_sha256=context["contract"]["reference_checkpoint_sha256"],
                   gates_sha256={name: plan["gates"][name]["sha256"] for name in GATES},
                   scientific_endpoint=True, automatic_continuation=False)
    P.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    offline = commands.add_parser("check-contract", help="read local source/config identity only; never launches")
    offline.add_argument("--contract", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--expected-code-sha", required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "check-contract":
        context = check_source_contract(args.contract)
        print(json.dumps({"offline_source_contract_valid": True, "runtime_readiness_checked": False,
                          "source_files": len(context["contract"]["scientific_source_sha256"]),
                          "config_300m": R.config_to_dict(context["config_300m"])}, indent=2))
    else:
        execute(args.plan, args.expected_code_sha, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
