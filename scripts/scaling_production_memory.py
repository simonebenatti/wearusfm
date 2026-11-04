"""S3: fresh 500-update production-path memory check, never a scientific endpoint.

API: run --plan PLAN --expected-code-sha SHA. The supervisor verifies the real
SLURM allocation and invokes four workers. Each calls unchanged train_jepa.main,
then records its own CUDA allocator peaks. report.json is written only after all
four workers and the pinned production logs prove the complete registered path.
No trainer hooks, resume, altered diagnostic cadence, submission or readiness.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "scaling-production-memory-20261104-v1"
PLAN_PROTOCOL = "scaling-production-memory-plan-20261104-v1"
RANK_PROTOCOL = "scaling-production-memory-rank-20261104-v1"
ARM, STEPS, MAX_SECONDS = "T100-G-300M", 500, 3600
EXECUTION_SOURCES = {"scripts/scaling_production_memory.py", "scripts/scaling_train.py", "scripts/scaling_preflight.py"}
ARTIFACT_NAMES = {"config": "config.json", "summary": "summary.json", "metrics": "metrics.jsonl",
                  "fixed_eval": "fixed_eval.jsonl", "fixed_set": "fixed_eval_set.json",
                  "pose_panel": "pose_panel.jsonl", "pose_set": "pose_panel_set.json",
                  "technical_marker": "TECHNICAL_ONLY.json", "technical_checkpoint": "checkpoint.pt"}
RUNTIME_FIELDS = ("python", "torch", "cuda")


def module(name):
    spec = importlib.util.spec_from_file_location("s3_" + name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


P = module("tokenizer_preflight")
require = P.require


def digest(value, n=64):
    return isinstance(value, str) and len(value) == n and value != "0" * n and all(c in "0123456789abcdef" for c in value)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    state = P.file_identity(path)
    return {k: state[k] for k in ("path", "sha256", "bytes")}


def checked_pin(value):
    require(isinstance(value, dict) and digest(value.get("sha256")) and
            isinstance(value.get("path"), str) and Path(value["path"]).is_absolute(), "absolute artifact pin required")
    state = P.file_identity(value["path"])
    require(state["sha256"] == value["sha256"] and
            ("bytes" not in value or state["bytes"] == value["bytes"]), "artifact SHA/size mismatch")
    return state


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite evidence")
    elif isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, list):
        for item in value: finite_tree(item)


def read_json(path, lines=False):
    value = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if lines else json.loads(Path(path).read_text())
    finite_tree(value)
    return value


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def runtime_values():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def runtime_comparison(current, reference):
    require(isinstance(current, dict) and set(current) == set(RUNTIME_FIELDS) | {"cudnn"} and
            all(isinstance(current[k], str) and current[k] for k in RUNTIME_FIELDS) and
            type(current["cudnn"]) is int and current["cudnn"] > 0, "complete CUDA runtime evidence required")
    require(reference.get("stopped") == "max_steps" and reference.get("steps") == 5000 and
            reference.get("parameters") == 28362013 and reference.get("world_size") == 4 and
            reference.get("code", {}).get("sha") == "afb644ea10ecc2a716160b4d8a4904e805c366fb",
            "runtime reference is not the historical T100-G endpoint")
    old = reference.get("environment", {})
    require(all(isinstance(old.get(k), str) and old[k] for k in RUNTIME_FIELDS), "historical runtime missing")
    available = list(RUNTIME_FIELDS)
    if old.get("cudnn") is not None:
        available.append("cudnn")
    require(all(current[k] == old[k] for k in available), "runtime differs from historical reference")
    return {"matched_fields": available, "unavailable_in_reference": [] if "cudnn" in available else ["cudnn"],
            "reference": {k: old[k] for k in available}}


def recipe(context, run_dir, seconds):
    require(type(seconds) is int and 1200 < seconds <= MAX_SECONDS and seconds % 900 == 0, "invalid S3 walltime")
    argv = list(context["comparison"]["arm_configurations"]["T100-G"]["validated_trainer_argv"])
    require(argv[0] == "scripts/train_jepa.py" and not any(x in argv for x in
            ("--init-from", "--branch-from", "--resume", "--time-limit-s")), "S3 cannot initialize or resume from a checkpoint")
    for flag, value in (("--preset", "300m"), ("--out-dir", str(run_dir)), ("--max-steps", str(STEPS))):
        require(argv.count(flag) == 1, "missing/duplicate recipe argument")
        argv[argv.index(flag) + 1] = value
    cli = module("train_jepa")
    cfg = cli.build_config(cli.build_parser().parse_args(argv[1:]))
    expected = replace(context["config_300m"], max_steps=STEPS)
    require(R.config_to_dict(cfg) == R.config_to_dict(expected), "technical recipe differs")
    return cfg, argv[1:] + ["--time-limit-s", str(seconds - 300)]


def resource_estimate(preflight, context):
    verified = module("scaling_train").check_preflight(preflight, context)
    seconds_per_step = max(verified["mean_update_seconds"][ARM],
                          3883.029760837555 / 5000 * max(1., verified["ratio300_to30"]))
    seconds = math.ceil((STEPS * seconds_per_step * 1.20 + 1200) / 900) * 900
    require(seconds <= MAX_SECONDS, "S3 measured estimate exceeds registered one-hour ceiling")
    return {"walltime_seconds": seconds, "gpu_hours_cap": 4 * seconds / 3600,
            "cpu_core_hours_cap": 32 * seconds / 3600, "seconds_per_step": seconds_per_step,
            "startup_diagnostics_checkpoint_reserve_seconds": 1200, "checkout_included": False}


def load_plan(path, expected_code):
    code = P.code_identity(expected_code)
    states = {"plan": P.file_identity(path)}
    plan = read_json(path)
    require(plan.get("protocol") == PLAN_PROTOCOL and plan.get("arm") == ARM and
            type(plan.get("steps")) is int and plan["steps"] == STEPS and plan.get("technical_only") is True,
            "wrong S3 plan/technical dose")
    for key in ("source_contract", "preflight", "reference_runtime_summary"):
        states[key] = checked_pin(plan.get(key))
    # This runs in the supervisor before output creation/torchrun, and again in
    # every worker before training; a changed runtime cannot consume the dose.
    runtime_comparison(runtime_values(), read_json(states["reference_runtime_summary"]["path"]))
    st = module("scaling_train")
    context = st.check_source_contract(states["source_contract"]["path"])
    states.update({"source:" + k: v for k, v in context["states"].items()})
    sources = plan.get("execution_source_sha256", {})
    require(set(sources) == EXECUTION_SOURCES, "missing S3 execution sources")
    for source, sha in sources.items():
        states[source] = checked_pin({"path": str(ROOT / source), "sha256": sha})
    for name, value in context["contract"]["inputs"].items():
        states["input:" + name] = checked_pin(value)
    allocation = plan.get("allocation", {})
    require(set(allocation) == {"nodes", "gpus", "cpus", "memory", "walltime_seconds"} and
            all(allocation.get(k) == v for k, v in {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "480G"}.items()),
            "wrong S3 allocation")
    for key in ("run_dir", "receipt_dir"):
        p = Path(plan[key])
        require(p.is_absolute() and p.resolve() == p and not p.is_symlink() and
                not any(p.is_relative_to(root) for root in (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"),
                                                         Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))),
                "canonical output outside code required")
    a, b = (Path(plan[k]) for k in ("run_dir", "receipt_dir"))
    require(not a.is_relative_to(b) and not b.is_relative_to(a), "technical run and receipts must be disjoint")
    estimate = resource_estimate(read_json(states["preflight"]["path"]), context)
    require(allocation["walltime_seconds"] >= estimate["walltime_seconds"], "S3 allocation below measured estimate")
    cfg, argv = recipe(context, a, allocation["walltime_seconds"])
    return plan, context, cfg, argv, states, code, estimate


def inspect_evidence(plan, artifacts, ranks, expected_config, contract_sha, code, job):
    """Pure evidence verifier: no GPU, no trainer call and no trusted coverage flags."""
    require(set(artifacts) == set(ARTIFACT_NAMES) and isinstance(ranks, list) and len(ranks) == 4,
            "complete production evidence and four rank files required")
    require(digest(code, 40) and str(job).isdigit() and int(job) > 0, "real code/job identity required")
    require(plan.get("protocol") == PLAN_PROTOCOL and plan.get("arm") == ARM and plan.get("steps") == STEPS and
            plan.get("technical_only") is True and plan.get("source_contract", {}).get("sha256") == contract_sha,
            "evidence plan identity differs")
    root, receipts = Path(plan["run_dir"]), Path(plan["receipt_dir"])
    require(root.is_absolute() and receipts.is_absolute() and root.resolve() == root and receipts.resolve() == receipts and
            not root.is_relative_to(receipts) and not receipts.is_relative_to(root), "invalid evidence directories")
    states = {}
    states["contract"] = checked_pin(plan["source_contract"])
    states["runtime_reference"] = checked_pin(plan.get("reference_runtime_summary"))
    reference = read_json(states["runtime_reference"]["path"])
    contract = read_json(states["contract"]["path"])
    stats_sha = contract["inputs"]["pose_stats"]["sha256"]
    for key, filename in ARTIFACT_NAMES.items():
        require(artifacts[key]["path"] == str(root / filename), "artifact outside registered technical run")
        states[key] = checked_pin(artifacts[key])
    require(not (root / "STOP").exists() and not list(root.glob("checkpoint_step*.pt")) and
            not list(root.glob("checkpoint_*decay.pt")), "STOP/scientific checkpoint present")
    data = {key: read_json(states[key]["path"], key in ("metrics", "fixed_eval", "pose_panel"))
            for key in ARTIFACT_NAMES if key != "technical_checkpoint"}
    config = data["config"]
    expected = dict(expected_config)
    expected["max_steps"] = STEPS
    require(config == expected and config["batch_size"] == 32 and config["accum_steps"] == 1 and
            config["prefetch"] is True and all(config[k] == 500 for k in
            ("eval_every", "pose_grad_every", "fixed_eval_every", "pose_panel_every", "ckpt_every")),
            "production config/cadence differs")
    marker = data["technical_marker"]
    require(marker == {"protocol": PROTOCOL, "technical_only": True, "fresh": True, "start_step": 0,
                       "steps": STEPS, "training_source_contract_sha256": contract_sha, "code_sha": code,
                       "slurm_job_id": str(job), "plan_sha256": canonical_hash(plan)}, "missing fresh technical-run marker")
    summary = data["summary"]
    required = {"stopped": "max_steps", "steps": STEPS, "world_size": 4, "batch_per_process": 8,
                "parameters": 292054847, "prefetch": True, "accum_steps": 1, "micro_batch": 8}
    require(all(summary.get(k) == v for k, v in required.items()) and summary.get("code", {}).get("sha") == code and
            summary.get("pose_stats_sha256") == stats_sha,
            "incomplete/wrong production summary")
    require(positive(summary.get("elapsed_s")) and positive(summary.get("gpu_max_memory_gib")), "missing real timing/memory")
    metrics = data["metrics"]
    require(len(metrics) == STEPS and [r.get("step") for r in metrics] == list(range(1, STEPS + 1)) and
            all(r.get("windows") == 32 and all(type(r.get(k)) in (int, float) and math.isfinite(r[k])
                for k in ("total", "pose", "grad_norm", "lr", "t_step_s")) for r in metrics),
            "missing contiguous 500 global updates")
    diag = ("shared_grad_pose", "shared_grad_anchors", "shared_grad_cos", "student_erank", "teacher_erank",
            "student_collapse_ratio", "teacher_collapse_ratio")
    require(all(type(metrics[-1].get(k)) in (int, float) and math.isfinite(metrics[-1][k]) for k in diag),
            "step500 production diagnostics not exercised")
    fixed, panel = data["fixed_eval"], data["pose_panel"]
    require([r.get("step") for r in fixed] == [0, STEPS] and all(r.get("batches") == 8 and
            type(r.get("total")) in (int, float) and math.isfinite(r["total"]) for r in fixed),
            "fixed evaluation must run at0 and500")
    fixed_set, pose_set = data["fixed_set"], data["pose_set"]
    require(fixed_set.get("seed") == [config["seed"], R.FIXED_EVAL_SEED] and
            len(fixed_set.get("windows", [])) == 8 * 32 and len(fixed_set.get("batch_sha256", [])) == 8 and
            all(isinstance(w, list) and len(w) == 4 for w in fixed_set["windows"]) and
            all(digest(h) for h in fixed_set["batch_sha256"]),
            "fixed evaluation support differs")
    n = config["pose_panel_users"] * config["pose_panel_recordings"] * config["pose_panel_windows"]
    require(n == 256 and len(pose_set.get("users", [])) == len(set(pose_set.get("users", []))) == 32 and
            len(pose_set.get("recordings", [])) == len(pose_set.get("batch_sha256", [])) == 64 and
            len(pose_set.get("windows", [])) == len(pose_set.get("window_users", [])) == n and
            all(digest(h) for h in pose_set["batch_sha256"]) and pose_set.get("pose_stats_sha256") == stats_sha,
            "pose panel support differs")
    ph = canonical_hash(pose_set)
    require([r.get("step") for r in panel] == [0, STEPS] and all(r.get("panel_sha256") == ph and
            r.get("n_windows") == r.get("n_windows_valid") == n and r.get("n_users") == 32 and
            len(r.get("per_window", [])) == n and positive(r.get("elapsed_s")) for r in panel),
            "pose panel must run on complete support at0 and500")
    for row in panel:
        require([w.get("user") for w in row["per_window"]] == pose_set["window_users"] and
                [w.get("window") for w in row["per_window"]] == [w[1:] for w in pose_set["windows"]],
                "pose panel rows differ from pinned set")
    seen, peaks, totals, reserved, runtime_by_rank = set(), [], [], [], []
    for i, item in enumerate(ranks):
        require(item["path"] == str(receipts / (f"rank_{i}.json")), "rank evidence path differs")
        states[f"rank_{i}"] = checked_pin(item)
        record = read_json(item["path"])
        require(record.get("protocol") == RANK_PROTOCOL and record.get("rank") == record.get("local_rank") == i and
                record.get("world_size") == 4 and record.get("worker_returned") is True and record.get("returncode") == 0 and
                record.get("code_sha") == code and record.get("slurm_job_id") == str(job) and
                record.get("training_source_contract_sha256") == contract_sha and
                record.get("run_dir") == str(root) and record.get("plan_sha256") == canonical_hash(plan),
                "rank identity/completion differs")
        require(record.get("rank") not in seen, "duplicate rank")
        seen.add(record["rank"])
        values = [record.get(k) for k in ("peak_allocated_bytes", "device_memory_bytes", "peak_reserved_bytes")]
        require(all(type(x) is int and x > 0 for x in values) and values[0] <= .9 * values[1] and
                values[2] >= values[0], "production memory gate failed")
        peaks.append(values[0]); totals.append(values[1]); reserved.append(values[2])
        runtime_comparison(record.get("runtime"), reference)
        runtime_by_rank.append(record["runtime"])
    runtime = runtime_by_rank[0]
    require(all(value == runtime for value in runtime_by_rank) and
            all(summary.get("environment", {}).get(k) == runtime[k] for k in RUNTIME_FIELDS),
            "rank/summary runtime differs")
    # Production summary reduces FP32 GiB before teardown. Rank peaks can include
    # the tiny final collective allocation, but must reconcile within1MiB.
    require(abs(summary["gpu_max_memory_gib"] * 2**30 - max(peaks)) <= 2**20, "summary/rank peaks do not reconcile")
    P.check_unchanged(states)
    return {"config": config, "world_size": 4, "global_batch_size": 32, "accum_steps": 1,
            "prefetch_exercised": True, "fixed_eval_exercised": True, "pose_panel_exercised": True,
            "pose_gradient_diagnostic_exercised": True, "steps": STEPS, "start_step": 0,
            "peak_allocated_bytes_by_rank": peaks, "device_memory_bytes_by_rank": totals,
            "peak_reserved_bytes_by_rank": reserved, "elapsed_s": summary["elapsed_s"],
            "runtime": runtime, "runtime_comparison": runtime_comparison(runtime, reference),
            "reference_runtime_summary": plan["reference_runtime_summary"],
            "coverage_steps": {"fixed_eval": [0, STEPS], "pose_panel": [0, STEPS], "gradient_diagnostics": [STEPS]}}


def validate_report(report, expected_config, contract_sha):
    require(report.get("protocol") == PROTOCOL and report.get("complete") is True and report.get("arm") == ARM and
            report.get("technical_only") is True and report.get("scientific_checkpoints_written") is False and
            report.get("training_source_contract_sha256") == contract_sha, "production report incomplete/wrong contract")
    plan_state = checked_pin(report.get("plan"))
    plan = read_json(plan_state["path"])
    require(set(plan.get("execution_source_sha256", {})) == EXECUTION_SOURCES, "missing evidence producer source pins")
    source_states = {p: checked_pin({"path": str(ROOT / p), "sha256": sha})
                     for p, sha in plan["execution_source_sha256"].items()}
    got = inspect_evidence(plan, report.get("artifacts", {}), report.get("ranks"), expected_config,
                           contract_sha, report.get("code_sha"), report.get("slurm_job_id"))
    require(all(report.get(k) == value for k, value in got.items()), "report claims differ from pinned production evidence")
    P.check_unchanged({"plan": plan_state, **source_states})
    return max(got["peak_allocated_bytes_by_rank"])


def worker(plan_path, expected_code, launch_sha):
    plan, context, cfg, argv, states, code, _ = load_plan(plan_path, expected_code)
    rank, local, world = (int(os.environ.get(k, "-1")) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 4 and 0 <= rank < 4 and local == rank, "four single-node ranks required")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and int(job) > 0, "real SLURM identity required")
    receipts, run_dir = Path(plan["receipt_dir"]), Path(plan["run_dir"])
    launch = checked_pin({"path": str(receipts / "launch.json"), "sha256": launch_sha})
    record = read_json(launch["path"])
    require(record["plan_sha256"] == states["plan"]["sha256"] and record["code_sha"] == code and
            record["slurm_job_id"] == job and record["trainer_argv"] == argv, "worker launch identity differs")
    require((run_dir / "TECHNICAL_ONLY.json").is_file() and not list(run_dir.glob("*.pt")) and not (run_dir / "STOP").exists(),
            "worker cannot resume a technical or scientific checkpoint")
    torch.cuda.set_device(local)
    torch.cuda.reset_peak_memory_stats(local)
    rc = module("train_jepa").main(argv)  # unchanged entry point owns DDP lifecycle
    torch.cuda.synchronize(local)
    P.check_unchanged(states)
    P.code_identity(code)
    result = {"protocol": RANK_PROTOCOL, "rank": rank, "local_rank": local, "world_size": world,
              "worker_returned": True, "returncode": rc, "code_sha": code, "slurm_job_id": job,
              "training_source_contract_sha256": states["source_contract"]["sha256"], "run_dir": str(run_dir),
              "plan_sha256": canonical_hash(plan),
              "runtime": runtime_values(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(local),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(local),
              "device_memory_bytes": torch.cuda.get_device_properties(local).total_memory}
    with (receipts / f"rank_{rank}.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    require(rc == 0, "trainer failed")


def execute(plan_path, expected_code):
    plan, context, cfg, argv, states, code, estimate = load_plan(plan_path, expected_code)
    module("scaling_train").verify_allocation(plan)
    run_dir, out = Path(plan["run_dir"]), Path(plan["receipt_dir"])
    require(not run_dir.exists() and not out.exists(), "fresh S3 outputs required; no resume/overwrite")
    job, sha = os.environ["SLURM_JOB_ID"], states["source_contract"]["sha256"]
    run_dir.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    marker = {"protocol": PROTOCOL, "technical_only": True, "fresh": True, "start_step": 0,
              "steps": STEPS, "training_source_contract_sha256": sha, "code_sha": code, "slurm_job_id": job,
              "plan_sha256": canonical_hash(plan)}
    P.write_json(run_dir / "TECHNICAL_ONLY.json", marker)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "pip_freeze_s3.txt").write_text(freeze)
    launch = {"protocol": PLAN_PROTOCOL, "plan_sha256": states["plan"]["sha256"], "code_sha": code,
              "slurm_job_id": job, "trainer_argv": argv, "resources": estimate, "verified_inputs": states}
    P.write_json(out / "launch.json", launch)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(ROOT / "scripts/scaling_production_memory.py"), "worker", "--plan", str(Path(plan_path).resolve()),
               "--expected-code-sha", code, "--launch-sha256", R.sha256_file(out / "launch.json")]
    with (out / "training.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "WEARUSFM_CODE_SHA": code, "WEARUSFM_CODE_DIR": str(ROOT)},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    artifacts = {key: pin(run_dir / name) for key, name in ARTIFACT_NAMES.items()}
    ranks = [pin(out / f"rank_{i}.json") for i in range(4)]
    evidence = inspect_evidence(plan, artifacts, ranks, R.config_to_dict(context["config_300m"]), sha, code, job)
    report = {"protocol": PROTOCOL, "complete": True, "arm": ARM, "technical_only": True,
              "scientific_checkpoints_written": False, "training_source_contract_sha256": sha,
              "code_sha": code, "slurm_job_id": job, "plan": pin(plan_path), "artifacts": artifacts,
              "ranks": ranks, **evidence}
    validate_report(report, R.config_to_dict(context["config_300m"]), sha)
    P.check_unchanged(states)
    P.code_identity(code)
    P.write_json(out / "report.json", report)  # final complete attestation, never before evidence
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--launch-sha256")
    args = parser.parse_args(argv)
    if args.mode == "run":
        execute(args.plan, args.expected_code_sha)
    else:
        worker(args.plan, args.expected_code_sha, args.launch_sha256)


if __name__ == "__main__":
    main()
