#!/usr/bin/env python3
"""Independent, stdlib-only audit of collected S3 evidence (no checkpoint load).

Run from any directory: python3 verify_collected.py [--directory DIRECTORY]
The successful JSON audit is printed to stdout; no files are written. Missing
or inconsistent evidence raises ValueError and never emits a successful audit.

All 15 copied remote files are basenames in DIRECTORY. collected_hashes.json:
  {"protocol":"scaling-s3-collected-hashes-20261104-v1",
   "job_id":"63527446", "code_sha":CODE,
   "files":{REMOTE_ABSOLUTE_PATH:{"sha256":HEX64,"bytes":INT,
                                   "local_file":"basename"}, ...}}
accounting.json (values returned as sacct strings or normalized integers):
  {"protocol":"scaling-s3-accounting-20261104-v1", "job_id":"63527446",
   "collected_at_utc":"ISO8601", "sacct_rows":[
     {"JobIDRaw":"63527446", "State":"COMPLETED", "ExitCode":"0:0",
      "ElapsedRaw":SECONDS, "AllocCPUS":32, "NNodes":1,
      "AllocTRES":"cpu=32,gres/gpu=4,mem=480G,node=1", "MaxRSS":""},
     ...optional .batch/.extern rows...]}
The main job is required and all supplied job steps must be completed 0:0.

This is intentionally NOT scaling_production_memory.validate_report: it checks
the small evidence copies independently, records the checkpoint's producer pin
without accessing it, and leaves the full guard to the producer and the future
scientific wrapper on a compute node. No new scientific performance criterion.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "results/infra/scaling_300m_20261104"
CODE = "0d5f0f90074bbaf1142be86d22490e360b632c63"
JOB = "63527446"
ARM = "T100-G-300M"
CONTRACT_SHA = "14b044de1df7be3e55c836685342d94ae48f9d072dc6784b33cca197dab47df1"
PLAN_SHA = "8a3b85416609bde2a684f04a9c73e94fdce03bb30f9b7ea30c72d6ef928f5351"
POLICY_SHA = "4f0f626e06cccf67d721ddcf6da14be0d5b822222d25538caf42322f721f9416"
PROTOCOL = "scaling-production-memory-20261104-v1"
PLAN_PROTOCOL = "scaling-production-memory-plan-20261104-v1"
RANK_PROTOCOL = "scaling-production-memory-rank-20261104-v1"
REMOTE_CODE = "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_memory_20261104_v1"
ARTIFACTS = {"config": "config.json", "summary": "summary.json", "metrics": "metrics.jsonl",
             "fixed_eval": "fixed_eval.jsonl", "fixed_set": "fixed_eval_set.json",
             "pose_panel": "pose_panel.jsonl", "pose_set": "pose_panel_set.json",
             "technical_marker": "TECHNICAL_ONLY.json", "technical_checkpoint": "checkpoint.pt"}
RUNTIME_FIELDS = ("python", "torch", "cuda")
DIAGNOSTICS = ("shared_grad_pose", "shared_grad_anchors", "shared_grad_cos", "student_erank",
               "teacher_erank", "student_collapse_ratio", "teacher_collapse_ratio")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, length=64):
    return (isinstance(value, str) and len(value) == length and value != "0" * length
            and all(c in "0123456789abcdef" for c in value))


def number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON evidence")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read(path, lines=False):
    text = Path(path).read_text()
    value = ([json.loads(line, object_pairs_hook=unique_object) for line in text.splitlines() if line.strip()]
             if lines else json.loads(text, object_pairs_hook=unique_object))
    finite_tree(value)
    return value


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular evidence: {path}")
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def same_pin(actual, expected, label):
    require(isinstance(actual, dict) and digest(actual.get("sha256")), f"invalid pin: {label}")
    require(actual["sha256"] == expected["sha256"], f"SHA mismatch: {label}")
    if "bytes" in expected:
        require(actual.get("bytes") == expected["bytes"], f"size mismatch: {label}")
    if "path" in expected:
        require(actual.get("path") == expected["path"], f"path mismatch: {label}")


def integer(value, label):
    require(type(value) is int or isinstance(value, str) and value.isdigit(), f"invalid integer: {label}")
    return int(value)


def check_accounting(accounting, elapsed):
    require(accounting.get("protocol") == "scaling-s3-accounting-20261104-v1" and
            accounting.get("job_id") == JOB, "accounting identity differs")
    stamp = datetime.fromisoformat(accounting["collected_at_utc"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "UTC accounting timestamp required")
    rows = accounting.get("sacct_rows")
    require(isinstance(rows, list) and rows, "sacct evidence missing")
    ids = [str(row.get("JobIDRaw")) for row in rows]
    require(len(ids) == len(set(ids)) and ids.count(JOB) == 1 and
            all(job == JOB or job.startswith(JOB + ".") for job in ids), "wrong/duplicate sacct job")
    require(all(row.get("State") == "COMPLETED" and row.get("ExitCode") == "0:0" for row in rows),
            "job or supplied step not COMPLETED 0:0")
    job = rows[ids.index(JOB)]
    seconds = integer(job.get("ElapsedRaw"), "ElapsedRaw")
    require(0 < seconds <= 3600 and elapsed <= seconds + 1, "job elapsed/summary exceeds allocation or differs")
    require(integer(job.get("AllocCPUS"), "AllocCPUS") == 32 and
            integer(job.get("NNodes"), "NNodes") == 1, "wrong observed CPU/node allocation")
    tres = dict(part.split("=", 1) for part in job.get("AllocTRES", "").split(",") if "=" in part)
    require(integer(tres.get("cpu"), "TRES cpu") == 32 and
            integer(tres.get("gres/gpu"), "TRES gpu") == 4, "wrong observed CPU/GPU TRES")
    return {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": seconds,
            "gpus": 4, "cpus": 32, "nodes": 1, "gpu_hours": seconds * 4 / 3600,
            "cpu_core_hours": seconds * 32 / 3600, "max_rss_by_sacct_row":
            {str(row["JobIDRaw"]): row.get("MaxRSS") for row in rows}}


def check_runtime(runtime, reference):
    require(isinstance(runtime, dict) and set(runtime) == set(RUNTIME_FIELDS) | {"cudnn"} and
            all(isinstance(runtime[k], str) and runtime[k] for k in RUNTIME_FIELDS) and
            type(runtime["cudnn"]) is int and runtime["cudnn"] > 0, "invalid CUDA runtime evidence")
    require(reference.get("stopped") == "max_steps" and reference.get("steps") == 5000 and
            reference.get("parameters") == 28362013 and reference.get("world_size") == 4 and
            reference.get("code", {}).get("sha") == "afb644ea10ecc2a716160b4d8a4904e805c366fb",
            "wrong historical runtime reference")
    fields = list(RUNTIME_FIELDS)
    if reference["environment"].get("cudnn") is not None:
        fields.append("cudnn")
    require(all(runtime[k] == reference["environment"][k] for k in fields), "runtime differs from historical control")
    return {"matched_fields": fields, "unavailable_in_reference": [] if "cudnn" in fields else ["cudnn"],
            "reference": {k: reference["environment"][k] for k in fields}}


def verify(directory):
    directory = Path(directory).resolve()
    # Local immutable plan/contracts are already present: no remote path access.
    plan_path, contract_path = BASE / "production_memory_plan.json", BASE / "training_source_contract.json"
    same_pin(identity(plan_path), {"sha256": PLAN_SHA}, "registered plan")
    same_pin(identity(contract_path), {"sha256": CONTRACT_SHA}, "registered source contract")
    policy_path = BASE / "production_memory_resource_policy.json"
    same_pin(identity(policy_path), {"sha256": POLICY_SHA}, "registered resource policy")
    plan, contract, policy = read(plan_path), read(contract_path), read(policy_path)
    require(plan["protocol"] == PLAN_PROTOCOL and plan["arm"] == ARM and plan["steps"] == 500 and
            plan["technical_only"] is True and plan["source_contract"]["sha256"] == CONTRACT_SHA,
            "registered plan identity differs")
    source_pins = {**contract["scientific_source_sha256"], **plan["execution_source_sha256"]}
    source_states = {}
    for source, sha in source_pins.items():
        source_states[str(ROOT / source)] = identity(ROOT / source)
        same_pin(source_states[str(ROOT / source)], {"sha256": sha}, source)
    baseline_path = ROOT / contract["config_30m"]["path"]
    same_pin(identity(baseline_path), {"sha256": contract["config_30m"]["sha256"]}, "historical config")
    expected_config = copy.deepcopy(read(baseline_path))
    expected_config["model"].update(dim=896, n_heads=14, local_layers=3, backbone_layers=18, decoder_layers=2)
    expected_config["max_steps"] = 500
    reference_path = ROOT / "results/infra/tokenizer_execution_20261103/5000/T100-G/summary.json"
    same_pin(identity(reference_path), {k: v for k, v in plan["reference_runtime_summary"].items() if k != "path"},
             "historical runtime summary")
    reference = read(reference_path)
    remote_run, remote_out = plan["run_dir"], plan["receipt_dir"]
    expected_files = {str(Path(remote_run) / name): name for key, name in ARTIFACTS.items()
                      if key != "technical_checkpoint"}
    expected_files.update({str(Path(remote_out) / name): name
                           for name in ["report.json", "launch.json", "training.log"] +
                           [f"rank_{rank}.json" for rank in range(4)]})
    manifest_path = directory / "collected_hashes.json"
    manifest_state = identity(manifest_path)
    manifest = read(manifest_path)
    require(manifest.get("protocol") == "scaling-s3-collected-hashes-20261104-v1" and
            manifest.get("job_id") == JOB and manifest.get("code_sha") == CODE and
            set(manifest.get("files", {})) == set(expected_files), "collection manifest identity or file set differs")
    checked = {}
    for remote, basename in expected_files.items():
        item = manifest["files"][remote]
        require(item.get("local_file") == basename and type(item.get("bytes")) is int and item["bytes"] > 0,
                f"invalid collected filename/size: {remote}")
        measured = identity(directory / basename)
        same_pin(measured, item, basename)
        checked[remote] = measured
    # Protect metadata and every evidence byte against a concurrent rewrite.
    snapshot = {str(directory / name): checked[remote] for remote, name in expected_files.items()}
    snapshot[str(manifest_path)] = manifest_state
    snapshot.update({str(path): identity(path) for path in (directory / "accounting.json",
                     plan_path, contract_path, policy_path, baseline_path, reference_path)})
    snapshot.update(source_states)
    report, launch = read(directory / "report.json"), read(directory / "launch.json")
    require(report.get("protocol") == PROTOCOL and report.get("complete") is True and
            report.get("arm") == ARM and report.get("technical_only") is True and
            report.get("scientific_checkpoints_written") is False and report.get("code_sha") == CODE and
            report.get("slurm_job_id") == JOB and report.get("training_source_contract_sha256") == CONTRACT_SHA,
            "report identity/completion differs")
    same_pin(report.get("plan"), {"path": REMOTE_CODE + "/results/infra/scaling_300m_20261104/production_memory_plan.json",
                                 **identity(plan_path)}, "report plan")
    require(set(report.get("artifacts", {})) == set(ARTIFACTS), "report artifact set differs")
    for key, name in ARTIFACTS.items():
        item = report["artifacts"][key]
        remote = str(Path(remote_run) / name)
        require(item.get("path") == remote and digest(item.get("sha256")) and
                type(item.get("bytes")) is int and item["bytes"] > 0, f"invalid artifact pin: {key}")
        if key != "technical_checkpoint":
            same_pin(checked[remote], {k: item[k] for k in ("sha256", "bytes")}, key)
    require(not (directory / "checkpoint.pt").exists(), "technical checkpoint is outside this collected audit")
    require(isinstance(report.get("ranks"), list) and len(report["ranks"]) == 4, "four rank pins required")
    for rank, item in enumerate(report["ranks"]):
        remote = str(Path(remote_out) / f"rank_{rank}.json")
        same_pin(item, {"path": remote, **checked[remote]}, f"rank {rank}")
    comparison_path = ROOT / contract["comparison"]["path"]
    same_pin(identity(comparison_path), {"sha256": contract["comparison"]["sha256"]}, "recipe comparison")
    snapshot[str(comparison_path)] = identity(comparison_path)
    argv = list(read(comparison_path)["arm_configurations"]["T100-G"]["validated_trainer_argv"])
    require(argv.pop(0) == "scripts/train_jepa.py", "wrong trainer entry point")
    for key, value in (("--preset", "300m"), ("--out-dir", remote_run), ("--max-steps", "500")):
        require(argv.count(key) == 1, "registered recipe argument missing/duplicate")
        argv[argv.index(key) + 1] = value
    argv += ["--time-limit-s", "3300"]
    require(launch.get("protocol") == PLAN_PROTOCOL and launch.get("code_sha") == CODE and
            launch.get("slurm_job_id") == JOB and launch.get("plan_sha256") == PLAN_SHA and
            launch.get("trainer_argv") == argv and launch.get("resources") == policy["estimate"],
            "launch recipe/identity/resources differs")
    verified = launch.get("verified_inputs", {})
    same_pin(verified.get("plan"), report["plan"], "launch plan")
    for key in ("source_contract", "preflight", "reference_runtime_summary"):
        same_pin(verified.get(key), plan[key], "launch " + key)
    for key, item in contract["inputs"].items():
        same_pin(verified.get("input:" + key), item, "launch input " + key)
    for key, sha in plan["execution_source_sha256"].items():
        same_pin(verified.get(key), {"path": REMOTE_CODE + "/" + key, "sha256": sha}, "launch source " + key)
    data = {key: read(directory / name, key in ("metrics", "fixed_eval", "pose_panel"))
            for key, name in ARTIFACTS.items() if key != "technical_checkpoint"}
    config, summary, metrics = data["config"], data["summary"], data["metrics"]
    require(config == expected_config, "production config differs from registered 300M recipe")
    require(config["batch_size"] == 32 and config["accum_steps"] == 1 and config["prefetch"] is True and
            all(config[key] == 500 for key in ("eval_every", "pose_grad_every", "fixed_eval_every", "pose_panel_every", "ckpt_every")),
            "production dose/batch/cadence differs")
    plan_canonical_sha = canonical_hash(plan)
    marker = {"protocol": PROTOCOL, "technical_only": True, "fresh": True, "start_step": 0, "steps": 500,
              "training_source_contract_sha256": CONTRACT_SHA, "code_sha": CODE,
              "slurm_job_id": JOB, "plan_sha256": plan_canonical_sha}
    require(data["technical_marker"] == marker, "fresh technical marker differs")
    stats_sha = contract["inputs"]["pose_stats"]["sha256"]
    required_summary = {"stopped": "max_steps", "steps": 500, "world_size": 4, "batch_per_process": 8,
                        "parameters": 292054847, "prefetch": True, "accum_steps": 1, "micro_batch": 8}
    require(all(summary.get(k) == v for k, v in required_summary.items()) and
            summary.get("code", {}).get("sha") == CODE and summary.get("code", {}).get("dir") == REMOTE_CODE and
            summary.get("pose_stats_sha256") == stats_sha and number(summary.get("elapsed_s"), True) and
            number(summary.get("gpu_max_memory_gib"), True), "production summary differs/incomplete")
    require(len(metrics) == 500 and [row.get("step") for row in metrics] == list(range(1, 501)) and
            all(row.get("windows") == 32 and all(number(row.get(k)) for k in
                ("total", "pose", "grad_norm", "lr", "t_step_s")) for row in metrics), "contiguous 500 global updates missing")
    for field in DIAGNOSTICS:
        require([row["step"] for row in metrics if field in row] == [500] and number(metrics[-1][field]),
                "diagnostic cadence/evidence differs: " + field)
    fixed, panel, fixed_set, pose_set = (data[k] for k in ("fixed_eval", "pose_panel", "fixed_set", "pose_set"))
    require([row.get("step") for row in fixed] == [0, 500] and all(row.get("batches") == 8 and
            number(row.get("total")) for row in fixed), "fixed evaluation incomplete")
    require(fixed_set.get("seed") == [config["seed"], 5151] and len(fixed_set.get("windows", [])) == 256 and
            len(fixed_set.get("batch_sha256", [])) == 8 and
            all(isinstance(w, list) and len(w) == 4 for w in fixed_set["windows"]) and
            all(digest(sha) for sha in fixed_set["batch_sha256"]), "fixed support differs")
    n = config["pose_panel_users"] * config["pose_panel_recordings"] * config["pose_panel_windows"]
    require(n == 256 and len(pose_set.get("users", [])) == len(set(pose_set.get("users", []))) == 32 and
            len(pose_set.get("recordings", [])) == len(pose_set.get("batch_sha256", [])) == 64 and
            len(pose_set.get("windows", [])) == len(pose_set.get("window_users", [])) == n and
            all(digest(sha) for sha in pose_set["batch_sha256"]) and pose_set.get("pose_stats_sha256") == stats_sha,
            "pose panel support differs")
    require([row.get("step") for row in panel] == [0, 500] and all(row.get("panel_sha256") == canonical_hash(pose_set) and
            row.get("n_windows") == row.get("n_windows_valid") == n and row.get("n_users") == 32 and
            len(row.get("per_window", [])) == n and number(row.get("elapsed_s"), True) for row in panel),
            "pose panel incomplete at 0/500")
    for row in panel:
        require([w.get("user") for w in row["per_window"]] == pose_set["window_users"] and
                [w.get("window") for w in row["per_window"]] == [w[1:] for w in pose_set["windows"]],
                "pose panel rows differ from fixed support")
    ranks = [read(directory / f"rank_{rank}.json") for rank in range(4)]
    for i, rank in enumerate(ranks):
        require(rank.get("protocol") == RANK_PROTOCOL and rank.get("rank") == rank.get("local_rank") == i and
                rank.get("world_size") == 4 and rank.get("worker_returned") is True and rank.get("returncode") == 0 and
                rank.get("code_sha") == CODE and rank.get("slurm_job_id") == JOB and
                rank.get("training_source_contract_sha256") == CONTRACT_SHA and rank.get("run_dir") == remote_run and
                rank.get("plan_sha256") == plan_canonical_sha, "rank identity/completion differs")
        peak, total, reserved = (rank.get(k) for k in ("peak_allocated_bytes", "device_memory_bytes", "peak_reserved_bytes"))
        require(all(type(v) is int and v > 0 for v in (peak, total, reserved)) and
                peak <= .9 * total and reserved >= peak, "rank memory gate failed")
        check_runtime(rank.get("runtime"), reference)
    runtime = ranks[0]["runtime"]
    require(all(rank["runtime"] == runtime for rank in ranks) and
            all(summary.get("environment", {}).get(k) == runtime[k] for k in RUNTIME_FIELDS), "rank/summary runtime differs")
    peaks = [rank["peak_allocated_bytes"] for rank in ranks]
    require(abs(summary["gpu_max_memory_gib"] * 2**30 - max(peaks)) <= 2**20,
            "summary/rank memory peaks do not reconcile within 1 MiB")
    evidence = {"config": config, "world_size": 4, "global_batch_size": 32, "accum_steps": 1,
                "prefetch_exercised": True, "fixed_eval_exercised": True, "pose_panel_exercised": True,
                "pose_gradient_diagnostic_exercised": True, "steps": 500, "start_step": 0,
                "peak_allocated_bytes_by_rank": peaks,
                "device_memory_bytes_by_rank": [rank["device_memory_bytes"] for rank in ranks],
                "peak_reserved_bytes_by_rank": [rank["peak_reserved_bytes"] for rank in ranks],
                "elapsed_s": summary["elapsed_s"], "runtime": runtime,
                "runtime_comparison": check_runtime(runtime, reference),
                "reference_runtime_summary": plan["reference_runtime_summary"],
                "coverage_steps": {"fixed_eval": [0, 500], "pose_panel": [0, 500], "gradient_diagnostics": [500]}}
    require(all(report.get(k) == value for k, value in evidence.items()), "report claims differ from collected evidence")
    accounting = check_accounting(read(directory / "accounting.json"), summary["elapsed_s"])
    for path, state in snapshot.items():
        same_pin(identity(path), state, "changed during audit: " + path)
    return {"protocol": "scaling-s3-independent-collected-audit-20261104-v1", "complete": True,
            "scope": "independent audit of small collected evidence; not full validate_report",
            "job_id": JOB, "code_sha": CODE, "training_source_contract_sha256": CONTRACT_SHA,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "verifier": {"path": str(Path(__file__).resolve().relative_to(ROOT)), **identity(__file__)},
            "collection_manifest": identity(manifest_path), "accounting": identity(directory / "accounting.json"),
            "report": identity(directory / "report.json"), "files_verified": len(checked),
            "source_files_verified": len(source_pins), "steps": 500, "global_windows_per_update": 32,
            "global_windows_processed": 16000, "ranks": 4, "coverage_steps": evidence["coverage_steps"],
            "production_memory_gate_passed": True, "peak_allocated_gib_by_rank": [p / 2**30 for p in peaks],
            "memory_fraction_by_rank": [r["peak_allocated_bytes"] / r["device_memory_bytes"] for r in ranks],
            "max_memory_fraction_allowed": .9, "summary_elapsed_seconds": summary["elapsed_s"],
            "runtime": runtime, "runtime_comparison": evidence["runtime_comparison"],
            "observed_allocation_and_cost": accounting,
            "technical_checkpoint": {"producer_pin": report["artifacts"]["technical_checkpoint"],
                                      "downloaded": False, "locally_hashed": False, "locally_loaded": False},
            "full_validate_report_local": False,
            "remaining_checks": ["Scientific wrapper repeats full validate_report on its compute node, including remote checkpoint hash.",
                                 "Remote absence of STOP/scientific checkpoints and checkpoint contents are not independently inspected here."],
            "authorizes_scientific_training": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
