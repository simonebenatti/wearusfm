"""Bind and run the four-GPU, two-reload diagnostic; this never releases training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import source_digest
from contrastive_release import checked_pin, file_sha, validate_review_binding
from contrastive_split_benchmark import checked_command

HERE = Path(__file__).resolve().parent
AREA = ROOT / "results/infra/contrastive_targets_20261118/cn_restart_probe_20261121"
PLAN = AREA / "plan.json"
BINDING = AREA / "release_binding.json"
REVIEW_PATH = "docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md"
REVIEW_REF = "refs/remotes/origin/worktree-claude-training-engine"
PROTOCOL = ROOT / "docs/contrastive_cn_restart_probe_20261121.md"
LAUNCHER = ROOT / "scripts/slurm/cn_restart_probe.sbatch"
LEG = HERE / "contrastive_cn_restart_leg.py"
COMPARE = HERE / "contrastive_cn_restart_compare.py"
FILES = (Path(__file__).resolve(), LEG, HERE / "contrastive_cn_restart_observe.py", COMPARE, LAUNCHER, PROTOCOL)
BLOCK = re.compile(r"^```cn-restart-review[ \t]*\n(.*?)\n```[ \t]*$", re.MULTILINE | re.DOTALL)


def write_new(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def package():
    plan = json.loads(PLAN.read_bytes())
    if (plan["schema"] != "cn-restart-probe-v1" or plan["status"] != "REVIEW_REQUIRED"
            or plan["training_approved"] is not False or plan["historical_status"] != "FAILED_REQUIRES_REVIEW"
            or source_digest() != plan["production_source_sha256"]):
        raise ValueError("Wrong scientific baseline or diagnostic plan")
    # The comparator also consumes archived rank traces and metrics. Pin their Git
    # trees to the published evidence rather than trusting the current working copy.
    evidence_paths = ("reloads_execution_20261121/CN_A", "benchmark_execution_20261121/CN",
                      "readout_execution_20261121/preflight_readout.json.gz")
    base = AREA.parent.relative_to(ROOT)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", plan["evidence_commit"], "--",
                    *[str(base / path) for path in evidence_paths]], check=True)
    manifest = {str(f.relative_to(ROOT)): file_sha(f) for f in FILES}
    manifest["production_source_sha256"] = plan["production_source_sha256"]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan, {"package_sha256": digest, "plan_sha256": file_sha(PLAN)}


def review_fields(body, pins):
    blocks = BLOCK.findall(body)
    if len(blocks) != 1:
        raise ValueError("Requires precisely one current cn-restart-review block")
    data = json.loads(blocks[0])
    commit = data.get("reviewed_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Full reviewed package commit required")
    if data != dict(schema="cn-restart-review-v1", status="APPROVED_FOR_DIAGNOSTIC",
                    reviewed_commit=commit, **pins):
        raise ValueError("Review does not approve this exact CN restart package")
    for path in (*FILES, PLAN):
        if git("show", f"{commit}:{path.relative_to(ROOT)}") != path.read_bytes():
            raise ValueError(f"File changed since Claude review: {path}")
    return data


def bind_review(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Full Claude review commit required")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, REVIEW_REF], check=True)
    body = git("show", f"{commit}:{REVIEW_PATH}")
    _, pins = package()
    review_fields(body.decode(), pins)
    write_new(BINDING, {"review_commit": commit, "original_review_path": REVIEW_PATH,
                        "review_sha256": hashlib.sha256(body).hexdigest(), "review_text": body.decode()})


def require_binding():
    plan, pins = package()
    if not BINDING.is_file():
        raise ValueError("New Claude review and published release_binding.json required")
    bound = json.loads(BINDING.read_bytes())
    body = bound["review_text"]
    if (bound["original_review_path"] != REVIEW_PATH or not re.fullmatch(r"[0-9a-f]{40}", bound["review_commit"])
            or hashlib.sha256(body.encode()).hexdigest() != bound["review_sha256"]):
        raise ValueError("Invalid review provenance")
    # bind_review authenticates the original on Claude's branch before freezing the
    # checkout. The deployment carries its exact bytes even on a single-branch remote.
    review_fields(body, pins)
    return plan


def commands(plan):
    job = json.loads(checked_pin(plan["historical_job"], repository=True).read_bytes())
    if job["arm"] != "CN" or job["repeat"] != "a" or job["fork_checkpoint"] != plan["fork"]:
        raise ValueError("Not the historical CN_A common-fork recipe")
    validate_review_binding(job["package_review"])
    for reference in ("fork", "ordinary_reference", "continuous_reference"):
        checked_pin(plan[reference])
    root = Path(plan["output"])
    if (root.parent != Path(os.environ["WORK"]).resolve() / "wearusfm_runs/results/contrastive_targets_20261118"
            or root.exists() or root.is_symlink()):
        raise ValueError("A fresh diagnostic output directory on WORK is required")
    common = ["--stage", "benchmark", "--time-limit-s", "1770"]
    # This authenticates the original runner contract. The entire diagnostic is
    # bounded separately by one 900 s allocation, including both workers and comparison.
    for key, flag in (("config", "--config"), ("release", "--release"), ("audit", "--audit-report"),
                      ("cohort", "--cohort"), ("calibration_report", "--calibration-report")):
        common += [flag, str(checked_pin(job[key]))]
    common += ["--calibration-run-dir", str(checked_pin(job["calibration_summary"]).parent)]
    for key in ("manifest", "scales", "window_seconds"):
        common += ["--" + key.replace("_", "-"), str(checked_pin(job["inputs"][key]))]
    checked_pin(job["inputs"]["pose_stats"])
    for path in job["roots"]:
        if not Path(path).is_dir():
            raise ValueError("Missing reviewed data root " + path)
        common += ["--root", path]
    cells = [[sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
              str(LEG), "--variant", variant, "--out-dir", str(root / variant), *common]
             for variant in plan["legs"]]
    compare = [sys.executable, str(COMPARE), "--ordinary", str(root / "ordinary"), "--warm", str(root / "warm"),
               "--historical-ordinary", str(Path(plan["ordinary_reference"]["path"]).parent),
               "--historical-continuous", str(Path(plan["continuous_reference"]["path"]).parent),
               "--report", str(root / "report.json")]
    return root, cells, compare, common


def run(check_only):
    plan = require_binding()
    start = time.monotonic()
    root, cells, compare, common = commands(plan)
    if check_only:
        # Parse both complete worker commands and the production runner before sbatch.
        from contrastive_cn_restart_leg import validate_argv
        for cell in cells:
            validate_argv(cell[cell.index(str(LEG)) + 1:], plan)
        subprocess.run([sys.executable, str(ROOT / "scripts/contrastive_run.py"), "--validate-only",
                        "--out-dir", str(root / "ordinary"), *common], check=True)
        print(json.dumps({"new_binding_valid": True, "leg_commands": cells, "compare": compare,
                          "execution_started": False}, indent=2))
        return
    from contrastive_job import validate_allocation
    validate_allocation("gpu")
    if os.environ.get("TORCH_DIST_INIT_BARRIER") != "1":
        raise ValueError("Missing production distributed barrier setting")
    root.mkdir(exist_ok=False)
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"], text=True)
    write_new(root / "allocation.json", {"node": socket.gethostname(), "gpu_inventory": gpu_info.splitlines(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "job_id": os.environ["SLURM_JOB_ID"],
              "execution_sha": os.environ["WEARUSFM_CODE_SHA"], "binding_sha256": file_sha(BINDING),
              "plan_sha256": file_sha(PLAN), "commands": cells, "comparison": compare})
    # Preserve a short tail for teardown and SLURM. Job walltime starts before Python.
    end_epoch = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(start + 875, time.monotonic() + end_epoch - time.time() - 25)
    phases = [*zip(plan["legs"], cells, strict=True), ("compare", compare)]
    completed = []
    with (root / "phase_timings.jsonl").open("x", buffering=1) as timings:
        for phase, command in phases:
            began = time.monotonic()
            timings.write(json.dumps({"phase": phase, "event": "start", "unix_s": time.time()}) + "\n")
            try:
                checked_command(command, deadline)
                if phase == "compare" and not (root / "report.json").is_file():
                    raise ValueError("Missing final comparison")
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                elapsed = time.monotonic() - began
                timings.write(json.dumps({"phase": phase, "event": "failed", "unix_s": time.time(),
                                          "elapsed_s": elapsed}) + "\n")
                write_new(root / "execution_failure.json", {"status": "INCONCLUSIVE", "phase": phase,
                          "completed_phases": completed, "elapsed_s": elapsed, "cause": str(exc),
                          "training_approved": False})
                raise
            timings.write(json.dumps({"phase": phase, "event": "completed", "unix_s": time.time(),
                                      "elapsed_s": time.monotonic() - began}) + "\n")
            completed.append(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--package", action="store_true")
    action.add_argument("--bind-review", metavar="REVIEW_COMMIT")
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.package:
        print(json.dumps(package()[1], indent=2))
    elif args.bind_review:
        bind_review(args.bind_review)
    else:
        run(args.validate_only)


if __name__ == "__main__":
    main()
