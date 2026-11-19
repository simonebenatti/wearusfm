"""Normal entry point: python -m pose_temponet COMMAND. No submission side effects."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from . import execution as E, provenance as P


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-plan")
    prepare.add_argument("--proposal", type=Path, default=Path("results/infra/pose_temponet_proposal_20261117/proposal.json"))
    prepare.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--plan", type=Path, required=True)
    release = commands.add_parser("release-plan", help="operator records the actual favorable operational review")
    release.add_argument("--plan", type=Path, required=True)
    release.add_argument("--review-commit", required=True)
    release.add_argument("--review-path", required=True)
    release.add_argument("--reviewed-code-sha", required=True)
    release.add_argument("--operator-checks", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    resources = commands.add_parser("resource-plan")
    resources.add_argument("--plan", type=Path, required=True)
    resources.add_argument("--cpu-receipt", type=Path, required=True)
    resources.add_argument("--gpu-receipt", type=Path, required=True)
    resources.add_argument("--operator-checks", type=Path, required=True,
                           help="fresh quota/lane checks taken after both benchmarks completed")
    resources.add_argument("--output", type=Path, required=True)
    materialize = commands.add_parser("materialize-plan")
    materialize.add_argument("--commit", required=True)
    materialize.add_argument("--repository-path", required=True)
    materialize.add_argument("--sha256", required=True)
    materialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--job-id", required=True)
    commands_parser = commands.add_parser("commands")
    commands_parser.add_argument("--plan", type=Path, required=True)
    commands_parser.add_argument("--checkout-sha", required=True)
    commands_parser.add_argument("--plan-commit", required=True)
    commands_parser.add_argument("--plan-repository-path", required=True)
    commands_parser.add_argument("--stage", action="append", required=True, choices=P.STAGES)
    commands_parser.add_argument("--completed-prerequisites", type=Path,
                                 help="pinned local receipt and sacct proofs for already completed predecessors")
    commands_parser.add_argument("--closed-reference-gate-output", type=Path,
                                 help="new local JSON artifact, required before generating controls commands")
    return p


def _write(path, value):
    path = path.absolute()
    return P.atomic_json(path, value, root=path.parent)


def _operator_checks(path):
    pin = P.file_pin(path.absolute())
    checks = P.read_json_pin(pin)
    for key in ("work_free_bytes", "pending_write_bytes"):
        P.require(type(checks.get(key)) is int and checks[key] >= 0, "explicit current quota accounting required")
    P.require(checks["work_free_bytes"] >= 12*1024**3+checks["pending_write_bytes"], "WORK quota gate fails")
    for key in ("no_duplicate_jobs", "serial_lane_checked", "reservations_checked", "balance_checked"):
        P.require(checks.get(key) is True, "operator check missing: "+key)
    _timestamp(checks.get("observed_at"), "operator evidence")
    return checks, pin


def _timestamp(value, name):
    P.require(isinstance(value, str) and bool(value), name+" needs a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(name+" has an invalid timestamp") from error
    P.require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
              name+" needs a timezone-aware timestamp")
    return parsed


def refresh_operator_checks(plan, operator_path, cpu_pin, gpu_pin):
    """Bind fresh accounting to authenticated benchmark completion times."""
    checks, evidence = _operator_checks(operator_path)
    observed = _timestamp(checks["observed_at"], "operator evidence")
    benchmarks = {}
    for stage, pin in (("cpu_benchmark", cpu_pin), ("gpu_benchmark", gpu_pin)):
        receipt = P.verified_receipt(pin, plan, stage=stage, verify_artifacts=False,
                                     receipt_mirror_dir=Path(pin["path"]).parent)
        finished = receipt.get("details", {}).get("finished_at")
        P.require(observed > _timestamp(finished, stage),
                  "operator checks must be newer than both completed benchmarks")
        benchmarks[stage] = {"receipt_sha256": receipt["receipt_sha256"], "finished_at": finished}
    out = copy.deepcopy(plan)
    out["operator_checks"], out["operator_evidence"] = checks, evidence
    out["resource_operator_freshness"] = {"observed_at": checks["observed_at"], "benchmarks": benchmarks}
    return out


def persist_closed_reference_gate(plan, plan_pin, proofs, output):
    """Archive the combined result, including a negative gate, before any send."""
    from .controls import GateNotPassed
    completed = E.verify_completed_predecessors(plan, proofs)
    envelope = {"schema": "pose-head-closed-reference-release-v1", "phase_plan": plan_pin,
                "completed_prerequisites": proofs}
    try:
        gate = E.verify_closed_form_release(plan, proofs, completed)
    except GateNotPassed as error:
        P.require(isinstance(error.gate, dict), "combined control failure lacks its measured gate")
        _write(output, {**envelope, "status": error.status, "gate": error.gate})
        raise
    return _write(output, {**envelope, "status": gate["status"], "gate": gate})


def release_plan(plan, *, review_commit, review_path, reviewed_code_sha, operator_checks):
    """Record, never infer, the review outcome read by the authorized operator."""
    for value in (review_commit, reviewed_code_sha):
        P.require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "full commit SHAs required")
    P.require(not Path(review_path).is_absolute() and ".." not in Path(review_path).parts,
              "review path must be repository relative")
    raw = subprocess.check_output(["git", "show", review_commit+":"+review_path], cwd=P.ROOT)
    review_text = raw.decode("utf-8")
    P.require(any(re.search(r"(?<![0-9a-f])"+re.escape(identity)+r"(?![0-9a-f])", review_text)
                  for identity in (reviewed_code_sha, plan["reader"]["source_sha256"])),
              "operational review must name the full reviewed code SHA or source SHA256")
    # Every source byte must be exactly the code snapshot that Claude reviewed.
    for path, digest in plan["reader"]["source_manifest"].items():
        old = subprocess.check_output(["git", "show", reviewed_code_sha+":"+path], cwd=P.ROOT)
        P.require(hashlib.sha256(old).hexdigest() == digest, "source differs from reviewed code: "+path)
    result = copy.deepcopy(plan)
    result["review"]["operational"] = {
        "status": "claude_review_converged", "commit": review_commit,
        "artifact": {"path": review_path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     "local_copy": review_path},
        "proposal_sha256": result["proposal"]["sha256"], "source_sha256": result["reader"]["source_sha256"],
        "reviewed_code_sha": reviewed_code_sha,
    }
    # A copy in the metadata commit makes the original text available in a
    # checkout whose Git object store does not include Claude's whole branch.
    destination = P.ROOT/review_path
    if destination.exists():
        P.require(destination.read_bytes() == raw, "existing review copy differs")
    else:
        P._atomic(destination, lambda stream: stream.write(raw), root=P.ROOT)
    result["reader"]["code_sha"] = reviewed_code_sha
    result["operator_checks"] = operator_checks
    result["status"] = "claude_review_converged"
    result["submission_released"] = True
    return P.validate_plan(result, require_review=True)


def materialize_plan(commit, repository_path, sha256, output):
    """Copy a committed phase plan on the allocated node, not through SSH edits."""
    P.require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None and
              re.fullmatch(r"[0-9a-f]{64}", sha256) is not None, "full plan commit/hash required")
    relative = Path(repository_path)
    P.require(not relative.is_absolute() and ".." not in relative.parts, "invalid committed plan path")
    raw = subprocess.check_output(["git", "show", commit+":"+relative.as_posix()], cwd=P.ROOT)
    P.require(hashlib.sha256(raw).hexdigest() == sha256, "committed phase plan hash differs")
    plan = json.loads(raw)
    P.validate_plan(plan, require_review=True)
    expected = Path(P.OUTPUT_ROOT)/"manifests"/(sha256+".json")
    P.require(output == expected, "materialized plan path must be its content hash")
    if output.exists():
        P.require(P.file_pin(output)["sha256"] == sha256, "existing materialized plan differs")
    else:
        try:
            P._atomic(output, lambda stream: stream.write(raw), root=P.OUTPUT_ROOT)
        except FileExistsError:
            P.require(P.file_pin(output)["sha256"] == sha256, "concurrent phase plan differs")
    return P.file_pin(output)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare-plan":
        plan = P.build_plan(args.proposal, jobs=E.job_graph())
        value = _write(args.output, plan)
    elif args.command == "inspect":
        plan = P.load_plan(args.plan)
        E.validate_graph(plan["jobs"])
        value = {"status": plan["status"], "review": plan["review"], "reader": plan["reader"]["source_sha256"],
                 "jobs": len(plan["jobs"]), "submission_released": plan["submission_released"],
                 "no_submission_performed": True}
    elif args.command == "release-plan":
        plan = P.load_plan(args.plan)
        checks, pin = _operator_checks(args.operator_checks)
        result = release_plan(plan, review_commit=args.review_commit, review_path=args.review_path,
                              reviewed_code_sha=args.reviewed_code_sha, operator_checks=checks)
        result["operator_evidence"] = pin
        value = _write(args.output, result)
    elif args.command == "resource-plan":
        from .runner import measured_plan
        plan = P.load_plan(args.plan, require_review=True)
        cpu_pin, gpu_pin = P.file_pin(args.cpu_receipt.absolute()), P.file_pin(args.gpu_receipt.absolute())
        plan = refresh_operator_checks(plan, args.operator_checks, cpu_pin, gpu_pin)
        result = measured_plan(plan, cpu_pin, gpu_pin)
        value = _write(args.output, result)
    elif args.command == "materialize-plan":
        value = materialize_plan(args.commit, args.repository_path, args.sha256, args.output)
    elif args.command == "commands":
        P.require("controls" not in args.stage or args.closed_reference_gate_output is not None,
                  "controls commands require --closed-reference-gate-output")
        P.require(re.fullmatch(r"[0-9a-f]{40}", args.plan_commit) is not None,
                  "full plan commit SHA required before command generation")
        relative = Path(args.plan_repository_path)
        P.require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
                  "plan path must be repository relative")
        plan = P.load_plan(args.plan, require_review=True)
        pin = P.file_pin(args.plan.absolute())
        committed = subprocess.check_output(["git", "show", args.plan_commit+":"+args.plan_repository_path], cwd=P.ROOT)
        P.require(hashlib.sha256(committed).hexdigest() == pin["sha256"], "plan has not been committed as declared")
        path = str(Path(P.OUTPUT_ROOT)/"manifests"/(pin["sha256"]+".json"))
        completed = (P.read_json_pin(P.file_pin(args.completed_prerequisites.absolute()))
                     if args.completed_prerequisites else {})
        # Do this before printing even the first command of a mixed stage list.
        # A rejected gate remains an immutable local artifact for the operator.
        if "controls" in args.stage:
            persist_closed_reference_gate(plan, pin, completed, args.closed_reference_gate_output)
        for job in plan["jobs"]:
            if job["stage"] in args.stage:
                command = E.submit_command(plan, job, checkout_sha=args.checkout_sha, plan_path=path,
                                           plan_sha256=pin["sha256"], completed_proofs=completed)
                for index, word in enumerate(command):
                    if word.startswith("--export="):
                        command[index] += f",TCN_PLAN_COMMIT={args.plan_commit},TCN_PLAN_REPO_PATH={args.plan_repository_path}"
                import shlex
                print(shlex.join(command))
        return 0
    else:
        from .runner import run_stage
        pin = P.file_pin(args.plan.absolute())
        P.require(pin["sha256"] == args.plan_sha256, "phase plan differs from submitted hash")
        P.require(os.environ.get("TCN_JOB_ID") == args.job_id, "job ID differs from launcher")
        plan = P.load_plan(args.plan, require_review=True)
        E.validate_graph(plan["jobs"])
        value = run_stage(plan, args.job_id)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
