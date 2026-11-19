"""Bind the new target cohort to an actual package review and completed audit.

No scheduler calls. Review/configuration documents remain immutable; release
artifacts are separate so neither the plan nor its review hashes itself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import PLAN, _read, source_digest

PREFLIGHT = ROOT / "results/infra/contrastive_targets_20261118/preflight_plan.json"
REVIEW_SCHEMA = "contrastive-target-release-review-v1"
BINDING_SCHEMA = "contrastive-target-review-binding-v1"
AUDIT_SCHEMA = "contrastive-audit-accounting-v1"
REVIEW_BLOCK = re.compile(r"```contrastive-release-review\s*\n(.*?)\n```", re.DOTALL)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path, *, relative=False):
    path = Path(path).resolve(strict=True)
    return {"path": str(path.relative_to(ROOT)) if relative else str(path),
            "sha256": file_sha(path), "bytes": path.stat().st_size}


def checked_pin(spec, *, repository=False):
    if not isinstance(spec, dict) or not {"path", "sha256"} <= set(spec):
        raise ValueError("complete artifact path and SHA256 required")
    path = Path(spec["path"])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if repository and not path.is_relative_to(ROOT.resolve()):
        raise ValueError("review/plans must be inside the frozen checkout")
    if file_sha(path) != spec["sha256"] or ("bytes" in spec and path.stat().st_size != spec["bytes"]):
        raise ValueError("release artifact pin mismatch: " + str(path))
    return path


def _full_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}", value) is not None


def source_paths():
    # Same ordered inventory as contrastive_prepare.source_digest(). New sibling
    # modules are included and cannot be hidden behind a historical review.
    paths = sorted((ROOT / "src/wearusfm").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("contrastive_*.py"))
    paths += sorted((ROOT / "scripts/slurm").glob("contrastive*.sbatch"))
    return paths + [ROOT / "scripts/train_jepa.py", ROOT / "scripts/slurm/lib_code_dir.sh"]


def git_bytes(commit, relative):
    if not _full_sha(commit) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("full commit and contained repository path required")
    try:
        return subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError("reviewed Git artifact unavailable: " + relative) from error


def verify_reviewed_sources(commit, expected):
    if not _full_sha(commit):
        raise ValueError("review must name the full source commit")
    digest = hashlib.sha256()
    for path in source_paths():
        relative = str(path.relative_to(ROOT))
        original = git_bytes(commit, relative)
        if original != path.read_bytes():
            raise ValueError("execution sources differ from reviewed commit: " + relative)
        digest.update(relative.encode() + b"\0" + hashlib.sha256(original).digest())
    if digest.hexdigest() != expected or expected != source_digest():
        raise ValueError("reviewed source inventory differs from execution digest")


def review_payload(text):
    blocks = REVIEW_BLOCK.findall(text)
    if len(blocks) != 1:
        raise ValueError("review must contain exactly one contrastive-release-review binding block")
    payload = json.loads(blocks[0])
    keys = {"schema", "status", "reviewed_commit", "source_sha256", "science_plan_sha256", "preflight_plan_sha256"}
    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":
        raise ValueError("review is not an approved target-cohort package review")
    if not _full_sha(payload["reviewed_commit"]):
        raise ValueError("review source commit must be a complete SHA")
    for name in ("source_sha256", "science_plan_sha256", "preflight_plan_sha256"):
        if not isinstance(payload[name], str) or re.fullmatch("[a-f0-9]{64}", payload[name]) is None:
            raise ValueError("review is missing an exact package hash")
    return payload


def validate_review_binding(binding, *, verify_git=True):
    if not isinstance(binding, dict) or binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("new cohort requires an authenticated package review binding")
    if not _full_sha(binding.get("review_commit")):
        raise ValueError("review provenance requires its full commit")
    review = checked_pin(binding["review"], repository=True)
    science = checked_pin(binding["science_plan"], repository=True)
    preflight = checked_pin(binding["preflight_plan"], repository=True)
    if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():
        raise ValueError("review binding points to another cohort")
    declared = review_payload(review.read_text())
    if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)
            or declared["preflight_plan_sha256"] != file_sha(preflight)):
        raise ValueError("review does not bind the current sources and both plans")
    plan = _read(preflight)
    if (plan.get("source_sha256") != declared["source_sha256"] or
            plan.get("science_plan", {}).get("sha256") != declared["science_plan_sha256"]):
        raise ValueError("preflight source/science-plan binding differs")
    if verify_git:
        verify_reviewed_sources(declared["reviewed_commit"], declared["source_sha256"])
        for path in (science, preflight):
            if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():
                raise ValueError("reviewed plan bytes differ from executable package")
    # The original review commit can live on another branch. Its original bytes
    # are checked by bind-review; this copy remains pinned on the deploy branch.
    # This is a provenance declaration, not a cryptographic author signature.
    if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):
        raise ValueError("review copy differs from authenticated original")
    return declared


def bind_review(review_commit, review_path, local_copy, science_plan=None, preflight_plan=None):
    science_plan = PLAN if science_plan is None else science_plan
    preflight_plan = PREFLIGHT if preflight_plan is None else preflight_plan
    original = git_bytes(review_commit, review_path)
    local_copy = Path(local_copy).resolve(strict=True)
    if original != local_copy.read_bytes():
        raise ValueError("review copy is not byte-identical to the original review commit")
    result = {"schema": BINDING_SCHEMA, "review_commit": review_commit,
              "original_review_path": review_path,
              "original_review_sha256": hashlib.sha256(original).hexdigest(),
              "review": pin(local_copy, relative=True),
              "review_payload": review_payload(original.decode()),
              "science_plan": pin(science_plan, relative=True),
              "preflight_plan": pin(preflight_plan, relative=True)}
    validate_review_binding(result)
    return result


def validate_audit_accounting(audit_path, accounting_pin_or_path, *, expected_job_id=None, expected_source=None):
    """No inherited TIMEOUT exception or scheduler overrun is admissible."""
    if accounting_pin_or_path is None:
        raise ValueError("audit accounting required before release")
    accounting_path = (checked_pin(accounting_pin_or_path) if isinstance(accounting_pin_or_path, dict)
                       else Path(accounting_pin_or_path).resolve(strict=True))
    accounting = _read(accounting_path)
    audit = _read(audit_path)
    source = expected_source or source_digest()
    if (accounting.get("schema") != AUDIT_SCHEMA or accounting.get("audit_sha256") != file_sha(audit_path)
            or accounting.get("source_sha256") != source or audit.get("source_sha256") != source):
        raise ValueError("audit accounting is not bound to this completed new-cohort audit")
    plan = _read(PREFLIGHT)
    expected_logical = [cell["id"] for cell in plan["cells"] if cell["mode"] == "audit"]
    if len(expected_logical) != 1 or accounting.get("logical_id") != expected_logical[0]:
        raise ValueError("audit accounting logical ID differs from the new cohort")
    job_id = accounting.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit() or (expected_job_id is not None and job_id != str(expected_job_id)):
        raise ValueError("audit accounting job ID missing or changed")
    sacct = checked_pin(accounting["sacct"])
    records = list(csv.DictReader(io.StringIO(sacct.read_text()), delimiter="|"))
    required = {"JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocCPUS", "TimelimitRaw"}
    if not records or not required <= set(records[0]):
        raise ValueError("audit sacct requires unambiguous allocation fields")
    allocation = [row for row in records if row["JobIDRaw"] == job_id]
    if len(allocation) != 1:
        raise ValueError("audit sacct requires one exact allocation row")
    row = allocation[0]
    try:
        elapsed, cpus, minutes = (int(row[name]) for name in ("ElapsedRaw", "AllocCPUS", "TimelimitRaw"))
    except (TypeError, ValueError) as error:
        raise ValueError("audit sacct elapsed/allocation is not integral") from error
    if (row["State"] != "COMPLETED" or row["ExitCode"] != "0:0" or not 0 <= elapsed <= 7200
            or cpus != 8 or minutes != 120):
        raise ValueError("audit requires COMPLETED 0:0, Elapsed <= 7200 s, 8 CPUs and 120 min")
    return {"accounting_sha256": file_sha(accounting_path), "sacct_sha256": file_sha(sacct),
            "audit_sha256": file_sha(audit_path), "job_id": job_id, "logical_id": expected_logical[0],
            "elapsed_seconds": elapsed, "state": row["State"], "exit_code": row["ExitCode"]}


def validate_cuda_prerequisite(spec, cohort, inputs, calibration_run_dir):
    """A benchmark cannot start with a missing/failed technical CUDA receipt."""
    from contrastive_benchmark_readout import _cuda_evidence
    from contrastive_prepare import build_review_configs, BASELINE
    from contrastive_rank_exit import validate_receipt
    path = checked_pin(spec)
    cfg = build_review_configs(_read(BASELINE))["calibration"]
    cuda = _cuda_evidence(path, cohort, inputs, cfg)
    if cuda.get("inputs", {}).get("calibration_checkpoint") != file_sha(Path(calibration_run_dir) / "checkpoint.pt"):
        raise ValueError("CUDA receipt checkpoint differs from the calibrated cohort")
    rank_exit_path = cuda.get("rank_exit_report_path")
    if not isinstance(rank_exit_path, str) or not rank_exit_path:
        raise ValueError("CUDA receipt lacks rank-exit prerequisite")
    rank_exit = Path(rank_exit_path)
    if file_sha(rank_exit) != cuda.get("rank_exit_report_sha256"):
        raise ValueError("CUDA rank-exit prerequisite pin changed")
    validate_receipt(rank_exit, expected_source_sha256=cohort["source_sha256"])
    return cuda


def make_job(binding, cell_id, evidence, out):
    """Materialize a stage from its fixed cell and already published evidence pins.

Physical input files are rehashed by contrastive_job on Leonardo. Upstream
receipts needed for release must be locally available and are checked now.
"""
    declared = validate_review_binding(binding)
    plan = _read(PREFLIGHT)
    cells = [row for row in plan["cells"] if row["id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("exact unique preflight cell required")
    cell = cells[0]
    if cell["mode"] not in {"audit", "calibration", "cuda", "full", "resume", "readout"}:
        raise ValueError("mode is outside the amended preflight")
    science = _read(PLAN)
    if evidence.get("inputs") is not None and {k: v.get("sha256") for k, v in evidence["inputs"].items()} != science["input_sha256"]:
        raise ValueError("new-cohort input pins differ from the science plan")
    job = dict(evidence)
    forbidden = {"status", "source_sha256", "mode", "package_review", "id", "out", "arm", "repeat"}
    if forbidden & set(evidence):
        raise ValueError("evidence cannot override the fixed released stage")
    job.update(status="APPROVED_FOR_PREFLIGHT", source_sha256=declared["source_sha256"], mode=cell["mode"],
               id=cell_id, package_review=binding, review=binding["review"], out=str(out), arm=cell.get("arm"))
    if cell.get("repeat") is not None:
        job["repeat"] = cell["repeat"]
    if cell["mode"] != "audit":
        audit = checked_pin(job["audit"])
        validate_audit_accounting(audit, job["audit_sacct"])
    return job


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind-review")
    bind.add_argument("--review-commit", required=True)
    bind.add_argument("--review-path", required=True)
    bind.add_argument("--review-copy", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    make = sub.add_parser("make-job")
    make.add_argument("--binding", type=Path, required=True)
    make.add_argument("--cell-id", required=True)
    make.add_argument("--evidence", type=Path, required=True)
    make.add_argument("--result-out", required=True)
    make.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check-binding")
    check.add_argument("--binding", type=Path, required=True)
    audit = sub.add_parser("check-audit")
    audit.add_argument("--audit", type=Path, required=True)
    audit.add_argument("--accounting", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "bind-review":
        write_new(args.out, bind_review(args.review_commit, args.review_path, args.review_copy))
    elif args.command == "make-job":
        write_new(args.out, make_job(_read(args.binding), args.cell_id, _read(args.evidence), args.result_out))
    elif args.command == "check-binding":
        print(json.dumps(validate_review_binding(_read(args.binding))))
    else:
        print(json.dumps(validate_audit_accounting(args.audit, args.accounting)))


if __name__ == "__main__":
    main()
