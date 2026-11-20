"""Skeptic probe (R4-M1): does make_job/validate_review_binding look at any budget or
approval field of the preflight plan?  Builds an isolated fixture repo (like
tests/cpu/test_contrastive_release.py) whose preflight declares a 0 GPUh cap, an
explicit 'Simone budget decision' prerequisite and submission_authorized=false,
then materialises a GPU calibration job from the review binding alone."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import contrastive_release as C


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def main():
    base = Path(sys.argv[1])
    root = Path(tempfile.mkdtemp(dir=base)) / "repo"
    root.mkdir()
    source = root / "scripts/contrastive_one.py"
    source.parent.mkdir()
    source.write_text("# reviewed execution\n")
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
    science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
    preflight = dump(root / "preflight.json", {
        "source_sha256": digest,
        "science_plan": {"path": "science.json", "sha256": C.file_sha(science)},
        "submission_authorized": False,
        "required_before_first_submission": ["registered Simone decision on the new GPU budget"],
        "resources": {"gpu_hours_cap": 0, "core_hours_cap": 0},
        "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200},
                  {"id": "new-calib", "mode": "calibration", "arm": None, "seconds": 1800, "gpus": 4}]})
    C.ROOT, C.PLAN, C.PREFLIGHT = root, science, preflight
    C.source_paths = lambda: [source]
    C.source_digest = lambda: hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()

    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
                                       cwd=root, stderr=subprocess.PIPE).decode().strip()
    git("init", "-q"); git("add", "."); git("commit", "-qm", "Reviewed package")
    code = git("rev-parse", "HEAD")
    payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": code, "source_sha256": digest,
               "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
    review = root / "review.md"
    review.write_text("# Review\n\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    git("add", "review.md"); git("commit", "-qm", "Review")
    binding = C.bind_review(git("rev-parse", "HEAD"), "review.md", review)
    audit = dump(root / "audit.json", {"source_sha256": C.source_digest(), "passed": True})
    tsv = root / "sacct.tsv"
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|COMPLETED|0:0|3900|8|120\n")
    receipt = dump(root / "accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                   "audit_sha256": C.file_sha(audit), "logical_id": "new-audit", "job_id": "123", "sacct": C.pin(tsv)})
    job = C.make_job(binding, "new-calib", {"audit": C.pin(audit), "audit_sacct": C.pin(receipt)}, "/remote/new-output")
    print(json.dumps({"mode": job["mode"], "id": job["id"], "status": job["status"],
                      "plan_gpu_hours_cap": 0, "plan_submission_authorized": False,
                      "binding_schema": job["package_review"]["schema"]}))


if __name__ == "__main__":
    main()
