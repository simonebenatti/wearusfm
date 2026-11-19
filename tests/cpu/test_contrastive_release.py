"""The current source and both plans, not an arbitrary old review, release jobs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_release as C


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "scripts/contrastive_one.py"
    source.parent.mkdir()
    source.write_text("# reviewed execution\n")
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
    science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
    preflight = dump(root / "preflight.json", {"source_sha256": digest, "science_plan": {"path": "science.json", "sha256": C.file_sha(science)},
                     "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200}]})
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.setattr(C, "PLAN", science)
    monkeypatch.setattr(C, "PREFLIGHT", preflight)
    monkeypatch.setattr(C, "source_paths", lambda: [source])
    monkeypatch.setattr(C, "source_digest", lambda: hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest())
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args], cwd=root, stderr=subprocess.PIPE).decode().strip()
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "Reviewed package")
    code = git("rev-parse", "HEAD")
    payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": code, "source_sha256": digest,
               "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
    review = root / "review.md"
    review.write_text("# Actual review\n\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    git("add", "review.md")
    git("commit", "-qm", "Actual package review")
    binding = C.bind_review(git("rev-parse", "HEAD"), "review.md", review)
    return {"root": root, "science": science, "preflight": preflight, "source": source,
            "payload": payload, "review": review, "binding": binding, "git": git}


def test_review_binding_checks_actual_git_source_and_two_plan_bytes(package):
    assert C.validate_review_binding(package["binding"])["reviewed_commit"] == package["payload"]["reviewed_commit"]
    # Later metadata commits are permissible; execution bytes remain identical.
    dump(package["root"] / "release.json", {"metadata_only": True})
    package["git"]("add", "release.json")
    package["git"]("commit", "-qm", "Release pins")
    C.validate_review_binding(package["binding"])


@pytest.mark.parametrize("kind", ["source", "science", "preflight", "review"])
def test_changed_source_plan_or_review_cannot_inherit_binding(package, kind):
    package[kind].write_text(package[kind].read_text() + "\n")
    with pytest.raises(ValueError, match="pin mismatch|current sources"):
        C.validate_review_binding(package["binding"])


def test_historical_review_is_insufficient_even_with_valid_file_pin(package):
    old = package["root"] / "old.md"
    old.write_text("APPROVED preflight on 17 November; historical sources and targets.\n")
    binding = copy.deepcopy(package["binding"])
    binding["review"] = C.pin(old, relative=True)
    binding["original_review_sha256"] = C.file_sha(old)
    with pytest.raises(ValueError, match="binding block"):
        C.validate_review_binding(binding)


def test_review_cannot_relabel_the_current_code_as_an_older_commit(package):
    binding = copy.deepcopy(package["binding"])
    payload = copy.deepcopy(package["payload"])
    payload["reviewed_commit"] = "0" * 40
    package["review"].write_text("```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    binding.update(review=C.pin(package["review"], relative=True), review_payload=payload,
                   original_review_sha256=C.file_sha(package["review"]))
    with pytest.raises(ValueError, match="Git artifact unavailable"):
        C.validate_review_binding(binding)


def test_bind_requires_review_bytes_from_its_original_commit(package):
    package["review"].write_text(package["review"].read_text() + "tampered")
    with pytest.raises(ValueError, match="byte-identical"):
        C.bind_review(package["binding"]["review_commit"], "review.md", package["review"])


def accounting(package, *, elapsed=7200, state="COMPLETED", exit_code="0:0", cpus=8, minutes=120):
    audit = dump(package["root"] / "audit.json", {"source_sha256": C.source_digest(), "passed": True})
    tsv = package["root"] / "sacct.tsv"
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
                   f"123|{state}|{exit_code}|{elapsed}|{cpus}|{minutes}\n"
                   "123.batch|COMPLETED|0:0|7199|8|120\n")
    receipt = dump(package["root"] / "accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                   "audit_sha256": C.file_sha(audit), "logical_id": "new-audit", "job_id": "123", "sacct": C.pin(tsv)})
    return audit, receipt, tsv


def test_audit_accepts_completed_exact_limit_and_binds_receipt_to_allocation(package):
    audit, receipt, _ = accounting(package)
    result = C.validate_audit_accounting(audit, receipt, expected_job_id="123")
    assert result["elapsed_seconds"] == 7200 and result["state"] == "COMPLETED"
    with pytest.raises(ValueError, match="job ID"):
        C.validate_audit_accounting(audit, receipt, expected_job_id="124")


@pytest.mark.parametrize("change", [{"elapsed": 7201}, {"elapsed": 7387}, {"state": "TIMEOUT"}, {"exit_code": "0:9"},
                                    {"cpus": 4}, {"minutes": 60}])
def test_audit_has_no_timeout_exception_or_slurm_grace_extension(package, change):
    audit, receipt, _ = accounting(package, **change)
    with pytest.raises(ValueError, match="COMPLETED 0:0"):
        C.validate_audit_accounting(audit, receipt)


@pytest.mark.parametrize("field,value", [("audit_sha256", "0" * 64), ("source_sha256", "0" * 64),
                                         ("logical_id", "old-audit"), ("job_id", "124")])
def test_audit_cannot_inherit_unrelated_receipt_or_allocation(package, field, value):
    audit, receipt, _ = accounting(package)
    data = json.loads(receipt.read_text()); data[field] = value; dump(receipt, data)
    with pytest.raises(ValueError):
        C.validate_audit_accounting(audit, receipt)


def test_audit_sacct_requires_unique_allocation_and_hash(package):
    audit, receipt, tsv = accounting(package)
    tsv.write_text(tsv.read_text() + "123|COMPLETED|0:0|7100|8|120\n")
    with pytest.raises(ValueError, match="pin mismatch"):
        C.validate_audit_accounting(audit, receipt)
    data = json.loads(receipt.read_text()); data["sacct"] = C.pin(tsv); dump(receipt, data)
    with pytest.raises(ValueError, match="one exact allocation"):
        C.validate_audit_accounting(audit, receipt)


def test_job_builder_releases_only_the_fixed_reviewed_cell(package):
    job = C.make_job(package["binding"], "new-audit", {"inputs": {"manifest": {"path": "/remote/data", "sha256": "1" * 64}}}, "/remote/new-output")
    assert job["mode"] == "audit" and job["id"] == "new-audit"
    assert job["status"] == "APPROVED_FOR_PREFLIGHT"
    with pytest.raises(ValueError, match="override"):
        C.make_job(package["binding"], "new-audit", {"mode": "training"}, "/remote/new-output")
    with pytest.raises(ValueError, match="unique"):
        C.make_job(package["binding"], "other-audit", {}, "/remote/new-output")


def test_audit_accounting_is_required_before_release(package):
    audit, _, _ = accounting(package)
    with pytest.raises(ValueError, match="audit accounting required"):
        C.validate_audit_accounting(audit, None)
