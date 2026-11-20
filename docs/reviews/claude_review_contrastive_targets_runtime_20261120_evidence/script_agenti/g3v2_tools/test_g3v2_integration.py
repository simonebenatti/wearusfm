"""Verifier-only integration probe (G-3): real binding/accounting through the consumers.

Not part of the reviewed package. Run with PYTHONPATH pointing at one clone.
"""
import copy
import hashlib
import json
import subprocess

import pytest

import contrastive_release as C
import contrastive_job as J
import contrastive_run as R


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "scripts/contrastive_one.py"
    source.parent.mkdir()
    source.write_text("# reviewed execution\n")
    def digest_now():
        return hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
    digest = digest_now()
    science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
    work = tmp_path / "work"
    (work / "wearusfm_runs" / "cohort").mkdir(parents=True)
    preflight = dump(root / "preflight.json", {
        "source_sha256": digest, "science_plan": {"path": "science.json", "sha256": C.file_sha(science)},
        "fresh_root": str(work / "wearusfm_runs" / "cohort"),
        "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200},
                  {"id": "new-calib", "mode": "calibration", "arm": None, "seconds": 1800}]})
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.setattr(C, "PLAN", science)
    monkeypatch.setattr(C, "PREFLIGHT", preflight)
    monkeypatch.setattr(C, "source_paths", lambda: [source])
    monkeypatch.setattr(C, "source_digest", digest_now)
    monkeypatch.setattr(J, "source_digest", digest_now)
    monkeypatch.setattr(J, "ROOT", root)
    monkeypatch.setattr(R, "source_digest", digest_now)
    monkeypatch.setenv("WORK", str(work))
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
                                       cwd=root, stderr=subprocess.PIPE).decode().strip()
    git("init", "-q"); git("add", "."); git("commit", "-qm", "Reviewed package")
    code = git("rev-parse", "HEAD")
    payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": code, "source_sha256": digest,
               "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
    review = root / "review.md"
    review.write_text("# Actual review\n\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    git("add", "review.md"); git("commit", "-qm", "Actual package review")
    binding = C.bind_review(git("rev-parse", "HEAD"), "review.md", review)
    # Another-package review: same schema, different preflight hash, committed in Git.
    other_payload = dict(payload, preflight_plan_sha256="0" * 64)
    other = root / "other_review.md"
    other.write_text("```contrastive-release-review\n" + json.dumps(other_payload) + "\n```\n")
    git("add", "other_review.md"); git("commit", "-qm", "Other package review")
    other_binding = copy.deepcopy(binding)
    other_binding.update(review_commit=git("rev-parse", "HEAD"), original_review_path="other_review.md",
                         original_review_sha256=C.file_sha(other), review=C.pin(other, relative=True),
                         review_payload=other_payload)
    return {"root": root, "work": work, "digest": digest, "binding": binding, "other": other_binding}


def accounting(pkg, name, *, state="COMPLETED", elapsed=7000):
    root = pkg["root"]
    audit = dump(root / name / "audit.json", {"source_sha256": C.source_digest(), "passed": True})
    tsv = root / name / "sacct.tsv"
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
                   f"123|{state}|0:0|{elapsed}|8|120\n")
    receipt = dump(root / name / "accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                   "audit_sha256": C.file_sha(audit), "logical_id": "new-audit", "job_id": "123", "sacct": C.pin(tsv)})
    return C.pin(audit), C.pin(receipt)


def dispatcher_job(pkg, binding, mode="audit", cell="new-audit"):
    return {"status": "APPROVED_FOR_PREFLIGHT", "mode": mode, "source_sha256": pkg["digest"], "id": cell, "arm": None,
            "package_review": binding, "review": binding["review"],
            "out": str(pkg["work"] / "wearusfm_runs" / "cohort" / "new")}


# ---- R-2, dispatcher -------------------------------------------------------------------------
def test_dispatcher_valid_binding_passes_binding_gate(pkg):
    with pytest.raises(KeyError, match="config"):   # stops later, at the absent config pin
        J.build_command(dispatcher_job(pkg, pkg["binding"]), "cpu", 7200)


def test_dispatcher_rejects_other_package_review(pkg):
    with pytest.raises(ValueError, match="review"):
        J.build_command(dispatcher_job(pkg, pkg["other"]), "cpu", 7200)


def test_dispatcher_rejects_job_review_not_matching_binding(pkg):
    job = dispatcher_job(pkg, pkg["binding"])
    job["review"] = pkg["other"]["review"]
    with pytest.raises(ValueError, match="review"):
        J.build_command(job, "cpu", 7200)


# ---- R-4, release builder and dispatcher -----------------------------------------------------
@pytest.mark.parametrize("state,elapsed", [("TIMEOUT", 7000), ("COMPLETED", 7201)])
def test_make_job_rejects_timeout_or_overrun_accounting(pkg, state, elapsed):
    audit, receipt = accounting(pkg, "bad", state=state, elapsed=elapsed)
    with pytest.raises(ValueError, match="COMPLETED 0:0"):
        C.make_job(pkg["binding"], "new-calib", {"audit": audit, "audit_sacct": receipt}, "/remote/out")


@pytest.mark.parametrize("state,elapsed", [("TIMEOUT", 7000), ("COMPLETED", 7201)])
def test_dispatcher_rejects_timeout_or_overrun_accounting(pkg, state, elapsed):
    good_audit, good_receipt = accounting(pkg, "good")
    job = dispatcher_job(pkg, pkg["binding"], "calibration", "new-calib")
    job.update(audit=good_audit, audit_sacct=good_receipt)
    with pytest.raises(KeyError, match="config"):   # valid accounting passes the gate
        J.build_command(job, "gpu", 1800)
    audit, receipt = accounting(pkg, "bad", state=state, elapsed=elapsed)
    job.update(audit=audit, audit_sacct=receipt)
    with pytest.raises(ValueError, match="COMPLETED 0:0"):
        J.build_command(job, "gpu", 1800)


# ---- R-2 and R-4, runtime --------------------------------------------------------------------
def runtime_release(pkg, tmp_path, monkeypatch, binding, receipt_pin, audit_pin):
    monkeypatch.setattr(R, "validate_input_bindings", lambda value: None)
    monkeypatch.setattr(R, "validate_audit", lambda *args: None)
    config = dump(tmp_path / "config.json", {"fixture": True})
    inputs = {name: dump(tmp_path / (name + ".json"), {}) for name in ("manifest", "scales", "window_seconds", "pose_stats")}
    release = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": pkg["digest"],
               "config_sha256": R.sha(config), "world_size": 4, "workers_per_rank": 6,
               "inputs": {k: R.sha(v) for k, v in inputs.items()}, "review_pin": binding["review_commit"],
               "package_review": binding, "audit_sacct": receipt_pin, "time_limit_s": 1770,
               "audit_sha256": audit_pin["sha256"]}
    return release, config, inputs


def test_runtime_accepts_valid_and_rejects_other_review_and_timeout(pkg, tmp_path, monkeypatch):
    audit, receipt = accounting(pkg, "good")
    release, config, inputs = runtime_release(pkg, tmp_path, monkeypatch, pkg["binding"], receipt, audit)
    assert R.validate_release(release, "calibration", config, inputs, time_limit_s=1770, audit_path=audit["path"]) is None
    other = dict(release, package_review=pkg["other"], review_pin=pkg["other"]["review_commit"])
    with pytest.raises(ValueError, match="review"):
        R.validate_release(other, "calibration", config, inputs, time_limit_s=1770, audit_path=audit["path"])
    relabel = dict(release, review_pin="f" * 40)
    with pytest.raises(ValueError, match="review"):
        R.validate_release(relabel, "calibration", config, inputs, time_limit_s=1770, audit_path=audit["path"])
    bad_audit, bad_receipt = accounting(pkg, "bad", state="TIMEOUT")
    timeout = dict(release, audit_sacct=bad_receipt, audit_sha256=bad_audit["sha256"])
    with pytest.raises(ValueError, match="COMPLETED 0:0"):
        R.validate_release(timeout, "calibration", config, inputs, time_limit_s=1770, audit_path=bad_audit["path"])


def test_make_job_rejects_other_package_review(pkg):
    assert C.make_job(pkg["binding"], "new-audit", {}, "/remote/out")["id"] == "new-audit"
    with pytest.raises(ValueError, match="review"):
        C.make_job(pkg["other"], "new-audit", {}, "/remote/out")
