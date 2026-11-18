"""Continuation operations: fail closed on missing quota, preserve verified bytes."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_continue as C


PROJECT = "12345 P /leonardo_work/IscrB_WearUsFM\n"
QUOTA = """Disk quotas for prj 12345 (pid 12345):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/leonardo_work
                100000 10000000 11000000 - 100 2000 3000 -
"""
POLICY = {"min_headroom_bytes": 8 * 1024**3, "min_headroom_files": 1000}


def evidence(quota=QUOTA):
    return {"work_root": C.WORK_ROOT, "checked_at_utc": "2026-11-04T17:00:00+00:00",
            "project_output": PROJECT, "quota_output": quota}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("row", ["/leonardo_work\n", C.WORK_ROOT + " "])
def test_project_quota_understands_wrapped_rows_and_soft_limit(row):
    values = C.check_quota_evidence(evidence(QUOTA.replace("/leonardo_work\n", row)), POLICY)
    assert values["project_id"] == 12345
    assert values["available_bytes"] == (10000000 - 100000) * 1024
    assert values["available_files"] == 1900


@pytest.mark.parametrize("change, match", [
    (("(pid 12345)", "(pid 999)"), "identity"),
    (("(pid 12345)", ""), "identity"),
    (("100000 10000000 11000000", "10000001* 10000000 11000000"), "byte quota"),
    (("100 2000 3000", "1500 2000 3000"), "inode quota"),
    (("100000 10000000 11000000", "100000 0 0"), "missing"),
    (("100000 10000000 11000000", "100M 10G 11G"), "non-numeric"),
])
def test_project_quota_does_not_trust_grace_human_units_or_wrong_identity(change, match):
    with pytest.raises(ValueError, match=match):
        C.check_quota_evidence(evidence(QUOTA.replace(*change)), POLICY)


def test_quota_pending_and_invalid_utc_fail_but_queue_age_alone_does_not():
    with pytest.raises(ValueError, match="pending"):
        C.check_quota_evidence(None, POLICY)
    value = evidence()
    value["checked_at_utc"] = "2026-11-04T18:00:00+01:00"
    with pytest.raises(ValueError, match="UTC"):
        C.check_quota_evidence(value, POLICY)
    value["checked_at_utc"] = "2025-01-01T00:00:00+00:00"
    # Freshness is a submit-time check; runtime queries the current quota again.
    assert C.check_quota_evidence(value, POLICY)["available_files"] == 1900


def test_submission_quota_age_is_explicit_and_does_not_depend_on_queue_delay():
    from datetime import datetime, timedelta
    now = datetime.fromisoformat(evidence()["checked_at_utc"])
    assert C.check_quota_evidence(evidence(), POLICY, submission_time=now + timedelta(hours=24))
    for at in (now - timedelta(seconds=1), now + timedelta(hours=24, seconds=1)):
        with pytest.raises(ValueError, match="24 hours"):
            C.check_quota_evidence(evidence(), POLICY, submission_time=at)


def test_live_quota_uses_only_project_commands_with_timeout(monkeypatch):
    calls = []
    def query(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return PROJECT if cmd[1] == "project" else QUOTA
    monkeypatch.setattr(C.subprocess, "check_output", query)
    result = C.live_quota(POLICY)
    assert result["available_files"] == 1900
    assert calls == [(["lfs", "project", "-d", C.WORK_ROOT], {"text": True, "timeout": 20}),
                     (["lfs", "quota", "-p", "12345", C.WORK_ROOT], {"text": True, "timeout": 20})]


def test_archive_is_exact_durable_and_does_not_overwrite(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"complete checkpoint including optimizer and rank RNG" * 100)
    expected = digest(source)
    destination = tmp_path / "work" / "checkpoint.pt"
    record = C.archive_checkpoint(source, destination, expected)
    assert digest(destination) == expected == digest(source)
    assert record["sha256"] == expected
    assert record["bytes"] == source.stat().st_size
    assert set(record["source_metadata_before_archive"]) == {"atime_ns", "mtime_ns", "ctime_ns"}
    assert "mtime + 40" in record["purge_estimate_basis"]
    assert not destination.with_name("checkpoint.pt.partial").exists()
    before = destination.stat().st_mtime_ns
    with pytest.raises(ValueError, match="already exists"):
        C.archive_checkpoint(source, destination, expected)
    reused = C.archive_checkpoint(source, destination, expected, allow_existing=True)
    assert reused["sha256"] == expected and destination.stat().st_mtime_ns == before


def test_archive_rejects_wrong_source_corrupt_existing_and_symlink(tmp_path):
    source, destination = tmp_path / "source.pt", tmp_path / "dest.pt"
    source.write_bytes(b"correct")
    with pytest.raises(ValueError, match="source differs"):
        C.archive_checkpoint(source, destination, "0" * 64)
    assert not destination.exists()
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="persistent checkpoint hash mismatch"):
        C.archive_checkpoint(source, destination, digest(source), allow_existing=True)
    link = tmp_path / "symlink.pt"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        C.archive_checkpoint(source, link, digest(source), allow_existing=True)


def test_partial_or_changed_source_is_not_published(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.pt", tmp_path / "dest.pt"
    source.write_bytes(b"original")
    expected = digest(source)
    original_copy = C.shutil.copyfileobj
    def mutate_after_copy(src, dst, length):
        original_copy(src, dst, length)
        source.write_bytes(b"changed")
    monkeypatch.setattr(C.shutil, "copyfileobj", mutate_after_copy)
    with pytest.raises(ValueError, match="source changed"):
        C.archive_checkpoint(source, destination, expected)
    assert not destination.exists()
    partial = destination.with_name("dest.pt.partial")
    assert partial.exists()
    monkeypatch.setattr(C.shutil, "copyfileobj", original_copy)
    with pytest.raises(FileExistsError):
        C.archive_checkpoint(source, destination, digest(source))
    assert partial.read_bytes() == b"original"


def test_released_operations_plan_has_review_and_pinned_real_quota():
    plan = json.loads((ROOT / "results/infra/consolidation_20261104/continuation_plan.json").read_text())
    assert plan["status"] == "claude_review_converged"
    assert plan["review_followup"]["review_commit"].startswith("0b2623a")
    evidence = plan["retention"]["quota_preflight"]
    assert digest(ROOT / evidence["source_file"]) == evidence["source_sha256"]
    assert C.check_quota_evidence(evidence, plan["retention"])["project_id"] == 20147512
    assert plan["retention"]["protected_steps"] == [19618, *C.PHASES]
    assert plan["training_gpu_hours_cap"] == 36


def test_even_converged_plan_cannot_archive_outside_slurm(tmp_path, monkeypatch):
    plan = {"protocol": C.PROTOCOL, "status": "claude_review_converged"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(C.T.P, "code_identity", lambda expected: expected)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    with pytest.raises(ValueError, match="compute-node"):
        C.execute(path, 30000, "a" * 40)
