"""Skeptic probe for G-3: do the consumers really call the R-2/R-4 validators?

Uses the existing fixtures, then restores the ORIGINAL validators (the
fixtures replace them with no-ops). A spy records each call and then rejects.
If a consumer call is removed, the spy is never called and the test fails.
"""
import sys
from pathlib import Path
import pytest

import contrastive_job as J
import contrastive_run as runner
import contrastive_release as release_checks
from test_contrastive_job import job_env  # noqa: F401  (fixture)
from test_contrastive_run import release_fixture


class Rejected(Exception):
    pass


def spy(calls, name):
    def fn(*args, **kwargs):
        calls.append(name)
        raise Rejected(name)
    return fn


@pytest.mark.parametrize("mode,kind,seconds", [("audit", "cpu", 7200), ("cuda", "gpu", 1800),
                                               ("readout", "cpu", 3600), ("full", "gpu", 1800)])
def test_dispatcher_calls_review_binding(job_env, monkeypatch, mode, kind, seconds):
    calls = []
    monkeypatch.setattr(release_checks, "validate_review_binding", spy(calls, "binding"))
    with pytest.raises(Rejected):
        J.build_command(job_env[0](mode), kind, seconds)
    assert calls == ["binding"]


@pytest.mark.parametrize("mode,kind,seconds", [("cuda", "gpu", 1800), ("readout", "cpu", 3600),
                                               ("calibration", "gpu", 1800), ("full", "gpu", 1800)])
def test_dispatcher_calls_audit_accounting_for_non_audit(job_env, monkeypatch, mode, kind, seconds):
    calls = []
    monkeypatch.setattr(release_checks, "validate_audit_accounting", spy(calls, "accounting"))
    with pytest.raises(Rejected):
        J.build_command(job_env[0](mode), kind, seconds)
    assert calls == ["accounting"]


def test_dispatcher_rejects_review_differing_from_binding(job_env):
    job = job_env[0]("audit")
    job["package_review"] = dict(job["package_review"], review={"path": "x", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="bound package review"):
        J.build_command(job, "cpu", 7200)


def test_runtime_calls_review_binding_and_accounting(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(release_checks, "validate_review_binding", spy(calls, "binding"))
    with pytest.raises(Rejected):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    monkeypatch.setattr(release_checks, "validate_review_binding", lambda value: None)
    monkeypatch.setattr(release_checks, "validate_audit_accounting", spy(calls, "accounting"))
    with pytest.raises(Rejected):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    assert calls == ["binding", "accounting"]


def test_runtime_rejects_review_pin_differing_from_binding(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    release["review_pin"] = "other-commit"
    with pytest.raises(ValueError, match="bound package review"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
