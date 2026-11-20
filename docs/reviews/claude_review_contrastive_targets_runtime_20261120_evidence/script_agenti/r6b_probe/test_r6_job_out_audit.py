"""r6 probe: J04/J05 behaviour (cohort root and audit accounting in the dispatcher)."""
import json
from pathlib import Path
import os
import pytest
import contrastive_job as J
import contrastive_release as release_checks
from test_contrastive_job import job_env  # noqa: F401

REAL_ACCOUNTING = release_checks.validate_audit_accounting


def test_output_outside_cohort_root_rejected(job_env):
    job = job_env[0]("full")
    pre = json.loads(Path(release_checks.PREFLIGHT).read_text())
    pre["fresh_root"] = str(Path(os.environ["WORK"]) / "wearusfm_runs" / "cohort")
    Path(release_checks.PREFLIGHT).write_text(json.dumps(pre))
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(job, "gpu", 1800)       # out = WORK/wearusfm_runs/full, outside cohort root
    job["out"] = pre["fresh_root"]
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(job, "gpu", 1800)       # cohort root itself


def test_audit_accounting_enforced_for_non_audit_modes(job_env, monkeypatch):
    monkeypatch.setattr(release_checks, "validate_audit_accounting", REAL_ACCOUNTING)
    with pytest.raises(ValueError, match="audit accounting is not bound"):
        J.build_command(job_env[0]("readout"), "cpu", 3600)
