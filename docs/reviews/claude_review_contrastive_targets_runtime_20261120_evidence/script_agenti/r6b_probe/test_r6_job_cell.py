"""r6 probe: dispatcher cell/review checks that the repository tests do not exercise (J01-J03)."""
import json
from pathlib import Path
import pytest
import contrastive_job as J
import contrastive_release as release_checks
from test_contrastive_job import job_env  # noqa: F401

REAL_BINDING = release_checks.validate_review_binding


@pytest.mark.parametrize("field,value", [("id", "CONTRAST-UNKNOWN"), ("arm", "CR"), ("repeat", "b")])
def test_cell_identity_mismatch_rejected(job_env, field, value):
    job = job_env[0]("resume")
    job[field] = value
    with pytest.raises(ValueError, match="reviewed fixed preflight cell|resume requires"):
        J.build_command(job, "gpu", 1800)


def test_review_copy_must_match_binding_and_binding_is_validated(job_env, monkeypatch):
    build, pin = job_env
    job = build("full")
    job["review"] = pin("other_review.json", {"other": True})
    with pytest.raises(ValueError, match="bound package review"):
        J.build_command(job, "gpu", 1800)
    monkeypatch.setattr(release_checks, "validate_review_binding", REAL_BINDING)
    with pytest.raises(ValueError, match="authenticated package review binding"):
        J.build_command(build("full"), "gpu", 1800)
