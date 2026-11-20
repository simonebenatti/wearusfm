"""r6 probe: J06-J08 behaviour (release binding fields and resume fork pins in the dispatcher)."""
import json
from pathlib import Path
import pytest
import contrastive_job as J
from test_contrastive_job import job_env  # noqa: F401


@pytest.mark.parametrize("field", ["fork_checkpoint", "continuous_summary", "package_review", "audit_sacct"])
def test_release_and_job_pins_must_agree(job_env, field):
    build, pin = job_env
    job = build("resume")
    release = json.loads(Path(job["release"]["path"]).read_text())
    release[field] = {"path": "/elsewhere", "sha256": "0" * 64}
    job["release"] = pin("resume/changed_release.json", release)
    with pytest.raises(ValueError, match="resume requires|review/audit accounting differs"):
        J.build_command(job, "gpu", 1800)


def test_resume_phase_is_repeat_specific(job_env):
    build, pin = job_env
    job = build("resume")
    release = json.loads(Path(job["release"]["path"]).read_text())
    release["phase"] = "resume_b"
    job["release"] = pin("resume/phase_release.json", release)
    with pytest.raises(ValueError, match="job stage"):
        J.build_command(job, "gpu", 1800)
