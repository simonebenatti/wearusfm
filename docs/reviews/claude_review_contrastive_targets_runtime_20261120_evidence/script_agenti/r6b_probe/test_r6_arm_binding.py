"""r6 probe: the reviewed cell arm is not bound to the executed config/release arm."""
import json
from pathlib import Path
import contrastive_job as J
import contrastive_release as release_checks
from test_contrastive_job import job_env  # noqa: F401  (fixture)


def _rebind(job, pin, mode, cell_arm, run_arm, repeat=None):
    pre = json.loads(Path(release_checks.PREFLIGHT).read_text())
    pre["cells"][0]["arm"] = cell_arm
    pre["cells"][0]["repeat"] = repeat
    Path(release_checks.PREFLIGHT).write_text(json.dumps(pre))
    job["arm"] = cell_arm
    job["repeat"] = repeat
    job["config"] = pin(mode + "/other_arm_config.json", {"contrastive": {"arm": run_arm}, "max_steps": 576})
    release = json.loads(Path(job["release"]["path"]).read_text())
    release.update(arm=run_arm, config_sha256=job["config"]["sha256"])
    job["release"] = pin(mode + "/other_arm_release.json", release)
    return job


def test_full_cell_C0_accepts_CN_config_and_release(job_env):
    build, pin = job_env
    job = _rebind(build("full"), pin, "full", "C0", "CN")
    command = J.build_command(job, "gpu", 1800)
    cfg = command[command.index("--config") + 1]
    print("\nC0 cell -> executed config arm:", json.loads(Path(cfg).read_text())["contrastive"]["arm"],
          "release arm:", json.loads(Path(command[command.index("--release") + 1]).read_text())["arm"])
    assert json.loads(Path(cfg).read_text())["contrastive"]["arm"] == "CN"


def test_reload_cell_CN_A_accepts_CR_config_and_release(job_env):
    build, pin = job_env
    job = _rebind(build("resume"), pin, "resume", "CN", "CR", repeat="a")
    command = J.build_command(job, "gpu", 1800)
    cfg = command[command.index("--config") + 1]
    print("\nCN-A cell -> executed config arm:", json.loads(Path(cfg).read_text())["contrastive"]["arm"])
    assert json.loads(Path(cfg).read_text())["contrastive"]["arm"] == "CR"
