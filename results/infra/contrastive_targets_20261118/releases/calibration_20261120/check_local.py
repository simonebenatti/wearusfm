"""Verify the production calibration package against byte-identical local inputs.

Pass --manifest, --scales, --window-seconds, --pose-stats and --audit JSON.
Only physical filesystem paths are translated for the local build_command.
The real compute mounts and fresh output must be checked again on Leonardo.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import contrastive_job as jobs
import contrastive_run as runner
from contrastive_release import checked_pin, make_job, validate_audit_accounting


def main():
    if not __debug__:
        raise RuntimeError("run without -O to enable local verification")
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("manifest", "scales", "window_seconds", "pose_stats", "audit")
    for name in names:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    local = {name: path.resolve(strict=True) for name, path in vars(parser.parse_args()).items()}
    audit = local.pop("audit")
    job = json.loads((HERE / "calibration.job.json").read_bytes())
    evidence = json.loads((HERE / "calibration_evidence.json").read_bytes())
    release = json.loads(checked_pin(job["release"]).read_bytes())
    config = checked_pin(job["config"])
    cfg = runner.validate_recipe(json.loads(config.read_bytes()), "calibration")
    assert job["arm"] is None and release["arm"] is None
    assert cfg.contrastive["arm"] == "C0" and cfg.contrastive["calibration_only"] and cfg.max_steps == 512
    assert not release["training_authorized"]
    bound_audit = dict(job["audit"], path=str(audit))
    checked_pin(bound_audit)
    accounting = validate_audit_accounting(audit, job["audit_sacct"], expected_job_id="64937087")
    candidate = make_job(job["package_review"], job["id"], dict(evidence, audit=bound_audit), job["out"])
    candidate["audit"] = evidence["audit"]
    assert candidate == job
    runner.validate_release(release, "calibration", config, local, time_limit_s=1770, audit_path=audit)
    runner.R.check_input_files(local["manifest"], local["window_seconds"], cfg.nested, release["inputs"]["manifest"])

    paths = {job["inputs"][name]["path"]: str(path) for name, path in local.items()}
    paths[job["audit"]["path"]] = str(audit)
    frozen_check = jobs.checked_file

    def local_file(spec):
        return frozen_check(dict(spec, path=paths.get(spec["path"], spec["path"])))

    with tempfile.TemporaryDirectory() as temp:
        roots = [(Path(temp) / name).resolve() for name in ("work", "scratch")]
        for folder in roots:
            folder.mkdir()
        if set(Path(p) for p in job["roots"]) != jobs.DATA_ROOTS:
            raise ValueError("published data roots differ from the frozen runner")
        with patch.dict(os.environ, WORK="/leonardo_work/IscrB_WearUsFM"), \
                patch.object(jobs, "checked_file", side_effect=local_file), \
                patch.object(jobs, "DATA_ROOTS", set(roots)):
            command = jobs.build_command(dict(job, roots=list(map(str, roots))), "gpu", 1800)
    assert command[:6] == [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc_per_node=4", str(ROOT / "scripts/contrastive_run.py")]
    for flag, value in {"--stage": "calibration", "--time-limit-s": "1770", "--out-dir": job["out"],
                        "--config": str(config), "--release": str(checked_pin(job["release"])),
                        "--audit-report": str(audit)}.items():
        assert command[command.index(flag) + 1] == value
    print(json.dumps({
        "status": "LOCAL_VALIDATION_PASSED", "job": job["id"],
        "job_sha256": runner.sha(HERE / "calibration.job.json"),
        "release_sha256": job["release"]["sha256"], "script_sha256": runner.sha(__file__),
        "evidence_sha256": runner.sha(HERE / "calibration_evidence.json"),
        "source_sha256": jobs.source_digest(), "inputs_sha256": {k: runner.sha(p) for k, p in local.items()},
        "audit_accounting": accounting, "stage": "calibration", "ranks": 4, "max_steps": 512,
        "trainer_seconds": 1770, "allocation_seconds": 1800, "training_authorized": False,
        "verified": ["binding", "make_job", "validate_recipe", "validate_release", "check_input_files",
                     "build_command_with_local_paths"],
        "scope": "Real source and input bytes. Absolute input paths and both data mounts translated for local "
                 "build_command. No execution. Physical mounts, release validation and fresh output to recheck on login."
    }, indent=2))


if __name__ == "__main__":
    main()
