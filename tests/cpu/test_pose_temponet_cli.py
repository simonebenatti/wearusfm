"""Exercise the public module entry point, without data access or submissions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from pose_temponet import __main__ as CLI
from pose_temponet import provenance as P


def run_cli(*args, **extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT/"src")+os.pathsep+str(ROOT/"scripts"),
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.update(extra)
    return subprocess.run([sys.executable, "-m", "pose_temponet", *map(str,args)], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=45)


def test_normal_entry_point_builds_inspects_and_refuses_unreviewed_run(tmp_path):
    plan_path = tmp_path/"plan.json"
    result = run_cli("prepare-plan", "--output", plan_path)
    assert result.returncode == 0, result.stderr
    pin = json.loads(result.stdout)
    assert pin["sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text())
    assert len(plan["jobs"]) == 24 and not plan["submission_released"]
    assert plan["review"]["documentary"]["status"] == "claude_review_converged"
    assert plan["review"]["operational"]["status"] == "REVIEW_REQUIRED"
    check = run_cli("inspect", "--plan", plan_path)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["no_submission_performed"]
    identifier = plan["jobs"][0]["id"]
    denied = run_cli("run", "--plan", plan_path, "--plan-sha256", pin["sha256"],
                     "--job-id", identifier, TCN_JOB_ID=identifier)
    assert denied.returncode != 0 and "has not been explicitly released" in denied.stderr
    assert not (tmp_path/"receipts").exists()
    repeated = run_cli("prepare-plan", "--output", plan_path)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr


def test_parser_offers_no_science_overrides_or_submit_subcommand():
    for arguments in (["submit"], ["run","--plan","x","--plan-sha256","f"*64,"--job-id","x","--epochs","3"],
                      ["run","--plan","x","--plan-sha256","f"*64,"--job-id","x","--seed","8"]):
        with pytest.raises(SystemExit):
            CLI.parser().parse_args(arguments)


def test_operator_quota_evidence_requires_other_pending_writes(tmp_path):
    path=tmp_path/"checks.json"
    checks={"work_free_bytes": 13*1024**3,"pending_write_bytes":2*1024**3,
            "no_duplicate_jobs":True,"serial_lane_checked":True,"reservations_checked":True,
            "balance_checked":True,"observed_at":"2026-11-19T12:00:00+01:00"}
    path.write_text(json.dumps(checks))
    with pytest.raises(ValueError, match="quota gate"):
        CLI._operator_checks(path)
    checks["work_free_bytes"] = 14*1024**3
    path.write_text(json.dumps(checks))
    value,pin=CLI._operator_checks(path)
    assert value==checks and pin==P.file_pin(path)
    del checks["pending_write_bytes"]
    path.write_text(json.dumps(checks))
    with pytest.raises(ValueError,match="current quota"):
        CLI._operator_checks(path)
