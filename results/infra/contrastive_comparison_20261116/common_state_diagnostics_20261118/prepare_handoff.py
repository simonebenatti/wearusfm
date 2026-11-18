"""Build the 3-launcher argv from a published, explicit execution commit.

The only subprocesses invoked are local Git reads. The calling launcher decides
when to submit the resulting commands through the shared submission ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
CAMPAIGN = "contrastive-common-20261118"
WORK = Path("/leonardo_work/IscrB_WearUsFM")
CHECKOUT = WORK / "wearusfm_campaigns" / CAMPAIGN
OUTPUT_ROOT = WORK / "wearusfm_runs/results/contrastive_common_diagnostics_20261118"
SUBMITTER = REPO.parent / "wearusfm-gpt-science/scripts/submit_job.py"
PYTHON = Path.home() / ".venvs/wearusfm-tok/bin/python"
CHECKOUT_ID = "CONTRAST-CHECKOUT-common-20261118"
CELLS = {"cuda": "CONTRAST-COMMON-cuda-20261118",
         "resume_CN": "CONTRAST-COMMON-resume-CN-20261118",
         "resume_CR": "CONTRAST-COMMON-resume-CR-20261118"}
LAUNCHER = "scripts/slurm/diagnose_contrastive_common.sbatch"
CHECKOUT_LAUNCHER = "scripts/slurm/campaign_checkout.sbatch"


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def prepare(frozen):
    if not re.fullmatch(r"[0-9a-f]{40}", frozen):
        raise ValueError("Supply the complete, published execution SHA.")
    if git("rev-parse", frozen + "^{commit}").decode().strip() != frozen:
        raise ValueError("Execution revision is not a commit.")
    bundle_files = (
        CHECKOUT_LAUNCHER, LAUNCHER, "scripts/diagnostics/contrastive_common_state_cuda.py",
        "scripts/diagnostics/contrastive_common_checkpoint.py",
        str((PACKAGE / "resources.json").relative_to(REPO)), str(Path(__file__).resolve().relative_to(REPO)),
    )
    pins = {}
    for name in bundle_files:
        frozen_bytes = git("show", f"{frozen}:{name}")
        if (REPO / name).read_bytes() != frozen_bytes:
            raise ValueError(f"Local delivery differs from the release SHA for {name}")
        pins[name] = sha(frozen_bytes)
    submitter_sha = sha(SUBMITTER.read_bytes())
    resources = json.loads((PACKAGE / "resources.json").read_text())
    if submitter_sha != resources["shared_submitter_sha256"]:
        raise ValueError("Common submitter differs from the audited version.")
    def submit_prefix(job_id, gpu_hours):
        return [str(PYTHON), str(SUBMITTER), "--id", job_id, "--agente", "gpt",
                "--campagna", "CONTRASTIVE-v2-20261116", "--gpu-ore", str(gpu_hours)]

    commands = {
        CHECKOUT_ID: submit_prefix(CHECKOUT_ID, 0) + [
            "--script-stdin", str(REPO / CHECKOUT_LAUNCHER), "--", "--cpus-per-task=4",
            "--time=00:04:00", "--mem=2G",
            f"--export=ALL,CAMPAIGN={CAMPAIGN},CAMPAIGN_SHA={frozen},EXPECTED_SHA={frozen}"],
    }
    after_verified_checkout = {}
    for cell, job_id in CELLS.items():
        base = submit_prefix(job_id, 2)
        transport = ["--script-stdin", str(REPO / LAUNCHER), "--",
                     f"--export=ALL,CODE_DIR={CHECKOUT},EXPECTED_SHA={frozen},DIAGNOSTIC_CELL={cell}"]
        commands[job_id] = base + ["--dopo", CHECKOUT_ID] + transport
        # This equivalent variant supports delayed submission after dependency
        # expiry. Use only after the ledger, sacct and frozen HEAD are verified.
        after_verified_checkout[job_id] = base + transport

    return {
        "frozen_commit": frozen, "submission_owner": "3-launcher", "campaign_dir": str(CHECKOUT),
        "deploy_command": ["git", "push", "leonardo", f"{frozen}:refs/heads/deploy/{CAMPAIGN}"],
        "deploy_ref": "refs/heads/deploy/" + CAMPAIGN, "commands": commands,
        "gpu_commands_after_verified_completed_checkout": after_verified_checkout,
        "local_transport_and_driver_sha256": pins, "submitter_sha256": submitter_sha,
        "reports": {cell: str(OUTPUT_ROOT / cell / "report.json") for cell in CELLS},
        "slurm_log_pattern": str(WORK / "wearusfm_runs/logs/contrastive_diag_%j.out"),
        "resources": resources,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-sha", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="print deployment and submission argv as JSON")
    args = parser.parse_args()
    print(json.dumps(prepare(args.frozen_sha), indent=2))
