"""Generate the two submission commands. Only an explicit login-check option contacts Leonardo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = HERE.parent.relative_to(ROOT)
AUDIT = PACKAGE / "audit_execution_20261120"
GLOO = PACKAGE / "gloo_p5_20261120"
REL = PACKAGE / "releases/calibration_20261120"
BINDING = PACKAGE / "releases/audit_20261120/review_binding.json"
LOGIN = HERE.relative_to(ROOT) / "login_check.sh"
GPU = Path("scripts/slurm/contrastive_preflight_gpu.sbatch")
CHECKOUT = Path("scripts/slurm/campaign_checkout.sbatch")
SUBMITTER = ROOT.parent / "wearusfm-gpt-science/scripts/submit_job.py"
PYTHON = Path.home() / ".venvs/wearusfm-tok/bin/python"
LEDGER = Path.home() / "wearusfm_local/jobs"
CAMPAIGN = "CONTRASTIVE-targets-v3-20261118"
DEPLOY = "contrastive-targets-calibration-20261120"
REF = "refs/heads/deploy/" + DEPLOY
DEST = Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns") / DEPLOY
CHECKOUT_ID = "CONTRAST-CHECKOUT-calibration-targets-v3-20261118"
CALIBRATION_ID = "CONTRAST-CALIB-targets-v3-20261118"
JOB = REL / "calibration.job.json"


def plan(sha):
    git = ["git", "-C", str(ROOT)]
    if not re.fullmatch("[0-9a-f]{40}", sha) or subprocess.check_output(git + ["rev-parse", sha + "^{commit}"]).decode().strip() != sha:
        raise ValueError("Complete published commit required")
    pins = {}
    for path in (Path(__file__).resolve().relative_to(ROOT), LOGIN, GPU, CHECKOUT, JOB, BINDING,
                 REL / "calibration.release.json", REL / "calibration_evidence.json",
                 AUDIT / "audit_accounting.json", AUDIT / "audit.sacct.tsv", GLOO / "full_c0/receipt.json",
                 GLOO / "reload/reload_result.json", Path("scripts/submit_job.py")):
        data = subprocess.check_output(git + ["show", f"{sha}:{path}"])
        if data != (ROOT / path).read_bytes():
            raise ValueError(f"File changed since release: {path}")
        pins[str(path)] = hashlib.sha256(data).hexdigest()
    if pins["scripts/submit_job.py"] != "814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82" or hashlib.sha256(SUBMITTER.read_bytes()).hexdigest() != pins["scripts/submit_job.py"]:
        raise ValueError("Shared submitter identity changed")
    submit = [str(PYTHON), str(SUBMITTER), "--agente", "gpt", "--campagna", CAMPAIGN]
    inspect = [str(PYTHON), str(Path(__file__).resolve()), "--frozen-sha", sha]
    return {
        "execution_commit": sha, "operator": "3-launcher", "campaign_directory": str(DEST), "pins": pins,
        "deploy": git + ["push", "leonardo", f"{sha}:{REF}"],
        "verify_deploy_sha": git + ["ls-remote", "--exit-code", "leonardo", REF],
        "before_checkout": inspect + ["--login-check", "checkout"],
        "checkout_only": submit + ["--id", CHECKOUT_ID, "--gpu-ore", "0", "--script-stdin", str(ROOT / CHECKOUT),
            "--", "--cpus-per-task=4", "--mem=2G", "--time=00:04:00",
            f"--export=ALL,CAMPAIGN={DEPLOY},CAMPAIGN_SHA={sha},EXPECTED_SHA={sha}"],
        "before_calibration": inspect + ["--login-check", "calibration"],
        "calibration_only": submit + ["--id", CALIBRATION_ID, "--gpu-ore", "2",
            "--", "--cpus-per-task=32", "--gres=gpu:4", "--mem=240G", "--time=00:30:00",
            f"--export=ALL,CODE_DIR={DEST},EXPECTED_SHA={sha},TORCH_DIST_INIT_BARRIER=1", str(DEST / GPU), str(JOB)],
        "requested": {"gpu_hours": 2, "core_hours": 16 + 4 * 4 / 60},
        "slurm_allowance": {"gpu_hours": .4, "core_hours": 3.6},
    }


def check(phase, sha):
    with (LEDGER / "ledger.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    logical = CHECKOUT_ID if phase == "checkout" else CALIBRATION_ID
    if any(row["id"] == logical for row in rows) or (LEDGER / logical).exists():
        raise ValueError(f"ID already reserved: {logical}")
    args = [phase, sha]
    if phase == "calibration":
        state = json.loads((LEDGER / CHECKOUT_ID / "state.json").read_text())
        numeric = str(state["job_slurm"])
        if not numeric.isdecimal() or state.get("campagna") != CAMPAIGN or state.get("id") != CHECKOUT_ID or state.get("stato") != "sottomesso" or state.get("codice_sha") != sha:
            raise ValueError("Checkout receipt does not match the frozen phase")
        if not any(row["id"] == CHECKOUT_ID and row["stato"] == "sottomesso" and row["job_slurm"] == numeric and row["codice_sha"] == sha for row in rows):
            raise ValueError("Checkout ledger record does not match its receipt")
        args.append(numeric)
    script = shlex.quote(f"{sha}:{LOGIN}")
    command = 'set -eo pipefail; git -C "$WORK/wearusfm" show ' + script + " | bash -s -- " + shlex.join(args)
    subprocess.run(["ssh", "leonardo", "bash -lc " + shlex.quote(command)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-sha", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--login-check", choices=("checkout", "calibration"))
    args = parser.parse_args()
    commands = plan(args.frozen_sha)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
    else:
        check(args.login_check, args.frozen_sha)
