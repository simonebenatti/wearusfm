"""Print pinned audit commands or run one explicitly selected read-only login check."""
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
RELEASE = PACKAGE / "releases/audit_20261120"
JOB = RELEASE / "audit.job.json"
CHECKOUT_SCRIPT = Path("scripts/slurm/campaign_checkout.sbatch")
AUDIT_SCRIPT = Path("scripts/slurm/contrastive_preflight_cpu.sbatch")
LOGIN_CHECK = HERE.relative_to(ROOT) / "login_check.sh"
SHARED_SUBMITTER = ROOT.parent / "wearusfm-gpt-science/scripts/submit_job.py"
SUBMITTER_SHA = "814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82"
PYTHON = Path.home() / ".venvs/wearusfm-tok/bin/python"
REGISTRY = Path.home() / "wearusfm_local/jobs"
CAMPAIGN = "contrastive-targets-audit-20261119"
PROJECT_CAMPAIGN = "CONTRASTIVE-targets-v3-20261118"
WORK = Path("/leonardo_work/IscrB_WearUsFM")
DEST = WORK / "wearusfm_campaigns" / CAMPAIGN
DEPLOY = "refs/heads/deploy/" + CAMPAIGN
CHECKOUT_ID = "CONTRAST-CHECKOUT-audit-targets-v3-20261118"
AUDIT_ID = "CONTRAST-AUDIT-targets-v3-20261118"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(frozen):
    if not re.fullmatch("[0-9a-f]{40}", frozen):
        raise ValueError("Use the complete published release SHA")
    git = ["git", "-C", str(ROOT)]
    if subprocess.check_output(git + ["rev-parse", frozen + "^{commit}"]).decode().strip() != frozen:
        raise ValueError("Release SHA must name a commit")
    files = [Path(__file__).resolve().relative_to(ROOT), LOGIN_CHECK, CHECKOUT_SCRIPT, AUDIT_SCRIPT,
             JOB, RELEASE / "review_binding.json", Path("scripts/submit_job.py")]
    pins = {}
    for path in files:
        content = subprocess.check_output(git + ["show", f"{frozen}:{path}"])
        if content != (ROOT / path).read_bytes():
            raise ValueError(f"Local file differs from frozen commit: {path}")
        pins[str(path)] = digest(content)
    if digest(SHARED_SUBMITTER.read_bytes()) != SUBMITTER_SHA or pins["scripts/submit_job.py"] != SUBMITTER_SHA:
        raise ValueError("Shared submitter differs from the verified tool")
    base = [str(PYTHON), str(SHARED_SUBMITTER)]
    common = ["--agente", "gpt", "--campagna", PROJECT_CAMPAIGN, "--gpu-ore", "0"]
    check = [str(PYTHON), str(Path(__file__).resolve()), "--frozen-sha", frozen]
    return {"execution_commit": frozen, "operator": "3-launcher", "campaign_directory": str(DEST),
            "deploy_ref": DEPLOY, "source_transport_sha256": pins,
            "deploy": git + ["push", "leonardo", f"{frozen}:{DEPLOY}"],
            "verify_deploy_sha": git + ["ls-remote", "--exit-code", "leonardo", DEPLOY],
            "check_immediately_before_checkout": check + ["--check-checkout-login"],
            "checkout_only": base + ["--id", CHECKOUT_ID] + common + [
                "--script-stdin", str(ROOT / CHECKOUT_SCRIPT), "--", "--cpus-per-task=4",
                "--mem=2G", "--time=00:04:00",
                f"--export=ALL,CAMPAIGN={CAMPAIGN},CAMPAIGN_SHA={frozen},EXPECTED_SHA={frozen}"],
            "check_immediately_before_audit": check + ["--check-audit-login"],
            "audit_only_after_verified_completed_checkout": base + ["--id", AUDIT_ID] + common + [
                "--", "--cpus-per-task=8", "--mem=30G", "--time=02:00:00",
                f"--export=ALL,CODE_DIR={DEST},EXPECTED_SHA={frozen}", str(DEST / AUDIT_SCRIPT), str(JOB)],
            "audit_output": json.loads((ROOT / JOB).read_bytes())["out"],
            "audit_phase_max": {"gpu_hours": 0, "core_hours": (4 * 4 + 8 * 120) / 60}}


def check_login(frozen, phase):
    logical = CHECKOUT_ID if phase == "checkout" else AUDIT_ID
    if (REGISTRY / logical).exists():
        raise ValueError(f"Logical ID already reserved: {logical}")
    ledger = REGISTRY / "ledger.tsv"
    if not ledger.is_file():
        raise ValueError("Shared ledger unavailable")
    with ledger.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if any(row["id"] == logical for row in rows):
        raise ValueError(f"Logical ID already recorded in ledger: {logical}")
    args = [phase, frozen]
    if phase == "audit":
        state = json.loads((REGISTRY / CHECKOUT_ID / "state.json").read_text())
        numeric = str(state["job_slurm"])
        if (not numeric.isdecimal() or state["stato"] != "sottomesso" or state["id"] != CHECKOUT_ID
                or state["codice_sha"] != frozen or state["campagna"] != PROJECT_CAMPAIGN):
            raise ValueError("Missing or inconsistent shared checkout receipt")
        if not any(row["id"] == CHECKOUT_ID and row["stato"] == "sottomesso" and
                   row["codice_sha"] == frozen and row["job_slurm"] == numeric for row in rows):
            raise ValueError("Checkout ledger identity disagrees with state receipt")
        args.append(numeric)
    spec = shlex.quote(f"{frozen}:{LOGIN_CHECK}")
    body = 'set -eo pipefail; git -C "$WORK/wearusfm" show ' + spec + " | bash -s -- " + shlex.join(args)
    subprocess.run(["ssh", "leonardo", "bash -lc " + shlex.quote(body)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-sha", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print argv; uses local Git and files only")
    mode.add_argument("--check-checkout-login", action="store_true", help="3-launcher read-only prerequisite checks")
    mode.add_argument("--check-audit-login", action="store_true", help="3-launcher read-only checkout and audit checks")
    args = parser.parse_args()
    plan = prepare(args.frozen_sha)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        check_login(args.frozen_sha, "checkout" if args.check_checkout_login else "audit")
