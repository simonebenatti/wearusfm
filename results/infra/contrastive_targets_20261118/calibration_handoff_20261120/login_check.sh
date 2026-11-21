#!/bin/bash
# 3-launcher runs this frozen blob on the login node. Both phases stop before torchrun.
set -euo pipefail
phase=${1:?checkout or calibration}
sha=${2:?full execution SHA}
[[ "$sha" =~ ^[0-9a-f]{40}$ ]]
campaign=contrastive-targets-calibration-20261120
repo="${WORK:?}/wearusfm"
dest="$WORK/wearusfm_campaigns/$campaign"
pkg=results/infra/contrastive_targets_20261118
release="$pkg/releases/calibration_20261120"
binding="$pkg/releases/audit_20261120/review_binding.json"
account="$pkg/audit_execution_20261120"
audit="$WORK/wearusfm_runs/results/contrastive_targets_20261118/audit.json"
out="${audit%/*}/calibration"
logical=CONTRAST-CALIB-targets-v3-20261118
[[ "$(git -C "$repo" rev-parse "refs/heads/deploy/$campaign^{commit}")" == "$sha" ]]
[[ -f "$audit" && ! -e "$out" && ! -L "$out" ]]
scratch_check=$(mktemp -d)
trap 'rm -rf -- "$scratch_check"' EXIT
old_checkout=$(sacct -X -n -P -j 64936161 --format=JobIDRaw,State,ExitCode,AllocCPUS,TimelimitRaw)
printf '%s\n' "$old_checkout"
[[ "$old_checkout" == '64936161|COMPLETED|0:0|4|4' ]]
sacct -X -P -j 64937087 --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw > "$scratch_check/audit.sacct.tsv"
[[ "$(git -C "$WORK/wearusfm_campaigns/contrastive-targets-audit-20261119" rev-parse HEAD)" == 9abde2951cd9b2a25aa699faceb93132de6732f8 ]]
if [[ "$phase" == checkout && $# == 2 ]]; then
    [[ ! -e "$dest" && ! -L "$dest" ]]
    mkdir "$scratch_check/tree"
    git -C "$repo" archive "$sha" -- scripts src/wearusfm "$pkg/plan.json" "$pkg/preflight_plan.json" \
        "$pkg/review_configs/calibration.config.json" "$pkg/releases" "$account/audit_accounting.json" \
        "$account/audit.sacct.tsv" results/infra/tokenizer_execution_20261103/19618/T100-G/config.json \
        results/infra/contrastive_comparison_20261116/review_response/coverage_manifest_reference.json \
        results/infra/contrastive_comparison_20261116/review_response/gradient_reference.json | tar -x -C "$scratch_check/tree"
    export GIT_DIR="$(git -C "$repo" rev-parse --absolute-git-dir)"
    cd "$scratch_check/tree"
elif [[ "$phase" == calibration && $# == 3 && "$3" =~ ^[0-9]+$ ]]; then
    row=$(sacct -X -n -P -j "$3" --format=JobIDRaw,State,ExitCode,JobName%80,AllocCPUS,TimelimitRaw)
    printf '%s\n' "$row"
    [[ "$row" == "$3|COMPLETED|0:0|CONTRAST-CHECKOUT-calibration-targets-v3-20261118|4|4" ]]
    export CODE_DIR="$dest" EXPECTED_SHA="$sha"
    source "$dest/scripts/slurm/lib_code_dir.sh"
    code_dir_setup
else
    echo 'Invalid phase or missing completed checkout ID' >&2
    exit 2
fi

cmp "$scratch_check/audit.sacct.tsv" "$account/audit.sacct.tsv"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 scripts/contrastive_release.py check-binding --binding "$binding"
python3 scripts/contrastive_release.py check-audit --audit "$audit" --accounting "$account/audit_accounting.json"
python3 scripts/contrastive_release.py make-job --binding "$binding" --cell-id "$logical" \
    --evidence "$release/calibration_evidence.json" --result-out "$out" --out "$scratch_check/job.json"
cmp "$release/calibration.job.json" "$scratch_check/job.json"
python3 - "$release/calibration.job.json" "$logical" "$out" <<'PY'
from pathlib import Path
import json
import subprocess
import sys

from contrastive_job import build_command
from contrastive_prepare import _read

job = _read(sys.argv[1])
command = build_command(job, "gpu", 1800)
release = _read(job["release"]["path"])
config = _read(job["config"]["path"])
runner = str(Path("scripts/contrastive_run.py").resolve())
if command[:6] != [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", runner]:
    raise ValueError("Unexpected dispatch")
if not (job["id"] == sys.argv[2] and job["out"] == sys.argv[3] and job["mode"] == "calibration"
        and job["arm"] is None and release["stage"] == "calibration" and release["arm"] is None
        and config["contrastive"]["arm"] == "C0" and config["contrastive"]["calibration_only"] is True
        and config["loader"]["rms_window_ms"] == 100 and config["loader"]["ms_fast_window_ms"] == 125):
    raise ValueError("Stage, arm, output or config mismatch")
print(json.dumps({"execution_argv": command, "job_arm": job["arm"], "release_arm": release.get("arm"),
                  "stage": release["stage"], "config_arm": config["contrastive"]["arm"],
                  "calibration_only": True}, indent=2), flush=True)
subprocess.run([sys.executable, runner, *command[6:], "--validate-only"], check=True)
PY
