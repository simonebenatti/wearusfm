#!/bin/bash
# Run from the exact deployed Git blob on the login node, once before each CPU submission.
set -euo pipefail
phase=${1:?checkout or audit}
sha=${2:?published release commit}
[[ "$sha" =~ ^[0-9a-f]{40}$ ]]
campaign=contrastive-targets-audit-20261119
repo="${WORK:?}/wearusfm"
dest="$WORK/wearusfm_campaigns/$campaign"
pkg=results/infra/contrastive_targets_20261118
release="$pkg/releases/audit_20261120"
checkout_id=CONTRAST-CHECKOUT-audit-targets-v3-20261118
[[ "$(git -C "$repo" rev-parse "refs/heads/deploy/$campaign^{commit}")" == "$sha" ]]
[[ ! -e "$WORK/wearusfm_runs/results/contrastive_targets_20261118/audit.json" ]]

if [[ "$phase" == checkout && $# == 2 ]]; then
    [[ ! -e "$dest" && ! -L "$dest" ]]
    # Temporary inspection of selected blobs, not a campaign checkout or canonical worktree change.
    inspect=$(mktemp -d)
    trap 'rm -rf -- "$inspect"' EXIT
    git -C "$repo" archive "$sha" -- src/wearusfm scripts "$pkg/plan.json" "$pkg/preflight_plan.json" \
        "$pkg/review_configs/calibration.config.json" "$release" \
        results/infra/tokenizer_execution_20261103/19618/T100-G/config.json \
        results/infra/contrastive_comparison_20261116/review_response/coverage_manifest_reference.json \
        results/infra/contrastive_comparison_20261116/review_response/gradient_reference.json | tar -x -C "$inspect"
    export GIT_DIR="$(git -C "$repo" rev-parse --absolute-git-dir)"
    cd "$inspect"
elif [[ "$phase" == audit && $# == 3 && "$3" =~ ^[0-9]+$ ]]; then
    row=$(sacct -X -n -P -j "$3" --format=JobIDRaw,State,ExitCode,JobName%80,AllocCPUS,TimelimitRaw)
    printf '%s\n' "$row"
    [[ "$row" == "$3|COMPLETED|0:0|$checkout_id|4|4" ]]
    export CODE_DIR="$dest" EXPECTED_SHA="$sha"
    source "$dest/scripts/slurm/lib_code_dir.sh"
    code_dir_setup
else
    echo 'Invalid phase or checkout receipt' >&2
    exit 2
fi

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/contrastive_release.py check-binding --binding "$release/review_binding.json"
# Validate the dispatched argv and recipe. The audit itself has no --validate-only;
# validate_recipe(calibration) is its actual runner validation; do not call audit.main here.
python3 - "$release/audit.job.json" <<'PY'
import json
import sys
from pathlib import Path

from contrastive_job import build_command
from contrastive_prepare import _read
from contrastive_run import validate_recipe

job = _read(sys.argv[1])
command = build_command(job, "cpu", 7200)
config = Path(job["config"]["path"])
cfg = validate_recipe(_read(config), "calibration")
checks = [
    job["id"] == "CONTRAST-AUDIT-targets-v3-20261118",
    job["mode"] == "audit" and job["arm"] is None and "release" not in job,
    job["package_review"] == _read(config.parents[1] / "releases/audit_20261120/review_binding.json"),
    cfg.contrastive["arm"] == "C0" and cfg.contrastive["calibration_only"] is True,
    cfg.loader.rms_window_ms == 100 and cfg.loader.ms_fast_window_ms == 125,
    Path(command[1]) == Path.cwd() / "scripts/contrastive_audit.py",
    Path(command[command.index("--config") + 1]) == config.resolve(),
    command[command.index("--steps") + 1] == "1024",
    command[command.index("--out") + 1] == job["out"],
]
if not all(checks):
    raise SystemExit("Audit mode, binding, config, arm, or generated runner command mismatch")
print(json.dumps({"job_id": job["id"], "job_arm": job["arm"], "config_arm": cfg.contrastive["arm"],
                  "calibration_only": cfg.contrastive["calibration_only"], "stage_release": job.get("release"),
                  "validated_argv_only": command, "job_started": False}, indent=2))
PY
