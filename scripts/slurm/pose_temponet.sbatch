#!/bin/bash
# Allocation is supplied by the measured job plan through submit_job.py.
set -euo pipefail
: "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${WORK:?}"
: "${TCN_PLAN:?}" "${TCN_PLAN_SHA256:?}" "${TCN_JOB_ID:?}"
: "${TCN_PLAN_COMMIT:?}" "${TCN_PLAN_REPO_PATH:?}"
[ "${SLURM_RESTART_COUNT:-0}" = 0 ] || { echo 'No automatic requeue/retry' >&2; exit 2; }
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -m pose_temponet materialize-plan --commit "$TCN_PLAN_COMMIT" \
    --repository-path "$TCN_PLAN_REPO_PATH" --sha256 "$TCN_PLAN_SHA256" --output "$TCN_PLAN"
exec python3 -m pose_temponet run --plan "$TCN_PLAN" --plan-sha256 "$TCN_PLAN_SHA256" --job-id "$TCN_JOB_ID"
