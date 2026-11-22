#!/bin/bash
# Diagnostics of the failed temporal control (D0 replay, D1 clean fit).
# Same environment, line for line, as scripts/slurm/pose_temponet.sbatch; the
# allocation is supplied by submit_job.py. No plan is materialized and nothing
# under the A1 output root is written.
set -euo pipefail
: "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${WORK:?}"
: "${TCND_STAGE:?}" "${TCND_JOB_ID:?}"
[ "${SLURM_RESTART_COUNT:-0}" = 0 ] || { echo 'No automatic requeue/retry' >&2; exit 2; }
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python3 -m tcn_temporal_diag "$TCND_STAGE"
