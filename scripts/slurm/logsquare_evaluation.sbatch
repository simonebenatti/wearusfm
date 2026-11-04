#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -J logsquare_evaluation
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_evaluation_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_evaluation_%j.out
# Submit only through shared submitter, after measured per-cell budget publication.
# Args: relative_plan step stage task_or_none new_out expected_plan_sha256
set -euo pipefail
[ "$#" -eq 6 ] || { echo 'six arguments required' >&2; exit 2; }
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"; STEP="$2"; STAGE="$3"; TASK="$4"; OUT="$5"; PLAN_SHA="$6"
case "$PLAN_REL" in /*|"") echo 'relative contained plan required' >&2; exit 2;; esac
case "$STEP:$STAGE:$TASK" in 5000:extract-pose:none|19618:extract-pose:none|19618:extract-classification:p1|19618:extract-classification:p2) ;; *) echo 'unregistered cell' >&2; exit 2;; esac
[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ] && [ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ] && [ "${SLURM_MEM_PER_NODE:-0}" -eq 102400 ] || { echo 'resource mismatch' >&2; exit 2; }
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' 'NF==4 {print $1*86400+$2*3600+$3*60+$4; next} NF==3 {print $1*3600+$2*60+$3; next} NF==2 {print $1*60+$2; next} {print 0}')
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 - "$CODE_DIR" "$PLAN_REL" "$WORK" <<'PY'
from pathlib import Path
import sys,torch
code=Path(sys.argv[1]).resolve(strict=True); plan=(code/sys.argv[2]).resolve(strict=True); work=Path(sys.argv[3]).resolve()
if not code.is_relative_to(work/'wearusfm_campaigns') or not plan.is_relative_to(code) or not plan.is_file(): raise SystemExit('frozen checkout/contained plan required')
if torch.cuda.device_count()!=1: raise SystemExit('one visible CUDA GPU required')
PY
ARGS=(--stage "$STAGE" --plan "$CODE_DIR/$PLAN_REL" --step "$STEP" --expected-code-sha "$EXPECTED_SHA" --expected-plan-sha256 "$PLAN_SHA" --out "$OUT" --device cuda --allocated-seconds "$SECS")
[ "$TASK" = none ] || ARGS+=(--task "$TASK")
exec python3 scripts/logsquare_evaluation.py "${ARGS[@]}"
