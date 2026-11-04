#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH -J logsquare_train
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_train_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_train_%j.out

# One fresh candidate; continuation resumes only its verified 5000-step state.
# No submission here. The shared submitter/ledger owns both logical job IDs.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'usage: logsquare_train.sbatch <relative plan> <5000|19618> <new receipt directory>' >&2
  exit 2
fi
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
PLAN_REL="$1"; STEP="$2"; OUT="$3"
case "$PLAN_REL" in /*|"") echo 'relative plan required' >&2; exit 2;; esac
case "$STEP" in 5000|19618) ;; *) exit 2;; esac
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 32 ]; then
  echo 'one node and 32 CPUs required' >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4; next}
  NF==3 {print $1*3600+$2*60+$3; next}
  NF==2 {print $1*60+$2; next}
  {print 0}')
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 - "$CODE_DIR" "$PLAN_REL" "$WORK" <<'PY'
from pathlib import Path
import sys
import torch
code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
work = Path(sys.argv[3]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan.is_relative_to(code):
    raise SystemExit('frozen campaign and contained plan required')
if torch.cuda.device_count() != 4:
    raise SystemExit('exactly four CUDA devices required')
PY
exec python3 scripts/logsquare_train.py --plan "$CODE_DIR/$PLAN_REL" --step "$STEP" \
  --expected-code-sha "$EXPECTED_SHA" --out "$OUT" --allocated-seconds "$SECS"
