#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J logsquare_infer_benchmark
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_infer_benchmark_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_infer_benchmark_%j.out
# Args: relative_plan new_out expected_plan_sha256. No hidden submissions.
set -euo pipefail
[ "$#" -eq 3 ] || { echo 'three arguments required' >&2; exit 2; }
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"; OUT="$2"; PLAN_SHA="$3"
case "$PLAN_REL" in /*|"") echo 'relative contained plan required' >&2; exit 2;; esac
[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ] && [ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ] && [ "${SLURM_MEM_PER_NODE:-0}" -eq 102400 ] || { echo 'resource mismatch' >&2; exit 2; }
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
[ "$ALLOC" = 00:30:00 ] || { echo 'benchmark capped at exactly1800 seconds' >&2; exit 2; }
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
exec python3 scripts/logsquare_infer_benchmark.py --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA" --expected-plan-sha256 "$PLAN_SHA" --out "$OUT"
