#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH -J tokenizer_evaluation
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_evaluation_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_evaluation_%j.out

# Thin launcher for the guarded teacher extraction controller; no scientific overrides.
# Pose: train+val, 45 min / 0.75 GPUh. P1: 30 min / 0.50 GPUh (submit --time=00:30:00).
# P2: 45 min / 0.75 GPUh. Classification exists only at step 19618.
# Submit through shared submit_job.py, with CODE_DIR and EXPECTED_SHA exported.
# Args: <relative plan> <arm> <5000|19618> <stage> <p1|p2|none> <new absolute output>.
set -euo pipefail
if [ "$#" -ne 6 ]; then
  echo 'usage: tokenizer_evaluation.sbatch <relative plan> <arm> <5000|19618> <validate|extract-pose|extract-classification> <p1|p2|none> <new absolute output>' >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
ARM="$2"
STEP="$3"
STAGE="$4"
TASK="$5"
OUT="$6"
case "$PLAN_REL" in /*|"") echo 'plan must be relative to code' >&2; exit 2;; esac
case "$ARM" in T25-L|T100-L|T100-G) ;; *) echo 'unregistered arm' >&2; exit 2;; esac
case "$STEP" in 5000|19618) ;; *) echo 'unregistered endpoint' >&2; exit 2;; esac
case "$STAGE:$TASK:$STEP" in
  validate:none:*|extract-pose:none:*) SECONDS_EXPECTED=2700;;
  extract-classification:p1:19618) SECONDS_EXPECTED=1800;;
  extract-classification:p2:19618) SECONDS_EXPECTED=2700;;
  *) echo 'invalid stage/task/endpoint combination' >&2; exit 2;;
esac
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo 'one node and eight CPUs per task required' >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4; next}
  NF==3 {print $1*3600+$2*60+$3; next}
  NF==2 {print $1*60+$2; next}
  {print 0}')
if [ "$SECS" -ne "$SECONDS_EXPECTED" ]; then
  echo "wrong extraction walltime: $ALLOC, expected $SECONDS_EXPECTED seconds" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

python3 - "$CODE_DIR" "$PLAN_REL" "$ARM" "$STEP" "$OUT" "$WORK" <<'PY'
from pathlib import Path
import json
import sys
import torch

code = Path(sys.argv[1]).resolve(strict=True)
plan_path = (code / sys.argv[2]).resolve(strict=True)
arm, step = sys.argv[3:5]
out_arg = Path(sys.argv[5])
out = out_arg.resolve()
work = Path(sys.argv[6]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan_path.is_relative_to(code) or not plan_path.is_file():
    raise SystemExit('frozen campaign and contained plan required')
if not out_arg.is_absolute() or out_arg != out or out_arg.exists() or out_arg.is_symlink():
    raise SystemExit('canonical absolute new output directory required')
if any(out.is_relative_to(root) for root in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
    raise SystemExit('output inside code')
plan = json.loads(plan_path.read_text())
if plan.get('protocol') != 'tokenizer-evaluation-20261103-v1' or arm not in plan.get('checkpoints', {}).get(step, {}):
    raise SystemExit('wrong extraction plan or missing checkpoint pin')
if torch.cuda.device_count() != 1:
    raise SystemExit('exactly one visible CUDA device required')
PY

module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARGS=(--stage "$STAGE" --plan "$CODE_DIR/$PLAN_REL" --arm "$ARM" --step "$STEP"
      --expected-code-sha "$EXPECTED_SHA" --out "$OUT" --device cuda)
if [ "$TASK" != none ]; then
  ARGS+=(--task "$TASK")
fi
exec python3 scripts/tokenizer_evaluation.py "${ARGS[@]}"
