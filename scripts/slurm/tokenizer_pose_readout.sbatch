#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH -J tokenizer_pose_readout
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_pose_readout_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_pose_readout_%j.out

# CPU-only wrapper; no new cache pins, scientific overrides or job submission.
# Caps from full_resource_policy.json: validate 2 CPU/30 min (1 coreh),
# benchmark 8 CPU/15 min (2 coreh), ridge 8 CPU/30 min (4 coreh), always <=30G.
# Override sbatch --cpus-per-task/--time for validate and benchmark.
# Optional compare: 2 CPU/30 min ceiling; NOT an additional reserved job.
# Ridge already writes the paired comparison. A separate compare needs an
# explicitly accounted execution; it does not enlarge the campaign CPU budget.
# Args: <relative plan> <stage> <new absolute output> [benchmark arm | cells report cells SHA256].
# CODE_DIR and EXPECTED_SHA identify a new frozen campaign checkout, never an active mutable tree.
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo 'usage: tokenizer_pose_readout.sbatch <relative plan> <validate|benchmark|ridge|compare> <new absolute output> [benchmark arm | cells report cells SHA256]' >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
STAGE="$2"
OUT="$3"
EXTRA=()
case "$PLAN_REL" in /*|"") echo 'plan must be relative to code' >&2; exit 2;; esac
case "$STAGE" in
  validate)
    [ "$#" -eq 3 ] || { echo 'validate takes no arm or cells' >&2; exit 2; }
    CPUS_EXPECTED=2
    SECONDS_CAP=1800
    ;;
  benchmark)
    [ "$#" -eq 4 ] || { echo 'benchmark requires an explicit arm' >&2; exit 2; }
    case "$4" in T25-L|T100-L|T100-G) ;; *) echo 'unregistered benchmark arm' >&2; exit 2;; esac
    EXTRA=(--arm "$4")
    CPUS_EXPECTED=8
    SECONDS_CAP=900
    ;;
  ridge)
    [ "$#" -eq 3 ] || { echo 'ridge fits all three arms and takes no arm or cells' >&2; exit 2; }
    CPUS_EXPECTED=8
    SECONDS_CAP=1800
    ;;
  compare)
    [ "$#" -eq 5 ] || { echo 'compare requires a completed cells report and its SHA256' >&2; exit 2; }
    case "$4" in /*) ;; *) echo 'cells report must be absolute' >&2; exit 2;; esac
    [[ "$5" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid cells SHA256' >&2; exit 2; }
    EXTRA=(--cells "$4" --cells-sha256 "$5")
    CPUS_EXPECTED=2
    SECONDS_CAP=1800
    ;;
  *) echo 'unregistered stage' >&2; exit 2;;
esac
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne "$CPUS_EXPECTED" ]; then
  echo "one node and $CPUS_EXPECTED CPUs per task required for $STAGE" >&2; exit 2
fi
if [ "${SLURM_MEM_PER_NODE:-0}" -le 0 ] || [ "${SLURM_MEM_PER_NODE:-0}" -gt 30720 ]; then
  echo 'positive node memory not exceeding 30G required' >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4; next}
  NF==3 {print $1*3600+$2*60+$3; next}
  NF==2 {print $1*60+$2; next}
  {print 0}')
if [ "$SECS" -le 0 ] || [ "$SECS" -gt "$SECONDS_CAP" ]; then
  echo "walltime $ALLOC exceeds the $STAGE ceiling of $SECONDS_CAP seconds or is invalid" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="$CPUS_EXPECTED" OPENBLAS_NUM_THREADS="$CPUS_EXPECTED" MKL_NUM_THREADS="$CPUS_EXPECTED"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""

python3 - "$CODE_DIR" "$PLAN_REL" "$OUT" "$WORK" <<'PY'
from pathlib import Path
import sys

code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
out_arg = Path(sys.argv[3])
out = out_arg.resolve()
work = Path(sys.argv[4]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan.is_relative_to(code) or not plan.is_file():
    raise SystemExit('frozen campaign checkout and contained plan required')
if not out_arg.is_absolute() or out_arg != out or out_arg.exists() or out_arg.is_symlink():
    raise SystemExit('canonical absolute new output directory required')
if any(out.is_relative_to(root) for root in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
    raise SystemExit('output inside code')
# The reader itself validates the complete cache grid, source hashes and plan
# semantics; benchmark opens only the requested train cache and train reference.
PY

module list 2>&1
python3 scripts/tokenizer_pose_readout.py "$STAGE" --plan "$CODE_DIR/$PLAN_REL" \
  --expected-code-sha "$EXPECTED_SHA" --out "$OUT" "${EXTRA[@]}"
code_dir_setup
