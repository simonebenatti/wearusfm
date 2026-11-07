#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH -J logsquare_pose_readout
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_pose_readout_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_pose_readout_%j.out

# No submission, overrides of science, or writes inside the frozen checkout.
# validate/compare: 2 CPUs/1800 s; benchmark: 8 CPUs/900 s; ridge: 8 CPUs/1800 s.
# Submitter passes matching --cpus-per-task and --time; every job <=30 GiB.
set -euo pipefail
if [ "$#" -ne 4 ] && [ "$#" -ne 6 ]; then
  echo 'usage: logsquare_pose_readout.sbatch <relative plan> <stage> <new absolute output> <plan sha256> [candidate report candidate sha256]' >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
STAGE="$2"
OUT="$3"
PLAN_SHA="$4"
case "$PLAN_REL" in /*|"") echo 'contained relative plan required' >&2; exit 2;; esac
[[ "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid plan SHA256' >&2; exit 2; }
EXTRA=()
case "$STAGE" in
  validate) CPUS=2; SECONDS_CAP=1800;;
  benchmark) CPUS=8; SECONDS_CAP=900;;
  ridge) CPUS=8; SECONDS_CAP=1800;;
  compare)
    CPUS=2; SECONDS_CAP=1800
    [ "$#" -eq 6 ] || { echo 'compare requires candidate report and SHA256' >&2; exit 2; }
    case "$5" in /*) ;; *) echo 'absolute candidate report required' >&2; exit 2;; esac
    [[ "$6" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid candidate SHA256' >&2; exit 2; }
    EXTRA=(--candidate-report "$5" --candidate-report-sha256 "$6")
    ;;
  *) echo 'unregistered stage' >&2; exit 2;;
esac
if [ "$STAGE" != compare ] && [ "$#" -ne 4 ]; then
  echo 'candidate report belongs only to compare' >&2; exit 2
fi
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne "$CPUS" ] || [ "${SLURM_NTASKS:-0}" -ne 1 ]; then
  echo "one node/task and $CPUS CPUs required" >&2; exit 2
fi
if [ "${SLURM_MEM_PER_NODE:-0}" -le 0 ] || [ "${SLURM_MEM_PER_NODE:-0}" -gt 30720 ]; then
  echo 'positive memory at most 30 GiB required' >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' 'NF==4 {print $1*86400+$2*3600+$3*60+$4; next} NF==3 {print $1*3600+$2*60+$3; next} NF==2 {print $1*60+$2; next} {print 0}')
if [ "$SECS" -ne "$SECONDS_CAP" ]; then
  echo "allocation must equal published $STAGE cap ($SECONDS_CAP seconds)" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
python3 - "$CODE_DIR" "$PLAN_REL" "$OUT" "$WORK" "$PLAN_SHA" <<'PY'
from pathlib import Path
import hashlib
import sys
code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
out_arg = Path(sys.argv[3])
out = out_arg.resolve()
work = Path(sys.argv[4]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan.is_relative_to(code) or not plan.is_file():
    raise SystemExit('frozen campaign checkout and contained plan required')
if hashlib.sha256(plan.read_bytes()).hexdigest() != sys.argv[5]:
    raise SystemExit('plan SHA256 differs')
if not out_arg.is_absolute() or out_arg != out or out_arg.exists() or out_arg.is_symlink():
    raise SystemExit('canonical absolute new output required')
if any(out.is_relative_to(root) for root in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
    raise SystemExit('output inside code')
PY
export OMP_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
python3 scripts/logsquare_pose_readout.py "$STAGE" --plan "$CODE_DIR/$PLAN_REL" \
  --expected-code-sha "$EXPECTED_SHA" --expected-plan-sha256 "$PLAN_SHA" --out "$OUT" "${EXTRA[@]}"
code_dir_setup
