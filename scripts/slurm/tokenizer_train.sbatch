#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH -J tokenizer_train
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_train_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/tokenizer_train_%j.out

# Three fresh arms, then own-state continuation. No scientific choices in launcher.
# Phase 5000: 90 min / 6 GPUh. Phase 19618: 240 min / 16 GPUh (additional 14618 steps).
# Submit via shared submit_job.py; controller validates the immutable phase receipt.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo 'usage: tokenizer_train.sbatch <relative plan> <arm> <5000|19618> <new receipt directory>' >&2
  exit 2
fi
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
PLAN_REL="$1"; ARM="$2"; STEP="$3"; OUT="$4"
case "$PLAN_REL" in /*|"") echo 'plan must be relative to code' >&2; exit 2;; esac
case "$ARM" in T25-L|T100-L|T100-G) ;; *) exit 2;; esac
case "$STEP" in 5000) SECONDS_EXPECTED=5400;; 19618) SECONDS_EXPECTED=14400;; *) exit 2;; esac
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
if [ "$SECS" -ne "$SECONDS_EXPECTED" ]; then
  echo "wrong phase walltime: $ALLOC" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

python3 - "$CODE_DIR" "$PLAN_REL" "$ARM" "$STEP" "$OUT" "$WORK" "$SECS" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
import torch

code = Path(sys.argv[1]).resolve(strict=True)
plan_path = (code / sys.argv[2]).resolve(strict=True)
arm, step = sys.argv[3:5]
out_arg = Path(sys.argv[5])
out = out_arg.resolve()
work = Path(sys.argv[6]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan_path.is_relative_to(code):
    raise SystemExit('frozen campaign and contained plan required')
if not out_arg.is_absolute() or out_arg.exists() or out_arg.is_symlink():
    raise SystemExit('absolute new receipt directory required')
if any(out.is_relative_to(r) for r in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
    raise SystemExit('output inside code')
plan = json.loads(plan_path.read_text())
if plan['allocations'][arm][step]['walltime_seconds'] != int(sys.argv[7]):
    raise SystemExit('allocation differs from plan')
if torch.cuda.device_count() != 4:
    raise SystemExit('exactly four CUDA devices required')
pin = plan['preflight_report']
report_path = (code / pin['path']).resolve(strict=True)
if hashlib.sha256(report_path.read_bytes()).hexdigest() != pin['sha256']:
    raise SystemExit('preflight hash mismatch')
report = json.loads(report_path.read_text())
if report.get('complete') is not True or report.get('paired_stream_query_targets_and_pose_support') is not True:
    raise SystemExit('preflight incomplete')
if report.get('batches') != 64 or report.get('windows_exposed_each_arm') != 2048:
    raise SystemExit('preflight dose mismatch')
for rel, sha in plan['scientific_source_sha256'].items():
    if hashlib.sha256((code / rel).read_bytes()).hexdigest() != sha:
        raise SystemExit('scientific source changed since preflight: ' + rel)
PY

exec python3 scripts/tokenizer_train.py --plan "$CODE_DIR/$PLAN_REL" \
  --arm "$ARM" --step "$STEP" --expected-code-sha "$EXPECTED_SHA" --out "$OUT"
