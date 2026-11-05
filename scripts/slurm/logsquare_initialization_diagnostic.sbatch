#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:15:00
#SBATCH -J logsquare_initialization_diagnostic
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_initialization_diagnostic_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_initialization_diagnostic_%j.out

# Diagnostic only: five fresh Python processes, CPU construction and CUDA copy;
# then the corrected launch up to the student's construction: the controller
# pre-check and the production fresh-phase command under torchrun with
# --verify-only. Four ranks build and verify their student and stop inside the
# constructor.
# No model reaches the trainer: no DDP wrapper, data forward, optimizer or training;
# only constructor synthetic calibration. No training restart here.
# 4 GPUs preserve the failed controller's allocation context; the construction
# children use only cuda:0, the torchrun dry run starts one rank per GPU.
# Submit only after diagnostic review, via submit_job.py and a fresh frozen checkout.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'usage: logsquare_initialization_diagnostic.sbatch <relative plan> <new absolute output>' >&2
  exit 2
fi
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
PLAN_REL="$1"
OUT="$2"
case "$PLAN_REL" in /*|"") echo 'relative plan required' >&2; exit 2;; esac
case "$OUT" in /*) ;; *) echo 'absolute output required' >&2; exit 2;; esac
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 32 ]; then
  echo 'one node and 32 CPUs required' >&2
  exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
# This short separate process cannot contaminate the diagnostic children's imports.
python3 - "$CODE_DIR" "$PLAN_REL" "$WORK" "$JOB_INFO" <<'PY'
import json
from pathlib import Path
import sys
import torch
code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
work = Path(sys.argv[3]).resolve()
fields = dict(word.split('=', 1) for word in sys.argv[4].split() if '=' in word)
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan.is_relative_to(code):
    raise SystemExit('frozen campaign and contained plan required')
if fields.get('TimeLimit') != '00:15:00' or torch.cuda.device_count() != 4:
    raise SystemExit('exactly fifteen minutes and four visible GPUs required')
p = json.loads(plan.read_text())
if (not p.get('cuda_required') or p.get('child_timeout_seconds') != 90 or p.get('rank_entry_timeout_seconds') != 300
        or len(p.get('variants', ())) != 6):
    raise SystemExit('production diagnostic requires CUDA, five bounded 90s children and one bounded 300s torchrun dry run')
PY
exec python3 scripts/logsquare_initialization_diagnostic.py \
  --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA" --out "$OUT"
