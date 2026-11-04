#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J consolidation_ft
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_ft_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_ft_%j.out
# FIT requires explicit --time=04:00:00 through the shared submitter.
set -euo pipefail
[ "$#" -ge 2 ] || { echo 'usage: consolidation_finetune.sbatch benchmark|fit OUT [FOLD REGIME]' >&2; exit 2; }
MODE="$1"; OUT="$2"
case "$MODE" in
  benchmark) [ "$#" -eq 2 ] || exit 2; EXPECTED_TIME=00:30:00; EXTRA=();;
  fit) [ "$#" -eq 4 ] || exit 2; EXPECTED_TIME=04:00:00; EXTRA=(--fold "$3" --regime "$4");;
  *) exit 2;;
esac
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ] && [ "${SLURM_CPUS_PER_TASK:-0}" -eq 8 ] || exit 2
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - "$EXPECTED_TIME" <<'PY'
import os, subprocess, sys, torch
info = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True).split()
if 'TimeLimit=' + sys.argv[1] not in info or torch.cuda.device_count() != 1:
    raise SystemExit('wrong time allocation or GPU count')
if int(os.environ.get('SLURM_MEM_PER_NODE', '0')) != 100 * 1024:
    raise SystemExit('allocation must be 100GiB')
PY
exec python3 scripts/consolidation_finetune.py \
  --plan results/infra/consolidation_20261104/finetune_plan.json --expected-code "$EXPECTED_SHA" \
  --mode "$MODE" --out "$OUT" "${EXTRA[@]}"
