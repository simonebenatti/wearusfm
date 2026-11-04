#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH -J consolidation
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_%j.out
set -euo pipefail
[ "$#" -eq 1 ] || { echo 'usage: consolidation_continue.sbatch 30000|39237|49046' >&2; exit 2; }
case "$1" in 30000|39237|49046) ;; *) echo 'unregistered continuation endpoint' >&2; exit 2;; esac
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ] && [ "${SLURM_CPUS_PER_TASK:-0}" -eq 32 ] || { echo 'allocation requires one node and 32 CPUs' >&2; exit 2; }
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - <<'PY'
import os, subprocess, torch
info = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True).split()
if 'TimeLimit=03:00:00' not in info or torch.cuda.device_count() != 4:
    raise SystemExit('allocation must be exactly 3h and four GPUs')
if int(os.environ.get('SLURM_MEM_PER_NODE', '0')) != 480 * 1024:
    raise SystemExit('allocation must be 480GiB')
PY
exec python3 scripts/consolidation_continue.py \
  --plan results/infra/consolidation_20261104/continuation_plan.json \
  --step "$1" --expected-code "$EXPECTED_SHA"
