#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -J scaling_continue49046
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_continue49046_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_continue49046_%j.out
set -euo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}"
[ "$#" -eq 2 ] || { echo 'Expected relative plan and new receipt directory' >&2; exit 2; }
case "$1" in /*|'') echo 'Plan must be relative to frozen checkout' >&2; exit 2;; esac
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python3 scripts/scaling_continue49046.py run --plan "$CODE_DIR/$1" --expected-code-sha "$EXPECTED_SHA" --out "$2"
