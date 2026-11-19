#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodelist=login08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH -J scaling_preserve49046
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_preserve49046_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_preserve49046_%j.out
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}"
[ "$#" -eq 3 ] || exit 2
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python3 scripts/scaling_preserve49046.py --plan "$1" --plan-sha "$2" --out "$3" --code "$EXPECTED_SHA"
