#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:27:00
#SBATCH -J scaling_p1_identity_diag
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_identity_diag_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_identity_diag_%j.out

# One approved diagnostic; one allocated GPU unused. 1500s application deadline,
# 1620s requested +180s scheduler grace = maximum .5GPUh /4 coreh. No retries.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'usage: <relative manifest> <manifest sha256> <new absolute diagnostic output>' >&2
  exit 2
fi
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
python3 scripts/scaling_p1_diagnostic_launch.py --code "$CODE_DIR" --manifest "$1" \
  --manifest-sha256 "$2" --out "$3" --job "$SLURM_JOB_ID" --job-info "$JOB_INFO"
module list 2>&1
exec python3 scripts/scaling_p1_identity_diagnostic.py --manifest "$CODE_DIR/$1" \
  --manifest-sha256 "$2" --out "$3" --expected-code-sha "$EXPECTED_SHA"
