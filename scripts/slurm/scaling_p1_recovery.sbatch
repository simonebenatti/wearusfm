#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH -J SCALING-300M-eval19618-p1-r2-20261116
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_recovery_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_recovery_%j.out

# Conditional single P1-r2 only. Two hours +180s scheduler grace =2.05GPUh.
# No retries. The frozen wrapper checks the positive diagnostic and all pins.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'usage: <relative recovery manifest> <manifest sha256>' >&2
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# CUDA check runs in a separate Python process: no import-order change in the
# following wrapper process or its unchanged numerical extraction subprocess.
python3 scripts/scaling_p1_recovery_launch.py --code "$CODE_DIR" --manifest "$1" \
  --manifest-sha256 "$2" --job "$SLURM_JOB_ID" --job-info "$JOB_INFO"
module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec python3 scripts/scaling_p1_recovery.py --manifest "$CODE_DIR/$1" \
  --manifest-sha256 "$2" --expected-code-sha "$EXPECTED_SHA"
