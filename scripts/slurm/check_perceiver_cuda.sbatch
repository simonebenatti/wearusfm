#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:05:00
#SBATCH -J perceiver_cuda
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/perceiver_cuda_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/perceiver_cuda_%j.out

# P2 safe_scores: regression CUDA float32/bf16 with and without activation
# checkpoint recomputation. Functional test only; <= 1 GPU * 5 min = 0.0834 GPUh.
set -euo pipefail
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/check_perceiver_cuda.py
