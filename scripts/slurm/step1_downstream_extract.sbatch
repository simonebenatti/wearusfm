#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J step1_downstream_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_downstream_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_downstream_extract_%j.out
# Arguments: expected commit, p1/p2, A_seed0/etc, runs root, NEW evaluation root.
# P2 MUST be submitted with --time=01:00:00. No resume/retry.
set -euo pipefail
EXPECTED="${1:?commit}"; TASK="${2:?p1/p2}"; ARM="${3:?arm}"; RUNS="${4:?runs}"; EVAL="${5:?eval}"
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step1_downstream.py --stage extract --expected-commit "$EXPECTED" --task "$TASK" --arm "$ARM" \
  --runs-root "$RUNS" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$WORK/data/raw/uci_emg"
