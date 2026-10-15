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
#SBATCH -J step1_random_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_extract_%j.out
# Args: full SHA, p1/p2, model seed, historical training root, NEW eval root.
# P2 requires sbatch --time=01:00:00. Frozen CODE_DIR mandatory; no resume/retry.
set -euo pipefail
export EXPECTED_SHA="${1:?commit}"
TASK="${2:?p1/p2}"; SEED="${3:?seed}"; RUNS="${4:?runs}"; EVAL="${5:?eval}"
: "${CODE_DIR:?frozen CODE_DIR required}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step1_random_downstream.py --stage extract --expected-commit "$EXPECTED_SHA" --task "$TASK" --model-seed "$SEED" \
  --runs-root "$RUNS" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$WORK/data/raw/uci_emg"
