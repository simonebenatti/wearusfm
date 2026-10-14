#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH -J step2c_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_extract_%j.out
# Step 2C (docs/step2c_frozen_branches.md): uscite dei soli rami (CPU, nessun encode raw) del run allenato e del passo 0 sulle finestre P1 o
# P2, poi le varianti RP, RW, RPW, RPW0 concatenate agli embedding gia' estratti del teacher d'origine. P2 va sottomesso con --time=03:00:00.
# Nessuna ripresa, nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/step2c_extract.sbatch <p1|p2> <arm> <runs A/B> \
#        <runs dei rami> <downstream> <eval>
set -euo pipefail
TASK="${1:?p1/p2}"; ARM="${2:?braccio}"; RUNS="${3:?runs A/B}"; BRANCH_RUNS="${4:?runs dei rami}"; DOWN="${5:?downstream}"; EVAL="${6:?eval}"
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step2c.py --stage extract --task "$TASK" --arm "$ARM" --expected-commit "$WEARUSFM_CODE_SHA" --runs-root "$RUNS" \
  --branch-runs "$BRANCH_RUNS" --downstream-root "$DOWN" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$WORK/data/raw/uci_emg"
