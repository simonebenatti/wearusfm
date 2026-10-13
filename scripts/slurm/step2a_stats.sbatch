#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH -J step2a_stats
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2a_stats_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2a_stats_%j.out
# Test 2A dello step 2 (docs/step2a_stats_probe.md): statistiche spettrali fisse sulle finestre P1 o P2 (CPU, nessuna GPU), poi le cache
# concatenate agli embedding gia' estratti dei teacher B. P2 va sottomesso con --time=02:00:00. Nessuna ripresa, nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/step2a_stats.sbatch <p1|p2> <runs> <downstream> <eval NUOVA>
set -euo pipefail
TASK="${1:?p1/p2}"; RUNS="${2:?runs A/B}"; DOWN="${3:?cartella downstream}"; EVAL="${4:?cartella nuova}"
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step2a.py --stage stats --task "$TASK" --expected-commit "$WEARUSFM_CODE_SHA" --runs-root "$RUNS" --downstream-root "$DOWN" \
  --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
  --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$WORK/data/raw/uci_emg"
