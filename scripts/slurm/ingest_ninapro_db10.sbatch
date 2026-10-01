#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH -J ingest_ninapro_db10
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db10_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db10_%j.out

# Passo 2: ingest di NinaPro DB10 = MeganePro, solo MDS1 (45 file S<NNN>_ex1.mat da ~1,5 GB, 12 EMG a 1926 Hz, ~56 h).
# Uso: sbatch scripts/slurm/ingest_ninapro_db10.sbatch --subjects 10 108     (collaudo: un normodotato e l'amputato col solo anello prossimale)
#      sbatch scripts/slurm/ingest_ninapro_db10.sbatch --skip-existing        (pieno, riprendibile)
# Lettura di un file: ~2 minuti (misurata sul login node il 01/10/2026); budget 2 h 40 min su 3 h; processato su $SCRATCH (~9 GB in int16).
# Memoria: emg di un soggetto in float64 ~0,8 GB piu' etichette e copie: 16 GB.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/ninapro_db10"
python3 scripts/ingest_ninapro_db10.py \
  --raw-root "$SCRATCH/data/raw/ninapro" \
  --out-root "$SCRATCH/data/processed/ninapro_db10" \
  --report "$WORK/wearusfm_runs/results/passo2/ninapro_db10_ingest_report_${SLURM_JOB_ID}.json" \
  --time-budget-s 9600 \
  "$@"
