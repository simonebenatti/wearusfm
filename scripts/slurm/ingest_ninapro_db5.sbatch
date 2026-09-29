#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH -J ingest_ninapro_db5
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db5_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db5_%j.out

# Passo 2: ingest di NinaPro DB5 (193 MB di zip, 10 soggetti, 200 Hz, 2 Myo, 16 canali).
# Uso: sbatch scripts/slurm/ingest_ninapro_db5.sbatch [--subjects 1 ...]  (senza argomenti: tutti)
# Raw letto da $SCRATCH (copia verificata per checksum il 29/09/2026); processato in $WORK.
# Il report va fuori dal repo ($WORK/wearusfm_runs/).

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs"
python3 scripts/ingest_ninapro_db5.py \
  --raw-root "$SCRATCH/data/raw/ninapro" \
  --out-root "$WORK/data/processed/ninapro_db5" \
  --report "$WORK/wearusfm_runs/results/passo2/ninapro_db5_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
