#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -J ingest_ninapro_std
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_std_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_std_%j.out

# Passo 2: ingest dei DB NinaPro a 12 elettrodi. Uso: sbatch scripts/slurm/ingest_ninapro_std.sbatch db4 [--subjects 1]
# Primo argomento = DB (db4). Raw da $SCRATCH (copia verificata per checksum il 29/09/2026);
# processato su $SCRATCH (per non aggravare la quota di $WORK: vedi CLAUDE.md), report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"
DB="$1"; shift

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/ninapro_${DB}"
python3 scripts/ingest_ninapro_std.py \
  --db "$DB" \
  --raw-root "$SCRATCH/data/raw/ninapro" \
  --out-root "$SCRATCH/data/processed/ninapro_${DB}" \
  --report "$WORK/wearusfm_runs/results/passo2/ninapro_${DB}_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
