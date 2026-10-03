#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -J measure_loader
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/measure_loader_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/measure_loader_%j.out

# Scale di sessione per tutte le sessioni di pretraining e misura del ritmo del dataloader (passo 6; autorizzato da Simone il 03/10/2026: «sì, via
# alla misura con le scale di sessione»; conferma della decisione 13, filtro nel dataloader). CPU, 0 GPU-ora. Solo lettura dei dati.
# Uso: sbatch scripts/slurm/measure_loader.sbatch <manifest.json.gz>
# Uscita fuori dal repo: $WORK/wearusfm_runs/results/passo6/loader_<job>/ (session_scales.json, throughput.json)

set -uo pipefail
MANIFEST="${1:?serve il percorso del manifest}"
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

OUT="$WORK/wearusfm_runs/results/passo6/loader_${SLURM_JOB_ID}"
mkdir -p "$OUT" "$WORK/wearusfm_runs/logs"
python3 scripts/measure_loader.py --manifest "$MANIFEST" \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --out-dir "$OUT" --workers "${SLURM_CPUS_PER_TASK:-4}" --batches 100 --batch-size 8
