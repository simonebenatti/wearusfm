#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -J fix_capgmyo
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/fix_capgmyo_grid_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/fix_capgmyo_grid_%j.out

# Correzione della griglia di CapgMyo nei sidecar (16 x 8; Simone, 07/10/2026, opzione A; fatti 14 e 26). CPU, 0 GPU-ora. Riscrive SOLO i
# metadata.json di CapgMyo (backup in OUT/backup), scrive il manifest v1.1 e le scale senza le bipolari di CapgMyo. Dopo: measure_loader.sbatch
# <OUT/manifest_v1_1.json.gz> <OUT/session_scales_without_capgmyo_bipolar.json> --bipolar-scales --skip-rate.
# Uso: sbatch scripts/slurm/fix_capgmyo_grid.sbatch [--dry-run]

set -uo pipefail
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
R="$WORK/wearusfm_runs/results"
OUT="$R/passo6/capgmyo_grid_fix"
mkdir -p "$WORK/wearusfm_runs/logs"
python3 scripts/fix_capgmyo_grid.py --manifest "$R/passo4/manifest_59253155.json.gz" --root "$WORK/data/processed" \
  --root "$SCRATCH/data/processed" --scales "$R/passo6/loader_59412542/session_scales.json" --out-dir "$OUT" "$@"
