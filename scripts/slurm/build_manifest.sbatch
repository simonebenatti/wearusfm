#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -J build_manifest
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/build_manifest_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/build_manifest_%j.out

# Manifest del pretraining (D9): legge sidecar e intestazioni degli array di tutti i dataset processati (nessun dato EMG caricato) e scrive il manifest.
# Uso: sbatch scripts/slurm/build_manifest.sbatch                                  (bozza con gli split di splits/draft/splits_draft.json)
#      sbatch scripts/slurm/build_manifest.sbatch --version manifest-v1 --splits <split firmati>   (SOLO dopo la firma di D9)
# Uscita fuori dal repo: $WORK/wearusfm_runs/results/passo4/manifest_<job>.json.gz

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo4" "$WORK/wearusfm_runs/logs"
python3 scripts/build_manifest.py \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --splits splits/draft/splits_draft.json \
  --out "$WORK/wearusfm_runs/results/passo4/manifest_${SLURM_JOB_ID}.json.gz" \
  "$@"
