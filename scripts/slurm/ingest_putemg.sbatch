#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ingest_putemg
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_putemg_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_putemg_%j.out

# Passo 2: ingest putEMG. Primo lancio: SOLO i primi 3 file (collaudo, formato HDF5
# pytables parsato a mano con h5py puro), prima di lanciare su tutti i 264.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 scripts/ingest_putemg.py \
  --raw-root "$WORK/data/raw/putemg/data-hdf5" \
  --out-root "$WORK/data/processed/putemg" \
  --limit 3 \
  --report "$WORK/wearusfm_runs/results/passo2/putemg_ingest_report_smoke.json"
