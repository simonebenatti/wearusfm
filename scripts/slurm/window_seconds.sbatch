#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -J window_seconds
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/window_seconds_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/window_seconds_%j.out

# Lunghezza media delle finestre per sessione (quote nel tempo, decisione 1 del 04/10/2026). Legge solo i sidecar. CPU, 0 GPU-ora.
# Uso: sbatch scripts/slurm/window_seconds.sbatch <manifest.json.gz>
# Uscita fuori dal repo: $WORK/wearusfm_runs/results/passo6/window_seconds_<job>.json

set -uo pipefail
MANIFEST="${1:?manifest}"
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$WORK/wearusfm_runs/results/passo6" "$WORK/wearusfm_runs/logs"
python3 scripts/window_seconds.py --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --out "$WORK/wearusfm_runs/results/passo6/window_seconds_${SLURM_JOB_ID}.json" --workers "${SLURM_CPUS_PER_TASK:-4}"
