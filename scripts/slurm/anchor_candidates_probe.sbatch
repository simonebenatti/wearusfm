#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J anchor_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/anchor_candidates_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/anchor_candidates_probe_%j.out

# Confronto dei target candidati, fase delle sonde (CPU, 0 GPU-ora). Uso: sbatch [--dependency=afterok:<job>] anchor_candidates_probe.sbatch <unita'.npz> <report.json>
set -uo pipefail
UNITS="${1:?file delle unita .npz}"; OUT="${2:?report .json}"
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$(dirname "$OUT")"
python3 scripts/anchor_candidates.py probe --units "$UNITS" --out "$OUT"
