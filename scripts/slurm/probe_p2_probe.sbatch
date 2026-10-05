#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J probe_p2_lin
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p2_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p2_probe_%j.out

# Metrica P2, fase delle sonde (CPU, 0 GPU-ora). Uso: sbatch [--dependency=afterok:<job>] probe_p2_probe.sbatch <checkpoint.pt> <cartella delle
# feature> <report.json> (il checkpoint serve solo al report)
set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle feature}"; OUT="${3:?report .json}"
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$(dirname "$OUT")"
python3 scripts/probe_p2.py --stage probe --checkpoint "$CKPT" --features "$FEAT" --out "$OUT"
