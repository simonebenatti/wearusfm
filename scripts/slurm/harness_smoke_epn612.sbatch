#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -J harness_smoke_epn612
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/harness_smoke_epn612_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/harness_smoke_epn612_%j.out

# Passo 5: primo collaudo dell'harness (splits/normalization/features Hudgins/baseline
# LDA/protocol) contro dati EPN-612 veri, non solo i test sintetici. EPN-612 e' escluso
# dal pretraining (v10 §2.3) - qui e' uno dei due benchmark "mai visto" del passo 5.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 bench/harness_smoke_epn612.py \
  --root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" \
  --n-users 30 --seed 0 \
  --out "$WORK/wearusfm_runs/results/passo5/epn612_smoke.json"
