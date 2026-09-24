#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -J harness_smoke_uci_emg
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/harness_smoke_uci_emg_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/harness_smoke_uci_emg_%j.out

# Passo 5: secondo collaudo dell'harness su dati reali, UCI-EMG. Esclude v10 §2.3 dal
# pretraining, uno dei due benchmark "mai visto" del passo 5.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 bench/harness_smoke_uci_emg.py \
  --root "$WORK/data/raw/uci_emg" \
  --window-samples 200 --stride-samples 100 \
  --n-subjects 36 --seed 0 \
  --out "$WORK/wearusfm_runs/results/passo5/uci_emg_smoke.json"
