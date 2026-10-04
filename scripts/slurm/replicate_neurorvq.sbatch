#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH -J replica_nrvq
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/replicate_neurorvq_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/replicate_neurorvq_%j.out

# Replica dei numeri di NeuroRVQ-EMG (passo 5; foglio docs/foglio_replica_neurorvq.md, firmato il 04/10/2026). 1 GPU.
# Uso: sbatch [--qos=boost_qos_dbg --time=00:30:00] scripts/slurm/replicate_neurorvq.sbatch <epn612|uci_emg> <uscita.json> [args extra]
#      (collaudo: args extra --seeds 0 --epochs 2)
# Uscita fuori dal repo: $WORK/wearusfm_runs/results/passo5/...

set -uo pipefail
DATASET="${1:?epn612 o uci_emg}"; OUT="${2:?uscita .json}"; shift 2
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
RAW="$WORK/data/raw"
if [ "$DATASET" = epn612 ]; then
  ROOTS=(--root "$RAW/epn612/EMG-EPN612 Dataset/trainingJSON" --root "$RAW/epn612/EMG-EPN612 Dataset/testingJSON")
else
  ROOTS=(--root "$RAW/uci_emg")
fi
python3 scripts/replicate_neurorvq.py --dataset "$DATASET" "${ROOTS[@]}" --checkpoint "$WORK/models/NeuroRVQ_EMG_foundation_model_v1.pt" \
  --out "$OUT" "$@"
