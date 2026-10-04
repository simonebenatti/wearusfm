#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J anchor_cand
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/anchor_candidates_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/anchor_candidates_%j.out

# Confronto dei target candidati per le ancore in frequenza, fase di estrazione (GPU; protocollo congelato in docs/decisioni.md, 04/10/2026).
# Uso: sbatch scripts/slurm/anchor_candidates.sbatch <checkpoint.pt> <uscita.npz>
# Le sonde (CPU): python3 scripts/anchor_candidates.py probe --units <uscita.npz> --out <report.json>, in un job su lrd_all_serial.

set -uo pipefail
CKPT="${1:?checkpoint}"; OUT="${2:?uscita .npz}"
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
R="$WORK/wearusfm_runs/results"
python3 scripts/anchor_candidates.py extract --manifest "$R/passo4/manifest_59253155.json.gz" --root "$WORK/data/processed" \
  --root "$SCRATCH/data/processed" --scales "$R/passo6/loader_59276700/session_scales.json" --rvq-codes "$R/passo6/rvq_codes" \
  --checkpoint "$CKPT" --repo-dir "$SCRATCH/external/NeuroRVQ_926e770" --tokenizer-checkpoint "$WORK/models/NeuroRVQ_EMG_tokenizer_v1.pt" \
  --out "$OUT"
