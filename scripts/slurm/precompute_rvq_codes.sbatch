#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH -J rvq_codes
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/rvq_codes_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/rvq_codes_%j.out

# Codici dell'ancora RVQ (D5b, livello 0 del ramo 0) per le sessioni di pretraining con l'ancora accesa (passo 6). 1 GPU, 8 CPU (3 processi per la
# vista canonica, al piu' 5 sessioni in volo: job 59303528 fuori memoria con 6 processi e risultati senza limite). Riprendibile: le sessioni gia' scritte si saltano. Il clone di NeuroRVQ (commit 926e770, CC BY-NC 4.0) sta FUORI dal repo.
# Uso: sbatch scripts/slurm/precompute_rvq_codes.sbatch <manifest.json.gz> <dataset[,dataset...]> [args extra]
# Uscita fuori dal repo: $WORK/wearusfm_runs/results/passo6/rvq_codes/ (un .npz per sessione, index_<job>.json).

set -uo pipefail
MANIFEST="${1:?manifest}"; DATASETS="${2:?dataset}"; shift 2
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo6/rvq_codes" "$WORK/wearusfm_runs/logs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 scripts/precompute_rvq_codes.py --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --datasets "$DATASETS" --repo-dir "$SCRATCH/external/NeuroRVQ_926e770" --checkpoint "$WORK/models/NeuroRVQ_EMG_tokenizer_v1.pt" \
  --out-dir "$WORK/wearusfm_runs/results/passo6/rvq_codes" --workers 3 --time-limit-s 10500 "$@"
