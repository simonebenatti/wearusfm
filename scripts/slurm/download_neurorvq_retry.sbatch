#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -J dl_neurorvq_retry
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_neurorvq_retry_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_neurorvq_retry_%j.out

# Ritenta SOLO il checkpoint NeuroRVQ, fallito nel job 58329241 per un
# ModuleNotFoundError (huggingface_hub non disponibile sul Python di sistema del login
# node). Fix: wget diretto, niente libreria Python. Vedi download_small_public.sbatch,
# gia' corretto per i lanci futuri - questo script serve solo a recuperare il pezzo
# mancante senza riscaricare 375 GB gia' presenti.

set -uo pipefail

fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "gia' presente: $dest"
    return 0
  fi
  echo "scarico: $url -> $dest"
  wget -q -c --tries=3 --timeout=120 -O "$dest" "$url" && echo "  ok" || echo "  FALLITO: $url"
}

NEURORVQ_DIR="$WORK/data/raw/neurorvq_checkpoint"
HF_BASE="https://huggingface.co/ntinosbarmpas/NeuroRVQ/resolve/main"
fetch "$HF_BASE/pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt" \
      "$NEURORVQ_DIR/NeuroRVQ_EMG_tokenizer_v1.pt"
fetch "$HF_BASE/pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt" \
      "$NEURORVQ_DIR/NeuroRVQ_EMG_foundation_model_v1.pt"
mkdir -p "$WORK/models"
cp -n "$NEURORVQ_DIR"/*.pt "$WORK/models/" 2>&1 || true

echo "=== riepilogo ==="
ls -la "$NEURORVQ_DIR" "$WORK/models"
