#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH -J ddp_equivalence
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ddp_equivalence_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ddp_equivalence_%j.out
# Prova (A) del protocollo w1_gpu_dati_v1 (docs/protocollo_gpu_dati_finestra1.md): equivalenza numerica 1 contro 4 GPU (NCCL) e accumulo, in
# float32 (TF32 spento) e bf16, su 8 batch globali fissi del preset 30M con le regole della finestra 1. Nessun training lungo, nessun
# checkpoint; un rapporto JSON in una cartella nuova. Costo: 4 GPU per la durata del job (<= 30 min su dbg = 2 GPU-ora).
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/ddp_equivalence.sbatch <manifest> <session_scales.json> \
#        <window_seconds.json> <rapporto NUOVO .json>
set -euo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?scale}"; WINDOW="${3:?durate}"; OUT="${4:?rapporto nuovo}"
case "$OUT" in /*) ;; *) echo "percorso assoluto richiesto"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
torchrun --standalone --nproc_per_node="$NGPU" scripts/ddp_equivalence.py --manifest "$MANIFEST" --scales "$SCALES" \
  --window-seconds "$WINDOW" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" --out "$OUT"
