#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J probe_memory
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_memory_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_memory_%j.out

# Sonda della memoria per GPU dei rung 30M/100M/300M sul caso peggiore (Hyser, 256 canali, finestre da 4 s), 4 e 8 finestre per GPU, fino al
# primo aggiornamento AdamW (collaudo dell'infrastruttura, docs/collaudo_infra_gpu_1.md). 1 GPU, al piu' 30 min = 0,5 GPU-ora (4 ore locali).
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/probe_memory.sbatch <manifest> <scales> <window_seconds> <out.json>

set -uo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?session_scales.json}"; WS="${3:?window_seconds.json}"; OUT="${4:?file di uscita}"
case "$OUT" in /*) ;; *) echo "OUT relativo: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"/*|"$WORK/wearusfm_campaigns"/*) echo "OUT dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 scripts/probe_memory.py --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" --scales "$SCALES" \
  --window-seconds "$WS" --out "$OUT" ${EXTRA_ARGS:-}
