#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -J sanity_jepa
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/sanity_jepa_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/sanity_jepa_%j.out

# Sanity JEPA (passo 6): preset `sanity` su emg2qwerty, valori firmati il 03/10/2026. 1 GPU, 30 min (0,5 GPU-ora per job). Riprende da
# checkpoint.pt nella stessa cartella: si concatenano job con --dependency=afterany sullo stesso RUN_DIR.
# Uso: sbatch scripts/slurm/sanity_jepa.sbatch <manifest.json.gz> <session_scales.json> <run_dir> [passi massimi]
# Uscita fuori dal repo: <run_dir> (metrics.jsonl, checkpoint.pt, summary.json, config.json), sotto $WORK/wearusfm_runs/runs/.

set -uo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?session_scales.json}"; RUN_DIR="${3:?cartella del run}"; MAX_STEPS="${4:-20000}"
case "$RUN_DIR" in "$WORK/wearusfm"/*) echo "RUN_DIR dentro il repo: rifiutato"; exit 2;; esac
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$RUN_DIR" "$WORK/wearusfm_runs/logs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 scripts/train_jepa.py --preset sanity --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --scales "$SCALES" --out-dir "$RUN_DIR" --max-steps "$MAX_STEPS" --num-workers 6 --time-limit-s 1560
