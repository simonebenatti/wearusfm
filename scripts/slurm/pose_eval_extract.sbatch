#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH -J pose_eval_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_extract_%j.out

# Emendamento 1 a pilot_posa_v1 (E1-E3): latenti per istante, predizione della testa e target di posa su un insieme di emg2pose (val, o
# sottoinsieme del train per E3) per un modello-endpoint. 1 A100, ~40 min di tetto (stima: ~100 finestre/s; val = ~19.500 finestre).
# Dal checkout di campagna (CODE_DIR). Uscite fuori dal repo.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/pose_eval_extract.sbatch <checkpoint.pt> <out.npz> --split val
#      ... --split train --n-users 40 --n-recordings 200

set -uo pipefail
CKPT="${1:?checkpoint}"; OUT="${2:?uscita .npz}"; shift 2
case "$OUT" in /*) ;; *) echo "uscita relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "uscita dentro il codice: rifiutata"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 scripts/pose_eval.py --stage extract --checkpoint "$CKPT" --out "$OUT" --device cuda \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --processed-root "$SCRATCH/data/processed/emg2pose" --pose-root "$SCRATCH/data/processed/emg2pose_pose" "$@"
