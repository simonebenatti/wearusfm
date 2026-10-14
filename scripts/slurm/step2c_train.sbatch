#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -J step2c_train
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_train_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_train_%j.out
# Step 2C dello step 2 (docs/step2c_frozen_branches.md): rami PSD e wavelet allenati sopra il teacher <arm> dell'A/B dello Step 1, congelato.
# QoS e tempo al momento della sottomissione:
#   collaudo (B_seed0): --qos=boost_qos_dbg --time=00:30:00, quarto argomento "collaudo" (cartella collaudo/, verdetto congelato);
#   run:                --qos=boost_qos_lprod --time=<run_allocation_h del verdetto del collaudo>.
# Il limite di tempo del training e' l'allocazione meno 5 minuti (salvataggio e controlli finali). Nessuna ripresa, nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> --qos=... --time=... scripts/slurm/step2c_train.sbatch \
#        <A_seed0|A_seed1|B_seed0|B_seed1> <runs A/B> <runs dei rami> [collaudo]
set -euo pipefail
ARM="${1:?braccio}"; RUNS="${2:?runs A/B}"; BRANCH_RUNS="${3:?runs dei rami}"; MODE="${4:-run}"
case "$ARM" in A_seed0|A_seed1|B_seed0|B_seed1) ;; *) echo "braccio non valido: $ARM"; exit 2;; esac
case "$MODE" in run) EXTRA=();; collaudo) EXTRA=(--collaudo);; *) echo "modo: run o collaudo"; exit 2;; esac
ALLOC=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(echo "$ALLOC" | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3;
                                       else if (NF==2) print $1*60+$2; else print 0 }')
if [ "$SECS" -le 600 ]; then echo "allocazione $ALLOC troppo corta"; exit 2; fi
if [ "$MODE" = collaudo ] && [ "$SECS" -gt 1800 ]; then echo "collaudo: al piu' 30 minuti (dbg), allocazione $ALLOC"; exit 2; fi
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/step2c.py --stage train --arm "$ARM" "${EXTRA[@]}" --expected-commit "$WEARUSFM_CODE_SHA" --runs-root "$RUNS" \
  --branch-runs "$BRANCH_RUNS" --time-limit-s $((SECS - 300)) --device cuda --num-workers 6 \
  --manifest "$WORK/wearusfm_runs/results/passo6/capgmyo_grid_fix/manifest_v1_1.json.gz" \
  --scales "$WORK/wearusfm_runs/results/passo6/loader_59623700/session_scales.json" \
  --window-seconds "$WORK/wearusfm_runs/results/passo6/window_seconds_59425316.json" \
  --calibration "$WORK/wearusfm_runs/results/passo6/step1_0710/calibration.npz" \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed"
