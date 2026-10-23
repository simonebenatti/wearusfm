#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -J branch_pilot_train
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/branch_pilot_train_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/branch_pilot_train_%j.out
# Pilot dei rami (docs/rami_pilota_obiettivo_v1.md, scripts/branch_pilot.py): un braccio nuovo (PWl, PWf, PWm) sul teacher B congelato, dal
# passo 0 del 2C (step2c_1410), stessi ingressi del 2C. QoS e tempo al momento della sottomissione:
#   collaudo (PWm, B_seed0): --qos=boost_qos_dbg --time=00:30:00, quarto argomento "collaudo" (prima PWf per un passo, poi PWm; verdetto);
#   run:                     --qos=boost_qos_lprod --time=<run_allocation_h del verdetto del collaudo>.
# Il limite di tempo del training e' l'allocazione meno 5 minuti (salvataggio e controlli finali). Nessuna ripresa, nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> --qos=... --time=... scripts/slurm/branch_pilot_train.sbatch \
#        <cartella del pilot> <PWl|PWf|PWm> <B_seed0|B_seed1> [collaudo]
# Cartella del pilot (assoluta, fuori dal codice): runs/ (training, ibridi, collaudo), cache/ (uscite dei rami), sonde/ (rapporti).
set -euo pipefail
PILOT="${1:?cartella del pilot}"; ARM="${2:?braccio}"; TEACHER="${3:?teacher}"; MODE="${4:-run}"
case "$PILOT" in /*) ;; *) echo "cartella del pilot relativa: rifiutata"; exit 2;; esac
case "$PILOT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "cartella del pilot dentro il codice: rifiutata"; exit 2;; esac
case "$ARM" in PWl|PWf|PWm) ;; *) echo "braccio non valido: $ARM"; exit 2;; esac
case "$TEACHER" in B_seed0|B_seed1) ;; *) echo "teacher non valido: $TEACHER"; exit 2;; esac
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
python scripts/branch_pilot.py --stage train --arm "$ARM" --teacher "$TEACHER" "${EXTRA[@]}" --expected-commit "$WEARUSFM_CODE_SHA" \
  --runs-root /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/step1_ab_0810 \
  --step2c-runs /leonardo_work/IscrB_WearUsFM/wearusfm_runs/runs/step2c_1410 --pilot-runs "$PILOT/runs" \
  --time-limit-s $((SECS - 300)) --device cuda --num-workers 6 \
  --manifest "$WORK/wearusfm_runs/results/passo6/capgmyo_grid_fix/manifest_v1_1.json.gz" \
  --scales "$WORK/wearusfm_runs/results/passo6/loader_59623700/session_scales.json" \
  --window-seconds "$WORK/wearusfm_runs/results/passo6/window_seconds_59425316.json" \
  --calibration "$WORK/wearusfm_runs/results/passo6/step1_0710/calibration.npz" \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed"
