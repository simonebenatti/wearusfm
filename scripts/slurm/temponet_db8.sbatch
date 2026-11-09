#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -J temponet_db8
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_db8_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_db8_%j.out
# Replica di TEMPONet su NinaPro DB8 (docs/temponet_db8_replication.md; criteri in docs/decisioni.md, voce del 09/11/2026). Una GPU.
#   collaudo: controllo eager/graph e tempi; soggetto 1, seme 0, 19 epoche con passo di training 32 e (solo fold1) 4;
#     QoS boost_qos_dbg, 30 minuti (al massimo 0,5 GPU-ore)
#     sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> temponet_db8.sbatch smoke <cache> <uscite> [cudnn]
#   completo: 12 soggetti x semi 0, 1, 2 (full solo col seme 0), 19 epoche, passo di training 8, un processo, poi la sintesi;
#     3 ore (al massimo 3 GPU-ore; emendamento del 09/11 in docs/decisioni.md)
#     sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> temponet_db8.sbatch full <cache> <uscite> [cudnn]
#   [cudnn]: deterministic (predefinito), default o benchmark; vedi scripts/temponet_db8_profile.py
# Le uscite vanno in una cartella NUOVA fuori dal codice; nessun riavvio automatico, nessun nuovo tentativo.
set -uo pipefail
MODE="${1:?smoke o full}"; CACHE="${2:?cartella della cache}"; OUT="${3:?cartella delle uscite}"; CUDNN="${4:-deterministic}"
case "$CUDNN" in deterministic|default|benchmark) ;; *) echo "modo cuDNN: deterministic, default o benchmark"; exit 2;; esac
case "$MODE" in smoke) EXPECTED_TIME=00:30:00;; full) EXPECTED_TIME=03:00:00;; *) echo "modo: smoke o full"; exit 2;; esac
for P in "$CACHE" "$OUT"; do case "$P" in /*) ;; *) echo "percorso relativo rifiutato: $P"; exit 2;; esac; done
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$SCRATCH/data/raw"/*) echo "uscite dentro il codice o nel raw: rifiutato"; exit 2;; esac
if [ "${SLURM_RESTART_COUNT:-0}" != 0 ]; then echo "riavvio automatico non ammesso"; exit 2; fi
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - "$EXPECTED_TIME" <<'PY' || exit 2
import os, subprocess, sys, torch
info = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True).split()
if 'TimeLimit=' + sys.argv[1] not in info:
    raise SystemExit(f'limite di tempo diverso da {sys.argv[1]}')
if torch.cuda.device_count() != 1:
    raise SystemExit(f'GPU visibili: {torch.cuda.device_count()}, attesa 1')
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| CUDA', torch.version.cuda)
PY
if [ "$MODE" = smoke ]; then
  # 0) profilo del passo (ms per passo con tre impostazioni cuDNN, kernel piu' costosi)
  mkdir -p "$OUT" && python3 scripts/temponet_db8_profile.py --cache "$CACHE" --out "$OUT/profile.json" || exit 1
  # 1) controllo eager/graph; soggetto 1, seme 0, tre modelli, 19 epoche, passo di training 32 (quello del criterio)
  python3 scripts/temponet_db8.py --cache "$CACHE" --out "$OUT/step32" --emg-unit uV --cudnn-mode "$CUDNN" --check-graph --subjects 1 \
    --seeds 0 --epochs 19 --workers 1 || exit 1
  python3 scripts/temponet_db8_summary.py "$OUT/step32" --seeds 0 || exit 1
  # 2) stesso soggetto e seme, solo fold1, passo di training 4: curva di validazione con 8 volte i passi per epoca
  python3 scripts/temponet_db8.py --cache "$CACHE" --out "$OUT/step4" --emg-unit uV --cudnn-mode "$CUDNN" --subjects 1 --seeds 0 \
    --epochs 19 --train-step 4 \
    --models fold1 --workers 1 || exit 1
else
  python3 scripts/temponet_db8.py --cache "$CACHE" --out "$OUT" --emg-unit uV --cudnn-mode "$CUDNN" --seeds 0 1 2 --workers 1 \
    --train-step 8 --full-seeds 0 --log-epochs --time-budget-s 9600 \
    || exit 1
  python3 scripts/temponet_db8_summary.py "$OUT" --seeds 0 1 2 || exit 1
fi
du -sh "$OUT"
