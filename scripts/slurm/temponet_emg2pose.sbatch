#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH -J temponet_emg2pose
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_emg2pose_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_emg2pose_%j.out
# TEMPONet su emg2pose (docs/temponet_emg2pose_plan.md; regole ed emendamento in docs/decisioni.md, 10/11/2026). Una GPU.
#   collaudo (passo 1): controllo eager/graph, training breve su 400 registrazioni di training, valutazione sulla sola validazione;
#     QoS boost_qos_dbg, 30 minuti (al massimo 0,5 GPU-ore)
#     sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> temponet_emg2pose.sbatch smoke <manifest> <uscite>
#   training (passo 2): seme 0, 19 epoche, controllo eager/graph prima; 3 ore e 30 (al massimo 3,5 GPU-ore); TRAIN_STRIDE=<blocchi> applica la
#     regola meccanica sulla velocita' del collaudo
#     sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>[,TRAIN_STRIDE=8] temponet_emg2pose.sbatch train <manifest> <uscite>
#   valutazione (passo 3): alpha e cancello sulla validazione, poi il test solo se il cancello passa; 45 minuti (al massimo 0,75 GPU-ore)
#     sbatch --time=00:45:00 --export=ALL,... temponet_emg2pose.sbatch eval <manifest> <uscite> <cartella del training>
#   studio delle loss (18/11/2026, docs/temponet_emg2pose_loss_plan_20261118.md): LOSS=l1|smoothl1|mse|mse_vel per train (default l1, la ricetta
#     di riferimento); TRAIN_TIME = limite del training, uguale a --time, fra quelli della tabella sotto (default 03:30:00); collaudo delle
#     quattro loss (controllo eager/graph e un'epoca breve ciascuna) su boost_qos_dbg, 30 minuti:
#     sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> temponet_emg2pose.sbatch losssmoke <manifest> <uscite>
#     sbatch --time=<TRAIN_TIME> --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>,LOSS=<loss>,TRAIN_TIME=<TRAIN_TIME> temponet_emg2pose.sbatch train ...
# Le uscite vanno in una cartella NUOVA fuori dal codice; nessun riavvio automatico, nessun nuovo tentativo.
set -uo pipefail
MODE="${1:?smoke, losssmoke, train o eval}"; MANIFEST="${2:?manifest del passo 0}"; OUT="${3:?cartella delle uscite}"; MODEL_DIR="${4:-}"
LOSS="${LOSS:-l1}"; TRAIN_TIME="${TRAIN_TIME:-03:30:00}"
case "$LOSS" in l1|smoothl1|mse|mse_vel) ;; *) echo "LOSS: l1, smoothl1, mse o mse_vel"; exit 2;; esac
# tempo massimo del training prima dell'ultima epoca: 600 s di margine col limite di 3 ore e 30 (come l'11/11), 900 s con i limiti piu' lunghi
case "$TRAIN_TIME" in 03:30:00) BUDGET=12000;; 04:30:00) BUDGET=15300;; 05:00:00) BUDGET=17100;; 05:30:00) BUDGET=18900;; 06:00:00) BUDGET=20700;;
  *) echo "TRAIN_TIME non ammesso: $TRAIN_TIME"; exit 2;; esac
case "$MODE" in smoke|losssmoke) EXPECTED_TIME=00:30:00;; train) EXPECTED_TIME=$TRAIN_TIME;; eval) EXPECTED_TIME=00:45:00;;
  *) echo "modo: smoke, losssmoke, train o eval"; exit 2;; esac
if [ "$MODE" = eval ] && [ -z "$MODEL_DIR" ]; then echo "eval: serve la cartella del training"; exit 2; fi
for P in "$MANIFEST" "$OUT" ${MODEL_DIR:+"$MODEL_DIR"}; do case "$P" in /*) ;; *) echo "percorso relativo rifiutato: $P"; exit 2;; esac; done
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$SCRATCH/data"/*) echo "uscite dentro il codice o nei dati: rifiutato"; exit 2;; esac
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
COMMON=(--csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" --processed-root "$SCRATCH/data/processed/emg2pose"
  --pose-root "$SCRATCH/data/processed/emg2pose_pose" --targets-manifest "$MANIFEST" --cudnn-mode benchmark)
echo "== $(date -Is) modo $MODE, codice $WEARUSFM_CODE_SHA in $WEARUSFM_CODE_DIR; uscite $OUT$([ "$MODE" = train ] && echo "; loss $LOSS, limite $TRAIN_TIME")"
if [ "$MODE" = smoke ]; then
  python3 scripts/temponet_emg2pose.py train "${COMMON[@]}" --out "$OUT/train" --check-graph --epochs 2 --max-train-recordings 400 \
    --max-val-recordings 60 --max-steps-per-epoch 4000 || exit 1
  python3 scripts/temponet_emg2pose.py eval "${COMMON[@]}" --out "$OUT/eval_val" --model-dir "$OUT/train" --val-only || exit 1
elif [ "$MODE" = losssmoke ]; then
  RC=0  # tutte e quattro le loss, anche dopo un fallimento: l'esito vale per loss (regola 4(a) del 18/11)
  for L in l1 smoothl1 mse mse_vel; do
    python3 scripts/temponet_emg2pose.py train "${COMMON[@]}" --out "$OUT/$L" --check-graph --loss "$L" --epochs 1 --max-train-recordings 400 \
      --max-val-recordings 60 --max-steps-per-epoch 8000 || { echo "collaudo di $L FALLITO"; RC=1; }
  done
  [ "$RC" = 0 ] || exit 1
elif [ "$MODE" = train ]; then
  python3 scripts/temponet_emg2pose.py train "${COMMON[@]}" --out "$OUT" --check-graph --time-budget-s "$BUDGET" \
    --train-stride-blocks "${TRAIN_STRIDE:-8}" --loss "$LOSS" || exit 1
else
  python3 scripts/temponet_emg2pose.py eval "${COMMON[@]}" --out "$OUT" --model-dir "$MODEL_DIR"
  RC=$?
  [ "$RC" = 0 ] || exit "$RC"
fi
du -sh "$OUT"
