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
#SBATCH -J train_ddp
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/train_ddp_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/train_ddp_%j.out

# Training DDP su un nodo (decisione del 09/10/2026: DDP su un nodo, 4 GPU; FSDP e piu' nodi fuori dal perimetro iniziale). torchrun lancia un
# processo per GPU; il batch del preset e' GLOBALE (sanity: 32 = 4 x 8). Costo: 4 GPU per la durata del job (30 min su boost_qos_dbg = 2
# GPU-ora). Ogni lancio va presentato con preventivo e autorizzato da Simone (budget «preparazione dell'infrastruttura», 25 GPU-ora).
# Le loss seguono il contratto delle loss distribuite (docs/contratto_loss_distribuite.md: numeratori locali, conteggi globali); accumulo con
# EXTRA_ARGS="--accum-steps K".
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>[,EXTRA_ARGS=...][,TIME_LIMIT_S=...][,BRANCH_FROM=...] \
#        scripts/slurm/train_ddp.sbatch <manifest.json.gz> <session_scales.json> <run_dir> [passi massimi]
# Uscita fuori dal codice: <run_dir> (metrics.jsonl, checkpoint.pt, summary.json, config.json).
# Catene della finestra 1 (15/10/2026, `scripts/wsd_chain.py` genera i comandi): su boost_usr_prod --qos=normal e --time=24:00:00 davanti allo
# script sovrascrivono le righe #SBATCH; TIME_LIMIT_S = durata del job meno il margine; ogni job riprende dal checkpoint della cartella.
# - BRANCH_FROM (facoltativo): checkpoint della fase stabile da cui parte un ramo del WSD (`--branch-from`); conta solo finche' la cartella
#   del run non ha un checkpoint, poi si riprende da li'. Se manca ancora (il tronco non e' arrivato a quel passo) il job esce con 3 e la
#   catena del ramo si ferma, senza partire da zero.
# - Un run gia' completo (summary.json con stopped = max_steps e almeno i passi chiesti) esce subito con 0: i job di scorta di una catena
#   non ricaricano modello e dati.

set -uo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?session_scales.json}"; RUN_DIR="${3:?cartella del run}"; MAX_STEPS="${4:-20000}"
case "$RUN_DIR" in /*) ;; *) echo "RUN_DIR relativo: rifiutato"; exit 2;; esac
case "$RUN_DIR" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
  "$WORK/wearusfm_campaigns"|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "RUN_DIR dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$RUN_DIR" "$WORK/wearusfm_runs/logs"
# ambiente registrato (seconda ondata, docs/pilot_posa_v1.md §3 (d)): moduli nel log del job e pip freeze nella cartella del run
echo "== moduli caricati"; module list 2>&1
python3 -m pip freeze 2>/dev/null | sort > "$RUN_DIR/pip_freeze_${SLURM_JOB_ID:-local}.txt" && echo "pip freeze: $(wc -l < "$RUN_DIR/pip_freeze_${SLURM_JOB_ID:-local}.txt") righe in $RUN_DIR"
if [ -f "$RUN_DIR/STOP" ]; then echo "STOP in $RUN_DIR: $(cat "$RUN_DIR/STOP")"; exit 0; fi
if [ -f "$RUN_DIR/summary.json" ] && python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s.get("stopped") == "max_steps" and (s.get("steps") or 0) >= int(sys.argv[2]) else 1)' "$RUN_DIR/summary.json" "$MAX_STEPS"; then
  echo "run gia' completo ($RUN_DIR, $MAX_STEPS passi): niente da fare"; exit 0
fi
BRANCH=()
if [ -n "${BRANCH_FROM:-}" ]; then
  if [ ! -f "$RUN_DIR/checkpoint.pt" ] && [ ! -f "$BRANCH_FROM" ]; then
    echo "ramo: $BRANCH_FROM non esiste ancora (il tronco non e' arrivato a quel passo): la catena del ramo si ferma"; exit 3
  fi
  BRANCH=(--branch-from "$BRANCH_FROM")
fi
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 8 CPU per GPU: 6 worker del dataloader per processo, come il sanity a 1 GPU
COMMON=(scripts/train_jepa.py --preset "${PRESET:-sanity}" --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed"
        --scales "$SCALES" --out-dir "$RUN_DIR" --num-workers 6 --distributed)
torchrun --standalone --nproc_per_node="$NGPU" "${COMMON[@]}" --max-steps "$MAX_STEPS" --time-limit-s "${TIME_LIMIT_S:-1200}" \
  ${BRANCH[@]+"${BRANCH[@]}"} ${EXTRA_ARGS:-}
rc=$?
# RESUME_TO (facoltativo, collaudi): ripresa nella stessa allocazione fino a quel passo, dal checkpoint appena scritto (stessi processi,
# worker e segmento: lo stream riprende la stessa sequenza)
if [ "$rc" -eq 0 ] && [ -n "${RESUME_TO:-}" ]; then
  echo "ripresa fino al passo $RESUME_TO"
  torchrun --standalone --nproc_per_node="$NGPU" "${COMMON[@]}" --max-steps "$RESUME_TO" --time-limit-s "${RESUME_TIME_LIMIT_S:-240}" \
    ${BRANCH[@]+"${BRANCH[@]}"} ${EXTRA_ARGS:-}
  rc=$?
fi
exit $rc
