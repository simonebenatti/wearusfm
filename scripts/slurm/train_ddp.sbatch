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
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>[,EXTRA_ARGS=...][,TIME_LIMIT_S=...] scripts/slurm/train_ddp.sbatch \
#        <manifest.json.gz> <session_scales.json> <run_dir> [passi massimi]
# Uscita fuori dal codice: <run_dir> (metrics.jsonl, checkpoint.pt, summary.json, config.json).

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
if [ -f "$RUN_DIR/STOP" ]; then echo "STOP in $RUN_DIR: $(cat "$RUN_DIR/STOP")"; exit 0; fi
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 8 CPU per GPU: 6 worker del dataloader per processo, come il sanity a 1 GPU
COMMON=(scripts/train_jepa.py --preset "${PRESET:-sanity}" --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed"
        --scales "$SCALES" --out-dir "$RUN_DIR" --num-workers 6 --distributed)
torchrun --standalone --nproc_per_node="$NGPU" "${COMMON[@]}" --max-steps "$MAX_STEPS" --time-limit-s "${TIME_LIMIT_S:-1200}" ${EXTRA_ARGS:-}
rc=$?
# RESUME_TO (facoltativo, collaudi): ripresa nella stessa allocazione fino a quel passo, dal checkpoint appena scritto (stessi processi,
# worker e segmento: lo stream riprende la stessa sequenza)
if [ "$rc" -eq 0 ] && [ -n "${RESUME_TO:-}" ]; then
  echo "ripresa fino al passo $RESUME_TO"
  torchrun --standalone --nproc_per_node="$NGPU" "${COMMON[@]}" --max-steps "$RESUME_TO" --time-limit-s "${RESUME_TIME_LIMIT_S:-240}" ${EXTRA_ARGS:-}
  rc=$?
fi
exit $rc
