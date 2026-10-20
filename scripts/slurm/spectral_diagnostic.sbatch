#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH -J spectral_diagnostic
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/spectral_diagnostic_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/spectral_diagnostic_%j.out

# Diagnostica dei rami PSD e wavelet (docs/rami_diagnostica_v1.md, scripts/spectral_diagnostic.py): sonde della validation interna sulle
# cache gia' estratte del 2C e della tranche downstream (sola lettura), rapporti fuori dal codice. CPU seriale, 8 core; 0 GPU-ora.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/spectral_diagnostic.sbatch <cartella dei rapporti> \
#        probe --task p1 --arm B_seed0 --sets R PW0 PW --redundancy        oppure        compare --out <esito.json>
# Le radici delle cache sono fisse (quelle del 2C): EVAL_ROOT e DOWNSTREAM_ROOT si possono sovrascrivere solo per un collaudo.

set -uo pipefail
OUT_DIR="${1:?cartella dei rapporti}"; STAGE="${2:?probe o compare}"; shift 2
EVAL_ROOT="${EVAL_ROOT:-/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_downstream_0910}"
case "$OUT_DIR" in /*) ;; *) echo "cartella dei rapporti relativa: rifiutata"; exit 2;; esac
case "$OUT_DIR" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$EVAL_ROOT"|"$EVAL_ROOT"/*|"$DOWNSTREAM_ROOT"|"$DOWNSTREAM_ROOT"/*)
  echo "rapporti dentro il codice o dentro le cache d'origine: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$OUT_DIR"
case "$STAGE" in
  probe) python3 scripts/spectral_diagnostic.py probe --eval-root "$EVAL_ROOT" --downstream-root "$DOWNSTREAM_ROOT" --out-dir "$OUT_DIR" "$@" ;;
  compare) python3 scripts/spectral_diagnostic.py compare --out-dir "$OUT_DIR" "$@" ;;
  *) echo "fase $STAGE sconosciuta"; exit 2 ;;
esac
