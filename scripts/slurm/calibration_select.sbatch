#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J calibration_select
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_select_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_select_%j.out

# Calibrazione C1-C4 (docs/calibrazione_c1_c4_selezione.md): ammissibilita' dei run e decisione della fase con le regole congelate
# (scripts/calibration_select.py select). CPU seriale, minuti; 0 GPU-ora. Ultimo nodo della catena di valutazione di una fase, dopo tutte
# le sonde. La decisione entra nel repo solo dopo la copia sul Mac (calibration_select.py merge) e il via di Simone alla fase successiva.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/calibration_select.sbatch <fase> <decisione.json> <registro.json> \
#        <piano di training | -> <lr> <run_dir> <rapporto | -> <passi> [<lr> <run_dir> <rapporto | -> <passi> ...]
# Il piano di training (relativo al checkout o assoluto) e' obbligatorio nelle fasi C1-C4; «-» come rapporto = run fermato prima della sonda.

set -uo pipefail
PHASE="${1:?fase}"; OUT="${2:?decisione}"; RECORD="${3:?registro}"; PLAN="${4:?piano di training o -}"; shift 4
for f in "$OUT" "$RECORD"; do
  case "$f" in /*) ;; *) echo "uscita relativa: rifiutata"; exit 2;; esac
  case "$f" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
    echo "uscita dentro il codice: rifiutata"; exit 2;; esac
done
if [ $(( $# % 4 )) -ne 0 ] || [ $# -eq 0 ]; then echo "candidati: gruppi di 4 argomenti (lr, run_dir, rapporto, passi)"; exit 2; fi
CANDS=()
while [ $# -gt 0 ]; do CANDS+=(--candidate "$1" "$2" "$3" "$4"); shift 4; done
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PLAN_ARGS=()
if [ "$PLAN" != "-" ]; then PLAN_ARGS=(--plan "$PLAN"); fi
python3 scripts/calibration_select.py select --phase "$PHASE" ${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"} --out "$OUT" --record "$RECORD" "${CANDS[@]}"
