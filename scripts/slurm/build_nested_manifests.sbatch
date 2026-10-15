#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -J build_nested
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/build_nested_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/build_nested_%j.out

# Manifest annidati per soggetto (D16) come file congelati, dal manifest congelato (GPT e Claude, 15/10/2026): uno per frazione, pesi rifatti
# con D9 sul sottoinsieme e residuo gestito senza superare il tetto (scripts/build_nested_manifests.py, che si ferma senza scrivere se
# un'autoverifica fallisce), poi il rapporto sulla distribuzione campionata (scripts/nested_manifest_report.py, uscita 1 se una verifica
# fallisce). CPU seriale, 1 core, minuti; 0 GPU-ora. Uscita fuori dal codice; un file gia' presente con contenuto diverso non si sovrascrive.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/build_nested_manifests.sbatch \
#        <manifest.json.gz> <cartella di uscita> <window_seconds.json>

set -uo pipefail
MANIFEST="${1:?manifest}"; OUT="${2:?cartella di uscita}"; WS="${3:?window_seconds.json}"
case "$OUT" in /*) ;; *) echo "cartella di uscita relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
  "$WORK/wearusfm_campaigns"|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "cartella di uscita dentro il codice: rifiutata"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$OUT"
python scripts/build_nested_manifests.py "$MANIFEST" "$OUT" || exit $?
python scripts/nested_manifest_report.py "$WS" "$MANIFEST" "$OUT"/*_nested_*.json.gz --json "$OUT/report.json"
