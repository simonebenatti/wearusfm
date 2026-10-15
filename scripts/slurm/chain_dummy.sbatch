#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH -J chain_dummy
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/chain_dummy_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/chain_dummy_%j.out

# Nodo fittizio per il collaudo della catena di valutazione (scripts/evaluation_chain.py, collaudo w1_valutazione_v1 del 15/10/2026): al
# posto di estrazioni, sonde e confronti di GPT, controlla che gli ingressi esistano e scrive un piccolo JSON d'uscita. Nessun calcolo, CPU,
# secondi; 0 GPU-ora. Esce con 3 se manca un ingresso.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/chain_dummy.sbatch <uscita.json> [ingressi...]

set -uo pipefail
OUT="${1:?uscita}"; shift
case "$OUT" in /*) ;; *) echo "uscita relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "uscita dentro il codice: rifiutata"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
for i in "$@"; do
  if [ ! -e "$i" ]; then echo "ingresso mancante: $i"; exit 3; fi
done
mkdir -p "$(dirname "$OUT")"
printf '{"job": "%s", "inputs": %d, "code_sha": "%s", "time": "%s"}\n' "${SLURM_JOB_ID:-locale}" "$#" "${WEARUSFM_CODE_SHA:-}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"
echo "nodo fittizio: $# ingressi presenti, uscita $OUT"
