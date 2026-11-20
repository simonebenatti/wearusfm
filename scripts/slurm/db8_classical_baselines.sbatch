#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -J db8_classical_baselines
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/db8_classical_baselines_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/db8_classical_baselines_%j.out
# Passo 1 della diagnosi di TEMPONet (docs/temponet_db8_diagnosis_plan.md; cancello in docs/decisioni.md, 09/11/2026): SVR RBF e MLP su
# Waveform Length, protocolli fold e full, 12 soggetti, sulla cache della replica. CPU, 0 GPU-ore, al massimo 32 core-ore. Uso:
#   sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/db8_classical_baselines.sbatch <cache> <uscite NUOVE>
set -uo pipefail
CACHE="${1:?cartella della cache}"; OUT="${2:?cartella delle uscite}"
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
python3 scripts/db8_classical_baselines.py --cache "$CACHE" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}" --time-budget-s 12600 || exit 1
du -sh "$OUT"
