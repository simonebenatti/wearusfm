#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J gate_d8
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/gate_d8_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/gate_d8_%j.out

# Gate di consistenza al ricampionamento (D8, passo 3), definizioni FIRMATE il 01/10/2026 (docs/proposta_gate_d8.md; filtro a muro firmato lo stesso
# giorno). Solo CPU, front-end all'inizializzazione: zero GPU-ora. Kaifosh processato, 20 finestre per utente, 3 semi, due casi (1 kHz e 200 Hz).
# Atteso (misurato sul Mac su finestre sintetiche da 16 canali): ~10 minuti di feature e ~1 minuto di sonda, piu' la lettura di Kaifosh (~7 GB).
#
# Uso: sbatch scripts/slurm/gate_d8.sbatch
#      sbatch --export=ALL,CODE_DIR=<checkout di campagna>,EXPECTED_SHA=<sha> <checkout>/scripts/slurm/gate_d8.sbatch --window-ms 150
#        (gate riaperto per il banco da 150 ms, R5 del pilot degli obiettivi, 24/10/2026: stesse soglie firmate, altro banco)
# Esito in $WORK/wearusfm_runs/results/step3/gate_<job>.json (fuori dal repo); feature intermedie in $SCRATCH. Uscita 1 = gate non superato.

set -uo pipefail
# codice congelato (CLAUDE.md, «Checkout di campagna»): CODE_DIR e EXPECTED_SHA, default il repo canonico come prima
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

KAIFOSH="${KAIFOSH_ROOT:-$SCRATCH/data/processed/kaifosh}"
echo "commit $(git rev-parse --short HEAD), Kaifosh da $KAIFOSH ($(find "$KAIFOSH" -name metadata.json | wc -l) sessioni), argomenti: $*"
mkdir -p "$WORK/wearusfm_runs/results/step3" "$WORK/wearusfm_runs/logs" "$SCRATCH/wearusfm_runs/step3"
python3 scripts/gate_d8.py \
  --processed-root "$KAIFOSH" \
  --out "$WORK/wearusfm_runs/results/step3/gate_${SLURM_JOB_ID}.json" \
  --work "$SCRATCH/wearusfm_runs/step3/gate_${SLURM_JOB_ID}_features" \
  --threads 2 \
  "$@"
