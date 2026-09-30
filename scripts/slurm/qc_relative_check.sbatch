#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -J qc_relative_check
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/qc_relative_check_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/qc_relative_check_%j.out

# Misura del QC dei canali con la regola RELATIVA di canale piatto (docs/decisioni.md, «Misura del QC dei canali», Simone 30/09/2026).
# Legge tutti i dataset processati sotto $WORK/data/processed e $SCRATCH/data/processed. I dataset gia' ingeriti con la regola relativa
# (NinaPro DB2/3/4/6/7, Hyser, ...) funzionano da CONTROLLO: per costruzione devono dare zero canali nuovi; se non e' cosi', il problema e' nello
# strumento e non nei dati, e ci si ferma prima di applicare qualsiasi cosa.
#
# Uso: sbatch scripts/slurm/qc_relative_check.sbatch                          (SOLA LETTURA: e' il primo passo)
#      sbatch scripts/slurm/qc_relative_check.sbatch --only kaifosh --apply   (corregge i flag qc_valid dei sidecar: solo dopo aver visto il report)
# Il report va fuori dal repo. I dati (data_int16.npy) non si toccano in nessun caso.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs"
python3 scripts/qc_relative_check.py \
  --scan "$WORK/data/processed" --scan "$SCRATCH/data/processed" \
  --report "$WORK/wearusfm_runs/results/passo2/qc_relative_check_${SLURM_JOB_ID}.json" \
  "$@"
