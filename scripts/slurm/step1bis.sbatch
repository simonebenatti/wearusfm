#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -J step1bis
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1bis_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1bis_%j.out

# Passo 1-bis: V1-V4 sul tokenizer NeuroRVQ-EMG congelato (D5a, congelata 29/09/2026).
# Uso: sbatch scripts/slurm/step1bis.sbatch [args extra di step1bis_run.py]
#      es. collaudo: sbatch --qos=boost_qos_dbg --time=00:30:00 scripts/slurm/step1bis.sbatch \
#            --smoke --n-groups 6 --v1-windows 4 --skip-v3 --datasets capgmyo camargo2021 --max-sessions-per-subject 1
#
# Il clone di NeuroRVQ (commit 926e770, CC BY-NC 4.0) sta FUORI dal repo, in $SCRATCH/external.
# Gli output vanno fuori dal repo.

set -euo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

REPO_DIR="$SCRATCH/external/NeuroRVQ_926e770"
# Il riferimento emg2pose e' un sottoinsieme MULTI-UTENTE (extract_emg2pose_reference.sbatch): il mini
# ha un solo utente e non permette V3. Nessun ripiego silenzioso sul mini.
EM_DIR="${EMG2POSE_DIR:-$SCRATCH/wearusfm_runs/emg2pose_reference}"
OUT_DIR="$WORK/wearusfm_runs/results/step1bis"
mkdir -p "$OUT_DIR" "$WORK/wearusfm_runs/logs"

if [ -z "$(find "$EM_DIR" -name '*.hdf5' -print -quit 2>/dev/null)" ]; then
  echo "ERRORE: nessun .hdf5 in $EM_DIR (lanciare prima scripts/slurm/extract_emg2pose_reference.sbatch)"
  exit 1
fi

python scripts/step1bis_run.py \
  --repo-dir "$REPO_DIR" \
  --checkpoint "$WORK/models/NeuroRVQ_EMG_tokenizer_v1.pt" \
  --processed-root "$WORK/data/processed" \
  --emg2pose-dir "$EM_DIR" \
  --emg2pose-csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --out "$OUT_DIR/step1bis_${SLURM_JOB_ID}.json" \
  --save-arrays-dir "$OUT_DIR/arrays_${SLURM_JOB_ID}" \
  "$@"
