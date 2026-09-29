#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J extract_emg2pose_ref
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/extract_emg2pose_ref_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/extract_emg2pose_ref_%j.out

# Passo 1-bis: sottoinsieme di riferimento di emg2pose (D5a, seed 0) da 16 utenti x 3
# registrazioni, estratto dal tar completo (463 GB, su $SCRATCH) in $SCRATCH (non in $WORK:
# quota). Serve perche' il mini ha un solo utente. Nessuna cancellazione, nessuna scrittura
# fuori da $SCRATCH/wearusfm_runs e da $WORK/wearusfm_runs.

set -euo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

CSV="$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv"
TAR="$SCRATCH/data/raw/emg2pose/emg2pose_dataset.tar"
OUT="$SCRATCH/wearusfm_runs/emg2pose_reference"
LIST="$WORK/wearusfm_runs/results/step1bis/emg2pose_reference_files.txt"
mkdir -p "$OUT" "$(dirname "$LIST")"

python scripts/select_emg2pose_reference.py --csv "$CSV" --n-users 16 --per-user 3 --seed 0 --out "$LIST"

# --occurrence=1: tar si ferma appena ha trovato tutti i membri dell'elenco
tar -xf "$TAR" -C "$OUT" --occurrence=1 -T "$LIST"

n_list=$(wc -l < "$LIST")
n_ok=$(find "$OUT" -name '*.hdf5' | wc -l)
echo "estratti $n_ok file (attesi $n_list)"
[ "$n_ok" = "$n_list" ] || { echo "ERRORE: conteggio diverso"; exit 1; }
du -sh "$OUT"
