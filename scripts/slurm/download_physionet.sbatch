#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_physionet
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_physionet_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_physionet_%j.out
#SBATCH --dependency=singleton

# Passo 1 (v10 §2.5): GRABMyo (9,4 GB) e Hyser (143 GB) da PhysioNet, open access.
# Pattern di download ufficiale PhysioNet (wget -r -N -c -np), riprendibile: se il
# walltime di 4h non basta per Hyser, risottomettere lo stesso script continua da dove
# si era fermato grazie a -N -c. --dependency=singleton evita due copie in parallelo
# dello stesso job (stesso nome, stesso utente) se risottomesso per errore.

set -uo pipefail
cd "$WORK"
RAW="$WORK/data/raw"

echo "=== GRABMyo ==="
mkdir -p "$RAW/grabmyo"
wget -r -N -c -np -nH --cut-dirs=3 -P "$RAW/grabmyo" \
  https://physionet.org/files/grabmyo/1.1.0/ 2>&1 | tail -50

echo "=== Hyser (HD-sEMG) - il piu' grande, 143 GB: puo' richiedere piu' sottomissioni ==="
mkdir -p "$RAW/hyser"
wget -r -N -c -np -nH --cut-dirs=3 -P "$RAW/hyser" \
  https://physionet.org/files/hd-semg/2.0.0/ 2>&1 | tail -50

echo "=== riepilogo ==="
du -sh "$RAW/grabmyo" "$RAW/hyser" 2>&1
