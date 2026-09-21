#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_emg2pose_full
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_emg2pose_full_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_emg2pose_full_%j.out
#SBATCH --dependency=singleton

# Passo 1 (v10 §2.5): emg2pose completo, 431 GiB - il piu' grande del corpus.
# Bassa priorita' (dopo emg2qwerty e i dataset piccoli), lanciato separatamente perche'
# richiedera' quasi certamente piu' sottomissioni da 4h per finire (curl -C - riprende).
# NON sottomettere finche' non sono confermati piccoli e emg2qwerty: e' un uso di banda
# e tempo molto piu' grande, non deve competere con quelli sul login/rete.

set -uo pipefail
cd "$WORK"
DEST="$WORK/data/raw/emg2pose"
mkdir -p "$DEST"

echo "=== emg2pose completo (431 GiB) ==="
curl -C - --retry 5 --retry-delay 30 \
  "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar" \
  -o "$DEST/emg2pose_dataset.tar"

echo "=== riepilogo ==="
du -sh "$DEST" 2>&1
