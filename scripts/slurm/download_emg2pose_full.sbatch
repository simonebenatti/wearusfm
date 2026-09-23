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
#
# 23/09/2026: spostato su $SCRATCH (scratch personale, non la quota di progetto $WORK
# da 1TB condivisa - vedi CLAUDE.md). Il tentativo precedente su $WORK si era davvero
# fermato a 311GB per OSError Disk quota exceeded (curl "Failed writing body"), non i
# 431GB dichiarati dal riepilogo finale (quel messaggio era testo fisso, non una verifica
# reale - rimosso). Il file parziale da 311GB viene spostato (mv) da $WORK a $SCRATCH
# alla prima esecuzione, cosi' curl -C - riprende da li' invece di riscaricare tutto.

set -uo pipefail
cd "$WORK"
DEST="$SCRATCH/data/raw/emg2pose"
mkdir -p "$DEST"

OLD_PARTIAL="$WORK/data/raw/emg2pose/emg2pose_dataset.tar"
if [ -f "$OLD_PARTIAL" ] && [ ! -f "$DEST/emg2pose_dataset.tar" ]; then
  echo "recupero parziale gia' scaricato da \$WORK ($(du -sh "$OLD_PARTIAL" | cut -f1)), evito di riscaricarlo"
  mv "$OLD_PARTIAL" "$DEST/emg2pose_dataset.tar"
fi

echo "=== emg2pose (target 431 GiB) ==="
curl -C - --retry 5 --retry-delay 30 \
  "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar" \
  -o "$DEST/emg2pose_dataset.tar"

echo "=== riepilogo ==="
ls -la "$DEST/emg2pose_dataset.tar" 2>&1
du -sh "$DEST" 2>&1
