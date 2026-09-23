#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_putemg
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_putemg_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_putemg_%j.out

# Passo 1 (v10 SS2.5): putEMG, solo emg_gestures in formato HDF5 (264 record su
# records.txt, 44 partecipanti). Niente emg_force/CSV/video/depth: non ci servono
# e pesano molto di piu' (Video-1080p da solo e' ~350GB).
#
# 23/09/2026: il downloader ufficiale del repo (putemg-downloader) e' rotto - usa
# un URL Nextcloud vecchio stile (".../s/<token>&files=records.txt", senza "?",
# 404 su Nextcloud moderno). Fix: WebDAV pubblico standard di Nextcloud funziona
# sempre per una condivisione pubblica - username = token di condivisione,
# password vuota. Questo script lo sostituisce per il nostro caso d'uso.
# Destinazione $WORK (non $SCRATCH): dataset piccolo (~11GB stimati), nessun
# rischio quota.

set -uo pipefail

TOKEN="45NY5snj0U4tgQz"
BASE="https://chmura.put.poznan.pl/public.php/webdav"
DEST="$WORK/data/raw/putemg/data-hdf5"
mkdir -p "$DEST"

echo "=== putEMG: elenco record (records.txt via WebDAV) ==="
curl -sf -u "${TOKEN}:" "$BASE/records.txt" -o /tmp/putemg_records_$$.txt
grep "^emg_gestures" /tmp/putemg_records_$$.txt > /tmp/putemg_gestures_$$.txt
N=$(wc -l < /tmp/putemg_gestures_$$.txt)
echo "record emg_gestures da scaricare: $N"

i=0
ok=0
fail=0
while IFS= read -r record; do
  i=$((i+1))
  dest="$DEST/${record}.hdf5"
  if [ -s "$dest" ]; then
    echo "[$i/$N] gia' presente: $record"
    ok=$((ok+1))
    continue
  fi
  echo "[$i/$N] scarico: $record"
  if wget -q -c --tries=3 --timeout=120 --auth-no-challenge \
       --user="$TOKEN" --password="" \
       -O "$dest" \
       "$BASE/Data-HDF5/${record}.hdf5"; then
    echo "  ok"
    ok=$((ok+1))
  else
    echo "  FALLITO: $record"
    fail=$((fail+1))
  fi
done < /tmp/putemg_gestures_$$.txt

rm -f /tmp/putemg_records_$$.txt /tmp/putemg_gestures_$$.txt

echo "=== riepilogo ==="
echo "fine: ok=$ok fail=$fail su $N record attesi"
du -sh "$DEST" 2>&1
