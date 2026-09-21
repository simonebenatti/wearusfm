#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_ninapro
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_ninapro_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_ninapro_%j.out

# Passo 1 (v10 §2.5): download NinaPro. Nessuna registrazione necessaria (verificato
# in sessione, 18/09/2026): le pagine di istruzioni linkano direttamente agli zip per
# soggetto. Il pattern del nome file cambia per DB, quindi si fa scraping degli href
# reali invece di indovinare un pattern comune - niente URL costruiti a mano.
#
# DB9 escluso (v10 §2.1: solo cinematica, nessun sEMG nuovo).

set -uo pipefail  # non -e: un DB che fallisce non deve fermare gli altri
cd "$WORK"

DEST="$WORK/data/raw/ninapro"
BASE="https://ninapro.hevs.ch"

for db in DB1 DB2 DB3 DB4 DB5 DB6 DB7 DB8 DB10; do
  out="$DEST/$db"
  mkdir -p "$out"
  echo "=== $db: scraping href .zip da $BASE/instructions/$db.html ==="

  page=$(curl -sL --max-time 30 "$BASE/instructions/$db.html")
  if [ -z "$page" ]; then
    echo "!!! $db: pagina istruzioni vuota o irraggiungibile, salto"
    continue
  fi

  # estrae gli href che finiscono in .zip, li rende assoluti rispetto a $BASE
  echo "$page" | grep -oE 'href="[^"]*\.zip"' | sed -E 's/href="//; s/"$//' | sort -u | while read -r href; do
    # risolve href assoluti, radice-relativi (/...) e relativi (../files/...) rispetto
    # alla pagina delle istruzioni, senza assumere quale forma useranno
    url=$(python3 -c "import urllib.parse,sys; print(urllib.parse.urljoin('$BASE/instructions/$db.html', '$href'))")
    fname=$(basename "$url")
    if [ -f "$out/$fname" ]; then
      echo "gia' presente: $fname"
      continue
    fi
    echo "scarico: $url"
    wget -q -c --tries=3 --timeout=60 -O "$out/$fname" "$url" && echo "  ok: $fname" || echo "  FALLITO: $fname"
  done
done

echo "=== riepilogo ==="
du -sh "$DEST"/* 2>&1
