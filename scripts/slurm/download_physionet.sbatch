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

# Passo 1 (v10 §2.5): GRABMyo (9,1 GB) e Hyser (135,2 GB) da PhysioNet, open access.
#
# Corretto il 22/09/2026: la prima versione usava `wget -r` (crawling ricorsivo file
# per file, migliaia di piccoli .dat/.hea WFDB) - troppo lento per il walltime di 4h
# (GRABMyo da solo avrebbe impiegato ~20h al ritmo osservato, ~130KB/s per l'overhead
# di migliaia di richieste piccole). PhysioNet offre uno zip unico per l'intero
# dataset (pagina "Download the ZIP file"), molto piu' veloce: usiamo quello.
#
# Riprendibile con wget -c se il walltime non basta; --dependency=singleton evita due
# copie in parallelo dello stesso job se risottomesso per errore.

set -uo pipefail
cd "$WORK"
RAW="$WORK/data/raw"

fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "gia' presente: $dest"
    return 0
  fi
  echo "scarico: $url -> $dest"
  wget -c --tries=5 --timeout=120 -O "$dest" "$url" && echo "  ok" || echo "  FALLITO: $url"
}

echo "=== GRABMyo (zip unico, 9,1 GB) ==="
fetch "https://physionet.org/content/grabmyo/get-zip/1.1.0/" "$RAW/grabmyo/grabmyo_1.1.0.zip"

echo "=== Hyser (HD-sEMG, zip unico, 135,2 GB - il piu' grande, probabile piu' sottomissioni) ==="
fetch "https://physionet.org/content/hd-semg/get-zip/2.0.0/" "$RAW/hyser/hyser_2.0.0.zip"

echo "=== riepilogo ==="
du -sh "$RAW/grabmyo" "$RAW/hyser" 2>&1
