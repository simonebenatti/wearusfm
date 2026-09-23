#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_camargo
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_camargo_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_camargo_%j.out

# Passo 1 (v10 SS2.2): Camargo 2021 (biomeccanica arti inferiori: EMG, IMU,
# goniometri), Mendeley Data, CC BY 4.0, 3 parti.
#
# 23/09/2026: Mendeley Data nasconde i link di download dietro JavaScript - non
# estraibili da una pagina statica. URL reali trovati verificando l'API pubblica
# di Mendeley (redirect verso bucket S3 prod-dcd-datasets-public-files), con
# richieste HEAD confermate 200 OK e Content-Length reale (vedi
# docs/dataset_access.md). Soggetti AB06-AB30 (25 totali, non 22 come riportato
# da una fonte terza/blog non ufficiale usata prima della verifica diretta).
# ~23.5GB totali, $WORK va bene (nessun rischio quota).

set -uo pipefail
DEST="$WORK/data/raw/camargo2021"
mkdir -p "$DEST"

fetch() {
  # fetch <url> <dest_path> - salta se gia' presente, riprendibile (-c), segue redirect
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "gia' presente: $dest"
    return 0
  fi
  echo "scarico: $url -> $dest"
  wget -q -c --tries=5 --retry-connrefused --timeout=120 -O "$dest" "$url" \
    && echo "  ok" || echo "  FALLITO: $url"
}

echo "=== Camargo 2021 Parte 1 (AB06-AB14, ~9.4GB) ==="
fetch "https://data.mendeley.com/public-files/datasets/fcgm3chfff/files/9c30f2e4-2d39-4d95-b691-1163c927455e/file_downloaded" \
      "$DEST/Subjects_Part1_AB06-AB14.zip"

echo "=== Camargo 2021 Parte 2 (AB15-AB24, ~9.5GB) ==="
fetch "https://data.mendeley.com/public-files/datasets/k9kvm5tn3f/files/2b976163-2870-458c-816d-30e07f17ad90/file_downloaded" \
      "$DEST/Subjects_Part2_AB15-AB24.zip"

echo "=== Camargo 2021 Parte 3 (AB25-AB30, ~4.6GB) ==="
fetch "https://data.mendeley.com/public-files/datasets/jj3r5f9pnf/files/2800004c-be9c-40fa-b9bd-50ad3ec1b085/file_downloaded" \
      "$DEST/Subjects_Part3_AB25-AB30.zip"

echo "=== README e metadati ==="
fetch "https://data.mendeley.com/public-files/datasets/fcgm3chfff/files/87d1e6e3-0f8e-4a59-b0a7-a22cc5cc2de4/file_downloaded" \
      "$DEST/README.txt"
fetch "https://data.mendeley.com/public-files/datasets/fcgm3chfff/files/d0f204d7-903c-4064-aff6-7080b2b5acb8/file_downloaded" \
      "$DEST/SubjectInfo.mat"

echo "=== riepilogo ==="
ls -la "$DEST"
du -sh "$DEST" 2>&1
