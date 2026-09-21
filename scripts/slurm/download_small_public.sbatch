#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_small
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_small_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_small_%j.out

# Passo 1 (v10 §2.5): dataset piccoli/medi ad accesso libero, piu' il checkpoint
# NeuroRVQ e la versione mini di emg2pose (serve al passo 1-bis, v10 §9).
# URL verificati in sessione (18/09/2026) dalle pagine ufficiali - vedi
# docs/dataset_access.md e docs/fatti_da_verificare.md.

set -uo pipefail
cd "$WORK"
RAW="$WORK/data/raw"

fetch() {
  # fetch <url> <dest_path> - salta se gia' presente, riprendibile (-c)
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "gia' presente: $dest"
    return 0
  fi
  echo "scarico: $url -> $dest"
  wget -q -c --tries=3 --timeout=120 -O "$dest" "$url" && echo "  ok" || echo "  FALLITO: $url"
}

echo "=== EPN-612 (Zenodo) ==="
fetch "https://zenodo.org/records/4421500/files/EMG-EPN612%20Dataset.zip?download=1" \
      "$RAW/epn612/EMG-EPN612_Dataset.zip"

echo "=== UCI-EMG (Lobov) ==="
fetch "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip" \
      "$RAW/uci_emg/emg_data_for_gestures.zip"

echo "=== CapgMyo (Figshare, 18 file) ==="
declare -A CAPGMYO_FILES=(
  [dba-s1.zip]=13277105 [dba-s2.zip]=13275896 [dba-s3.zip]=13275953 [dba-s4.zip]=13275962
  [dba-s5.zip]=13275965 [dba-s6.zip]=13275983 [dba-s7.zip]=13276016 [dba-s8.zip]=13276019
  [dba-s9.zip]=13276022 [dba-s10.zip]=13276934 [dba-s11.zip]=13277147 [dba-s12.zip]=13277000
  [dba-s13.zip]=13277015 [dba-s14.zip]=13277027 [dba-s15.zip]=13276136 [dba-s16.zip]=13276139
  [dba-s17.zip]=13276142 [dba-s18.zip]=13276148
)
for fname in "${!CAPGMYO_FILES[@]}"; do
  fid="${CAPGMYO_FILES[$fname]}"
  fetch "https://ndownloader.figshare.com/files/$fid" "$RAW/capgmyo/$fname"
done

echo "=== emg2qwerty (S3, repo archiviato - priorita' alta v10 §2.5) ==="
fetch "https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz" \
      "$RAW/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"

echo "=== emg2pose-mini (sanity check per il passo 1-bis, v10 §9) ==="
fetch "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar" \
      "$RAW/emg2pose_mini/emg2pose_dataset_mini.tar"
fetch "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv" \
      "$RAW/emg2pose_mini/emg2pose_metadata.csv"

echo "=== checkpoint NeuroRVQ (HuggingFace) - copia anche su \$WORK/models/ (CLAUDE.md/piano §3) ==="
NEURORVQ_DIR="$WORK/data/raw/neurorvq_checkpoint"
mkdir -p "$NEURORVQ_DIR"
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
dest = pathlib.Path('$NEURORVQ_DIR')
for fname in ['pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt',
              'pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt']:
    try:
        path = hf_hub_download(repo_id='ntinosbarmpas/NeuroRVQ', filename=fname)
        shutil.copy(path, dest / pathlib.Path(fname).name)
        print('ok:', fname)
    except Exception as e:
        print('FALLITO:', fname, '-', e)
"
mkdir -p "$WORK/models"
cp -n "$NEURORVQ_DIR"/*.pt "$WORK/models/" 2>&1 || true

echo "=== putEMG: clona il downloader ufficiale (v10 §2.2, biolab-put/putemg-downloader) ==="
if [ ! -d "$RAW/putemg/putemg-downloader" ]; then
  mkdir -p "$RAW/putemg"
  git clone --depth 1 https://github.com/biolab-put/putemg-downloader "$RAW/putemg/putemg-downloader" \
    && echo "  clonato: eseguire manualmente lo script del downloader (richiede scelta dei soggetti/file)" \
    || echo "  FALLITO clone putemg-downloader"
fi

echo "=== riepilogo ==="
du -sh "$RAW"/* 2>&1
