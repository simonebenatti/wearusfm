#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_kaifosh
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_kaifosh_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_kaifosh_%j.out

# Passo 1 (v10 SS2.1, D3a deciso 23/09/2026): dataset "Discrete Gestures" di
# Kaifosh et al. (Meta, Nature 2025). Licenza CC-BY-NC-4.0 (Non-Commerciale,
# accettata esplicitamente in docs/decisioni.md D3a - diversa dal resto del
# corpus, perlopiu' CC BY 4.0).
#
# Il comando CLI ufficiale del repo (`python -m
# generic_neuromotor_interface.scripts.download_data`) richiederebbe clonare e
# installare il pacchetto Python di Meta solo per un download HTTP diretto -
# non necessario: il meccanismo sottostante e' un semplice tar su S3 pubblico,
# stesso bucket di emg2pose/emg2qwerty (fb-ctrl-oss), nessuna autenticazione.
# Verificato con HEAD request, 23/09/2026: 33.361.018.880 byte (~31 GiB).

set -uo pipefail
DEST="$WORK/data/raw/kaifosh_discrete_gestures"
mkdir -p "$DEST"

fetch() {
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

BASE="https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data/data/discrete_gestures"

echo "=== Discrete Gestures: corpus.csv (metadati, 108KB) ==="
fetch "$BASE/discrete_gestures_corpus.csv" "$DEST/discrete_gestures_corpus.csv"

echo "=== Discrete Gestures: full_data.tar (~31 GiB) ==="
fetch "$BASE/full_data.tar" "$DEST/full_data.tar"

echo "=== riepilogo ==="
ls -la "$DEST"
du -sh "$DEST" 2>&1
