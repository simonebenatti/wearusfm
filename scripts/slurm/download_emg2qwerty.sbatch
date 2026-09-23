#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J dl_emg2qwerty
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_emg2qwerty_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_emg2qwerty_%j.out

# Passo 1 (v10 §2.5): emg2qwerty, ri-scaricato dopo essere stato cancellato il
# 23/09/2026 per liberare la quota di $WORK durante la crisi di spazio (vedi
# CLAUDE.md e docs/dataset_access.md). Va su $SCRATCH come Hyser ed emg2pose,
# non su $WORK, per non rischiare di superare di nuovo la quota di progetto.
# Sottomesso con --dependency=afterok sul job di emg2pose: parte solo dopo che
# emg2pose e' finito con successo, per non competere per banda/rete.

set -uo pipefail
cd "$WORK"
RAW="$SCRATCH/data/raw"

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

echo "=== emg2qwerty (S3, ri-scaricato dopo cancellazione per quota) ==="
fetch "https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz" \
      "$RAW/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"

echo "=== riepilogo ==="
ls -la "$RAW/emg2qwerty/emg2qwerty-data-2021-08.tar.gz" 2>&1
du -sh "$RAW/emg2qwerty" 2>&1
