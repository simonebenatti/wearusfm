#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH -J copy_raw_scratch
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/copy_raw_scratch_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/copy_raw_scratch_%j.out

# 29/09/2026: $WORK e' a 1023GB su 1000GB di quota soft (project quota, vedi CLAUDE.md).
# Sposta il RAW su $SCRATCH (personale, nessuna quota, purge a 40 giorni) invece di
# cancellare o chiedere quota: solo il raw GIA' INGERITO o facilmente ri-scaricabile
# (ogni dataset ha il suo script di download nel repo). Il PROCESSATO (il corpus
# durevole) resta su $WORK.
#
# NON entrano: csl_hdemg (link di download personali via email), zhang2026 (scaricato
# a mano dal browser per un blocco anti-bot), camargo2021_emg_converted (richiede
# MATLAB sul Mac per rigenerarlo), epn612 e uci_emg (letti dall'harness da $WORK).
#
# QUESTO SCRIPT NON CANCELLA NULLA: solo copia + verifica (numero di file e byte).
# Rimuovere gli originali da $WORK e' un passo separato, dopo conferma di Simone.

set -uo pipefail

SRC="$WORK/data/raw"
DST="$SCRATCH/data/raw"
mkdir -p "$DST"

DATASETS=(camargo2021 camargo2021_emg_only grabmyo putemg capgmyo ninapro kaifosh_discrete_gestures)

ok=0
bad=0
for d in "${DATASETS[@]}"; do
  echo "=== $d ==="
  if [ ! -d "$SRC/$d" ]; then
    echo "MANCANTE in sorgente: $SRC/$d"
    bad=$((bad+1))
    continue
  fi
  if ! rsync -a "$SRC/$d/" "$DST/$d/"; then
    echo "FALLITO rsync: $d"
    bad=$((bad+1))
    continue
  fi
  n_src=$(find "$SRC/$d" -type f | wc -l)
  n_dst=$(find "$DST/$d" -type f | wc -l)
  b_src=$(find "$SRC/$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  b_dst=$(find "$DST/$d" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  if [ "$n_src" = "$n_dst" ] && [ "$b_src" = "$b_dst" ]; then
    echo "VERIFICATO $d: $n_src file, $b_src byte identici"
    ok=$((ok+1))
  else
    echo "MISMATCH $d: file $n_src vs $n_dst, byte $b_src vs $b_dst"
    bad=$((bad+1))
  fi
done

echo "=== riepilogo: verificati=$ok problemi=$bad su ${#DATASETS[@]} dataset ==="
du -sh "$DST" 2>&1
