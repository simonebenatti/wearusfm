#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J delete_verified_raw
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/delete_verified_raw_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/delete_verified_raw_%j.out

# DA LANCIARE A MANO DA SIMONE (l'agente non esegue cancellazioni definitive).
# Toglie da $WORK/data/raw gli originali dei 7 dataset GIA' copiati su $SCRATCH, ma SOLO se per
# ciascuno, al momento del lancio: (1) ogni file esiste sulla copia con la stessa dimensione
# (rsync --size-only, veloce), (2) la copia ha almeno i file dell'originale (puo' averne di piu': ninapro su $SCRATCH contiene anche DB scaricati
# direttamente li'; criterio approvato da Simone il 04/10/2026, «approvo tutto», punto 16; prima il numero doveva coincidere), (3) nessun file della copia e'
# piu' vecchio di 24 ore (cioe' il rinfresco del purge e' stato fatto). Il confronto per
# CHECKSUM e' quello di verify_touch_scratch_raw.sbatch (lento, ~0,6 GB/min): controllare nel suo
# log che ogni dataset riporti "CHECKSUM IDENTICI" PRIMA di lanciare questo script.
# Cosa non e' MAI toccato: csl_hdemg, zhang2026, camargo2021_emg_converted, epn612, uci_emg,
# emg2pose_mini, neurorvq_checkpoint, e qualunque cartella fuori dall'elenco qui sotto.

set -uo pipefail

SRC="$WORK/data/raw"
DST="$SCRATCH/data/raw"
DATASETS=(camargo2021 camargo2021_emg_only grabmyo putemg capgmyo ninapro kaifosh_discrete_gestures)

deleted=0
kept=0
for d in "${DATASETS[@]}"; do
  echo "=== $d ==="
  case "$d" in
    camargo2021|camargo2021_emg_only|grabmyo|putemg|capgmyo|ninapro|kaifosh_discrete_gestures) ;;
    *) echo "RIFIUTATO: $d non e' nell'elenco"; kept=$((kept+1)); continue ;;
  esac
  if [ ! -d "$SRC/$d" ]; then echo "gia' assente in $SRC: $d"; continue; fi
  if [ ! -d "$DST/$d" ]; then echo "COPIA ASSENTE: non cancello $d"; kept=$((kept+1)); continue; fi
  diff_out=$(rsync -rn --size-only --itemize-changes "$SRC/$d/" "$DST/$d/" 2>&1)
  if [ -n "$diff_out" ]; then echo "DIFFERENZE: non cancello $d"; echo "$diff_out" | head -10; kept=$((kept+1)); continue; fi
  n_src=$(find "$SRC/$d" -type f | wc -l)
  n_dst=$(find "$DST/$d" -type f | wc -l)
  if [ "$n_src" -gt "$n_dst" ]; then echo "COPIA CON MENO FILE ($n_src vs $n_dst): non cancello $d"; kept=$((kept+1)); continue; fi
  [ "$n_src" -lt "$n_dst" ] && echo "la copia ha $((n_dst - n_src)) file in piu' dell'originale (ammesso): ogni file dell'originale c'e', stessa dimensione"
  old=$(find "$DST/$d" -type f -mtime +0 | wc -l)
  if [ "$old" != "0" ]; then echo "COPIA NON RINFRESCATA ($old file vecchi): non cancello $d"; kept=$((kept+1)); continue; fi
  echo "verifiche superate ($n_src file identici e freschi): cancello $SRC/$d"
  rm -rf -- "${SRC:?}/${d:?}"
  deleted=$((deleted+1))
done

echo "=== riepilogo: cancellati=$deleted conservati=$kept su ${#DATASETS[@]} ==="
lfs quota -hp "$(lfs project -d /leonardo_work/IscrB_WearUsFM | awk '{print $1}')" /leonardo_work/IscrB_WearUsFM
