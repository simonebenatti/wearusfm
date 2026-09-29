#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J verify_touch_raw
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/verify_touch_raw_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/verify_touch_raw_%j.out

# 29/09/2026: dopo la copia del raw su $SCRATCH (job 59019349, 7 dataset VERIFICATI per numero
# di file e byte). Qui: (1) confronto per CONTENUTO (checksum, rsync -c) fra originale e copia;
# (2) rinfresco della data di modifica delle copie (il purge di $SCRATCH conta 40 giorni dalla
# ultima modifica, e `rsync -a` ha preservato le date vecchie); (3) verifica che nessun file
# della copia sia piu' vecchio di 24 ore.
#
# QUESTO SCRIPT NON CANCELLA NULLA. La cancellazione degli originali e' un passo separato
# (scripts/slurm/delete_verified_raw.sbatch) che lancia Simone.

set -uo pipefail

SRC="$WORK/data/raw"
DST="$SCRATCH/data/raw"
ALL=(camargo2021 camargo2021_emg_only grabmyo putemg capgmyo ninapro kaifosh_discrete_gestures)
# Senza argomenti: tutti e 7. Con argomenti: solo quelli (per dividere il lavoro in piu' job:
# il confronto per checksum di 168 GB e' lento, ~0,6 GB/min sul primo run).
if [ "$#" -gt 0 ]; then DATASETS=("$@"); else DATASETS=("${ALL[@]}"); fi
for d in "${DATASETS[@]}"; do
  case " ${ALL[*]} " in *" $d "*) ;; *) echo "RIFIUTATO: $d non e' nell'elenco"; exit 2 ;; esac
done

ok=0
bad=0
for d in "${DATASETS[@]}"; do
  echo "=== $d ==="
  if [ ! -d "$SRC/$d" ] || [ ! -d "$DST/$d" ]; then
    echo "MANCANTE: $SRC/$d o $DST/$d"
    bad=$((bad+1)); continue
  fi
  # -r senza -a: niente confronto di date/permessi; -c: confronto per checksum; -n: nulla viene scritto.
  diff_out=$(rsync -rnc --itemize-changes "$SRC/$d/" "$DST/$d/" 2>&1)
  if [ -n "$diff_out" ]; then
    echo "DIFFERENZE $d (prime 20 righe):"
    echo "$diff_out" | head -20
    bad=$((bad+1)); continue
  fi
  n_src=$(find "$SRC/$d" -type f | wc -l)
  n_dst=$(find "$DST/$d" -type f | wc -l)
  if [ "$n_src" != "$n_dst" ]; then
    echo "NUMERO FILE DIVERSO $d: $n_src vs $n_dst"
    bad=$((bad+1)); continue
  fi
  echo "CHECKSUM IDENTICI $d: $n_src file"
  find "$DST/$d" -type f -exec touch {} +
  old=$(find "$DST/$d" -type f -mtime +0 | wc -l)
  if [ "$old" != "0" ]; then
    echo "ANCORA VECCHI dopo il touch: $old file in $d"
    bad=$((bad+1)); continue
  fi
  echo "RINFRESCATO $d: nessun file piu' vecchio di 24 ore"
  ok=$((ok+1))
done

echo "=== riepilogo: pronti=$ok problemi=$bad su ${#DATASETS[@]} dataset ==="
