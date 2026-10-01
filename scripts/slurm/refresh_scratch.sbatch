#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH -J refresh_scratch
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/refresh_scratch_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/refresh_scratch_%j.out

# Rinfresco delle date dei file su $SCRATCH (purge automatico a 40 giorni; decisioni.md, «Dati su $SCRATCH: rinfresco periodico»).
# Per ogni cartella: (1) quanti file sono piu' vecchi di 30 giorni; (2) `touch -c -h` su tutti i file e le cartelle (aggiorna accesso e modifica;
# -c non crea nulla, -h non segue i link); (3) verifica che nessun file sia piu' vecchio di 1 giorno. NON CANCELLA E NON SCRIVE CONTENUTI.
#
# Uso: sbatch scripts/slurm/refresh_scratch.sbatch                 (default: data/raw, data/processed, wearusfm_runs, external)
#      sbatch scripts/slurm/refresh_scratch.sbatch data/raw/emg2qwerty data/raw/ninapro     (solo queste, percorsi relativi a $SCRATCH)
# Visto il 01/10/2026: il tar grezzo di emg2qwerty aveva data 25/08/2021 e i .mat di DB8 date dal 2019 (wget conserva la data del server).

set -uo pipefail
cd "$SCRATCH"
DEFAULT=(data/raw data/processed wearusfm_runs external)
if [ "$#" -gt 0 ]; then TARGETS=("$@"); else TARGETS=("${DEFAULT[@]}"); fi

bad=0
for t in "${TARGETS[@]}"; do
  case "$t" in /*|*..*) echo "RIFIUTATO: $t (serve un percorso relativo a \$SCRATCH, senza ..)"; bad=$((bad+1)); continue ;; esac
  [ -e "$t" ] || { echo "MANCANTE: $t"; bad=$((bad+1)); continue; }
  for d in "$t"/* ; do
    [ -e "$d" ] || continue
    n=$(find "$d" -type f | wc -l)
    old=$(find "$d" -type f -mtime +30 | wc -l)
    oldest=$(find "$d" -type f -printf "%TY-%Tm-%Td\n" | sort | head -1)
    find "$d" -exec touch -c -h {} +
    still=$(find "$d" -type f -mtime +0 | wc -l)
    echo "$d: file=$n piu_vecchi_di_30_giorni_prima=$old data_piu_vecchia_prima=$oldest ancora_vecchi_dopo=$still"
    [ "$still" = "0" ] || bad=$((bad+1))
  done
done
echo "=== fine: cartelle con problemi = $bad ==="
[ "$bad" = "0" ]
