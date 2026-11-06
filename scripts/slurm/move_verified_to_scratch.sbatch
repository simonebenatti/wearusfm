#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH -J move_work_to_scratch
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/move_work_to_scratch_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/move_work_to_scratch_%j.out

# Spostamento verificato da $WORK a $SCRATCH di un elenco CHIUSO di voci (06/11/2026). Istruzione di Simone nella
# sottochat 2, quando la quota di progetto di $WORK scendeva di circa 1,4 GiB l'ora per le scritture di altri utenti:
# «anzi io non libero niente. sposta tu la roba sullo scratch al massimo».
#
# Per ogni voce: (1) la sorgente deve esistere, non essere un link e non contenere link o file speciali; la
# destinazione non deve esistere; (2) copia su $SCRATCH SENZA conservare le date, cosi' i file nascono con la data di
# oggi (il purge di $SCRATCH conta 40 giorni dalla modifica); (3) confronto fra originale e copia di elenco dei file,
# dimensioni e sha256 di OGNI file; (4) controllo che nessun file della copia abbia una data piu' vecchia di 24 ore;
# (5) SOLO se tutto coincide l'originale viene tolto da $WORK. Una voce che fallisce un controllo resta dov'e', la sua
# copia parziale resta su $SCRATCH per l'ispezione, e il job esce con 1. Nulla viene sovrascritto.
#
# Le due voci di questo elenco non hanno copie uniche: `data/raw/neurorvq_checkpoint` e' identica a `$WORK/models`
# (la copia letta dal codice: scripts/anchor_candidates.py, precompute_rvq_codes.py, replicate_neurorvq.py);
# `EMG-EPN612_Dataset.zip` e' l'archivio pubblico gia' estratto in `data/raw/epn612/EMG-EPN612 Dataset`, che e' la
# cartella letta dall'harness e NON viene toccata. Le destinazioni stanno sotto `$SCRATCH/data/raw`, dentro il
# perimetro predefinito di scripts/slurm/refresh_scratch.sbatch.
#
# MOVE_MODE=dry-run (predefinito): solo letture, nessuna scrittura. MOVE_MODE=move: esegue.
# Prova in una cartella finta: MOVE_TEST_ROOT=<radice sotto $SCRATCH/tmp> con dentro work/ e scratch/;
# MOVE_TEST_CORRUPT=content|size altera la copia prima della verifica, per mostrare che un originale non verificato
# non viene tolto. Fuori dalla prova le due variabili non hanno effetto sui percorsi reali.
#
# Uso (dal Mac, con il submitter comune; lo script va a sbatch su stdin):
#   python scripts/submit_job.py --id <ID> --agente claude --campagna infra_storage_20261106 --gpu-ore 0 \
#     --script-stdin scripts/slurm/move_verified_to_scratch.sbatch -- --partition=lrd_all_serial --nodes=1 --ntasks=1 \
#     --cpus-per-task=1 --mem=2G --time=00:20:00 --no-requeue --export=ALL,MOVE_MODE=move

set -uo pipefail
export LC_ALL=C

MODE="${MOVE_MODE:-dry-run}"
case "$MODE" in dry-run|move) ;; *) echo "MOVE_MODE non valido: $MODE"; exit 2 ;; esac

if [ -n "${MOVE_TEST_ROOT:-}" ]; then
  case "$MOVE_TEST_ROOT" in
    /leonardo_scratch/large/userexternal/sbenatti/tmp/?*) ;;
    *) echo "MOVE_TEST_ROOT deve stare sotto /leonardo_scratch/large/userexternal/sbenatti/tmp/"; exit 2 ;;
  esac
  case "$MOVE_TEST_ROOT" in *..*) echo "MOVE_TEST_ROOT con '..'"; exit 2 ;; esac
  W="$MOVE_TEST_ROOT/work"; S="$MOVE_TEST_ROOT/scratch"; OUT="$MOVE_TEST_ROOT/out"; TEST=1
else
  W=/leonardo_work/IscrB_WearUsFM
  S=/leonardo_scratch/large/userexternal/sbenatti
  OUT=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/storage_move_20261106
  TEST=0
fi

# Elenco chiuso: percorso relativo a $WORK, uguale al percorso relativo a $SCRATCH.
ITEMS=(
  "data/raw/neurorvq_checkpoint"
  "data/raw/epn612/EMG-EPN612_Dataset.zip"
)

listing() {  # dimensione e percorso relativo di ogni file, in ordine
  if [ -d "$1" ]; then (cd "$1" && find . -type f -printf '%s %p\n' | sort -k2)
  else printf '%s ./%s\n' "$(stat -c %s -- "$1")" "$(basename -- "$1")"; fi
}

hashes() {  # sha256 di ogni file, percorsi relativi, in ordine
  if [ -d "$1" ]; then (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum)
  else (cd "$(dirname -- "$1")" && sha256sum -- "./$(basename -- "$1")"); fi
}

move_one() {
  local rel="$1" src="$W/$1" dst="$S/$1" tag kind n_src b_src old victim
  tag="$(printf '%s' "$rel" | tr '/ ' '__')"
  echo "=== $rel"
  case "$src" in
    "$W/data/raw/neurorvq_checkpoint"|"$W/data/raw/epn612/EMG-EPN612_Dataset.zip") ;;
    *) echo "RIFIUTATO: fuori dall'elenco chiuso: $src"; return 1 ;;
  esac
  if [ -L "$src" ]; then echo "RIFIUTATO: la sorgente e' un link: $src"; return 1; fi
  if [ ! -e "$src" ]; then echo "MANCANTE in sorgente: $src"; return 1; fi
  if [ -e "$dst" ] || [ -L "$dst" ]; then echo "RIFIUTATO: la destinazione esiste gia': $dst"; return 1; fi
  if [ -d "$src" ]; then
    kind=cartella
    if [ -n "$(find "$src" -mindepth 1 ! -type f ! -type d -print -quit)" ]; then
      echo "RIFIUTATO: link o file speciali dentro $src"; return 1
    fi
  elif [ -f "$src" ]; then
    kind=file
  else
    echo "RIFIUTATO: ne' file ne' cartella: $src"; return 1
  fi
  n_src=$(listing "$src" | wc -l)
  b_src=$(listing "$src" | awk '{s+=$1} END {printf "%d", s+0}')
  echo "sorgente ($kind): $n_src file, $b_src byte"
  if [ "$n_src" -lt 1 ]; then echo "RIFIUTATO: sorgente senza file"; return 1; fi
  if [ "$MODE" != move ]; then
    echo "DRY-RUN: copierei in $dst, confronterei elenco, dimensioni e sha256, e solo dopo toglierei $src"
    return 0
  fi

  mkdir -p -- "$(dirname -- "$dst")" "$OUT" || { echo "FALLITA la creazione delle cartelle"; return 1; }
  listing "$src" > "$OUT/$tag.source.sizes" || { echo "FALLITO l'elenco dell'originale"; return 1; }
  if [ "$kind" = cartella ]; then cp -r --preserve=mode -- "$src" "$dst"; else cp --preserve=mode -- "$src" "$dst"; fi \
    || { echo "FALLITA la copia: originale non toccato"; return 1; }

  if [ "$TEST" = 1 ] && [ -n "${MOVE_TEST_CORRUPT:-}" ]; then  # solo nella cartella finta
    if [ -d "$dst" ]; then victim="$(find "$dst" -type f | sort | head -1)"; else victim="$dst"; fi
    case "$MOVE_TEST_CORRUPT" in
      size) printf 'x' >> "$victim" ;;
      content) printf 'X' | dd of="$victim" bs=1 count=1 conv=notrunc 2>/dev/null ;;
      *) echo "MOVE_TEST_CORRUPT non valido"; return 1 ;;
    esac
    echo "PROVA: copia alterata ($MOVE_TEST_CORRUPT) in $victim"
  fi

  listing "$dst" > "$OUT/$tag.copy.sizes" || { echo "FALLITO l'elenco della copia: originale non toccato"; return 1; }
  cmp -s "$OUT/$tag.source.sizes" "$OUT/$tag.copy.sizes" \
    || { echo "DIFFERENZE di elenco o di dimensioni: originale non toccato"; return 1; }
  hashes "$src" > "$OUT/$tag.source.sha256" || { echo "FALLITO lo sha256 dell'originale: originale non toccato"; return 1; }
  hashes "$dst" > "$OUT/$tag.copy.sha256" || { echo "FALLITO lo sha256 della copia: originale non toccato"; return 1; }
  if [ "$(wc -l < "$OUT/$tag.source.sha256")" != "$n_src" ] || [ "$(wc -l < "$OUT/$tag.copy.sha256")" != "$n_src" ]; then
    echo "SHA256 INCOMPLETI: originale non toccato"; return 1
  fi
  cmp -s "$OUT/$tag.source.sha256" "$OUT/$tag.copy.sha256" \
    || { echo "DIFFERENZE di contenuto (sha256): originale non toccato"; return 1; }
  old=$(find "$dst" -type f -mtime +0 | wc -l)
  if [ "$old" != 0 ]; then echo "COPIA NON FRESCA ($old file con data vecchia): originale non toccato"; return 1; fi
  echo "VERIFICATO: $n_src file, $b_src byte, sha256 identici, copia con la data di oggi"

  rm -rf --one-file-system -- "$src" || { echo "FALLITA la rimozione di $src"; return 1; }
  if [ -e "$src" ] || [ -L "$src" ]; then echo "ANCORA PRESENTE dopo la rimozione: $src"; return 1; fi
  echo "SPOSTATO: $src -> $dst"
}

quota() { [ "$TEST" = 1 ] || lfs quota -q -p 20147512 /leonardo_work/IscrB_WearUsFM; }

echo "modo: $MODE; da $W a $S; $(date '+%Y-%m-%d %H:%M:%S %Z') su $(hostname)"
echo "--- quota di progetto prima (kB usati, limite)"; quota
ok=0; bad=0
for rel in "${ITEMS[@]}"; do
  if move_one "$rel"; then ok=$((ok+1)); else bad=$((bad+1)); fi
done
echo "--- quota di progetto dopo"; quota
echo "=== riepilogo ($MODE): riuscite=$ok problemi=$bad su ${#ITEMS[@]} voci; $(date '+%H:%M:%S') ==="
[ "$bad" -eq 0 ]
