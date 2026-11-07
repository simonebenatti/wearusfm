#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:40:00
#SBATCH -J preserve_long_to_work
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/preserve_long_to_work_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/preserve_long_to_work_%j.out

# Copia verificata da $SCRATCH a $WORK delle uscite della valutazione lunga della linea 2 (07/11/2026), prima della
# cancellazione automatica a 40 giorni di $SCRATCH. Decisione di Simone, 07/11/2026, alla domanda «Conservo le uscite
# della valutazione lunga su $WORK prima della purge di meta' dicembre?»: «Tutto, 19,6 GiB». Raccomandazione di GPT
# consolidamento in docs/reviews/consolidation_long_preservation_20261107.md (commit 57c4d7e): conservare l'intera
# radice e un manifest che mappa originale e copia con sha256 e byte uguali; i piani congelati continuano a nominare i
# percorsi originali su $SCRATCH.
#
# Elenco CHIUSO di una voce: `wearusfm_runs/results/consolidation_long_20261105` (LONG_ROOT), stesso percorso relativo
# sotto $WORK. SOLO COPIA: l'originale su $SCRATCH non viene rinominato, linkato, toccato ne' rimosso; nulla viene
# sovrascritto. Per la voce: (1) la sorgente deve esistere, essere una cartella, non essere un link e non contenere
# link o file speciali; la destinazione non deve esistere; (2) lo spazio libero di $WORK dopo la copia deve restare
# sopra la soglia C-1 della linea 2 (10,724 GiB, docs/decisioni.md); (3) copia conservando modi e date; (4) confronto
# fra originale e copia di elenco dei file, dimensioni, albero delle cartelle e sha256 di OGNI file; (5) ricevuta con
# conteggi, byte e sha256 dei manifest. Un controllo fallito lascia la copia dov'e' per l'ispezione e fa uscire il
# job con 1: niente retry automatico, nuova decisione.
#
# PRESERVE_MODE=dry-run (predefinito): solo letture, nessuna scrittura. PRESERVE_MODE=copy: esegue.
# Prova in una cartella finta: PRESERVE_TEST_ROOT=<radice sotto $SCRATCH/tmp> con dentro scratch/ e work/;
# PRESERVE_TEST_CORRUPT=content|size altera la copia prima della verifica. Fuori dalla prova non hanno effetto.
#
# Uso (dal Mac, con il submitter comune; lo script va a sbatch su stdin):
#   python scripts/submit_job.py --id <ID> --agente claude --campagna consolidation_long_20261105 --gpu-ore 0 \
#     --script-stdin scripts/slurm/preserve_long_to_work.sbatch -- --partition=lrd_all_serial --nodes=1 --ntasks=1 \
#     --cpus-per-task=1 --mem=2G --time=00:40:00 --no-requeue --export=ALL,PRESERVE_MODE=copy,EXPECTED_SHA=<commit>

set -uo pipefail
export LC_ALL=C

MODE="${PRESERVE_MODE:-dry-run}"
case "$MODE" in dry-run|copy) ;; *) echo "PRESERVE_MODE non valido: $MODE"; exit 2 ;; esac

ITEMS=( "wearusfm_runs/results/consolidation_long_20261105" )
MIN_FREE_KB=11244930   # 10,724 GiB in kB: soglia C-1 della linea 2, da rispettare DOPO la copia

if [ -n "${PRESERVE_TEST_ROOT:-}" ]; then
  case "$PRESERVE_TEST_ROOT" in
    /leonardo_scratch/large/userexternal/sbenatti/tmp/?*) ;;
    *) echo "PRESERVE_TEST_ROOT deve stare sotto /leonardo_scratch/large/userexternal/sbenatti/tmp/"; exit 2 ;;
  esac
  case "$PRESERVE_TEST_ROOT" in *..*) echo "PRESERVE_TEST_ROOT con '..'"; exit 2 ;; esac
  S="$PRESERVE_TEST_ROOT/scratch"; W="$PRESERVE_TEST_ROOT/work"; OUT="$PRESERVE_TEST_ROOT/out"; TEST=1
else
  S=/leonardo_scratch/large/userexternal/sbenatti
  W=/leonardo_work/IscrB_WearUsFM
  OUT=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/storage_preserve_20261107
  TEST=0
fi

listing() {  # dimensione e percorso relativo di ogni file, in ordine
  (cd "$1" && find . -type f -printf '%s %p\n' | sort -k2)
}

hashes() {  # sha256 di ogni file, percorsi relativi, in ordine
  (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum)
}

folders() {  # albero delle cartelle, percorsi relativi, in ordine
  (cd "$1" && find . -type d | sort)
}

quota() { [ "$TEST" = 1 ] || lfs quota -q -p 20147512 /leonardo_work/IscrB_WearUsFM; }

free_kb() {  # kB liberi nella quota di progetto di $WORK (quota - usati); nella prova un valore grande
  if [ "$TEST" = 1 ]; then echo 999999999999; return 0; fi
  lfs quota -q -p 20147512 /leonardo_work/IscrB_WearUsFM | tr '\n' ' ' | awk '{ printf "%d\n", $3 - $2 }'
}

copy_one() {
  local rel="$1" src="$S/$1" dst="$W/$1" tag n_src b_src free need after victim
  tag="$(printf '%s' "$rel" | tr '/ ' '__')"
  echo "=== $rel"
  case "$rel" in
    wearusfm_runs/results/consolidation_long_20261105) ;;
    *) echo "RIFIUTATO: fuori dall'elenco chiuso: $rel"; return 1 ;;
  esac
  if [ -L "$src" ]; then echo "RIFIUTATO: la sorgente e' un link: $src"; return 1; fi
  if [ ! -d "$src" ]; then echo "RIFIUTATO: sorgente assente o non cartella: $src"; return 1; fi
  if [ -e "$dst" ] || [ -L "$dst" ]; then echo "RIFIUTATO: la destinazione esiste gia': $dst"; return 1; fi
  if [ -n "$(find "$src" -mindepth 1 ! -type f ! -type d -print -quit)" ]; then
    echo "RIFIUTATO: link o file speciali dentro $src"; return 1
  fi
  n_src=$(listing "$src" | wc -l)
  b_src=$(listing "$src" | awk '{s+=$1} END {printf "%d", s+0}')
  echo "sorgente (cartella): $n_src file, $b_src byte"
  if [ "$n_src" -lt 1 ]; then echo "RIFIUTATO: sorgente senza file"; return 1; fi
  free=$(free_kb) || { echo "FALLITA la lettura della quota"; return 1; }
  need=$(( (b_src + 1023) / 1024 ))
  after=$(( free - need ))
  echo "quota di progetto: $free kB liberi, servono $need kB, dopo la copia $after kB (soglia $MIN_FREE_KB kB)"
  if [ "$after" -lt "$MIN_FREE_KB" ]; then echo "RIFIUTATO: spazio insufficiente sotto la soglia C-1"; return 1; fi
  if [ "$MODE" != copy ]; then
    echo "DRY-RUN: copierei in $dst conservando modi e date, confronterei elenco, dimensioni, cartelle e sha256, e scriverei i manifest in $OUT; l'originale resta com'e'"
    return 0
  fi

  mkdir -p -- "$(dirname -- "$dst")" "$OUT" || { echo "FALLITA la creazione delle cartelle"; return 1; }
  listing "$src" > "$OUT/$tag.source.sizes" || { echo "FALLITO l'elenco dell'originale"; return 1; }
  folders "$src" > "$OUT/$tag.source.dirs" || { echo "FALLITO l'albero dell'originale"; return 1; }
  cp -r --preserve=mode,timestamps -- "$src" "$dst" \
    || { echo "FALLITA la copia: copia parziale lasciata in $dst per l'ispezione, originale intatto"; return 1; }

  if [ "$TEST" = 1 ] && [ -n "${PRESERVE_TEST_CORRUPT:-}" ]; then  # solo nella cartella finta
    victim="$(find "$dst" -type f | sort | head -1)"
    case "$PRESERVE_TEST_CORRUPT" in
      size) printf 'x' >> "$victim"; echo "PROVA: copia alterata (size) in $victim" ;;
      content) printf 'X' | dd of="$victim" bs=1 count=1 conv=notrunc 2>/dev/null
               echo "PROVA: copia alterata (content) in $victim" ;;
      *) echo "PRESERVE_TEST_CORRUPT non valido"; return 1 ;;
    esac
  fi

  listing "$dst" > "$OUT/$tag.copy.sizes" || { echo "FALLITO l'elenco della copia"; return 1; }
  cmp -s "$OUT/$tag.source.sizes" "$OUT/$tag.copy.sizes" \
    || { echo "DIFFERENZE di elenco o di dimensioni: copia lasciata in $dst per l'ispezione"; return 1; }
  folders "$dst" > "$OUT/$tag.copy.dirs" || { echo "FALLITO l'albero della copia"; return 1; }
  cmp -s "$OUT/$tag.source.dirs" "$OUT/$tag.copy.dirs" \
    || { echo "DIFFERENZE nell'albero delle cartelle: copia lasciata in $dst per l'ispezione"; return 1; }
  hashes "$src" > "$OUT/$tag.source.sha256" || { echo "FALLITO lo sha256 dell'originale"; return 1; }
  hashes "$dst" > "$OUT/$tag.copy.sha256" || { echo "FALLITO lo sha256 della copia"; return 1; }
  if [ "$(wc -l < "$OUT/$tag.source.sha256")" != "$n_src" ] || [ "$(wc -l < "$OUT/$tag.copy.sha256")" != "$n_src" ]; then
    echo "SHA256 INCOMPLETI: copia lasciata in $dst per l'ispezione"; return 1
  fi
  cmp -s "$OUT/$tag.source.sha256" "$OUT/$tag.copy.sha256" \
    || { echo "DIFFERENZE di contenuto (sha256): copia lasciata in $dst per l'ispezione"; return 1; }
  {
    echo "job: ${SLURM_JOB_ID:-prova} su $(hostname), $(date '+%Y-%m-%d %H:%M:%S %Z'); commit atteso: ${EXPECTED_SHA:-}"
    echo "originale: $src"
    echo "copia: $dst"
    echo "file: $n_src; byte: $b_src; sha256 identici; modi e date conservati; originale non toccato"
    (cd "$OUT" && sha256sum -- "$tag.source.sizes" "$tag.copy.sizes" "$tag.source.dirs" "$tag.copy.dirs" "$tag.source.sha256" "$tag.copy.sha256")
  } > "$OUT/$tag.receipt.txt" || { echo "FALLITA la ricevuta (copia verificata)"; return 1; }
  echo "VERIFICATO: $n_src file, $b_src byte, sha256 identici; copia in $dst, originale intatto; ricevuta $OUT/$tag.receipt.txt"
}

echo "modo: $MODE; da $S a $W; commit atteso: ${EXPECTED_SHA:-}; $(date '+%Y-%m-%d %H:%M:%S %Z') su $(hostname)"
echo "--- quota di progetto prima (kB usati, limite)"; quota
ok=0; bad=0
for rel in "${ITEMS[@]}"; do
  if copy_one "$rel"; then ok=$((ok+1)); else bad=$((bad+1)); fi
done
echo "--- quota di progetto dopo"; quota
echo "=== riepilogo ($MODE): riuscite=$ok problemi=$bad su ${#ITEMS[@]} voci; $(date '+%H:%M:%S') ==="
[ "$bad" -eq 0 ]
