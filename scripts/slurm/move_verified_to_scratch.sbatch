#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH -J move_work_to_scratch
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/move_work_to_scratch_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/move_work_to_scratch_%j.out

# Spostamento verificato da $WORK a $SCRATCH di un elenco CHIUSO di voci (06/11/2026). Istruzioni di Simone nella
# sottochat 2, quando la quota di progetto di $WORK scendeva per le scritture di altri utenti: «anzi io non libero
# niente. sposta tu la roba sullo scratch al massimo» (fra le 18:18 e le 18:29) e, alle 20:03, «Sì, spostale» alla
# domanda «Sposto sullo scratch anche le feature di sonda di ottobre (18,1 GiB)?».
#
# Due elenchi chiusi, scelti con MOVE_SET:
#   duplicates (predefinito; eseguito il 06/11, job 63688163): `data/raw/neurorvq_checkpoint`, identica a
#     `$WORK/models` (la copia letta dal codice), e `data/raw/epn612/EMG-EPN612_Dataset.zip`, archivio pubblico gia'
#     estratto in `data/raw/epn612/EMG-EPN612 Dataset` (cartella letta dall'harness, NON toccata). Nessuna copia unica.
#   october-features: le quattro cartelle di feature di sonda di ottobre sotto `wearusfm_runs/results/passo6`
#     (`step2c_1410`, `step1_downstream_0910`, `step1_random_20261014_r3`, `step2a_20261014`). Sono COPIE UNICHE di dati
#     derivati. Il resto di `passo6` (manifest, scale, finestre, `rvq_codes`, le altre cartelle) non si tocca: lo
#     leggono i launcher correnti. Al vecchio percorso resta un link alla copia, cosi' gli script di ottobre che
#     nominano quei percorsi continuano a funzionare per l'utente proprietario dello scratch.
#
# Per ogni voce: (1) la sorgente deve esistere, non essere un link e non contenere link o file speciali; la
# destinazione non deve esistere; (2) copia su $SCRATCH SENZA conservare le date, cosi' i file nascono con la data di
# oggi (il purge di $SCRATCH conta 40 giorni dalla modifica); (3) confronto fra originale e copia di elenco dei file,
# dimensioni, albero delle cartelle e sha256 di OGNI file; (4) controllo che nessun file della copia abbia una data
# piu' vecchia di 24 ore; (5) SOLO se tutto coincide l'originale viene tolto da $WORK. Con october-features il passo
# (5) e' in tre tempi: l'originale viene rinominato a lato nella stessa cartella, al suo posto va il link, si rilegge
# l'elenco dei file ATTRAVERSO il link, e solo allora l'originale rinominato viene tolto; se il link non e' valido
# viene tolto e l'originale torna al suo nome. Una voce che fallisce un controllo resta dov'e', la sua copia resta su
# $SCRATCH per l'ispezione, e il job esce con 1. Nulla viene sovrascritto.
#
# Le destinazioni hanno lo stesso percorso relativo sotto $SCRATCH (`data/raw/...`, `wearusfm_runs/results/passo6/...`),
# dentro il perimetro predefinito di scripts/slurm/refresh_scratch.sbatch: i giri di rinfresco non vanno saltati.
#
# MOVE_MODE=dry-run (predefinito): solo letture, nessuna scrittura. MOVE_MODE=move: esegue.
# Prova in una cartella finta: MOVE_TEST_ROOT=<radice sotto $SCRATCH/tmp> con dentro work/ e scratch/;
# MOVE_TEST_CORRUPT=content|size altera la copia prima della verifica, MOVE_TEST_CORRUPT=link altera il link prima del
# suo controllo: mostrano che un originale non verificato non viene tolto. Fuori dalla prova non hanno effetto.
#
# Uso (dal Mac, con il submitter comune; lo script va a sbatch su stdin):
#   python scripts/submit_job.py --id <ID> --agente claude --campagna infra_storage_20261106 --gpu-ore 0 \
#     --script-stdin scripts/slurm/move_verified_to_scratch.sbatch -- --partition=lrd_all_serial --nodes=1 --ntasks=1 \
#     --cpus-per-task=1 --mem=2G --time=01:00:00 --no-requeue --export=ALL,MOVE_MODE=move,MOVE_SET=october-features

set -uo pipefail
export LC_ALL=C

MODE="${MOVE_MODE:-dry-run}"
case "$MODE" in dry-run|move) ;; *) echo "MOVE_MODE non valido: $MODE"; exit 2 ;; esac

# Elenchi chiusi: percorso relativo a $WORK, uguale al percorso relativo a $SCRATCH. LINK=1: al vecchio percorso
# resta un link alla copia (ammesso solo per cartelle).
SET="${MOVE_SET:-duplicates}"
case "$SET" in
  duplicates)
    LINK=0
    ITEMS=(
      "data/raw/neurorvq_checkpoint"
      "data/raw/epn612/EMG-EPN612_Dataset.zip"
    ) ;;
  october-features)
    LINK=1
    ITEMS=(
      "wearusfm_runs/results/passo6/step2c_1410"
      "wearusfm_runs/results/passo6/step1_downstream_0910"
      "wearusfm_runs/results/passo6/step1_random_20261014_r3"
      "wearusfm_runs/results/passo6/step2a_20261014"
    ) ;;
  *) echo "MOVE_SET non valido: $SET"; exit 2 ;;
esac

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

listing() {  # dimensione e percorso relativo di ogni file, in ordine (anche attraverso un link a una cartella)
  if [ -d "$1" ]; then (cd "$1" && find . -type f -printf '%s %p\n' | sort -k2)
  else printf '%s ./%s\n' "$(stat -c %s -- "$1")" "$(basename -- "$1")"; fi
}

hashes() {  # sha256 di ogni file, percorsi relativi, in ordine
  if [ -d "$1" ]; then (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum)
  else (cd "$(dirname -- "$1")" && sha256sum -- "./$(basename -- "$1")"); fi
}

folders() {  # albero delle cartelle, percorsi relativi, in ordine
  (cd "$1" && find . -type d | sort)
}

restore() {  # dopo un link non valido: toglie il link e rimette l'originale al suo nome
  local src="$1" aside="$2"
  if [ -L "$src" ]; then rm -f -- "$src"; fi
  if [ -e "$src" ] || [ -L "$src" ]; then echo "NON RIPRISTINATO: $src e' occupato; l'originale e' in $aside"; return 1; fi
  mv -- "$aside" "$src" || { echo "NON RIPRISTINATO: l'originale e' in $aside"; return 1; }
  echo "originale rimesso al suo posto: $src"
}

move_one() {
  local rel="$1" src="$W/$1" dst="$S/$1" tag kind n_src b_src old victim aside
  tag="$(printf '%s' "$rel" | tr '/ ' '__')"
  aside="$src.__in_uscita_${SLURM_JOB_ID:-prova$$}"
  echo "=== $rel"
  case "$SET:$rel" in
    duplicates:data/raw/neurorvq_checkpoint) ;;
    duplicates:data/raw/epn612/EMG-EPN612_Dataset.zip) ;;
    october-features:wearusfm_runs/results/passo6/step2c_1410) ;;
    october-features:wearusfm_runs/results/passo6/step1_downstream_0910) ;;
    october-features:wearusfm_runs/results/passo6/step1_random_20261014_r3) ;;
    october-features:wearusfm_runs/results/passo6/step2a_20261014) ;;
    *) echo "RIFIUTATO: fuori dall'elenco chiuso: $SET:$rel"; return 1 ;;
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
  if [ "$LINK" = 1 ]; then
    if [ "$kind" != cartella ]; then echo "RIFIUTATO: per questo elenco sono ammesse solo cartelle: $src"; return 1; fi
    if [ -e "$aside" ] || [ -L "$aside" ]; then echo "RIFIUTATO: esiste gia' $aside"; return 1; fi
  fi
  n_src=$(listing "$src" | wc -l)
  b_src=$(listing "$src" | awk '{s+=$1} END {printf "%d", s+0}')
  echo "sorgente ($kind): $n_src file, $b_src byte"
  if [ "$n_src" -lt 1 ]; then echo "RIFIUTATO: sorgente senza file"; return 1; fi
  if [ "$MODE" != move ]; then
    if [ "$LINK" = 1 ]; then
      echo "DRY-RUN: copierei in $dst, confronterei elenco, dimensioni, cartelle e sha256, metterei un link al vecchio percorso e solo dopo toglierei l'originale"
    else
      echo "DRY-RUN: copierei in $dst, confronterei elenco, dimensioni e sha256, e solo dopo toglierei $src"
    fi
    return 0
  fi

  mkdir -p -- "$(dirname -- "$dst")" "$OUT" || { echo "FALLITA la creazione delle cartelle"; return 1; }
  listing "$src" > "$OUT/$tag.source.sizes" || { echo "FALLITO l'elenco dell'originale"; return 1; }
  if [ "$kind" = cartella ]; then
    folders "$src" > "$OUT/$tag.source.dirs" || { echo "FALLITO l'albero dell'originale"; return 1; }
  fi
  if [ "$kind" = cartella ]; then cp -r --preserve=mode -- "$src" "$dst"; else cp --preserve=mode -- "$src" "$dst"; fi \
    || { echo "FALLITA la copia: originale non toccato"; return 1; }

  if [ "$TEST" = 1 ] && [ -n "${MOVE_TEST_CORRUPT:-}" ]; then  # solo nella cartella finta
    if [ -d "$dst" ]; then victim="$(find "$dst" -type f | sort | head -1)"; else victim="$dst"; fi
    case "$MOVE_TEST_CORRUPT" in
      size) printf 'x' >> "$victim"; echo "PROVA: copia alterata (size) in $victim" ;;
      content) printf 'X' | dd of="$victim" bs=1 count=1 conv=notrunc 2>/dev/null
               echo "PROVA: copia alterata (content) in $victim" ;;
      link) : ;;  # agisce piu' avanti, sul link
      *) echo "MOVE_TEST_CORRUPT non valido"; return 1 ;;
    esac
  fi

  listing "$dst" > "$OUT/$tag.copy.sizes" || { echo "FALLITO l'elenco della copia: originale non toccato"; return 1; }
  cmp -s "$OUT/$tag.source.sizes" "$OUT/$tag.copy.sizes" \
    || { echo "DIFFERENZE di elenco o di dimensioni: originale non toccato"; return 1; }
  if [ "$kind" = cartella ]; then
    folders "$dst" > "$OUT/$tag.copy.dirs" || { echo "FALLITO l'albero della copia: originale non toccato"; return 1; }
    cmp -s "$OUT/$tag.source.dirs" "$OUT/$tag.copy.dirs" \
      || { echo "DIFFERENZE nell'albero delle cartelle: originale non toccato"; return 1; }
  fi
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

  if [ "$LINK" = 1 ]; then
    mv -- "$src" "$aside" || { echo "FALLITA la rinomina a lato: originale non toccato"; return 1; }
    if ! ln -s -- "$dst" "$src"; then echo "FALLITA la creazione del link: $src"; restore "$src" "$aside"; return 1; fi
    if [ "$TEST" = 1 ] && [ "${MOVE_TEST_CORRUPT:-}" = link ]; then  # solo nella cartella finta
      rm -f -- "$src"; ln -s -- "$dst.__assente" "$src"; echo "PROVA: link alterato in $src"
    fi
    if [ "$(readlink -- "$src")" != "$dst" ] || ! listing "$src" > "$OUT/$tag.link.sizes" 2>/dev/null \
       || ! cmp -s "$OUT/$tag.source.sizes" "$OUT/$tag.link.sizes"; then
      echo "LINK NON VALIDO: $src"; restore "$src" "$aside"; return 1
    fi
    echo "LINK VERIFICATO: $src -> $dst, stesso elenco di file attraverso il link"
    rm -rf --one-file-system -- "$aside" \
      || { echo "FALLITA la rimozione di $aside (copia verificata, link al suo posto)"; return 1; }
    if [ -e "$aside" ] || [ -L "$aside" ]; then echo "ANCORA PRESENTE dopo la rimozione: $aside"; return 1; fi
    echo "SPOSTATO: $src -> $dst (al vecchio percorso resta un link)"
  else
    rm -rf --one-file-system -- "$src" || { echo "FALLITA la rimozione di $src"; return 1; }
    if [ -e "$src" ] || [ -L "$src" ]; then echo "ANCORA PRESENTE dopo la rimozione: $src"; return 1; fi
    echo "SPOSTATO: $src -> $dst"
  fi
}

quota() { [ "$TEST" = 1 ] || lfs quota -q -p 20147512 /leonardo_work/IscrB_WearUsFM; }

echo "modo: $MODE; elenco: $SET; da $W a $S; $(date '+%Y-%m-%d %H:%M:%S %Z') su $(hostname)"
echo "--- quota di progetto prima (kB usati, limite)"; quota
ok=0; bad=0
for rel in "${ITEMS[@]}"; do
  if move_one "$rel"; then ok=$((ok+1)); else bad=$((bad+1)); fi
done
echo "--- quota di progetto dopo"; quota
echo "=== riepilogo ($MODE, $SET): riuscite=$ok problemi=$bad su ${#ITEMS[@]} voci; $(date '+%H:%M:%S') ==="
[ "$bad" -eq 0 ]
