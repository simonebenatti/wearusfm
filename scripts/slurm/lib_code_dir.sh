#!/bin/bash
# Codice congelato per un job (decisione del 09/10/2026, CLAUDE.md, «Checkout di campagna»). Da includere nei launcher, prima di qualunque
# comando Python:
#
#   source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
#
# Variabili (sbatch --export=ALL,CODE_DIR=...,EXPECTED_SHA=...):
#   CODE_DIR      checkout da usare; default $WORK/wearusfm (il repo canonico, come prima di questa regola);
#   EXPECTED_SHA  commit atteso, 40 caratteri. Obbligatorio per un checkout di campagna (CODE_DIR diverso dal canonico), facoltativo sul
#                 canonico (se c'e', si verifica).
# Controlli: CODE_DIR e' un checkout git; HEAD = EXPECTED_SHA; nessuna modifica e nessun file non tracciato. Ogni comando git deve uscire
# con 0: un errore (anche con stdout vuoto, come `git status` fallito) ferma il job, non vale mai come checkout pulito (review GPT del
# 09/10/2026). Poi `cd` in CODE_DIR, src/ di CODE_DIR in testa a PYTHONPATH (cosi' anche un'installazione editabile del venv non vince),
# niente bytecode scritto nel checkout, e WEARUSFM_CODE_DIR / WEARUSFM_CODE_SHA esportati per la provenienza nei risultati.

code_dir_setup() {
  local canonical="${WORK:?WORK non definita}/wearusfm"
  CODE_DIR="${CODE_DIR:-$canonical}"
  CODE_DIR="${CODE_DIR%/}"
  local inside head status
  if ! inside=$(git -C "$CODE_DIR" rev-parse --is-inside-work-tree 2>/dev/null) || [ "$inside" != "true" ]; then
    # dentro .git rev-parse esce con 0 ma risponde "false"
    echo "CODE_DIR=$CODE_DIR: non e' un checkout git" >&2; return 2
  fi
  if ! head=$(git -C "$CODE_DIR" rev-parse HEAD); then
    echo "CODE_DIR=$CODE_DIR: git rev-parse HEAD fallito: rifiutato" >&2; return 2
  fi
  if [ "$CODE_DIR" != "$canonical" ] && [ -z "${EXPECTED_SHA:-}" ]; then
    echo "checkout di campagna $CODE_DIR senza EXPECTED_SHA: rifiutato" >&2; return 2
  fi
  if [ -n "${EXPECTED_SHA:-}" ]; then
    case "$EXPECTED_SHA" in
      [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
      *) echo "EXPECTED_SHA=$EXPECTED_SHA: non e' uno SHA" >&2; return 2;;
    esac
    if [ "${#EXPECTED_SHA}" -ne 40 ]; then echo "EXPECTED_SHA deve avere 40 caratteri" >&2; return 2; fi
    if [ "$head" != "$EXPECTED_SHA" ]; then
      echo "CODE_DIR=$CODE_DIR e' a $head, atteso $EXPECTED_SHA: rifiutato" >&2; return 2
    fi
  fi
  if ! status=$(git -C "$CODE_DIR" status --porcelain --untracked-files=all); then
    echo "CODE_DIR=$CODE_DIR: git status fallito, pulizia non verificabile: rifiutato" >&2; return 2
  fi
  if [ -n "$status" ]; then
    echo "CODE_DIR=$CODE_DIR non e' pulito (modifiche o file non tracciati): rifiutato" >&2
    printf '%s\n' "$status" | head -20 >&2
    return 2
  fi
  if [ "$CODE_DIR" != "$canonical" ]; then  # un checkout di campagna e' in sola lettura: se no e' rimasto a meta' (job di checkout fallito)
    local writable
    if ! writable=$(find "$CODE_DIR" -path "$CODE_DIR/.git" -prune -o ! -type l \( -perm -0200 -o -perm -0020 -o -perm -0002 \) -print); then
      echo "CODE_DIR=$CODE_DIR: verifica della sola lettura fallita: rifiutato" >&2; return 2
    fi
    if [ -n "$writable" ]; then
      echo "checkout di campagna $CODE_DIR scrivibile (incompleto o modificato): rifiutato" >&2
      printf '%s\n' "$writable" | head -20 >&2
      return 2
    fi
  fi
  cd "$CODE_DIR" || return 2
  export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONDONTWRITEBYTECODE=1
  export WEARUSFM_CODE_DIR="$CODE_DIR" WEARUSFM_CODE_SHA="$head"
  echo "codice: $CODE_DIR @ $head"
}
