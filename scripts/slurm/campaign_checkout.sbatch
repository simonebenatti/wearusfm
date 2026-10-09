#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -J campaign_checkout
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_%j.out

# Checkout di campagna congelato (eccezione approvata da Simone il 09/10/2026; CLAUDE.md, «Checkout di campagna»; CPU, 0 GPU-ora).
# Crea $WORK/wearusfm_campaigns/<nome> come worktree git staccato (detached) del repo canonico $WORK/wearusfm, al commit <sha>, che deve essere
# GIA' pubblicato su leonardo (di norma con `git push leonardo deploy/<nome>` dal Mac: un riferimento di deploy non tocca il working tree del
# canonico, quindi si puo' fare anche durante una campagna). Non modifica il working tree del canonico; scrive solo i metadati del worktree
# in $WORK/wearusfm/.git/worktrees/. Un checkout esistente non si tocca mai: se la cartella c'e', il job si rifiuta.
# Dopo la creazione: worktree bloccato (`git worktree lock`, niente prune) e file in sola lettura.
#
# Uso (dal Mac, via leonardo-ops; lo script si passa su stdin, quindi non serve che sia gia' nel canonico):
#   ssh leonardo "sbatch --export=ALL,CAMPAIGN=<nome>,CAMPAIGN_SHA=<sha a 40 caratteri>" < scripts/slurm/campaign_checkout.sbatch
# Poi i launcher: sbatch --export=ALL,CODE_DIR=$WORK/wearusfm_campaigns/<nome>,EXPECTED_SHA=<sha> <launcher> ...

set -uo pipefail
CANONICAL="$WORK/wearusfm"
NAME="${CAMPAIGN:?CAMPAIGN (nome della campagna)}"
SHA="${CAMPAIGN_SHA:?CAMPAIGN_SHA (commit, 40 caratteri)}"
case "$NAME" in *[!A-Za-z0-9._-]*|"") echo "nome di campagna non valido: $NAME"; exit 2;; esac
case "$NAME" in .*) echo "nome di campagna non valido: $NAME"; exit 2;; esac
if [ "${#SHA}" -ne 40 ] || [ -n "$(printf '%s' "$SHA" | tr -d '0-9a-f')" ]; then echo "CAMPAIGN_SHA deve essere uno SHA da 40 caratteri"; exit 2; fi
DEST="$WORK/wearusfm_campaigns/$NAME"
if [ -e "$DEST" ]; then echo "$DEST esiste gia': un checkout di campagna non si modifica. Nessuna azione."; exit 2; fi
if ! git -C "$CANONICAL" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo "commit $SHA non presente in $CANONICAL: va prima pubblicato (git push leonardo deploy/$NAME)"; exit 2
fi
mkdir -p "$WORK/wearusfm_campaigns"
git -C "$CANONICAL" worktree add --detach "$DEST" "$SHA" || { echo "git worktree add fallito"; exit 1; }
git -C "$CANONICAL" worktree lock --reason "campagna $NAME, commit $SHA (job $SLURM_JOB_ID)" "$DEST" || { echo "git worktree lock fallito"; exit 1; }
HEAD_NOW=$(git -C "$DEST" rev-parse HEAD)
if [ "$HEAD_NOW" != "$SHA" ]; then echo "HEAD $HEAD_NOW diverso da $SHA"; exit 1; fi
if [ -n "$(git -C "$DEST" status --porcelain --untracked-files=all)" ]; then echo "checkout non pulito subito dopo la creazione"; exit 1; fi
# sola lettura sui file del codice (il file .git del worktree resta com'e')
find "$DEST" -path "$DEST/.git" -prune -o -type f -exec chmod a-w {} +
find "$DEST" -path "$DEST/.git" -prune -o -type d -exec chmod a-w {} +
echo "checkout di campagna pronto: $DEST @ $SHA"
echo "files: $(git -C "$DEST" ls-files | wc -l); dimensione: $(du -sh "$DEST" | cut -f1)"
