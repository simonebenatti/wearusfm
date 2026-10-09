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
# Ogni passo deve riuscire (codice d'uscita controllato: mkdir, worktree add/lock, rev-parse, status, find/chmod) e alla fine si verifica
# che nessun file o cartella del codice (escluso il file .git del worktree) sia ancora scrivibile; altrimenti il job esce con 1 senza il
# messaggio "pronto" (review GPT del 09/10/2026), quindi un launcher con `--dependency=afterok` non parte. Un checkout rimasto a meta' non
# si ripara: la cartella resta (quel nome di campagna non e' piu' utilizzabile) e si controlla a mano.
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
fallito() {  # nessun messaggio di successo e uscita 1
  echo "checkout di campagna NON pronto: $*"
  if [ -e "$DEST" ]; then echo "$DEST resta incompleto: nessun launcher deve usarlo; controllo e rimozione a mano"; fi
  exit 1
}
mkdir -p "$WORK/wearusfm_campaigns" || fallito "mkdir $WORK/wearusfm_campaigns"
git -C "$CANONICAL" worktree add --detach "$DEST" "$SHA" || fallito "git worktree add"
git -C "$CANONICAL" worktree lock --reason "campagna $NAME, commit $SHA (job $SLURM_JOB_ID)" "$DEST" || fallito "git worktree lock"
HEAD_NOW=$(git -C "$DEST" rev-parse HEAD) || fallito "git rev-parse HEAD"
if [ "$HEAD_NOW" != "$SHA" ]; then fallito "HEAD $HEAD_NOW diverso da $SHA"; fi
STATUS=$(git -C "$DEST" status --porcelain --untracked-files=all) || fallito "git status (pulizia non verificabile)"
if [ -n "$STATUS" ]; then fallito "checkout non pulito subito dopo la creazione"; fi
# sola lettura sui file del codice (il file .git del worktree resta com'e'); i link simbolici non hanno permessi propri
find "$DEST" -path "$DEST/.git" -prune -o -type f -exec chmod a-w {} + || fallito "find/chmod a-w sui file"
find "$DEST" -path "$DEST/.git" -prune -o -type d -exec chmod a-w {} + || fallito "find/chmod a-w sulle cartelle"
WRITABLE=$(find "$DEST" -path "$DEST/.git" -prune -o ! -type l \( -perm -0200 -o -perm -0020 -o -perm -0002 \) -print) \
  || fallito "find di verifica della sola lettura"
if [ -n "$WRITABLE" ]; then
  echo "restano scrivibili:"; printf '%s\n' "$WRITABLE" | head -20
  fallito "codice non in sola lettura"
fi
echo "checkout di campagna pronto: $DEST @ $SHA"
echo "files: $(git -C "$DEST" ls-files | wc -l); dimensione: $(du -sh "$DEST" | cut -f1)"
