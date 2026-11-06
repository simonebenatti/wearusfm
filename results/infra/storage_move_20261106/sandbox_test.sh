# Prova in una cartella finta dello script scripts/slurm/move_verified_to_scratch.sbatch (06/11/2026).
# Si esegue sul nodo di login: `ssh leonardo 'bash -s' < questo file`, dopo aver sostituito il segnaposto di B64 con il
# contenuto dello script in base64 (`base64 < scripts/slurm/move_verified_to_scratch.sbatch | tr -d '\n'`).
# Tocca solo la propria cartella sotto $SCRATCH/tmp; pochi secondi di CPU. Esito del 06/11: sandbox_test_20261106.log.
set -u
B64="__BASE64_OF_scripts/slurm/move_verified_to_scratch.sbatch__"
BASE=/leonardo_scratch/large/userexternal/sbenatti/tmp/move_test_20261106_$(date +%H%M%S)
mkdir -p "$BASE" && cd "$BASE" || exit 9
echo "$B64" | base64 -d > "$BASE/move.sbatch"
echo "sandbox: $BASE on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "script sha256: $(sha256sum < "$BASE/move.sbatch" | cut -c1-64)"
bash -n "$BASE/move.sbatch" && echo "syntax ok (bash $BASH_VERSION)"
mk() {
  mkdir -p "$1/work/data/raw/neurorvq_checkpoint" "$1/work/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" "$1/work/models" "$1/scratch/data/raw"
  printf 'alpha-model-%s\n' "$(seq 1 2000 | tr '\n' ' ')" > "$1/work/data/raw/neurorvq_checkpoint/NeuroRVQ_EMG_foundation_model_v1.pt"
  printf 'beta-tokenizer-%s\n' "$(seq 1 3000 | tr '\n' ' ')" > "$1/work/data/raw/neurorvq_checkpoint/NeuroRVQ_EMG_tokenizer_v1.pt"
  cp "$1/work/data/raw/neurorvq_checkpoint/"* "$1/work/models/"
  printf 'zip-archive-%s\n' "$(seq 1 5000 | tr '\n' ' ')" > "$1/work/data/raw/epn612/EMG-EPN612_Dataset.zip"
  printf '{"user": 1}\n' > "$1/work/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON/user1.json"
  find "$1/work" -exec touch -d '2026-09-21 13:56' {} +
}
snap() { (cd "$1" && find . \( -type f -o -type l \) -print0 | sort -z | xargs -0 -r sha256sum 2>/dev/null; cd "$1" && find . | sort) | sha256sum | cut -c1-16; }
run() { (MOVE_TEST_ROOT="$1" MOVE_MODE="$2" MOVE_TEST_CORRUPT="${3:-}" bash "$BASE/move.sbatch" > "$1.log" 2>&1; echo $?); }
FAILS=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got '$2' expected '$3'"; FAILS=$((FAILS+1)); fi; }
there() { if [ -e "$1" ] || [ -L "$1" ]; then echo yes; else echo no; fi; }

R=$BASE/t1; mk $R; before=$(snap $R/work); sb=$(snap $R/scratch)
check "t1 dry-run exit" "$(run $R dry-run)" 0
check "t1 work unchanged" "$(snap $R/work)" "$before"
check "t1 scratch unchanged" "$(snap $R/scratch)" "$sb"
check "t1 no manifests folder" "$(there $R/out)" no
check "t1 default mode is dry-run" "$( (MOVE_TEST_ROOT=$R bash $BASE/move.sbatch | grep -c '^DRY-RUN') )" 2

R=$BASE/t2; mk $R; before=$(snap $R/work)
check "t2 altered content: exit" "$(run $R move content)" 1
check "t2 work unchanged" "$(snap $R/work)" "$before"
check "t2 both stopped at sha256" "$(grep -c 'DIFFERENZE di contenuto (sha256): originale non toccato' $R.log)" 2
check "t2 nothing moved" "$(grep -c '^SPOSTATO' $R.log)" 0

R=$BASE/t3; mk $R; before=$(snap $R/work)
check "t3 altered size: exit" "$(run $R move size)" 1
check "t3 work unchanged" "$(snap $R/work)" "$before"
check "t3 both stopped at sizes" "$(grep -c 'DIFFERENZE di elenco o di dimensioni: originale non toccato' $R.log)" 2

R=$BASE/t4; mk $R
h_model=$(cd $R/work/data/raw/neurorvq_checkpoint && sha256sum * | sha256sum | cut -c1-16)
h_zip=$(sha256sum < $R/work/data/raw/epn612/EMG-EPN612_Dataset.zip | cut -c1-16)
keep=$(snap "$R/work/data/raw/epn612/EMG-EPN612 Dataset"); models=$(snap $R/work/models)
check "t4 move: exit" "$(run $R move)" 0
check "t4 folder source gone" "$(there $R/work/data/raw/neurorvq_checkpoint)" no
check "t4 zip source gone" "$(there $R/work/data/raw/epn612/EMG-EPN612_Dataset.zip)" no
check "t4 folder copy identical" "$(cd $R/scratch/data/raw/neurorvq_checkpoint && sha256sum * | sha256sum | cut -c1-16)" "$h_model"
check "t4 zip copy identical" "$(sha256sum < $R/scratch/data/raw/epn612/EMG-EPN612_Dataset.zip | cut -c1-16)" "$h_zip"
check "t4 extracted folder untouched" "$(snap "$R/work/data/raw/epn612/EMG-EPN612 Dataset")" "$keep"
check "t4 models untouched" "$(snap $R/work/models)" "$models"
check "t4 copies have today's date" "$(find $R/scratch -type f -mtime +0 | wc -l)" 0
check "t4 eight manifests" "$(ls $R/out | wc -l)" 8
check "t4 parents kept" "$(there $R/work/data/raw/epn612)$(there $R/work/data/raw)" yesyes
echo "--- t4 log"; cat $R.log; echo "--- t4 manifests"; ls -l $R/out | awk '{print $5, $9}'
after4=$(snap $R)
check "t5 second run: exit" "$(run $R move)" 1
check "t5 says missing twice" "$(grep -c '^MANCANTE in sorgente' $R.log)" 2
check "t5 nothing changed" "$(snap $R)" "$after4"

R=$BASE/t6; mk $R; ln -s /etc/hostname $R/work/data/raw/neurorvq_checkpoint/link; before=$(snap $R/work/data/raw/neurorvq_checkpoint)
check "t6 link inside folder: exit" "$(run $R move)" 1
check "t6 folder refused" "$(grep -c '^RIFIUTATO: link o file speciali' $R.log)" 1
check "t6 folder untouched" "$(snap $R/work/data/raw/neurorvq_checkpoint)" "$before"
check "t6 no partial copy of the folder" "$(there $R/scratch/data/raw/neurorvq_checkpoint)" no

R=$BASE/t7; mk $R; mkdir -p $R/scratch/data/raw/neurorvq_checkpoint; before=$(snap $R/work/data/raw/neurorvq_checkpoint)
check "t7 destination exists: exit" "$(run $R move)" 1
check "t7 refused" "$(grep -c '^RIFIUTATO: la destinazione esiste' $R.log)" 1
check "t7 folder untouched" "$(snap $R/work/data/raw/neurorvq_checkpoint)" "$before"

R=$BASE/t8; mk $R; mv $R/work/data/raw/neurorvq_checkpoint $R/work/data/raw/real; ln -s real $R/work/data/raw/neurorvq_checkpoint; before=$(snap $R/work/data/raw/real)
check "t8 source is a link: exit" "$(run $R move)" 1
check "t8 refused" "$(grep -c "^RIFIUTATO: la sorgente e' un link" $R.log)" 1
check "t8 target of the link untouched" "$(snap $R/work/data/raw/real)" "$before"
check "t8 link still there" "$(there $R/work/data/raw/neurorvq_checkpoint)" yes

check "t9 test root outside the sandbox refused" "$( (MOVE_TEST_ROOT=/tmp/x MOVE_MODE=move bash $BASE/move.sbatch >/dev/null 2>&1; echo $?) )" 2
check "t9 invalid mode refused" "$( (MOVE_TEST_ROOT=$BASE/t1 MOVE_MODE=delete bash $BASE/move.sbatch >/dev/null 2>&1; echo $?) )" 2
echo "FAILS=$FAILS"
