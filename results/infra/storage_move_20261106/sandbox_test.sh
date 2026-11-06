# Prova in una cartella finta dello script scripts/slurm/move_verified_to_scratch.sbatch (06/11/2026).
# Si esegue sul nodo di login: `ssh leonardo 'bash -s' < questo file`, dopo aver sostituito il segnaposto di B64 con il
# contenuto dello script in base64 (`base64 < scripts/slurm/move_verified_to_scratch.sbatch | tr -d '\n'`).
# Tocca solo la propria cartella sotto $SCRATCH/tmp; pochi secondi di CPU. Esiti del 06/11: sandbox_test_20261106.log
# (prima versione dello script, elenco `duplicates`) e sandbox_test_v2_20261106.log (due elenchi, prove t* e u*).
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
run() { (MOVE_TEST_ROOT="$1" MOVE_MODE="$2" MOVE_TEST_CORRUPT="${3:-}" MOVE_SET="${4:-}" bash "$BASE/move.sbatch" > "$1.log" 2>&1; echo $?); }
FAILS=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got '$2' expected '$3'"; FAILS=$((FAILS+1)); fi; }
there() { if [ -e "$1" ] || [ -L "$1" ]; then echo yes; else echo no; fi; }

echo "=== elenco duplicates (predefinito)"
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
check "t4 ten manifests" "$(ls $R/out | wc -l)" 10
check "t4 parents kept, no link left" "$(there $R/work/data/raw/epn612)$(there $R/work/data/raw)/$(find $R/work -type l | wc -l)" yesyes/0
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
check "t9 invalid list refused" "$( (MOVE_TEST_ROOT=$BASE/t1 MOVE_MODE=move MOVE_SET=tutto bash $BASE/move.sbatch >/dev/null 2>&1; echo $?) )" 2

echo "=== elenco october-features: quattro cartelle con copie uniche, link al vecchio percorso"
P=wearusfm_runs/results/passo6
FOUR="step2c_1410 step1_downstream_0910 step1_random_20261014_r3 step2a_20261014"
NEIGH="window_seconds_59425316.json derived_v1_1 rvq_codes step1_random_20261014 step1_random_20261014_r2"
mk2() {
  local w="$1/work/$P"
  mkdir -p "$w/step2c_1410/p1/A_seed0" "$w/step2c_1410/p2/B_seed1/branches_trained" "$w/step2c_1410/p2/vuota" \
           "$w/step1_downstream_0910/p1/A_seed0" "$w/step1_random_20261014_r3/p1/Random_seed0" "$w/step2a_20261014/p1/S" \
           "$w/derived_v1_1" "$w/rvq_codes/x" "$w/step1_random_20261014" "$w/step1_random_20261014_r2" \
           "$1/scratch/wearusfm_runs/results"
  printf 'feat-a-%s\n' "$(seq 1 4000 | tr '\n' ' ')" > "$w/step2c_1410/p1/A_seed0/ninapro_db2.npz"
  printf 'feat-b-%s\n' "$(seq 1 2500 | tr '\n' ' ')" > "$w/step2c_1410/p2/B_seed1/branches_trained/ninapro_db6.npz"
  printf '{"cmp": 1}\n' > "$w/step2c_1410/comparison.json"
  printf 'down-%s\n' "$(seq 1 3000 | tr '\n' ' ')" > "$w/step1_downstream_0910/p1/A_seed0/ninapro_db2.npz"
  printf '{"cmp": 2}\n' > "$w/step1_downstream_0910/comparison.json"
  printf 'teacher-%s\n' "$(seq 1 1500 | tr '\n' ' ')" > "$w/step1_random_20261014_r3/p1/Random_seed0/initial_teacher.pt"
  printf '{"cmp": 3}\n' > "$w/step1_random_20261014_r3/comparison.json"
  printf 'spec-%s\n' "$(seq 1 1200 | tr '\n' ' ')" > "$w/step2a_20261014/p1/S/ninapro_db2.npz"
  printf '{"cmp": 4}\n' > "$w/step2a_20261014/comparison.json"
  printf '{"window": 1}\n' > "$w/window_seconds_59425316.json"
  printf 'manifest\n' > "$w/derived_v1_1/manifest_v1_1_nested_x.json.gz"
  printf 'codes\n' > "$w/rvq_codes/x/codes.npz"
  printf 'keep-r1\n' > "$w/step1_random_20261014/keep.json"
  printf 'keep-r2\n' > "$w/step1_random_20261014_r2/keep.json"
  find "$1/work" -exec touch -d '2026-10-14 19:03' {} +
}
# contenuto e albero di una cartella, anche attraverso un link
content() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum; cd "$1" && find . -type d | sort) | sha256sum | cut -c1-16; }
neigh() { for n in $NEIGH; do if [ -d "$1/$n" ]; then content "$1/$n"; else sha256sum < "$1/$n" | cut -c1-16; fi; done | tr '\n' ' '; }
links() { find "$1" -type l | wc -l; }
asides() { find "$1" -name '*__in_uscita_*' | wc -l; }
declare -A C

R=$BASE/u1; mk2 $R; before=$(snap $R/work); sb=$(snap $R/scratch)
check "u1 dry-run exit" "$(run $R dry-run '' october-features)" 0
check "u1 work unchanged" "$(snap $R/work)" "$before"
check "u1 scratch unchanged" "$(snap $R/scratch)" "$sb"
check "u1 no manifests folder" "$(there $R/out)" no
check "u1 four dry-run lines" "$(grep -c '^DRY-RUN' $R.log)" 4
echo "--- u1 log"; cat $R.log

R=$BASE/u2; mk2 $R; before=$(snap $R/work)
check "u2 altered content: exit" "$(run $R move content october-features)" 1
check "u2 work unchanged" "$(snap $R/work)" "$before"
check "u2 four stopped at sha256" "$(grep -c 'DIFFERENZE di contenuto (sha256): originale non toccato' $R.log)" 4
check "u2 nothing moved, no link, nothing aside" "$(grep -c '^SPOSTATO' $R.log)/$(links $R/work)/$(asides $R/work)" 0/0/0

R=$BASE/u3; mk2 $R; before=$(snap $R/work)
check "u3 altered size: exit" "$(run $R move size october-features)" 1
check "u3 work unchanged" "$(snap $R/work)" "$before"
check "u3 four stopped at sizes" "$(grep -c 'DIFFERENZE di elenco o di dimensioni: originale non toccato' $R.log)" 4
check "u3 nothing moved, no link, nothing aside" "$(grep -c '^SPOSTATO' $R.log)/$(links $R/work)/$(asides $R/work)" 0/0/0

R=$BASE/u4; mk2 $R; before=$(snap $R/work)
check "u4 altered link: exit" "$(run $R move link october-features)" 1
check "u4 four invalid links detected" "$(grep -c '^LINK NON VALIDO' $R.log)" 4
check "u4 four originals put back" "$(grep -c '^originale rimesso al suo posto' $R.log)" 4
check "u4 work as before, no link, nothing aside" "$(snap $R/work)/$(links $R/work)/$(asides $R/work)" "$before/0/0"
check "u4 nothing moved" "$(grep -c '^SPOSTATO' $R.log)" 0
echo "--- u4 log"; cat $R.log

R=$BASE/u5; mk2 $R
for d in $FOUR; do C[$d]=$(content $R/work/$P/$d); done
nb=$(neigh $R/work/$P)
check "u5 move: exit" "$(run $R move '' october-features)" 0
for d in $FOUR; do
  check "u5 $d: old path is a link to the copy" "$(readlink $R/work/$P/$d)" "$R/scratch/$P/$d"
  check "u5 $d: same content and tree through the link" "$(content $R/work/$P/$d)" "${C[$d]}"
  check "u5 $d: the copy is a real folder" "$( [ -d $R/scratch/$P/$d ] && [ ! -L $R/scratch/$P/$d ] && echo yes)" yes
done
check "u5 empty folder kept in the copy" "$(there $R/scratch/$P/step2c_1410/p2/vuota)" yes
check "u5 neighbours untouched" "$(neigh $R/work/$P)" "$nb"
check "u5 exactly four links, nothing aside" "$(links $R/work)/$(asides $R/work)" 4/0
check "u5 copies have today's date" "$(find $R/scratch -type f -mtime +0 | wc -l)" 0
check "u5 28 manifests" "$(ls $R/out | wc -l)" 28
check "u5 four moved, four links verified" "$(grep -c '^SPOSTATO' $R.log)/$(grep -c '^LINK VERIFICATO' $R.log)" 4/4
echo "--- u5 log"; cat $R.log; echo "--- u5 manifests"; ls -l $R/out | awk '{print $5, $9}'
echo "--- u5 passo6 in work"; ls -l $R/work/$P | awk '{print $1, $9, $10, $11}'
after5=$(snap $R)
check "u6 second run: exit" "$(run $R move '' october-features)" 1
check "u6 four refused as links" "$(grep -c "^RIFIUTATO: la sorgente e' un link" $R.log)" 4
check "u6 nothing changed" "$(snap $R)" "$after5"

R=$BASE/u7; mk2 $R; ln -s /etc/hostname $R/work/$P/step2a_20261014/p1/link; before=$(snap $R/work/$P/step2a_20261014)
check "u7 link inside one folder: exit" "$(run $R move '' october-features)" 1
check "u7 that folder refused" "$(grep -c '^RIFIUTATO: link o file speciali' $R.log)" 1
check "u7 that folder untouched, no copy of it" "$(snap $R/work/$P/step2a_20261014)/$(there $R/scratch/$P/step2a_20261014)" "$before/no"
check "u7 the other three moved" "$(grep -c '^SPOSTATO' $R.log)" 3

R=$BASE/u8; mk2 $R; mkdir -p $R/scratch/$P/step2c_1410; before=$(snap $R/work/$P/step2c_1410)
check "u8 destination exists: exit" "$(run $R move '' october-features)" 1
check "u8 refused" "$(grep -c '^RIFIUTATO: la destinazione esiste' $R.log)" 1
check "u8 folder untouched, not a link" "$(snap $R/work/$P/step2c_1410)/$( [ -L $R/work/$P/step2c_1410 ] && echo link || echo real)" "$before/real"

R=$BASE/u9; mk2 $R; mkdir $R/work/$P/step2c_1410.__in_uscita_777; before=$(snap $R/work/$P/step2c_1410)
check "u9 name aside already taken: exit" "$( (SLURM_JOB_ID=777 MOVE_TEST_ROOT=$R MOVE_MODE=move MOVE_SET=october-features bash $BASE/move.sbatch > $R.log 2>&1; echo $?) )" 1
check "u9 refused before copying" "$(grep -c "^RIFIUTATO: esiste gia'" $R.log)/$(there $R/scratch/$P/step2c_1410)" 1/no
check "u9 folder untouched" "$(snap $R/work/$P/step2c_1410)" "$before"

R=$BASE/u10; mk2 $R; rm -r $R/work/$P/step2a_20261014; printf 'not a folder\n' > $R/work/$P/step2a_20261014
check "u10 a file where a folder is expected: exit" "$(run $R move '' october-features)" 1
check "u10 refused, file untouched, no copy" "$(grep -c '^RIFIUTATO: per questo elenco sono ammesse solo cartelle' $R.log)/$(cat $R/work/$P/step2a_20261014)/$(there $R/scratch/$P/step2a_20261014)" "1/not a folder/no"
echo "FAILS=$FAILS"
