date "+%d/%m/%Y %H:%M:%S"; hostname
W=/leonardo_work/IscrB_WearUsFM
R=$W/wearusfm_runs/results/contrastive_targets_20261118
echo "--- cartella di uscita della prova (deve mancare)"
ls -ld "$R/cn_restart_probe_20261121" 2>&1
echo "--- checkout di campagna (deve mancare)"
ls -ld "$W/wearusfm_campaigns/cn-restart545-20261121" 2>&1
echo "--- riferimenti storici"
ls -l "$R/fork/CN/checkpoint.pt" "$R/benchmark/CN/checkpoint.pt" "$R/reloads/CN_A/checkpoint.pt" 2>&1
ls "$R" 2>&1 | head -30
echo "--- canonico: ramo estratto e HEAD"
git -C "$W/wearusfm" rev-parse --abbrev-ref HEAD 2>&1; git -C "$W/wearusfm" rev-parse HEAD 2>&1
git -C "$W/wearusfm" status --porcelain --untracked-files=all 2>&1 | head -5; echo "(fine status)"
echo "--- commit 8c72655c presente nel canonico?"
git -C "$W/wearusfm" cat-file -e 8c72655c3d894c82fd4918ed523baf0ac9c69e23^{commit} 2>&1 && echo presente || echo ASSENTE
echo "--- riferimenti deploy recenti"
git -C "$W/wearusfm" for-each-ref --sort=-creatordate --format='%(creatordate:short) %(refname) %(objectname:short)' refs/heads/deploy refs/tags 2>&1 | head -8
echo "--- hook e configurazione di ricezione"
ls "$W/wearusfm/.git/hooks" 2>&1 | grep -v '\.sample$' | head; git -C "$W/wearusfm" config --get receive.denyCurrentBranch 2>&1; echo "(fine hook)"
echo "--- quota di progetto"
lfs project -d "$W" 2>&1 | head -2
PID=$(lfs project -d "$W" 2>/dev/null | awk '{print $1}' | head -1); echo "ID=$PID"
lfs quota -hp "$PID" "$W" 2>&1 | head -6
echo "--- coda dell'utente"
squeue -u "$USER" -h -o '%i|%j|%T|%r|%M|%l|%P|%q' 2>&1 | head -20
echo "--- reservation"
scontrol show reservation 2>&1 | grep -E 'ReservationName|StartTime|State' | head -12
echo "--- qos dbg"
sacctmgr -nP show qos boost_qos_dbg format=Name,MaxWall,MaxSubmitPU,MaxJobsPU,MaxTRESPU 2>&1 | head -3
