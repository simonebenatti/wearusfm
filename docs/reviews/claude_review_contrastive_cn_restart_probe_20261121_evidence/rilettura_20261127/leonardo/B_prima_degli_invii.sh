# Blocco B: controlli prima degli invii. Sola lettura.
date "+%d/%m/%Y %H:%M:%S"
W=/leonardo_work/IscrB_WearUsFM
R=$W/wearusfm_runs/results/contrastive_targets_20261118
echo "== cartelle che devono mancare"
ls -d "$R/cn_restart_probe_20261121" 2>/dev/null && echo "ATTENZIONE: la cartella di uscita esiste" || echo "cartella di uscita: assente"
ls -d "$W/wearusfm_campaigns/cn-restart545-20261121" 2>/dev/null && echo "il checkout di campagna esiste" || echo "checkout di campagna: assente"
echo "== riferimenti storici"
ls -l "$R/benchmark/CN/checkpoint_step0000544.pt" "$R/benchmark/CN/checkpoint.pt" "$R/reloads/CN_A/checkpoint.pt" | awk '{print $5, $9}'
echo "== quota di progetto"
PID=$(lfs project -d "$W" | awk '{print $1}' | head -1); echo "ID di progetto: $PID"
lfs quota -hp "$PID" "$W" | tail -2
echo "== saldo del progetto"
saldo -b 2>&1 | grep -E "account|IscrB_WearUsFM|^-" | head -8
echo "== job dell'utente"
squeue -u "$USER" -h -o '%i|%j|%T|%r|%P|%q|%l' | head -20
echo "== job dell'account in esecuzione (solo il numero)"
echo "in esecuzione: $(squeue -A IscrB_WearUsFM -h -t RUNNING | wc -l) | in attesa: $(squeue -A IscrB_WearUsFM -h -t PENDING | wc -l)"
echo "== limiti delle code"
sacctmgr -nP show qos boost_qos_dbg format=Name,MaxWall,MaxSubmitPU,MaxJobsPU | head -2
echo "== prenotazioni attive o future (nome, inizio, fine)"
scontrol show reservation | grep -E "^ReservationName" | awk '{print $1, $2, $3}'
