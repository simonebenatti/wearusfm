# Sola lettura su Leonardo per il recupero v3 della continuazione 300M 5.000 -> 19.618 (linea 1).
# Stesse sezioni delle letture del 06/11 (make_training_plan.py le legge), piu': cartelle padre (le uscite del recupero
# devono mancare), checkout e riferimenti di deploy, partizione seriale, sacct -P originale degli otto job gia' eseguiti
# (checkout v1, training fallito, quattro diagnosi, checkout e pre-controllo v2). Nessuna scrittura fuori da /tmp.
# Uso dal Mac, poi divisione in due file a @@@@SPLIT@@@@ e `make_training_plan.py <listing> <sha256> --check`:
#   ssh -o ProxyCommand=none -o ControlMaster=no -o ControlPath=none leonardo 'bash -s' < remote_read.sh > <log>
set -u
W=/leonardo_work/IscrB_WearUsFM
S=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs
L=/tmp/claude_v2_listing_$$.txt
# Listing
{
echo "=== date"; date '+%Y-%m-%d %H:%M:%S %Z'; hostname
echo "=== run 300M a 5000"; ls -l --time-style=long-iso $S/runs/scaling_300m_20261104/T100-G-300M-5000
echo "=== cartella runs scaling"; ls -l --time-style=long-iso $S/runs/scaling_300m_20261104
echo "=== ricevute 5000"; ls -l --time-style=long-iso $W/wearusfm_runs/results/scaling_300m_20261104/scientific5000_v1
echo "=== cartella ricevute scaling"; ls -l --time-style=long-iso $W/wearusfm_runs/results/scaling_300m_20261104
echo "=== cartella ricevute scaling su scratch"; ls -l --time-style=long-iso $S/results/scaling_300m_20261104
echo "=== controllo 30M T100-G"; ls -l --time-style=long-iso $S/runs/tokenizer_20261103/T100-G
echo "=== cache controllo 19618"; for d in pose p1 p2; do echo "--- $d"; ls -l --time-style=long-iso $S/results/tokenizer_evaluation_20261103/19618/T100-G/$d; done
echo "=== riferimenti posa"; ls -l --time-style=long-iso $S/results/pilot_posa/B19618_20261028/B/pose
echo "=== campagne scaling"; ls -d $W/wearusfm_campaigns/scaling_300m_*
echo "=== deploy refs scaling"; git -C $W/wearusfm for-each-ref 'refs/heads/deploy/scaling_300m_*' --format='%(refname) %(objectname)'
echo "=== qos normal"; sacctmgr show qos normal -P -n format=Name,Priority,MaxTRES,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES,Flags
echo "=== assoc"; sacctmgr show assoc user=$USER -P -n format=Account,Partition,QOS,DefaultQOS,GrpTRES,MaxJobs
echo "=== partition"; scontrol show partition boost_usr_prod | tr ' ' '\n' | grep -E '^(PartitionName|AllowQos|QoS|DefaultTime|MaxNodes|MaxTime|State|TotalNodes|DefMemPerCPU|MaxMemPerNode)='
echo "=== partition seriale"; scontrol show partition lrd_all_serial | tr ' ' '\n' | grep -E '^(PartitionName|MaxTime|State|TotalNodes|DefMemPerCPU|MaxMemPerNode|MaxCPUsPerNode)='
echo "=== reservations"; scontrol show reservation
echo "=== squeue mio"; squeue -u $USER -h -o '%i|%j|%P|%q|%T|%M|%l|%S|%r|%k'
echo "=== saldo"; saldo -b
echo "=== sinfo boost"; sinfo -h -p boost_usr_prod -o '%T %D' | sort | uniq -c
echo "=== quota work"; lfs quota -hp 20147512 $W
} > "$L" 2>&1
cat "$L"
echo "@@@@SPLIT@@@@"
# Sha256
{
echo "=== date inizio"; date '+%Y-%m-%d %H:%M:%S %Z'
echo "=== job training 5000"; sacct -j 63532932 -X -P -o JobID,JobName,Partition,QOS,State,ExitCode,Elapsed,Timelimit,Submit,Start,End,ReqTRES,AllocTRES,NodeList
echo "=== job training fallito 63696864 e diagnosi (sola lettura)"; sacct -j 63696864,63860499,63861166,63861825,63864244 -X -P -o JobID,JobName,Partition,QOS,State,ExitCode,Elapsed,Timelimit,Submit,Start,End,AllocTRES,NodeList
echo "=== sacct -P degli otto job gia' eseguiti (passi compresi)"; sacct -P -j 63695634,63696864,63860499,63861166,63861825,63864244,63974034,63974196 --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,NodeList,Partition,Start,End,Timelimit,AllocTRES
echo "=== cartelle del tentativo v1 (non ingressi)"; ls -l --time-style=long-iso $S/runs/scaling_300m_20261104/T100-G-300M-19618-from5000 $S/results/scaling_300m_20261104/continuation19618_v1
echo "=== dimensione di un checkout di campagna"; du -sh $W/wearusfm_campaigns/scaling_300m_continuation19618_20261106_v1
echo "=== sha256 run 300M a 5000"; (cd $S/runs/scaling_300m_20261104/T100-G-300M-5000 && sha256sum *)
echo "=== sha256 ricevute 5000"; (cd $W/wearusfm_runs/results/scaling_300m_20261104/scientific5000_v1 && sha256sum *)
echo "=== sha256 controllo 30M"; (cd $S/runs/tokenizer_20261103/T100-G && sha256sum checkpoint_step0019618.pt checkpoint.pt config.json summary.json fixed_eval_set.json pose_panel_set.json origin_sessions.json)
echo "=== sha256 cache controllo 19618"; (cd $S/results/tokenizer_evaluation_20261103/19618/T100-G && sha256sum pose/train_subset.npz pose/train_subset.npz.provenance.json pose/val.npz pose/val.npz.provenance.json pose/receipt.json p1/ninapro_db2.npz p1/ninapro_db2.npz.provenance.json p1/ninapro_db3.npz p1/ninapro_db3.npz.provenance.json p1/ninapro_db6.npz p1/ninapro_db6.npz.provenance.json p1/receipt.json p2/epn612.npz p2/epn612.npz.provenance.json p2/uci_emg.npz p2/uci_emg.npz.provenance.json p2/receipt.json)
echo "=== sha256 riferimenti posa"; (cd $S/results/pilot_posa/B19618_20261028/B/pose && sha256sum train_subset.npz val.npz)
echo "=== sha256 ingressi del training"; sha256sum $W/wearusfm_runs/results/passo6/derived_v1_1/manifest_v1_1_nested_1.json.gz $W/wearusfm_runs/results/passo6/loader_59623700/session_scales.json $W/wearusfm_runs/results/passo6/window_seconds_59425316.json /leonardo_scratch/large/userexternal/sbenatti/data/processed/emg2pose_pose/pose_stats.json
echo "=== date fine"; date '+%Y-%m-%d %H:%M:%S %Z'
} 2>&1
rm -f "$L"
