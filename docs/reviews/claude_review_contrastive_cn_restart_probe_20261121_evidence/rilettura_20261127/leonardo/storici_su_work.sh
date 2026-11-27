date "+%d/%m/%Y %H:%M:%S"
R=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118
echo "WORK risolto: $(readlink -f "$WORK")"
ls -l "$R/benchmark/CN/checkpoint_step0000544.pt" | awk '{print "fork:", $5, $9}'
cd "$R/reloads/CN_A" && sha256sum trace-0.jsonl trace-1.jsonl trace-2.jsonl trace-3.jsonl numeric_profile.rank0.json numeric_profile.rank1.json numeric_profile.rank2.json numeric_profile.rank3.json metrics.jsonl config.json start.json summary.json | sed 's#  #  CN_A/#'
cd "$R/benchmark/CN" && sha256sum trace-0.jsonl trace-1.jsonl trace-2.jsonl trace-3.jsonl numeric_profile.rank0.json numeric_profile.rank1.json numeric_profile.rank2.json numeric_profile.rank3.json metrics.jsonl trace-0.ddp.jsonl trace-1.ddp.jsonl trace-2.ddp.jsonl trace-3.ddp.jsonl config.json summary.json | sed 's#  #  CN/#'
