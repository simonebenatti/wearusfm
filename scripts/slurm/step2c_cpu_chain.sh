#!/bin/bash
# Step 2C (docs/step2c_frozen_branches.md): sottomissione della catena CPU dal Mac (via leonardo-ops), dopo i quattro run dei rami.
# 8 estrazioni (teacher x P1/P2), poi 16 job di sonda (due varianti ciascuno), ognuno `afterok` della sua estrazione; l'ultimo (P2, B_seed1,
# RPW,RPW0) fa anche il confronto (COMPARE=1) ed e' `afterok` della sua estrazione e degli altri 15 job di sonda (submit_job.py accetta una
# sola dipendenza: questa si passa a sbatch esplicita). Si ferma alla prima sottomissione non riuscita (nessun retry). Con DRY=1 stampa i
# comandi senza eseguirli. Uso: [DRY=1] bash scripts/slurm/step2c_cpu_chain.sh <sha del checkout> [STEP2A=<cartella del 2A concluso>]
set -euo pipefail
SHA="${1:?sha del checkout di campagna}"
CAMPAIGN=step2c-20261014
C=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/$CAMPAIGN
RUNS=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/step1_ab_0810
BR=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/runs/step2c_1410
DOWN=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_downstream_0910
EVAL=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410
EXP="ALL,CODE_DIR=$C,EXPECTED_SHA=$SHA"
ARMS="A_seed0 A_seed1 B_seed0 B_seed1"
LAST=""
N=0

sub() {  # submit_job.py con gli argomenti dati; LAST = ID SLURM del job sottomesso
  if [ "${DRY:-0}" = "1" ]; then
    N=$((N + 1)); LAST="DRY$N"
    echo "python3 scripts/submit_job.py $*"
    return 0
  fi
  local out
  if ! out=$(python3 scripts/submit_job.py "$@" 2>&1); then echo "$out"; echo "STOP: submit_job.py uscito con errore"; exit 1; fi
  echo "$out"
  case "$out" in
    *": sottomesso, job "*) LAST="${out##*job }";;
    *) echo "STOP: stato diverso da sottomesso"; exit 1;;
  esac
}

EXT_FINAL=""
for T in p1 p2; do
  TIME=01:30:00
  if [ "$T" = p2 ]; then TIME=03:00:00; fi
  for A in $ARMS; do
    sub --id "step2c_extract_${T}_$A" --agente claude --campagna "$CAMPAIGN" --gpu-ore 0 -- --time=$TIME --export="$EXP" \
      "$C/scripts/slurm/step2c_extract.sbatch" "$T" "$A" "$RUNS" "$BR" "$DOWN" "$EVAL"
    if [ "$T" = p2 ] && [ "$A" = B_seed1 ]; then EXT_FINAL="$LAST"; fi
  done
done
DEPS=""
for T in p1 p2; do
  TIME=02:00:00
  if [ "$T" = p2 ]; then TIME=02:30:00; fi
  for A in $ARMS; do
    for V in main rami; do
      if [ "$T" = p2 ] && [ "$A" = B_seed1 ] && [ "$V" = main ]; then continue; fi
      VARS=RPW,RPW0
      if [ "$V" = rami ]; then VARS=RP,RW; fi
      sub --id "step2c_probe_${T}_${A}_$V" --agente claude --campagna "$CAMPAIGN" --gpu-ore 0 --dopo "step2c_extract_${T}_$A" -- \
        --time=$TIME --export="$EXP" "$C/scripts/slurm/step2c_probe.sbatch" "$T" "$A" "$VARS" "$RUNS" "$BR" "$DOWN" "$EVAL"
      DEPS="$DEPS:$LAST"
    done
  done
done
FINAL_EXP="$EXP,COMPARE=1"
if [ -n "${STEP2A:-}" ]; then FINAL_EXP="$FINAL_EXP,STEP2A=$STEP2A"; fi
sub --id step2c_probe_p2_B_seed1_main --agente claude --campagna "$CAMPAIGN" --gpu-ore 0 -- --time=03:00:00 \
  --dependency="afterok:$EXT_FINAL$DEPS" --export="$FINAL_EXP" "$C/scripts/slurm/step2c_probe.sbatch" p2 B_seed1 RPW,RPW0 \
  "$RUNS" "$BR" "$DOWN" "$EVAL"
echo "catena sottomessa: 8 estrazioni, 16 job di sonda (confronto nell'ultimo)"
