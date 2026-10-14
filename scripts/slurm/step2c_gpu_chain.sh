#!/bin/bash
# Step 2C (docs/step2c_frozen_branches.md): estrazioni e sonde sui nodi GPU (Simone, 14/10/2026: «opzione 2, vai su GPU», «Anche le estrazioni
# su GPU»), dal Mac via leonardo-ops. La partizione CPU (`lrd_all_serial`) ammette 10 job in coda e 8 CPU in esecuzione per utente: in serie la
# catena richiederebbe ~34 h. Qui ogni job prende un quarto di nodo GPU (1 A100, 8 CPU) e gira in parallelo; la GPU resta inattiva (estrazione
# dei rami e sonde sklearn sulla CPU), stessi launcher (`step2c_extract.sbatch`, `step2c_probe.sbatch`) col solo cambio di partizione.
# Le estrazioni elencate in KEEP (gia' partite sulla CPU, `step2c_extract_<task>_<teacher>`) non si rifanno: le loro sonde ne dipendono. Ogni
# sonda (due varianti) e' `afterok` della sua estrazione; l'ultima (P2, B_seed1, RPW,RPW0) fa il confronto ed e' `afterok` della sua estrazione
# e delle altre 15. Tempi piu' larghi per le varianti con 1.024 colonne. Si ferma alla prima sottomissione non riuscita; DRY=1 stampa soltanto.
# Uso: [DRY=1] [KEEP="p1_A_seed0"] bash scripts/slurm/step2c_gpu_chain.sh <sha> [STEP2A=<cartella del 2A concluso>]
set -euo pipefail
SHA="${1:?sha del checkout di campagna}"
KEEP=" ${KEEP:-} "
CAMPAIGN=step2c-20261014
C=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/$CAMPAIGN
RUNS=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/step1_ab_0810
BR=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/runs/step2c_1410
DOWN=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_downstream_0910
EVAL=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410
EXP="ALL,CODE_DIR=$C,EXPECTED_SHA=$SHA"
GPU=(-p boost_usr_prod --qos=boost_qos_lprod --gres=gpu:1 --cpus-per-task=8 --mem=60G)
LAST=""
N=0

sub() {  # sub <ID logico> <argomenti di submit_job.py>; LAST = ID SLURM del job. Riprendibile: un ID gia' sottomesso si riusa (non si
  # risottomette), uno rifiutato da sbatch si risottomette con --dopo-rifiuto (stato letto dal registro di submit_job.py)
  local id="$1"; shift
  local st extra=()
  st=$(python3 scripts/submit_job.py --stato "$id" | cut -f2,3)
  case "$st" in
    sottomesso*) LAST="${st#*$'\t'}"; echo "$id: gia' sottomesso, job $LAST (riusato)"; return 0;;
    rifiutato*) extra=(--dopo-rifiuto);;
    None*) ;;
    *) echo "STOP: $id in stato $st"; exit 1;;
  esac
  if [ "${DRY:-0}" = "1" ]; then
    N=$((N + 1)); LAST="DRY$N"
    echo "python3 scripts/submit_job.py --id $id ${extra[*]+${extra[*]}} $*"
    return 0
  fi
  local out
  if ! out=$(python3 scripts/submit_job.py --id "$id" ${extra[@]+"${extra[@]}"} "$@" 2>&1); then
    echo "$out"; echo "STOP: submit_job.py uscito con errore"; exit 1
  fi
  echo "$out"
  case "$out" in
    *": sottomesso, job "*) LAST="${out##*job }";;
    *) echo "STOP: stato diverso da sottomesso"; exit 1;;
  esac
}

extract_id() {  # ID logico dell'estrazione di cui dipendono le sonde di <task>_<teacher>
  case "$KEEP" in *" $1 "*) echo "step2c_extract_$1";; *) echo "step2c_gextract_$1";; esac
}

EXT_FINAL=""
for T in p1 p2; do
  TIME=03:00:00  # P1: ~54 min per modello misurati sulla CPU di lrd_all_serial (60491778), due modelli
  if [ "$T" = p2 ]; then TIME=03:00:00; fi
  for A in B_seed0 B_seed1 A_seed0 A_seed1; do  # prima la famiglia B
    case "$KEEP" in *" ${T}_$A "*) continue;; esac
    sub "step2c_gextract_${T}_$A" --agente claude --campagna "$CAMPAIGN" --gpu-ore 1 -- "${GPU[@]}" --time=$TIME --export="$EXP" \
      "$C/scripts/slurm/step2c_extract.sbatch" "$T" "$A" "$RUNS" "$BR" "$DOWN" "$EVAL"
    if [ "$T" = p2 ] && [ "$A" = B_seed1 ]; then EXT_FINAL="$LAST"; fi
  done
done
if [ -z "$EXT_FINAL" ]; then echo "STOP: l'estrazione P2 di B_seed1 deve essere sottomessa da questo script"; exit 1; fi
DEPS=""
for T in p1 p2; do
  for A in B_seed0 B_seed1 A_seed0 A_seed1; do
    for V in main rami; do
      if [ "$T" = p2 ] && [ "$A" = B_seed1 ] && [ "$V" = main ]; then continue; fi
      if [ "$V" = main ]; then VARS=RPW,RPW0; TIME=02:30:00; else VARS=RP,RW; TIME=02:00:00; fi
      if [ "$T" = p2 ]; then
        if [ "$V" = main ]; then TIME=03:00:00; else TIME=02:30:00; fi
      fi
      sub "step2c_gprobe_${T}_${A}_$V" --agente claude --campagna "$CAMPAIGN" --gpu-ore 1 --dopo "$(extract_id "${T}_$A")" -- \
        "${GPU[@]}" --time=$TIME --export="$EXP" "$C/scripts/slurm/step2c_probe.sbatch" "$T" "$A" "$VARS" "$RUNS" "$BR" "$DOWN" "$EVAL"
      DEPS="$DEPS:$LAST"
    done
  done
done
FINAL_EXP="$EXP,COMPARE=1"
if [ -n "${STEP2A:-}" ]; then FINAL_EXP="$FINAL_EXP,STEP2A=$STEP2A"; fi
sub step2c_gprobe_p2_B_seed1_main --agente claude --campagna "$CAMPAIGN" --gpu-ore 1 -- "${GPU[@]}" --time=03:30:00 \
  --dependency="afterok:$EXT_FINAL$DEPS" --export="$FINAL_EXP" "$C/scripts/slurm/step2c_probe.sbatch" p2 B_seed1 RPW,RPW0 \
  "$RUNS" "$BR" "$DOWN" "$EVAL"
echo "catena sottomessa sui nodi GPU (confronto nell'ultimo job)"
