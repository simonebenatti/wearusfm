#!/bin/bash
# Sottomette una catena di N job uguali collegati con --dependency=afterok (eseguito sul login node: solo sbatch, nessun calcolo).
# Ogni job riprende dal checkpoint della stessa cartella del run; un job che fallisce (exit != 0 o TIMEOUT di SLURM) ferma la catena; un allarme o
# una perdita non finita scrivono STOP nella cartella del run e i job successivi escono subito (scripts/slurm/sanity_jepa.sbatch).
# Su boost_qos_dbg al massimo 2 job sottomessi (MaxSubmitPU = 2, anche quelli in attesa): catene lunghe su boost_usr_prod (--qos e --time
# davanti allo script li sovrascrivono: SBATCH_QOS / SBATCH_TIMELIMIT nell'ambiente, o modificare lo sbatch).
# Uso: bash scripts/slurm/submit_chain.sh N scripts/slurm/sanity_jepa.sbatch <argomenti dello sbatch...>
set -euo pipefail
N="${1:?numero di job}"; shift
SCRIPT="${1:?sbatch}"; shift
prev=$(sbatch --parsable "$SCRIPT" "$@")
echo "$prev"
for _ in $(seq 2 "$N"); do
  prev=$(sbatch --parsable --dependency=afterok:"$prev" "$SCRIPT" "$@")
  echo "$prev"
done
