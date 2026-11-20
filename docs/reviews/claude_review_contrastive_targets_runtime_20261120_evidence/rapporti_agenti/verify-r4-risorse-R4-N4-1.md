# Scettico n. 1 — rilievo R4-N4 (r4-risorse): «I 1770 s interni non valgono per la cella CUDA»

Esito: **NON confutato**. Gravita' confermata: **NOTA** (solo documentazione, nessun effetto su tetto, cancelli o esiti).

## Riproduzione (comandi miei, copia a 0efa4d45; fbfde797..0efa4d45 aggiunge solo delivery_pin.json)
- `git show fbfde797:docs/contrastive_targets_execution_20261119.md | sed -n 91p`:
  «L'audit richiede `--time=02:00:00`. I job GPU mantengono1800s richiesti e1770s interni.»
  Nessun'altra riga del prontuario 19/11 restringe l'affermazione (grep di 1770/interni/time-limit: solo la riga 91).
- `scripts/contrastive_job.py:119-128` (fbfde797): il comando `cuda` (torchrun + contrastive_cuda_check.py) non contiene
  `--time-limit-s`; quello di calibration/full/resume (`:158-159`) passa `--time-limit-s 1770`.
- Prova funzionale con la fixture del test rivisto (`tests/cpu/test_contrastive_job.py::job_env`), file mio in
  rev_runtime/tmp: `J.build_command(job,'gpu',1800)` stampa
  `cuda -> contrastive_cuda_check.py time-limit-s None`; `calibration/full/resume -> contrastive_run.py time-limit-s 1770`
  (4 passed in 2,79 s).
- `grep -n "time-limit\|time_limit" scripts/contrastive_cuda_check.py` -> nessun risultato (exit 1): l'argparse
  (`:207-218`) non ha un parametro di limite; `trainer_stop_checks` (`:179`, `:196`) chiama `R.train` senza `time_limit_s`,
  che in `src/wearusfm/training/run.py:1233` vale `None` (il controllo `out_of_time` a `:1609` e' quindi spento).
- Limiti effettivi della cella CUDA: probe rank-exit `--timeout-s 60` con scadenza `time.monotonic()+90`
  (`contrastive_job.py:56-57`; `contrastive_rank_exit.py:23` COMPLETION_SECONDS=60, CLEANUP_SECONDS=20),
  timeout NCCL `timedelta(minutes=10)` (`contrastive_cuda_check.py:255`, timeout delle collettive, non limite di parete),
  e il walltime SLURM (`scripts/slurm/contrastive_preflight_gpu.sbatch:10` `--time=00:30:00`, invariato nel diff).

## Motivi di confutazione cercati e scartati
- Wrapper esterno (`timeout 1770`) nel launcher GPU: assente (sbatch esegue direttamente `contrastive_job.py`).
- Limite interno in `numeric_profile` o nel probe: solo timeout di sottoprocessi brevi (5 s, 60/90 s), non 1770 s.
- Piani: `preflight_plan.json` porta solo `seconds: 1800` per cella, nessun campo di limite interno; nessuna precisazione.
- Prontuario storico 16/11 (`docs/contrastive_preflight_execution_20261116.md:53-54,59`) separava correttamente il comando
  CUDA (senza limite) da `contrastive_run.py ... --time-limit-s 1770`, ma il prontuario 19/11 dichiara (riga 9) di prevalere
  sui prontuari del 16-18/11: la frase generale della riga 91 resta imprecisa per la cella CUDA. «mantengono» richiama la
  continuita' con lo storico, ma la lettera attribuisce i 1770 s a tutti i job GPU.

## Gravita'
- Nessun effetto sul tetto: `ALLOCATED_SECONDS['cuda']=1800` (`contrastive_job.py:17`) e il controllo
  `allocated_seconds != ALLOCATED_SECONDS.get(mode)` (`:67`) sul TimeLimit letto da scontrol (sbatch `:22-24`) fissano
  l'allocazione; il margine SLURM e' gia' contabilizzato (360 s per allocazione, prontuario `:105-107`).
- Durata storica della cella CUDA lontana dal limite: 141 s (64746651, `docs/reviews/contrastive_cuda_bf16_results_20261118.md:3`),
  67 s (64706211 FAILED, `docs/reviews/contrastive_cuda_failure_20261117.md:5`).
- Nessun cancello di lettura dipende da un arresto interno della CUDA. Quindi NOTA, come proposto.

## Correzione proposta (condivisa)
Riga 91: «1770 s interni per calibrazione, full e reload (`contrastive_run.py --time-limit-s 1770`); la cella CUDA ha solo
il probe rank-exit (60/90 s), il timeout NCCL di 10 minuti e il walltime SLURM di 1800 s».
