# Scettico n. 1 su R4-N3 (r4-risorse) — avanzamento

## Passo 1: prove citate riprodotte
- `sed -n 175,177p docs/contrastive_targets_execution_20261119.md` (identico a fbfde797 e 0efa4d45: il diff fbfde797..0efa4d45 tocca solo delivery_pin.json):
  "Per i checkout successivi `--dopo <ID_PREREQUISITO>` precede il separatore `--`." Presente.
- `sed -n 7852,7854p docs/decisioni.md`: "Dipendenza già completata e scaduta nel controller SLURM ... checkout 64652383 COMPLETED 0:0 (11 s), ma MinJobAge=30 sec ... Job dependency problem". Presente.
- Ricevuta archiviata `results/infra/contrastive_comparison_20261116/submission_20261117/audit_dependency_rejected_state.json`: sbatch con `--dependency=afterok:64652383` -> stderr "Batch job submission failed: Job dependency problem", stato `rifiutato`, job_slurm null; `dependency_diagnosis.json`: MinJobAge 30 s, DependencyParameters kill_invalid_depend.
- `scripts/submit_job.py:209-216`: `--dopo` legge solo il ledger (stato `sottomesso` + job_slurm) e aggiunge sempre `--dependency=afterok:<id>`; nessun controllo che il job sia ancora nel controller. `classify` (181-187): uscita !=0 con "sbatch: error" -> `rifiutato`; `reserve` (140-161): un `rifiutato` si ripresenta solo con `--dopo-rifiuto`.

## Passo 2: il prerequisito di un checkout successivo e' sempre gia' concluso (fbfde797)
- `scripts/contrastive_release.py:240-242` (make_job): per ogni cella non-audit richiede `validate_audit_accounting`, cioe' TSV sacct dell'audit con `State=COMPLETED`, `0:0` (righe 155-191). Il job JSON deve stare nel checkout congelato (`scripts/contrastive_job.py:179-180`, "job specification must be in frozen checkout"; sbatch: `--job "$CODE_DIR/$1"`). Quindi lo SHA del checkout di calibrazione esiste solo dopo la fine dell'audit.
- Reload: `scripts/contrastive_run.py:247-265` (validate_target_fork) esige summary del full con `steps==576`, `stopped==max_steps` e pin del checkpoint 544 nella release -> checkout dei reload solo dopo i full conclusi. Full: `validate_cuda_prerequisite` (release 194-211) esige la ricevuta CUDA pinnata. Prontuario 182-187 e 141-145 lo dichiarano ("Le fasi successive pubblicano pin solo dopo le ricevute effettive"; "I sei checkout congelano solo le nuove evidenze tra fasi").
- Fra fine del prerequisito e sbatch del checkout ci sono raccolta, pin, commit, push e deploy: ben oltre MinJobAge 30 s.

## Passo 3: simulazione del submitter (clone non toccato; registro e finto ssh in rev_runtime/tmp/r4n3_sim)
Finto ssh che risponde come il controller del 17/11 a un afterok verso job rimosso. Esito:
- `--dopo CONTRAST-AUDIT-targets-v3-20261118` sul checkout di calibrazione -> sbatch emesso con `--dependency=afterok:64999001`, esito `rifiutato` (exit 3), "Job dependency problem".
- ripresentazione senza `--dopo-rifiuto` -> "gia' prenotato, stato 'rifiutato'. Si risottomette solo con --dopo-rifiuto." (exit 1); con `--dopo-rifiuto` e senza `--dopo` -> sottomesso.

## Passo 4: ricerca di motivi per confutare
- Nessun controllo altrove: submit_job.py non interroga sacct/scontrol per `--dopo`; nessun generatore di comandi nei moduli nuovi (grep di submit_job/--dopo in scripts/contrastive_*.py e tests: nulla).
- Il prontuario non contiene avvertenze su MinJobAge/dipendenze scadute (grep: nulla) e alla riga 9 "prevale sui prontuari operativi del 16-18/11", che invece contenevano l'avvertenza corretta (docs/reviews/contrastive_calibration_release_20261117.md:52-54; docs/contrastive_handoff_20261117.md:229). I diagnostici 365cb1e avevano `gpu_commands_after_verified_completed_checkout` (variante senza dipendenza).
- Attenuanti (non confutano): righe 178-179 rinviano i comandi definitivi alla consegna post-review; nessun costo (17/11: job_slurm null, audit_job_created false); precedente 17/11 accetta la ripresentazione esplicita con `--dopo-rifiuto`.

## Esito
Non confutato. Il rilievo e' riprodotto in tutte le parti verificabili senza cluster; il comportamento di SLURM e' quello archiviato il 17/11 (audit_dependency_rejected_state.json, dependency_diagnosis.json). Gravita' NOTA confermata: indicazione documentale che porta a un rifiuto innocuo e recuperabile, nessun job, nessun costo, nessun cancello falsificato; i comandi definitivi non sono ancora emessi.
Non verificato: comportamento live di SLURM/MinJobAge attuale su Leonardo (vietato l'accesso); riletto solo dall'archivio del 17/11 e dai precedenti 04-05/11.
