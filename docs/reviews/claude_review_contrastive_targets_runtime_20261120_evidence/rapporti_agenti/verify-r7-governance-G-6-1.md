# verify-r7-governance G-6 (scettico n. 1)

Esito: **NON confutato**. Gravita' reale: **NOTA** (confermata).

## Riproduzione (comandi miei, fbfde797 = 0efa4d45 per gli script: `git diff --stat fbfde797 0efa4d45` tocca solo delivery_pin.json)
- `git grep -n expected_job_id fbfde797 -- scripts tests docs results`: solo `contrastive_release.py:155` (firma), `:172` (controllo) e `tests/cpu/test_contrastive_release.py:111,114`.
- Chiamanti di produzione, tutti senza `expected_job_id`: `contrastive_job.py:81` (build_command), `contrastive_run.py:195` (validate_release), `contrastive_release.py:242` (make_job), `contrastive_benchmark_readout.py:421` (read_preflight, solo `expected_source`), CLI `check-audit` `contrastive_release.py:281`.
- `test_contrastive_historical_callsites.py` sostituisce il gate con un finto che accetta solo `(audit_path, sacct_path, *, expected_source)`: la firma senza ID e' quella fissata dal test.
- `contrastive_audit.py` (invariato nel diff): 0 occorrenze di «slurm»; il report ha inputs, config_sha256, source_sha256, descriptor_budget, world_size, workers_per_rank. `contrastive_job.py` legge `SLURM_*` solo in validate_allocation (r.165-167) e non scrive l'ID. L'ID compare solo nel nome del log `contrastive_cpu_%j.out` (sbatch r.10), non controllato dal gate.
- Prova: `rev_runtime/tmp/g6v1/tests/test_g6_declared_job_id.py` (riusa la fixture `package` del test di release). Con la firma di produzione, lo stesso report di audit passa con un accounting che dichiara un'allocazione qualsiasi (`job_id` 999 e riga TSV `999|COMPLETED|0:0|100|8|120`); lo rifiuta solo `expected_job_id="123"`, che nessuno passa. Esito: `24 passed in 6.47s` (20 del file di release + 4 miei, compresi i 3 casi del secondo test).

## Motivi cercati per confutare (non sufficienti)
- R-4 di 94f5094 (r.130-137, 295) chiedeva solo State/ExitCode/Elapsed da sacct: il legame report->allocazione non era richiesto. La risposta r.13 («lega audit, report, sorgenti, ID e TSV sacct») e' letteralmente vera per cio' che il receipt collega: hash del report, sorgenti, ID logico contro il piano, ID SLURM contro una riga unica del TSV. Il test negativo di ID con rilevanza di produzione esiste: `job_id` "124" senza riga nel TSV, test_contrastive_release.py:125-131.
- Prontuario r.141-142: l'accounting e' un JSON dell'operatore («ID logico e SLURM, pin del TSV sacct»). Non dichiara pero' che l'attribuzione all'allocazione sia una dichiarazione. Il limite analogo e' dichiarato solo per la review (prontuario r.130-131, release.py:129-131).
- Il rischio si restringe: AllocCPUS=8 e TimelimitRaw=120 escludono checkout (4/4), GPU (32/30), readout (60) e audit storici (60, sorgente diversa); il mio secondo test lo conferma. submit_job.py:12 vieta di riusare un ID. Resta possibile un errore dell'operatore: ID di un altro job seriale da 8 CPU e 2 h, oppure secondo audit sotto un ID nuovo. Anche `logical_id` e' dichiarato e non viene confrontato col Comment di sacct.

## Gravita'
Nessuna catena scorretta dimostrata senza errore o falsificazione dell'operatore. Restano due problemi di chiarezza: un parametro non collegato (il test `..._binds_receipt_to_allocation` suggerisce un legame che la produzione non ha) e un limite non dichiarato. Quindi **NOTA**. Rimedio sensato: scrivere `SLURM_JOB_ID` nel report o in una ricevuta di contrastive_job, oppure dichiarare il limite nel prontuario.
