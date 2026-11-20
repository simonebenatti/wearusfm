# Scettico n. 2 — rilievo R3-M3 (r3-release)

Rilievo: l'ID SLURM dell'accounting dell'audit non e' ancorato all'esecuzione reale dell'audit.
Area di lavoro: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/vr3m3_2

## Progressi
- [avvio] letti contrastive_release.py (155-192, 215-243), contrastive_job.py (62-188), sbatch preflight cpu/gpu, contrastive_audit.py (invariato nel diff), prontuario righe 121-146.

## Riproduzione propria (moduli reali della copia, nessuna mutazione; fixture in tmp/vr3m3_2/work)
Script: tmp/vr3m3_2/scripts/repro.py; uscita: tmp/vr3m3_2/repro.out. ROOT = copia, source 40d39f56... = preflight, ID audit CONTRAST-AUDIT-targets-v3-20261118.
- S1 report audit + accounting/TSV veri del job (ipotetico) 64900000, COMPLETED 7387 s: RESPINTO ("audit requires COMPLETED 0:0, Elapsed <= 7200 s ...").
- S2 STESSO report (stessi byte) + coppia accounting/TSV coerente di un altro job 64900001 (COMPLETED 0:0, 3600 s, 8 CPU, 120 min): ACCETTATO.
- S3 come S2 con expected_job_id="64900000": RESPINTO ("job ID missing or changed"): il controllo esiste ma solo a mano.
- S4 job_id 64900001 con TSV di 64900000: RESPINTO solo per incoerenza interna ("one exact allocation row").
- S5/S6 coppie di un checkout (4 CPU/4 min) o di un job GPU (32 CPU/30 min): RESPINTE dalla forma.
- CLI di produzione `contrastive_release.py check-audit` (unico CLI del gate): S1+coppia di 64900001 -> exit 0, JSON con job_id 64900001; il CLI non ha alcuna opzione per l'ID atteso (--help: solo --audit, --accounting).
- pytest tests/cpu/test_contrastive_release.py -k audit: 13 passed in 4,12 s. Il caso parametrizzato job_id-124 di test_audit_cannot_inherit_unrelated_receipt_or_allocation passa solo perche' il TSV contiene righe 123 (incoerenza interna), non per un legame con l'esecuzione.

## Tentativi di confutazione (tutti falliti)
- Controllo altrove: git grep a fbfde797, chiamanti di validate_audit_accounting = contrastive_job.py:81, contrastive_release.py:242 e 281, contrastive_run.py:195, contrastive_benchmark_readout.py:421; expected_job_id compare solo nella definizione (release:155,172) e nei test (test_contrastive_release.py:111,114). Nessun SLURM_* in contrastive_audit/release/run/prepare; in contrastive_job solo validate_allocation (165-167, CPU/memoria, non l'ID).
- Produttori: nessuno script produce l'accounting JSON o il TSV (grep "TimelimitRaw|ElapsedRaw|contrastive-audit-accounting"): li costruisce a mano 3-launcher.
- Dichiarazioni: prontuario 140-146 ("accounting JSON ... ID logico e SLURM, pin del TSV sacct") e risposta R-4 (response:13, "lega audit, report, sorgenti, ID e TSV sacct") descrivono un legame; nessun limite dichiarato del tipo "ID SLURM attestato dall'operatore" (la clausola di provenienza non crittografica, prontuario 129-131 e release.py:129-131, riguarda solo la review).
- Prassi del progetto (aggrava): logsquare_pose_readout.py:314 (JobIDRaw == report.slurm_job_id, registrato a 390) e scaling_readers.sbatch:194 ancorano la riga sacct all'ID registrato dal job stesso.
- Ancora disponibile e non usata: submit_job.py:204-205 e 224 impongono --job-name=<ID logico> e --comment=wearusfm:<ID logico>; il gate non richiede JobName/Comment.
- Imprecisione del rilievo, non confutante: il dispatcher stampa prima di execv un JSON con command/job_sha256/source_sha256 (contrastive_job.py:186-187), che finisce nel log contrastive_cpu_%j.out (contrastive_preflight_cpu.sbatch:10-11): traccia manuale con l'ID nel nome del file, ma nessun gate la legge.
- Precisione: per l'accettazione accidentale l'altro job deve avere anche AllocCPUS 8 e TimelimitRaw 120 (non solo COMPLETED entro 7200 s); le altre fasi della coorte (checkout, GPU, readout 60 min) sono respinte.

## Gravita'
- Con input onesti e corretti la catena e' corretta; serve un errore di identificazione dell'operatore (o un atto deliberato, che pero' nemmeno la correzione proposta impedisce del tutto: il TSV resta testo fornito dall'operatore). Niente BLOCCANTE.
- Difetto reale di codice (parametro previsto e mai collegato, ancore JobName/slurm_job_id disponibili, test e risposta R-4 che descrivono un legame piu' forte): MINORE confermato.

## Esito: NON confutato (refuted=false), gravita' MINORE.
