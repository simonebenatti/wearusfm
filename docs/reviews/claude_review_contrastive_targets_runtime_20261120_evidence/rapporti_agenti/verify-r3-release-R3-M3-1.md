# verify-r3-release-R3-M3-1 — scettico n. 1 sul rilievo R3-M3

Avvio: 2026-11-20. Compito: confutare R3-M3 (ID SLURM dell'accounting non ancorato all'esecuzione dell'audit).
Copia di riferimento: rev_targets/clone a 0efa4d45 (sorgenti = fbfde797). Solo lettura; esperimenti in rev_runtime/tmp.

## Riproduzione propria (clone tmp/v3m3 a 0efa4d45; script tmp/v3m3_scripts/repro.py; uscita tmp/v3m3_repro.out)
- Moduli reali (ROOT = clone, source 40d39f56..., ID audit CONTRAST-AUDIT-targets-v3-20261118 dal preflight).
- S1 accounting vero del job audit (TIMEOUT 7387 s): RESPINTO. S2 stesso report + accounting/TSV di un altro job 64900001 (COMPLETED 0:0, 7000 s, 8 CPU, 120 min): ACCETTATO (anche come pin dict).
- S3 stesso caso con expected_job_id="64900000": RESPINTO ("job ID missing or changed") -> il controllo esiste ma nessun chiamante lo usa.
- S4 collisioni accidentali con job di altre fasi (checkout 4 CPU/4 min, readout 8 CPU/60 min, GPU 32 CPU/30 min, 8 CPU/240 min): tutte RESPINTE; resta esposto solo un job 8 CPU/120 min COMPLETED <= 7200 s.
- S5 make_job della calibrazione con binding reale (review sintetica committata nel clone tmp, bind_review ok): accounting dell'altro job ACCETTATO, accounting vero TIMEOUT RESPINTO.
- S6 dispatcher build_command (calibrazione, gpu, 1800): con l'accounting dell'altro job supera il gate (si ferma dopo, al controllo dell'output); con quello vero si ferma al gate.
- J1 (tmp/v3m3_scripts/jobname.py): TSV dell'altro job con colonne JobName/Comment di un ALTRO ID logico ("ALTRO-LAVORO-8CPU-2H"): ACCETTATO; l'esito riporta logical_id CONTRAST-AUDIT-targets-v3-20261118 (preso dal piano, non dal record SLURM).
- pytest tests/cpu/test_contrastive_release.py nel clone tmp: 22 passed in 5,37 s.

## Tentativi di confutazione
- Controllo altrove? NO. Chiamanti a fbfde797 (git grep): contrastive_job.py:81, contrastive_release.py:242 e 281, contrastive_run.py:195, contrastive_benchmark_readout.py:421; nessuno passa expected_job_id. validate_release (contrastive_run.py:164-195), build_command (contrastive_job.py:129-142) e read_preflight (readout 405-447) non confrontano job_id con nulla; il readout lo copia in 'audit_accounting'. Nessun codice contrastivo legge ledger, log SLURM o JobName.
- Percorso non di produzione? NO: il gate e' il cancello R-4 in make_job, dispatcher, runtime della calibrazione e lettura finale.
- Dichiarazione esplicita del limite? NO. Prontuario 140-146: l'accounting contiene "ID logico e SLURM", nessuna menzione di un ID solo dichiarato. Risposta R-4 (response:13): "lega audit, report, sorgenti, ID e TSV sacct". Il test test_audit_accepts_completed_exact_limit_and_binds_receipt_to_allocation usa expected_job_id="123", che la produzione non passa: la documentazione descrive un legame piu' forte di quello attivo.
- Prassi del progetto (aggrava, non confuta): logsquare_pose_readout.py:314 (linea 3) ancora JobIDRaw al slurm_job_id del report; i report registrano os.environ["SLURM_JOB_ID"] in scaling_preflight.py:269, pose_*_readout, scaling_pose_readout.py:350 ecc. contrastive_audit.py:181-185 no.
- Ancora disponibile ma non usato: submit_job.py:224 impone --job-name=<ID logico> e --comment=wearusfm:<ID logico>; il gate accetta colonne extra ma non richiede/confronta JobName (J1).
- Imprecisione del rilievo (non confuta): il dispatcher stampa prima di execv un JSON con command (incluso --out dell'audit), job_sha256 e source_sha256 nel log contrastive_cpu_%j.out (contrastive_job.py:186-187; sbatch righe 10-11): una traccia manuale con l'ID nel nome del file esiste, ma nessun controllo la legge e non contiene SLURM_JOB_ID.

## Gravita'
- Limiti delle collisioni accidentali (S4): serve un job dello stesso utente con 8 CPU, TimelimitRaw 120, COMPLETED 0:0, <= 7200 s; le altre fasi della coorte sono respinte.
- Una falsificazione deliberata del TSV (testo dell'operatore) non e' comunque impedibile dal codice: l'ancora protegge solo dall'errore di identificazione. Con input corretti la catena e' corretta: niente BLOCCANTE.
- Difetto reale di codice (parametro previsto e mai collegato; ancore JobName/slurm_job_id disponibili e non usate), non solo documentazione: MINORE confermato.
- Correzione minima possibile senza toccare il report dell'audit: richiedere la colonna JobName nel TSV e JobName == logical_id (imposto da submit_job). Cambia comunque l'inventario (nuova review).

## Esito: NON confutato. Gravita' MINORE confermata.
- Pulizia: cloni tmp/v3m3, v3m3_work e basetemp v3m3_pt cancellati; conservati tmp/v3m3_scripts/ e tmp/v3m3_repro.out. Copia di riferimento a 0efa4d45, pulita.
