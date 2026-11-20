# verify r4-risorse R4-N7 — scettico n. 1

Inizio: 2026-11-20 03:26. Copia rev_targets/clone, HEAD 0efa4d45 (scripts identici a fbfde797:
sha256 di contrastive_release.py 389cf29c... in entrambi; fbfde797..0efa4d45 aggiunge solo delivery_pin.json).

## Prove statiche (03:30)
- fbfde797:scripts/contrastive_release.py:215-243: make_job; docstring 216-220 (righe 218-219 come citato);
  verifiche: validate_review_binding (221), cella unica/modo (222-228), pin input vs piano (229-231),
  chiavi proibite (233-235), solo per mode != audit checked_pin(audit) + validate_audit_accounting (240-242).
  Nessun controllo di cuda_report, calibration_report, calibration_checkpoint, cohort.
- git grep validate_cuda_prerequisite: definita release.py:195, unica chiamata contrastive_run.py:208-209
  (validate_release, stage benchmark). Nessun test la chiama direttamente.
- git grep "calibration_report\|cohort" scripts/contrastive_cuda_check.py: vuoto (exit 1).
- calibration_accepted compare solo come etichetta 'depends' (targets_plan.py:92-93, preflight_plan.json:62,81,100,119);
  nessuno script legge 'depends'.
- contrastive_job.build_command ramo cuda (fbfde797:scripts/contrastive_job.py:119-128): validate_recipe(config,'calibration'),
  pin del checkpoint; nessuna lettura del rapporto di calibrazione. Il launcher GPU esegue build_command dentro l'allocazione.
- contrastive_cuda_check.py:243-247: verifica solo che il checkpoint sia quello a pesi fissi (step 512, niente optimizer,
  teacher = student iniziale), non PASSED_FOR_REVIEW / failure_reasons.

## Riproduzione dinamica (03:29)
Script rev_runtime/tmp/verify_r4n7_1_scripts/repro.py (pacchetto sintetico come la fixture di
tests/cpu/test_contrastive_release.py, moduli importati dalla copia, venv wearusfm-tok, -B). Output:
- 1 make_job CUDA con rapporto di calibrazione FAILED_REQUIRES_REVIEW e checkpoint arbitrario -> cuda APPROVED_FOR_PREFLIGHT
- 2a make_job full con ricevuta CUDA passed=False -> full APPROVED_FOR_PREFLIGHT; 2b senza cuda_report -> accettato
- 3 validate_cuda_prerequisite sulla stessa ricevuta -> "CUDA/NCCL check missing, failed or from different source/cohort/inputs"
- 4 make_job CUDA senza audit -> KeyError 'audit' (l'unica ricevuta a monte richiesta e' l'audit)
- 5 contrastive_job.build_command cuda (stesse sostituzioni della fixture job_env) con calibrazione fallita: comando costruito,
  nessun --calibration-report ne' --cohort.
Cartella di lavoro cancellata; copia pulita (status vuoto, HEAD 0efa4d45).

## Motivi cercati per confutare (03:35)
- Lettura stretta del docstring: prontuario:145 "Il gate e' eseguito nella release, nel dispatcher e nel runtime" usa
  "release" per il gate dell'audit, e validate_audit_accounting dice "audit accounting required before release": in quel
  lessico le ricevute "needed for release" sono l'accounting, che make_job controlla. Ma prontuario:186-187 chiede che la
  ricevuta CUDA "esista e sia pinnata prima di costruire i job full" (make_job) e make_job non lo verifica (2b): il
  docstring resta ambiguo/fuorviante per le celle full. Nessuna dichiarazione esplicita che CUDA e calibrazione accettata
  siano controllate solo sul nodo GPU o dall'operatore.
- Controllo altrove prima dell'allocazione: nessuno. checkout (campaign_checkout.sbatch) e submit_job.py non leggono
  ricevute; afterok vede solo il codice d'uscita (la calibrazione esce 0 anche con gate falliti, perche' il rapporto e'
  calcolato dopo da contrastive_prepare calibration-report). contrastive_run --validate-only potrebbe verificare su CPU ma
  nessun prontuario lo prescrive.
- Percorso non di produzione: no, make-job e' il comando di preparazione del prontuario (righe 152-159).
- Mitigazioni reali (non confutano): per le celle full la calibrazione accettata e' verificata da macchina
  (contrastive_run.py:47 validate_recipe -> calibrated_configs -> authenticate_calibration; :59 validate_cohort) e la CUDA
  da validate_cuda_prerequisite (:208-209), prima del training ma dentro l'allocazione; la lettura finale
  (benchmark_readout.py:398, :422) ricontrolla entrambe. Una ricevuta CUDA da calibrazione non accettata non e' mai
  utilizzabile (pin del checkpoint e initial_sha legati alla coorte autenticata): nessuna evidenza falsa accettata.

## Esito
Non confutato. Fatti riprodotti. Gravita' NOTA confermata (chiarezza/documentazione e cancello procedurale; spreco solo con
errore dell'operatore, nessun tetto superato: la cella CUDA e' gia' nel preventivo).
Precisazioni: il limite "al piu' 2 GPUh" va inteso per allocazione (con OverTimeLimit+KillWait del piano fino a 2,4 GPUh,
il CUDA non ha limite interno); con ricevuta CUDA fallita possono partire fino a tre full, ciascuno rifiutato all'avvio
(secondi-minuti). "Solo in contrastive_run" vale prima dei benchmark: dopo, la lettura finale ricontrolla.
Fine 03:36.
