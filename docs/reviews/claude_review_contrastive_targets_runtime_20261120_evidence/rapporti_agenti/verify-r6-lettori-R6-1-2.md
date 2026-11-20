# Verifica scettica n. 2 del rilievo R6-1 (r6-lettori)

Inizio: 2026-11-20 04:02:10

## Lettura del codice (fbfde797)
- job.py:76-79: unico controllo dell'arm = etichetta job['arm'] contro cells[0]['arm']; job['arm'] non e' usato altrove (git grep: solo job.py:77 e release.py:237).
- job.py:129-159 (full/resume): config letto solo come hash (checked_file); payload release confrontato su status/stage/config_sha256/source/time/inputs/audit/phase/package_review/audit_sacct/cuda/cohort/calibration/fork/continuous_summary; nessun confronto di payload['arm'] o del contenuto del config con la cella. Il repeat invece e' legato (expected_phase, 133/139).
- release.py:229-243 make_job: arm copiato dalla cella, evidenza non puo' sovrascriverlo, ma config/release vengono dall'evidenza senza confronto di braccio.
- run.py: nessun riferimento a cella/job/id; validate_release confronta release.arm con config.arm (205-206); validate_cohort accetta qualunque membro della coorte; validate_target_fork lega fork al braccio della release, non alla cella.
- launcher sbatch contrastive_preflight_gpu/cpu: nessun controllo di braccio o di ID logico.

## Riproduzione indipendente (04:09)
Probe mio: rev_runtime/tmp/sk2r61_scripts/probe_arm_binding.py, clone temporaneo sk2r61_clone a 0efa4d45 (review di prova committata solo li').
Binding REALE (bind_review + validate_review_binding con verifica Git), audit accounting REALE, celle REALI del preflight_plan, make_job REALE.
Stub solo DATA_ROOTS e validate_input_bindings (dati di Leonardo), estranei al braccio.
Esito:
- A  BENCH-C0 + config/release CN: ACCETTATO -> config eseguito CN, release CN phase full (contrastive_run.py).
- A0 BENCH-C0 + C0: accettato (controllo positivo).
- A1 job.arm=CN sulla cella C0: RIFIUTATO 'job differs from the reviewed fixed preflight cell' (controllo solo sull'etichetta).
- A2 make_job con evidence.arm: RIFIUTATO 'evidence cannot override'.
- B  RELOAD-CN-A + config/release CR (fork CR): ACCETTATO.
- B1 RELOAD-CN-A + release phase resume_b: RIFIUTATO (il repeat e' legato).
- B2 RELOAD-CN-A + config/release C0 phase resume_a: ACCETTATO da build_command (il runtime lo rifiuterebbe: run.py:213 'only CN/CR').
- C  BENCH-CR + config/release C0: ACCETTATO.
=> Rilievo RIPRODOTTO: il dispatcher lega solo l'etichetta del braccio.

## Lettura a valle e ricerca di motivi di confutazione
- Readout: probe sk2r61_scripts/probe_readout_content.py -> config CN nello slot C0: 'ValueError: incomplete/unmatched benchmark C0'; config C0 nello stesso slot supera il controllo (si ferma dopo, su numeric_profile assente). Attribuzione per contenuto confermata (readout:205): nessun esito scientifico attribuito male.
- Nessun controllo altrove: launcher sbatch (solo TimeLimit da scontrol), lib_code_dir.sh, campaign_checkout.sbatch, submit_job.py (ID logico solo in --job-name/--comment, mai confrontato con job['id']), run.py (nessun accesso a cella/ID).
- Nessuna dichiarazione nel prontuario, nella risposta alla review o nei piani che accetti il limite; il prompt per Claude dice che release/job delle fasi successive sono metadati preparati dopo le ricevute (nessuna review per fase prevista dal codice).
- Test del repository: fixture job_env con arm None e config '{}' per full; nessun test di braccio config/release vs cella. test_contrastive_job.py + test_contrastive_release.py: 52 passati (copia invariata).
- Percorso di produzione: make_job -> contrastive_job.build_command -> contrastive_run.py e' proprio quello delle 7 celle GPU full/reload: non e' un percorso inutilizzato.

## Gravita'
- Non BLOCCANTE: serve un errore dell'operatore; il readout rifiuta per contenuto; nessun tetto superato in automatico (failure_policy: nessun retry, serve nuova decisione).
- Non NOTA: difetto di codice reale e riproducibile; il controllo sull'etichetta e' vacuo per l'esecuzione e asimmetrico rispetto al repeat (legato alla phase).
- MINORE confermata. Precisazione: '2 GPU-h consumate' e' il massimo allocato (4 GPU x 1800 s); il consumo reale e' l'elapsed (full storico 661-775 s in bf16 automatico; reload 32 update molto meno). Il resto della catena (CN/CR full, reload) prosegue valido; manca solo la cella sbagliata, da rifare con nuova autorizzazione.
- Contesto (non valutato qui): anche l'ID logico del ledger (job-name) non e' confrontato con job['id'] dal launcher.

## Esito
refuted=false, gravita' MINORE.
