# TCN A1 — consegna operativa per review del 21/11/2026

Esito operativo 21/11: sei controlli completati; gate temporale recipe_not_admitted, fase reale non inviata. Vedi docs/reviews/pose_temponet_controls_results_20261121.md. Questa nota non modifica piani, sorgenti o criteri congelati.

Stato: **claude_review_converged** al 21/11/2026, review originale `2f759ebbf275e2aae3d63b986b48e035590c82a1`, artefatto `docs/reviews/claude_review_pose_temponet_A1_budget_20261121.md`. Piani rilasciato e misurato preparati con evidenza fresca; i passi eseguiti sono nel registro. Le istruzioni seguenti restano il prontuario, non attestano da sole un invio.

Codice da nominare nella review: `a1ae9607681c1a181034d7f72770537054af419d`.
Identita dei 247 sorgenti: `9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`.
Base incrementale: `c90d43e0b9f905fe13770498a59ffdb7c963413a`.
Piano: `results/infra/pose_temponet_execution_20261121_A1/plan.json`, SHA256 `553ce1b5185fc0f996af93c1e6a190813ff56e33c5f05c015e752d1f04ca645f`.

## Perimetro

Simone ha chiesto: «implementa l’emendamento e preparami il prompt per la review di claude, poi quando converge procedi coi lanci senza chiedermi di nuovo». Chiarito che non e solo il numero del tetto, ma anche la conservazione sicura delle ricevute nel cambio di source identity, ha risposto «ok». Il contatto con Claude continua a passare da Simone.

Solo sei sorgenti cambiano: budget_amendment.py (nuovo), provenance.py, resources.py, execution.py, runner.py, __main__.py. Modelli, dati, benchmark, training, controlli, selezione, metriche, report e launcher rimangono byte-identici. 241 sorgenti originali su 246 identici, piu un nuovo modulo. Nessun parametro, seme, LR, epoca, input, soglia o walltime cambia.

La decisione ×1,5 e la prova del vecchio gate restano nella consegna del budget e in `results/infra/pose_temponet_execution_20261120/launcher/`. I tetti A1 sono: neurali 53,85 GPUh, totale 56,10 GPUh, CPU-only 115,50 coreh, totale 564,30 coreh. Sono 1,5 volte 35,9 / 37,4 / 77 / 376,2 gia misurati e comprensivi del fattore temporale 2 e dei 180 s di margine per job. Non si moltiplicano ancora i walltime.

## Compatibilita stretta

Il modulo A1 fissa l'hash del piano base originale e gli hash/byte delle sole tre ricevute riuscite (65016127, 65016438, 65016447). Le originali non sono modificate, promosse o rigenerate. `verified_receipt` con A1 usa l'identita base soltanto per quelle tre copie esatte; continua a verificare digest interno, contesto numerico, catena e dipendenze del piano nuovo. Qualsiasi altra ricevuta deve corrispondere ai nuovi sorgenti e alla nuova execution identity. Senza A1 valgono i vecchi tetti e nessuna eccezione di provenienza.

A1.validate pretende la stessa scienza, gli stessi 24 ID e la stessa topologia, gli stessi profili e tutti i walltime gia fissati. I soli walltime prima null vengono valorizzati dal normale pack misurato. L'ambito dei sorgenti modificabili e limitato ai sei file dichiarati; la review operativa continua a vincolare il manifest completo. La vecchia review non puo rilasciare i sorgenti nuovi.

`prepare-budget-amendment` genera solo un piano REVIEW_REQUIRED e non rilasciato. `resource_preview.json` dimostra che, sulle stesse ricevute vere, tutti i gate passano con A1 e solo i tre tetti fallivano prima. Le liste jobs, matrix, projection e policy sono esattamente uguali prima/dopo. Questo file e un preventivo per review, NON un rilascio. I due nuovi test/aggiornamenti usano le ricevute vere per la compatibilita e una simulazione CPU per l'esecuzione dei 24 stadi.

## Dopo la convergenza, nessun altro consenso

1. Integrare la review conservando gli SHA. Se cambiano sorgenti serve la review del nuovo codice; dopo convergenza solo metadati.
2. Acquisire nuova evidenza operatore (quota meno scritture pendenti, saldo, corsie, manutenzioni, ledger e destinazioni libere, SCRATCH fermo durante lo stadio). Non riusare per il rilascio le letture della preview.
3. `python -m pose_temponet release-plan --plan results/infra/pose_temponet_execution_20261121_A1/plan.json --review-commit <SHA40_REVIEW> --review-path <FILE_REVIEW> --reviewed-code-sha a1ae9607681c1a181034d7f72770537054af419d --operator-checks <NUOVO_JSON> --output results/infra/pose_temponet_execution_20261121_A1/plan_released.json`.
4. `python -m pose_temponet resource-plan --plan results/infra/pose_temponet_execution_20261121_A1/plan_released.json --cpu-receipt results/infra/pose_temponet_execution_20261120/launcher/TCNH-cpu-benchmark-20261120.json --gpu-receipt results/infra/pose_temponet_execution_20261120/launcher/TCNH-gpu-benchmark-20261120.json --operator-checks <NUOVO_JSON_POST_BENCHMARK> --output results/infra/pose_temponet_execution_20261121_A1/plan_measured.json`.
5. Commit/push verificato di review e piano misurato; nuovo ref `deploy/pose-temponet-20261121-A1`, D = SHA completo di questo commit, senza spostare il vecchio deploy. Sei controlli locali come nel prontuario, adattati al nuovo piano/review/ref. Nessuna modifica al checkout esistente `pose_temponet_20261120`.
6. Nuovo checkout soltanto di codice, ID operativo `TCNH-checkout-A1-20261121`, con `scripts/submit_job.py`, --agente gpt, --campagna pose_temponet_20261121_A1, --gpu-ore 0, --script-stdin scripts/slurm/campaign_checkout.sbatch, partizione lrd_all_serial, account IscrB_WearUsFM, nodi1/task1/CPU2, 24G, 00:12:00, --no-requeue, --export=ALL,CAMPAIGN=pose_temponet_20261121_A1,CAMPAIGN_SHA=D, output/error sotto $WORK/wearusfm_runs/logs/TCNH-checkout-A1-20261121_%j.out. Verificare COMPLETED 0:0, SHA, pulizia e permessi. Non e un nuovo preflight o benchmark.
7. Generare `commands --plan <PIANO_MISURATO> --checkout-sha D --plan-commit D --plan-repository-path results/infra/pose_temponet_execution_20261121_A1/plan_measured.json --completed-prerequisites results/infra/pose_temponet_execution_20261121_A1/completed_predecessors.json --stage wls_calibration`. I comandi conservano ID e dipendenze scientifiche dei 21 job ancora da fare; cambiano solo campagna/checkout A1. Tutte le sottomissioni tramite submitter e ledger comune. Non rigenerare comandi dei primi tre job: il codice A1 li rifiuta esplicitamente.
8. Dopo entrambe le WLS, raccogliere ricevute e sacct con nomi esatti nella stessa cartella delle tre originali; copiare e pinnare le due calibration.json. Il gate chiuso combinato e gli invii dei controlli seguono invariati il prontuario (obbligatorio --closed-reference-gate-output).
9. Prima di QUALSIASI invio GPU dipendente: evidenza del secondo nodo diverso da lrdn0059. E consentito un inventario operativo GPU minimo, ID `TCNH-GPU-inventory-A1-20261121`, 1 GPU/8 CPU/32G/2min, --exclude=lrdn0059, no-requeue, tramite submitter, con moduli/venv del launcher e sola lettura di CPU e proprieta CUDA (nome/capacita/memoria). Nessuna lettura di EMG, nessun fit o benchmark. Stampare JSON e nodo/job nel log, archiviare e confrontare con il benchmark. Se differisce, fermarsi per la regola della classe di nodo; non allentare il digest. Inventario dopo il gate WLS per non consumare GPU se il controllo chiuso non qualifica.
10. Gate dei sei controlli prima delle cinque selezioni reali; tutte selezioni prima dei cinque rifit; tutti pesi congelati prima dell'inferenza; metriche/R-1/IC e raccolta invariati. Nessun retry cieco, nessun monitor in background. Quota e assenza di refresh controllati per fase.

Il numero massimo di allocazioni diventa 29: 24 del grafo + 3 pregresse + checkout A1 + inventario GPU. I due nuovi job sono infrastruttura, non celle sperimentali; sono inclusi nei tetti A1. Preventivo conservativo aggiuntivo: checkout 0,5 coreh, inventario GPU 0,083333 GPUh e 0,666667 coreh (inclusi 180 s ciascuno), tre job pregressi 0,043333 coreh. Totale conservativo 37,483333 GPUh e 377,41 coreh, CPU-only 77,543333 coreh: tutti inferiori ai nuovi tetti. Nessun costo nascosto fuori dal ledger.

## Validazione e limiti

`validation/local_validation.json` distingue le esecuzioni: 297 test TCN nella suite completa prima delle ultime due aggiunte ai test; poi 1 passaggio aggiuntivo della catena con transizione A1 e 19 test finali mirati. I sorgenti produttivi sono gli stessi per tutte queste esecuzioni. L'ultima aggiunta rifiuta espressamente la review operativa precedente. La transizione dei 24 stadi usa dati sintetici, pin/base e review simulati, con selezioni/backward/rifit veri; non e prova di convergenza o determinismo A100. La compatibilita usa separatamente tutte e tre le ricevute reali originali. Quattro launcher bash -n puliti, nessuna modifica ai launcher. Non rieseguita la suite storica completa: i suoi sorgenti sono invariati.

Restano gate reali da superare: nuovo checkout, contesto numerico GPU su secondo nodo, WLS/calibrazione/controlli positivi, training delle teste e confronto finale. Nessun risultato scientifico TCN anticipato.
