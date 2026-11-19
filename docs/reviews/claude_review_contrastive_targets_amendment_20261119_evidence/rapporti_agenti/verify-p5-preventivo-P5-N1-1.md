# Verifica scettica n. 1 del rilievo P5-N1 (audit senza scadenza interna ne' gate meccanico su TIMEOUT)

Copia: clone @ 438c8d65 (sola lettura). Rapporto incrementale.

## Riproduzioni mie
- `grep -n add_argument scripts/contrastive_audit.py`: righe 152-158, solo --config/--manifest/--root/--scales/--window-seconds/--steps/--out. Nessuna opzione di tempo. Riprodotto.
- `git grep -n -E 'TIMEOUT|ElapsedRaw|elapsed_seconds' -- 'scripts/contrastive_*.py'`: nessun risultato (rc=1). Anche `sacct|COMPLETED|Elapsed` su scripts/contrastive_*.py e scripts/slurm/contrastive_*: nessun risultato. Riprodotto.
- contrastive_job.py:62-68: allocated_seconds usato solo nel confronto con ALLOCATED_SECONDS (riga 67, unica modifica della consegna insieme alla riga 17); riga 104: comando audit con solo `--steps 1024`. Le fasi GPU hanno invece tempi interni fissi (1770/1800, righe 118, 136, 138).
- contrastive_audit.py:176-191: rapporto scritto una sola volta con open("x") dopo il replay completo e il ricontrollo dei pin; un solo print finale (189). Nessun gestore di SIGTERM in src/ o nel percorso dell'audit (`git grep 'signal.signal|SIGTERM'`: solo rank_exit e split_benchmark). Dopo SIGTERM il processo muore: nessun rapporto.
- contrastive_run.py:91-109 validate_audit(path, inputs, calibration_config): firma senza stato o durata; controlla sorgente, input, passi/geometria, config_sha256, pannello e copertura. validate_release (161-210) e read_preflight del lettore (contrastive_benchmark_readout.py:320-334) idem: solo contenuto. Il campo `audit_recovery` della vecchia release non e' letto da alcun codice (git grep: nulla).
- Storico 64652559 (accounting_and_integrity.txt): job TIMEOUT, .batch COMPLETED 0:0, 3787 s su 3600, rapporto scritto alle 09:10:55 (1 s prima della fine); OverTimeLimit 3 min, KillWait 180 s. Dimostra che un job TIMEOUT puo' lasciare un rapporto completo che il cancello di contenuto accetterebbe. slurm.out: solo la riga del comando e quella finale, nessun avanzamento.
- Vecchio audit contro la ricetta nuova (script tmp/p5n1_skeptic/old_audit_vs_new.py): config_sha256 vecchio ab50bc11... contro atteso nuovo 7471e608...; sorgente ab2d6460... contro 643a8937.... Il riuso del VECCHIO rapporto e' quindi rifiutato meccanicamente; manca solo il controllo dello stato SLURM del NUOVO audit.

## Motivi cercati per confutare
1. "Tutto o niente" e' fail-closed. Oltre l'uccisione (circa 7.380 s piu' il ritardo del controllo periodico) non resta nessun rapporto, quindi non c'e' un audit_sha256 da pinnare e non si puo' costruire la release di calibrazione (contrastive_job.py:119-125 e 69; contrastive_run.py:181-184). Un rapporto parziale sarebbe comunque rifiutato (steps==1024). L'unico varco e' la finestra fra 7.200 s e l'uccisione (almeno 187 s, come nello storico), dove il rapporto e' completo e ha lo stesso contenuto di un run COMPLETED, perche' il replay e' deterministico. L'effetto riguarda solo la politica e il consuntivo (circa 0,4-0,5 core-ore), non la ricetta, il riuso o l'attribuzione.
2. Il gate procedurale e' una scelta documentata e gia' applicata. contrastive_audit_results_20261117.md:64-66: «Il runtime verifica il contenuto del rapporto, ma non lo stato SLURM». contrastive_calibration_release_20261117.md:21-22: «requisito di job principale COMPLETED 0:0», con deroga che «Non vale per altri TIMEOUT». Nello storico verification.json ha execution_gate FAILED_TIME_LIMIT e calibration_released false, fino all'autorizzazione di Simone. Nessun build_release.py storico legge sacct.
3. La non eredita' riferita al vecchio rapporto e' meccanica (config e sorgente diversi, vedi sopra). Resta procedurale solo il trattamento di un TIMEOUT nuovo.

## Perche' non lo confuto
- Il pacchetto nuovo non riscrive il requisito COMPLETED 0:0 per l'audit nuovo. Ci sono solo emendamento:35 («nessuna eredita' dell'eccezione TIMEOUT») e preflight_plan failure_policy. `git grep` su results/infra/contrastive_targets_20261118 e sull'emendamento non trova COMPLETED, sacct o elapsed.
- «controlla il tempo effettivo» (emendamento:75) e' implementato solo come TimeLimit letto da scontrol (launcher CPU:20-22, dispatcher:67), non come durata. Si puo' leggere come un controllo della durata che non esiste.
- La mancanza di righe di avanzamento era gia' lamentata il 17/11 (audit_results:48-49) e non e' stata corretta. E' un limite reale di diagnosi se l'audit va in TIMEOUT.

## Esito
Fatti riprodotti, rilievo NON confutato. Gravita' NOTA confermata, senza BLOCCANTE o MINORE: architettura invariata e documentata, rischio basso, correzione documentale nella release. Basta il requisito State=COMPLETED, ExitCode 0:0 del job principale, che implica la durata entro il TimeLimit. Le righe di avanzamento restano facoltative.
Precisazioni per il revisore: la soglia di 7.380 s e' approssimata (lo storico e' finito 7 s oltre limite + OverTimeLimit senza essere ucciso). Fra 7.200 s e l'uccisione nasce un TIMEOUT con rapporto valido: e' il caso che conta.
Non verificato: stima della durata del nuovo audit (3.900-4.000 s, massimo 6.032 s), configurazione SLURM attuale (OverTimeLimit e KillWait letti il 17/11).
