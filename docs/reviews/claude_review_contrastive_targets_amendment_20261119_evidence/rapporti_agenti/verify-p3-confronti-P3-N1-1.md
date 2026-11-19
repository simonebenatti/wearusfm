# Scettico n.1 - rilievo P3-N1 (NOTA)

Inizio: 2026-11-19 10:21:23; copia clone @ 438c8d65

## Passo 1: lettori del piano (grep)
- grep -rn -e target_cli_args -e target_recipe_amendment -e new_recipe -e old_receipts_reusable -e required_new_evidence scripts src tests: nessuna occorrenza (rc=1).
- PLAN (contrastive_prepare.py:30) letto solo in policy_reference (r.76: coverage.manifest_reference, calibration.historical_reference), validate_input_bindings (r.107: input_sha256), _reader_provenance (r.219: pin del file).
- Le occorrenze di 'arms' in benchmark_readout/cuda_check sono dizionari dei report, non del piano.
- Nessun test legge results/infra/contrastive_targets_20261118/{preflight_plan,local_validation}.json o review_configs/.

## Passo 2: coerenza odierna (script mio ../tmp/skep1_p3n1_coherence.py)
- build_review_configs(baseline): C0/CN/CR/calibration = (100.0, 125.0, step1_psd_power_shape_energy_fast125_v3).
- plan.common, plan.arms.{C0,CN,CR} (rms/fast e target_cli_args), target_recipe_amendment.new_recipe: tutti uguali ai config generati (True). arms non portano target_version (il rilievo non lo afferma).
- historical_baseline dichiarata 25/50/v1 == baseline ripristinata con with_legacy_target_windows.
- review_configs/{C0,CN,CR,calibration}.config.json byte-identici all'uscita del builder (stessa serializzazione di `prepare`).
- sha256(plan.json) = 6081832627cee150... == preflight_plan.science_plan.sha256; nessun codice legge preflight_plan.json.
- Sorgenti identici fra 0efdc29c e 438c8d65 (git diff --quiet): le righe 133-135 citate valgono per la consegna.

## Passo 3: mutazioni in clone temporaneo (../tmp/skep1_p3n1_mut @ 438c8d65)
- Base: 10 file contrastivi, 200 passed (19,4 s).
- Mutazione A, SOLO piano: common/arms/target_cli_args/new_recipe portati a 150/200/'bogus_v9' (20 righe). Builder resta 100/125/v3.
  Esito: 200 passed sugli stessi 10 file + 55 passed sugli altri 6 file contrastivi (callsite_oracle, common_checkpoint, common_state_cuda, mmap_compatibility, rank_exit, runtime). Nessun test ferma la divergenza piano->config. CONFERMATO.
- Mutazione B, SOLO builder: riga 133 rms 100.->150. Esito: 1 failed (test_contrastive_prepare.py::test_review_configs_share_recipe_and_training_weights_are_not_executable, righe 116-120 con letterali 100/125/v3), 111 passed.
  Precisazione: una deriva nel builder e' fermata dal test letterale; resta scoperta la deriva originata nel piano o un cambio coordinato builder+test che dimentichi il piano.

## Passo 4: motivi cercati per confutare
- Emendamento r.20-22: "La rappresentazione CLI esplicita e' ..." e "Il dispatcher contrastivo usa i config JSON pinnati, non i default di una CLI implicita": i campi del piano sono documentali per scelta dichiarata. Conferma la natura del rilievo, non lo smentisce (manca solo il legame automatico).
- Convenzione preesistente: tutte le chiavi di plan.common storiche (lr, batch_global, pesi, schedule, ...) sono documentali; il builder codifica i valori attesi a r.118-130. Le chiavi nuove (ms_fast_window_ms, rms_window_ms, target_cli_args, target_version) seguono lo stesso schema: non e' una regressione introdotta dalla consegna.
- Pin del piano: _reader_provenance (r.219) lo include nel report di calibrazione; authenticate_calibration confronta la ricevuta col ricalcolo -> una modifica successiva del piano cambia identita' e verrebbe rifiutata, ma come cambio, non come incoerenza.
- local_validation.json registra controlli puntuali "C0cli_explicit", "...storedconfigmatchesgenerator", "...explicit100/125/v3"; lo script che li ha prodotti non e' nel repository (git grep: i nomi compaiono solo in local_validation.json). Verifica una tantum, non regressione.
- Il percorso d'esecuzione non dipende dal piano: contrastive_job.build_command usa config pinnati (checked_file) e per audit/cuda validate_recipe contro build_review_configs; calibration_report confronta con build_review_configs; nessun rischio di ricetta sbagliata.

## Esito
NON CONFUTATO. Rilievo riprodotto; gravita' NOTA corretta (nessun effetto sull'esecuzione; legame piano<->config solo per revisione). Precisazione da riportare: il builder e' gia' legato ai letterali 100/125/v3 da test_contrastive_prepare.py:116-120; la lacuna riguarda il solo lato piano.
- Pulizia: rimossi basetemp skep1_bt_* e clone skep1_p3n1_mut; resta solo lo script skep1_p3n1_coherence.py. Copia di review pulita (git status vuoto).
