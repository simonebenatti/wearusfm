# Pacchetto eseguibile TCN frozen — 19/11/2026

Protocollo documentale convergente con Claude `ad9f5141`; review del codice e
del piano operativo ancora richiesta. Nessun invio o deploy.

- `plan.json`: derivato dall'entry point pubblico; 24 ID/output distinti,
  dipendenze e risorse; sedici walltime neurali null finché non misurati.
- `tests_runtime.json/.txt`: 204 nuovi test passati, 17 file.
- `tests_historical.json/.txt`: 125 passati e un failure preesistente.
- `baseline_failure.json`: riproduzione isolata sulla base `0e56c8b1`; test,
  programma e piano storico invariati. Non è un difetto introdotto nel lettore.
- `delivery_pin.json`: SHA dei contenuti e hash dei file della consegna.

Il JSON della proposta originale è mantenuto byte-identico per la review:
i suoi campi di stato descrivono la consegna documentale. Lo stato operativo
corrente è quello del nuovo `plan.json`, non un aggiornamento retroattivo del
vecchio snapshot. Nessun benchmark A100, score reale o gate positivo è
dichiarato superato dalle fixture locali.

La formula dei costi non è un preventivo misurato. Il benchmark deve validare
entro i tetti tutte le estensioni, memoria, I/O, quota e job packaging; il piano
misurato e le ricevute vanno pubblicati prima dei fit. Nessuna riduzione della
matrice o nuovo tentativo viene fatta automaticamente.

Istruzioni operative: `docs/pose_temponet_execution_20261119.md`.
Prompt: `docs/prompt_claude_pose_temponet_runtime_review_20261119.md`.
