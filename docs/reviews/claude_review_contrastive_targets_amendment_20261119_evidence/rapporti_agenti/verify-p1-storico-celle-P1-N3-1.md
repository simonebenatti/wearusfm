# Verifica scettica P1-N3 (test storici di manomissione vacui a HEAD)

Copia: clone a 438c8d65. Inizio: lettura di tests/cpu/test_contrastive_historical_calibration.py e scripts/contrastive_prepare.py.

- Ordine dei controlli a HEAD: authenticate_calibration (prepare.py:375-391) controlla status/failure_reasons, poi chiama calibration_report(run_dir) PRIMA del confronto ricevuta/ricalcolo; calibration_report (prepare.py:266-274) confronta config.json con la calibrazione 100/125/v3 attesa subito dopo l'hash della baseline, prima di summary, provenance, validate_historical_evidence e metrics.jsonl.

## Riproduzione indipendente (script mio: tmp/skep1_p1n3/scripts/probe.py <checkout> <workdir>)
- HEAD 438c8d65 (digest 643a8937...): ricevuta storica non modificata, weight, selection, reader, source e calibration_report
  con metrics.jsonl +"\n" e senza: TUTTI `ValueError: calibration run config differs from the frozen 512-batch recipe`
  sollevato a contrastive_prepare.py:274 (calibration_report). La ricevuta non e' mai confrontata (prepare.py:383-390 irraggiungibile).
- 8905e346 (lettore congelato, digest 46938f31...): non modificata ACCEPTED (ricevuta e run); weight/selection/reader/source ->
  `historical calibration receipt changed` (contrastive_source_compat.py:186); metrics +"\n" -> `historical calibration run bytes changed` (:194).
  Li' i 5 test discriminano (pytest a 8905e346: 7 passed).
- e388a1de (merge di main prima della consegna): tutti i casi, anche non modificati, `current execution source inventory/pins differ`
  (compat.py:162); pytest: i 2 test positivi FALLISCONO, i 5 di manomissione passano. La vacuita' nasce col merge delle finestre;
  la consegna 0efdc29c ha spostato in run_frozen solo i due positivi e lasciato i 5 negativi sul lettore corrente.

## Mutazioni (clone temporaneo a 438c8d65, tmp/skep1_p1n3/scripts/mutate.py)
- M1: tolti il confronto ricevuta/ricalcolo (prepare.py:390), i due controlli di manomissione dell'helper (compat.py:186, :194)
  e il controllo dell'ordine delle metriche (prepare.py:303). I 5 test + test_unmodified: 6 passed -> vacui confermati.
- M2 (M1 + tolto anche il gate config prepare.py:274): i 5 test passano ancora (rifiuto a compat.py:162, inventario sorgenti),
  fallisce solo test_unmodified... per il suo match. Con evidenze reali la logica di manomissione e' irraggiungibile a HEAD su due livelli.
- Con M1 falliscono invece test_contrastive_source_compat.py::test_any_historical_evidence_mutation_is_rejected
  [config.json, summary.json, metrics.jsonl, receipt, reader_provenance] e test_contrastive_prepare.py::test_receipt_is_recomputed_not_trusted_and_all_arms_share_it
  (6 failed): la logica e' coperta altrove in modo discriminante.

## Motivi di confutazione cercati
- Controllo altrove: SI, parziale. Helper e manifest identici fra 8905e346 e 438c8d65 (`git diff --stat` vuoto su
  scripts/contrastive_source_compat.py e compatibility_20261118/manifest.json): i rami di manomissione del lettore congelato
  sono gli stessi testati sinteticamente a HEAD (test_contrastive_source_compat.py:103-120). Correzione al rilievo: la frase
  «nel lettore congelato sono coperti solo audit e source del summary» vale per i test run_frozen, ma omette questa copertura sintetica.
- Percorso di produzione: il percorso storico a HEAD rifiuta per costruzione; nessun effetto sulla nuova coorte (nessuna evidenza vecchia accettata).
- Dichiarazioni: emendamento e local_validation non attribuiscono a questi test la prova della manomissione; local_validation dice
  solo «old real calibration and audit receipts rejected», vero. La descrizione della consegna («test storici spostati al proprio
  reader congelato») pero' non vale per questi 5.

## Esito
Rilievo NON confutato: i 5 casi sono vacui a HEAD (passano anche con tutta la logica di manomissione rimossa). Gravita' reale NOTA
(chiarezza dei test, nessuna lacuna di copertura: logica coperta da test sintetici discriminanti e verificata a 8905e346 sulle evidenze reali).
Baseline HEAD non mutata: 16 passed (file storico + 6 sintetici + 1 prepare). Clone principale pulito.
- Correzione riferimento: il test sintetico e' test_contrastive_source_compat.py:106-121 (parametrize 106, raises 120-121), non 103-120.
