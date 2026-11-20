# Scettico n. 1 su R5-5 (r5-test): accounting e contesto numerico, controlli non isolati

Avvio: 2026-11-20 04:50. Copia 0efa4d45 (contenuti fbfde797). Rilievo MINORE (+ NOTA su 222-223).
Script miei in rev_runtime/tmp/sk1_r55_scripts/, clone usa e getta rev_runtime/tmp/sk1_r55_clone.

## Lettura del codice (fbfde797)
- release.py:155-192 validate_audit_accounting; righe citate coincidono: 164-165 (schema, audit_sha, source
  dell'accounting e source interno dell'audit), 172 (job_id str numerico), 176-177 (colonne obbligatorie).
- Chiamanti in produzione: release.py:242 (make_job), :281 (CLI validate-audit), contrastive_job.py:81,
  contrastive_run.py:195, readout.py:421 (expected_source=coorte). Nessun chiamante passa expected_job_id.
- readout.py: numeric_context_across_runs (345-389, assert a 374) e' chiamato a 451 PRIMA di _run_evidence
  (465, 483; assert a 220, coerenza fra rank 222-223); se lo stato non e' PASSED read_preflight ritorna a 462.
- _cuda_evidence (285-323, assert a 301) chiamato da readout.py:422 e da release.validate_cuda_prerequisite (202).

## Riproduzione (mut.py mio; sostituzione esatta, 1 occorrenza verificata, ripristino con git checkout, albero pulito dopo ogni prova)
- Nota di harness: un primo giro con percorso del clone relativo caricava `wearusfm` dall'installazione editabile
  (ModuleNotFoundError contrastive_objectives) -> scartato (targeted_INVALID_relative_pythonpath.jsonl); rifatto con percorso assoluto.
- Test mirati (file che osservano il modulo: 5 per release.py = 94 test; 7 per readout.py = 136 test):
  controlli positivi UCCISI: K1 State COMPLETED (release.py:187) -> 1 failed [change2]; K2 differenze del contesto
  (readout.py:383) -> 1 failed test_numerical_context_must_match...; K3 criterio fisso CUDA (readout.py:321) -> 1 failed
  test_cuda_receipt_binds.... Mutazioni del rilievo, tutte SOPRAVVISSUTE: C4d (165, source interno dell'audit) 94 passed;
  C4g (177, colonne) 94 passed; C4h (172, job_id numerico) 94 passed; C5c (374) 136 passed; C5f (222) 136 passed;
  C5m (301) 136 passed. Extra mio X220 (assert del profilo in _run_evidence, 220) 136 passed.

## Sonde di comportamento (tmp/sk1_r55_scripts/test_probe_r55.py; mutanti compilati in memoria, clone sk1_r55_probe mai modificato; 8 passed, log probe.log)
- A0 caso valido -> ACCEPTED (job 123, 7000 s).
- A1 C4g (177): TSV senza TimelimitRaw / State / JobIDRaw: originale ValueError "unambiguous allocation fields";
  mutante KeyError 'TimelimitRaw'/'State'/'JobIDRaw'. Nessun chiamante cattura ValueError (grep: nessun except in
  job/run/release/readout) -> il mutante chiude comunque: mutante EQUIVALENTE quanto ad accettazione (cambia solo il messaggio).
- A2 C4h (172): job_id "123.batch" con TSV in formato fixture (riga .batch con TimelimitRaw 120): originale "job ID missing
  or changed", mutante ACCETTA la riga di step. Con TSV in formato Leonardo reale (sacct.txt del job 63495005 in
  results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/initial_gate: righe .batch/.extern con TimelimitRaw vuoto)
  il mutante chiude ("elapsed/allocation is not integral"); job_id int o None -> "one exact allocation row". Nessun
  chiamante di produzione passa expected_job_id. Difesa in profondita': con sacct genuino il mutante e' innocuo.
- A3 C4d (165): audit con source diversa + accounting con source corrente e sha dell'audit: originale rifiuta, mutante
  ACCETTA in validate_audit_accounting. Barriera storica `validate_historical_evidence('audit', ...)` con i sorgenti
  attuali -> "current execution source inventory/pins differ": run.validate_audit (run.py:97-99, chiamata a 191 prima
  di 195) e readout 413-415 (prima di 421) rifiutano comunque. Non ridondante solo nel rilascio (make_job release.py:242,
  build_command job.py:81): per la cella cuda (contrastive_cuda_check non rilegge l'audit) un errore dell'operatore
  passerebbe il rilascio e sarebbe fermato solo dal readout finale (spreco di una cella, nessuna evidenza accettata).
- B1 C5c (374): 28 ricevute coerenti con TF32 matmul acceso, fixture del readout: originale INCONCLUSIVE_REQUIRES_REVIEW
  (28 "required profile unavailable", 0 differenze, nessun braccio letto); mutante PASSED_FOR_REVIEW. Confermato.
- B2: _run_evidence vero (220) sulle stesse ricevute -> NumericProfileUnavailable (eccezione, non INCONCLUSIVE): confermata
  la seconda barriera del flusso reale. Mutante X220 (220 rimosso) -> accetta.
- B3 222-223: ricevute costruite ad arte (rank0 cudnn_sdp_enabled=None, rank1-3 chiave assente): contesto PASSED (confronto
  per campo con .get), _run_evidence vero -> "production numeric flags differ across ranks". Quindi 222-223 NON e'
  irraggiungibile alla lettera; lo e' per ricevute del produttore (numeric_flags() emette sempre le stesse 19 chiavi).
- B4 C5m (301): rapporto CUDA con TF32 acceso sui 4 rank: originale NumericProfileUnavailable, mutante accetta.

## Intero insieme della ricevuta (29 file, 465 test), mutazioni singole
- C5m -> 465 passed (121,80 s); C4d -> 465 passed (121,92 s); C5c -> 465 passed (121,95 s). (in corso: C4g, C4h, C5f, X220, C5c+X220)
- Completato (tmp/sk1_r55_scripts/full.jsonl, albero pulito dopo ogni prova): C4g 465 passed; C4h 465 passed; C5f 465 passed;
  X220 465 passed; **C5c+X220 insieme (entrambe le barriere del profilo del readout tolte) 465 passed**.
- Esaustivita': tutti i file di test che citano release/readout/_run_evidence/_cuda_evidence/numeric_context (10) sono
  nei 29 della ricevuta; nessun altro test li osserva.

## Motivi cercati per confutare
- Controllo altrove: solo parziale. C4d e' coperto a runtime dalla barriera storica (run.py:97-99/191) e nel readout
  (413-415), non nel rilascio della cella cuda. C5c e' coperto da 220, ma 220 stesso non ha test (X220 e C5c+X220
  sopravvivono a 465/465). C5m: il produttore (cuda_check.py:281, 302) impone il profilo, anch'esso senza test (C7e/C7f di r5).
- Percorso non usato in produzione: no; readout, make_job, dispatcher e runtime sono la catena del prontuario (righe 140-146).
- Dichiarazione esplicita: no. Prontuario 23 e 40 e risposta N-3 dichiarano che il profilo richiesto mancante/non
  disponibile rende inconcluso il confronto; R-4 elenca test negativi (TIMEOUT, exit, durata, pin, ID, duplicati) senza
  source interno, colonne o formato del job_id. Nessuna accettazione della lacuna.
- Precedente della scala comune: M-1 della review 94f5094 ("tre barriere di secondo livello senza test") = MINORE.

## Imprecisioni del rilievo (non cambiano l'esito)
- 222-223 "irraggiungibile" e' falso alla lettera (B3: raggiungibile con chiave None in un rank e assente negli altri);
  e' ridondante per le ricevute del produttore. Resta NOTA.
- C4g e' un mutante equivalente quanto ad accettazione (KeyError chiude comunque); C4h chiude con il formato sacct reale
  di Leonardo (step con TimelimitRaw vuoto) e accetta una riga di step solo con TSV sintetico come quello della fixture.
  Queste due voci valgono come NOTA; il peso MINORE viene da C4d, C5c(+220) e C5m.
- L'affermazione su 220 ("lo ferma con un'eccezione invece di INCONCLUSIVE") e' corretta (B2: NumericProfileUnavailable,
  RuntimeError non catturata); ma anche 220 e' senza test, il che rafforza il rilievo.

## Esito
- refuted = false. Gravita' reale MINORE (composito): il codice attuale contiene ed esegue tutte le guardie e rifiuta in
  ogni sonda; nessuna evidenza non valida e' accettata oggi, quindi non BLOCCANTE. Manca la protezione dalle regressioni:
  una regressione singola su 165, 374 o 301, o doppia su 374+220, passerebbe la ricevuta da 465 test.

Pulizia: cloni sk1_r55_clone{,2,3}, sk1_r55_probe e basetemp sk1r55_bt_* cancellati; restano in tmp/sk1_r55_scripts solo
script e risultati (52 KB: mut.py, test_probe_r55.py, targeted.jsonl, full.jsonl, probe.log). Copia di sola lettura
pulita a 0efa4d45. Fine: 2026-11-20 05:06.
