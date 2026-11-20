# Verifica scettica n. 2 del rilievo R5-1 (r5-test)

Avvio: 2026-11-20 04:2x. Copia di sola lettura a 0efa4d45 (codice fbfde797). Clone di lavoro e script miei in
rev_runtime/tmp/sk2_r51*, nessun uso dei risultati o degli script di r5 (letti solo per sapere quali barriere).

## Lettura del codice e dei test (fbfde797)
- Unico file di test che esercita le barriere vere: tests/cpu/test_contrastive_release.py (11 funzioni, 157 righe).
  test_contrastive_job.py:26-27 e test_contrastive_run.py:72-74 sostituiscono validate_review_binding e
  validate_audit_accounting con lambda; test_contrastive_targets_readout.py:112 e
  test_contrastive_historical_callsites.py:112 sostituiscono validate_audit_accounting. Nessun test chiama make_job
  fuori da test_contrastive_release.py (grep build_command/make_job/bind_review/check-binding).
- Fixture `package`: payload sempre status APPROVED, un solo blocco, piani della fixture (C.PLAN/C.PREFLIGHT).

## Mutazioni mie (script rev_runtime/tmp/sk2_r51_scripts/mut.py, clone rev_runtime/tmp/sk2_r51_clone a 0efa4d45)
Ogni mutazione da sola, sostituzione esatta (una occorrenza), ripristino con git checkout e albero pulito verificato.
### Solo tests/cpu/test_contrastive_release.py (base senza mutazioni: 22 passed)
- SOPRAVVIVONO (22 passed): M00 neutra (commento: cambia il digest); M01 status (96); M02 blocchi >=1 invece di ==1 (92);
  M03 piani di altra coorte (114); M04 hash dichiarati dei piani (117-118, resta il confronto della sorgente);
  M05 legame interno del preflight (121-123); M06 chiamata verify_reviewed_sources (125); M06b confronto byte sorgenti (83-84);
  M06c inventario/digest (86); M07 byte dei piani in git (126-128); M08 copia+payload (132), M08a solo payload, M08b solo sha
  originale; M09 make_job senza validate_review_binding (221); M10 input di make_job contro il piano (230);
  M11 pin dentro il checkout (47); M12 forma in git_bytes (68); M13 chiavi extra (96); M14 formato hash (101);
  M15 review_commit completo (109).
- UCCISA: M02b (controllo del numero di blocchi tolto del tutto) -> test_historical_review_is_insufficient_even_with_valid_file_pin
  (zero blocchi -> IndexError invece di ValueError). Quindi lo zero e' coperto, il caso "due blocchi" no.

### Cinque file che toccano contrastive_release (release, job, run, targets_readout, historical_callsites; base 94 passed)
- Stessi esiti: tutte 94 passed tranne M02b (1 failed, lo stesso test). (M12-M15 in corso)

### Insieme completo della ricevuta (29 file, 465 test)
- Mutante combinato (M01+M13 riga 96, M02, M03, M04, M05, M06, M06b, M06c, M07, M08, M09, M10, M11, M12, M14, M15 insieme):
  **1 failed, 464 passed** (393 s); unico fallito test_review_cannot_relabel_the_current_code_as_an_older_commit, che cade
  solo perche' M06 e M07 sono tolti insieme (provato: M06+M07 da soli -> 1 failed sul file release; M06 o M07 da solo -> 22 passed).
  Quindi la provenienza Git e' coperta in blocco (commit inesistente), non barriera per barriera.
- Otto barriere centrali tolte insieme (M01+M02+M03+M04+M05+M08+M09+M10) -> 22 passed su test_contrastive_release.py.

### Sonde negative mie (rev_runtime/tmp/sk2_r51_scripts/test_sk2_release_probes.py, fixture `package` riusata)
- Codice non mutato: 21 passed -> le barriere funzionano a fbfde797 (status REVIEW_REQUIRED, due blocchi, esempio APPROVED
  prima del blocco vero REVIEW_REQUIRED, piani identici in altra cartella, review di un'altra coorte via bind_review con
  science_plan/preflight_plan, hash preflight dichiarato errato, source interno del preflight errato, reviewed_commit con
  sorgenti diverse e piani uguali, piani diversi al reviewed_commit, payload/sha originale alterati, make_job con binding
  invalido, input fuori piano, copia della review fuori dal checkout, commit abbreviato, chiave extra, hash non esadecimale,
  review_commit abbreviato nel binding).
- Ogni mutante centrale e' ucciso da una sonda di comportamento (DID NOT RAISE): M01, M02, M03, M04, M05, M06, M07, M08,
  M08a, M08b, M09, M10, M11, M13, M15 -> le mutazioni NON sono equivalenti, la lacuna e' reale e chiudibile.
- Equivalenti o quasi (solo il messaggio cambia, il rifiuto resta): M06c riga 86 (nessuna sonda la uccide: ridondante con
  83-84 e 117), M06b righe 83-84 (rifiuto a 86), M12 riga 68 (rifiuto a 109 o da git), M14 riga 101 (rifiuto a 117).
  Per questi controlli di forma "senza test" non e' una lacuna di comportamento.

### Aggiornamento esiti (04:50)
- Cinque file (94 test): TUTTE le mutazioni singole M00-M15 (comprese M12, M13, M14, M15) -> 94 passed, tranne M02b
  (1 failed). Il combinato su questi cinque file -> 1 failed (relabel, per M06+M07 insieme). Albero pulito dopo ogni prova.
- Insieme completo 465, mutazione singola M01 status (riga 96) -> **465 passed** (364 s): riprodotta la prova citata nel rilievo.
- Parsing reale di review_payload (funzione pura, copia di sola lettura): un blocco -> accettato; status REVIEW_REQUIRED ->
  "review is not an approved..."; esempio + blocco vero -> "exactly one ... binding block"; esempio in blockquote + vero ->
  JSONDecodeError (sottoclasse di ValueError, rifiuto); wrapper ````text del prompt e menzione `inline` -> accettati.
  La nota finale del rilievo ("anche un esempio citato nel testo conta") e' esatta ed e' un rifiuto sicuro.
- Ridondanze a valle (riducono l'effetto di M09/M10 in produzione): contrastive_job.build_command:70 richiama
  validate_review_binding e :110 validate_input_bindings (contrastive_prepare.py:99-109 confronta gli input con
  PLAN.input_sha256). Ma quelle chiamate nei test sono sostituite da lambda (test_contrastive_job.py:26-27), e tutte le
  barriere di review_payload/validate_review_binding sono un punto unico usato da bind_review, make_job, build_command e
  validate_release: una regressione li', non vista dai test, passerebbe a tutta la catena.
- Insieme completo 465, mutazione singola M02 (blocchi >=1, riga 92) -> **465 passed** (164 s).
- Pacchetto vero (clone usa e getta, review copia NON committata alla radice, binding costruito a mano, verify_git attivo;
  script rev_runtime/tmp/sk2_r51_scripts/real_package_probe.py): source_digest() == delivery_pin (40d39f56...);
  APPROVED con reviewed_commit fbfde797 o 0efa4d45 -> accettato; REVIEW_REQUIRED -> rifiutato; esempio + blocco vero ->
  "exactly one"; reviewed_commit 438c8d65 -> "execution sources differ from reviewed commit: scripts/contrastive_benchmark_readout.py";
  hash preflight dichiarato errato -> "current sources and both plans"; payload o sha originale del binding alterati ->
  "review copy differs"; review_commit abbreviato -> "requires its full commit". Clone pulito dopo la prova.
  Conferma indipendente che oggi le barriere funzionano: la lacuna riguarda solo i test (protezione dalle regressioni).
- Insieme completo 465, combinato di 13 barriere (M01+M13, M02, M03, M04, M05, M06, M08, M09, M10, M11, M12, M14, M15;
  restano attive solo 126-128 e 83-86) -> **465 passed** (136 s).

## Ricerca di motivi per confutare
- Nessun altro test esercita le barriere vere: job/run/readout/callsites le sostituiscono con lambda; nessun test chiama
  make_job, bind_review o check-binding fuori da test_contrastive_release.py.
- Nessun documento dichiara una copertura alternativa: prontuario righe 121-131 descrive le regole del blocco (un unico
  blocco, status APPROVED, hash), la risposta R-2 afferma i rifiuti senza test che li isolino; il prompt per Claude chiede
  "un unico blocco" e mostra il modello dentro un wrapper ````text (accettato dal parser).
- Attenuanti reali, che non confutano: il codice e' corretto oggi (sonde sul pacchetto vero); una modifica di
  contrastive_release.py cambia source_digest e richiede una nuova review; M09/M10 hanno difese a valle nel dispatcher.
- Precisazioni al rilievo: (1) "un solo blocco": coperto il caso zero (M02b uccisa), scoperto il caso due blocchi;
  (2) righe 68, 101 e 86 sono controlli ridondanti (mutanti equivalenti salvo il messaggio): per loro l'assenza di test non
  e' una lacuna di comportamento; (3) la provenienza Git e' coperta in blocco (commit inesistente) dal test relabel.

## Esito
NON confutato. Riprodotto con strumenti miei: M01 e M02 da sole -> 465 passed; 13 barriere insieme -> 465 passed; ogni
barriera centrale e' un mutante non equivalente (le mie sonde lo uccidono), quindi la lacuna e' reale e il fix proposto e'
fattibile con la fixture esistente. Gravita' MINORE confermata: nessun effetto oggi (barriere funzionanti sul pacchetto vero),
ma una regressione di status/blocco/coorte/legami passerebbe inosservata a tutta la catena (funzione unica).
Pulizia: cloni sk2_r51_clone{,2,3} e basetemp cancellati; restano solo script/risultati (64 KB) in rev_runtime/tmp/sk2_r51_scripts.
Copia di sola lettura pulita a 0efa4d45. Fine: 2026-11-20 04:5x.
