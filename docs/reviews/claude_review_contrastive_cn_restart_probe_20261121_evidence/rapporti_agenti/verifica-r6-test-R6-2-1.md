# Verifica avversaria di R6-2 (compare() non esercitato dalla suite)

Esito: il rilievo **regge** come lacuna di test. La gravita' resta **importante**, non bloccante.

## Eseguito
- Ho tracciato le righe di `scripts/diagnostics/contrastive_cn_restart_compare.py` eseguite dalle quattro suite `test_cn_restart_{package,leg,observe,compare}.py`. Ho usato un plugin `sys.settrace` messo nella mia cartella (`agents/verifica-r6-test-R6-2-1/plug/linetrace_plugin.py`); il clone non e' stato toccato.
- Il risultato e' in `agents/verifica-r6-test-R6-2-1/hits.json`. Le righe eseguite di `compare()` sono 215, 216, 218, 219, 220 e 221. Di `history()` sono eseguite tutte le righe da 128 a 135. `history()` esce alla riga 135 per lo SHA del checkpoint assente, perche' il test passa cartelle inesistenti. **Le righe 222-304 non vengono mai eseguite.** Restano quindi senza alcuna esecuzione l'inventario dei parametri, la fedelta' dell'ordinaria, il ciclo per rank/tensore, le soglie, il piazzamento e la chiamata finale a `classify`. Ogni mutazione in quel tratto sopravvive per costruzione. Questo conferma, con una prova piu' forte, le 17 mutazioni sopravvissute del revisore.
- `LIMIT` e `DENOM_FLOOR` (righe 24-25) non sono collegati al piano da nessun test. `test_cn_restart_package.py:67-68` controlla solo i valori nel `plan.json`. Oggi i valori coincidono (1e-6 e 1e-12).

## Dedotto dal codice
- Non c'e' un controesempio che renda bloccante il codice attuale. L'e2e del revisore, con artefatti veri di CN_A e CN, porta al `classify` atteso in tre scenari. Il commit e' congelato, quindi le regressioni future ipotizzate non toccano questo run.
- Il rischio concreto riguarda il raccordo con i file prodotti sulla GPU:
  - L'e2e sintetizza da se' i record `boundary.rank*.json` e `post_update.rank*.json`.
  - Sostituisce `load_observation`, `history`, `checkpoint` e `file_sha`.
  - Per questo non verifica che le chiavi scritte da observe/leg coincidano con quelle lette da compare.
  - B-1 (`mmap=True` con `Path`) e' proprio un difetto di questo raccordo, e nessun test l'ha intercettato.
- Nessun altro controllo del diagnostico sostituisce i test. Le `require` trasformano gli errori di raccordo in INCONCLUSIVE, ma solo dopo aver consumato l'unico tentativo GPU.

## Gravita'
Importante. La probabilita' e' incerta e la lacuna e' mitigabile con un test. Non e' bloccante, perche' manca un controesempio sul codice attuale.

## Correzione minima
1. Aggiungere un test end-to-end di `compare()` sul modello dell'e2e del revisore. I record boundary/post_update vanno prodotti dalle funzioni di scrittura vere di leg/observe, e non scritti a mano. Il test va parametrizzato su SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED per differenza locale, HYPOTHESIS_UNSUPPORTED per eccesso (scarti appena sopra e appena sotto 1e-6) e INCONCLUSIVE.
2. Aggiungere `assert compare.LIMIT == plan["relative_gradient_roundoff_limit"]` e `assert compare.DENOM_FLOOR == plan["gradient_norm_floor"]`.

## Verifiche positive
- `classify`, `compare_vector`, `checkpoint_differences`, `layouts`, `check_boundary`, `post_update_hashes`, `branch_traces` e `metrics_evidence` sono eseguite dalla suite.
- `main()` rende INCONCLUSIVE con `training_approved=False` e `FAILED_REQUIRES_REVIEW`.
