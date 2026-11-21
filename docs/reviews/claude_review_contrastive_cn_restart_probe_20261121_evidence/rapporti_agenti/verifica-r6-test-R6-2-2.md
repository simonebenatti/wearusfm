# Verifica avversaria di R6-2 (compare() non coperto dalla suite)

Esito: il rilievo regge nel merito. La gravita' scende da "importante" a "minore".

## Cosa ho eseguito
- Ho letto `tests/cpu/test_cn_restart_compare.py`. Gli unici test che arrivano a `compare()` passano da `main()` con cartelle inesistenti (righe 72-80) e si fermano subito nell'eccezione, che da' INCONCLUSIVE. Gli altri test chiamano solo funzioni di supporto: `classify`, `compare_vector`, `checkpoint_differences`, `layouts`, `metrics_evidence`, `branch_traces`, `check_boundary` e `post_update_hashes`. Nessuno esegue il corpo delle righe 214-304.
- Ho applicato la mutazione C9 su una copia nella cartella di lavoro: alla riga 287, `any_local |= bool(...)` diventa `any_local |= False`. Risultato: `13 passed in 1.26s` su `test_cn_restart_compare.py`. La mutazione sopravvive, come diceva il revisore. Dopo la prova ho cancellato la copia.
- Le costanti nel codice coincidono con il piano: `LIMIT = 1e-6` e `DENOM_FLOOR = 1e-12` (compare.py:24-25), contro `relative_gradient_roundoff_limit: 1e-06` e `gradient_norm_floor: 1e-12` (plan.json:41-42). Un test non le lega, ma oggi sono uguali.

## Perche' non e' importante e non e' bloccante
1. Manca un controesempio sul codice candidato. L'end-to-end del revisore (`agents/r6-test/e2e/e2e_compare.py`) usa artefatti veri di CN_A e CN e l'inventario del modello vero. Da' gli esiti attesi per PARTIAL_SUPPORT, SUPPORTED e HYPOTHESIS_UNSUPPORTED. Non ho trovato nessun percorso per cui il codice attuale attribuisca male l'esito.
2. Per questo run, le "regressioni future" non sono un rischio. Il codice e' congelato nel pacchetto: il validatore `--package` ne fissa le impronte e il checkout e' congelato. Mutazioni come C1 o C9 non possono arrivare nel run vero senza una nuova release, e quindi senza una nuova review.
3. Un errore di collegamento in compare non costa altre GPU. `compare` e' un passo di post-elaborazione CPU, eseguito dopo le due celle sugli output che restano in `root` (contrastive_cn_restart.py:133-136, 170-171). Se compare uscisse INCONCLUSIVE per un errore di collegamento, osservazioni e checkpoint resterebbero su disco. Servirebbe una correzione rivista e una nuova esecuzione CPU, non un secondo tentativo GPU. Questo l'ho dedotto dal codice e non l'ho eseguito.
4. B-1 si trova in `load_observation` (observe.py). Si manifesta solo col torch 2.2 del cluster: un test end-to-end sul Mac (torch 2.14) non lo intercetterebbe, a meno di simulare il comportamento di 2.2 con uno shim. L'end-to-end proposto non avrebbe quindi preso il difetto bloccante gia' trovato.

## Correzione proposta (minima, non necessaria per la release)
- Promuovere l'end-to-end del revisore a test parametrizzato in `test_cn_restart_compare.py`, con gli esiti SUPPORTED, PARTIAL, UNSUPPORTED locale, UNSUPPORTED per eccesso (appena sopra e appena sotto 1e-6) e INCONCLUSIVE.
- Aggiungere `assert compare.LIMIT == plan["relative_gradient_roundoff_limit"]` e `assert compare.DENOM_FLOOR == plan["gradient_norm_floor"]`.
- Utile, ma non condizione di release: si puo' includere nel giro di correzione di B-1, senza un giro di review a parte.
