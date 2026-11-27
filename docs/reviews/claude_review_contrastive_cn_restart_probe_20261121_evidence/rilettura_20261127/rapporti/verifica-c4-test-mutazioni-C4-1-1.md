# Verifica avversaria del rilievo C4-1 (commit 8c72655c)

Esito: il rilievo **regge come lacuna dei test**. La gravita' scende da importante a **minore**: il codice congelato e' corretto e la lacuna non puo' cambiare l'esito del run vero.

## Eseguito
Ho lavorato su una copia `git archive` di 8c72655c in `agents/verifica-c4-test-mutazioni-C4-1-1/wt`. Il clone non e' stato toccato e alla fine `cmp` conferma che coincide con la copia ripristinata.
- Base: i quattro file compare/observe/end_to_end/leg danno 61 test passati. Nella copia senza `.git` falliscono tre test di package, un effetto della copia che non dipende da compare.py. `test_c4_extra.py` passa 6 su 6.
- Mutazione C61 (`any_local |= False`, compare.py:366): 61 test su 61 passano. Il caso extra `local_diff` fallisce e restituisce **SUPPORTED** invece di HYPOTHESIS_UNSUPPORTED.
- Mutazione C64 (`FORK` al posto di `FORK + 1`, compare.py:382): 61 test su 61 passano. Il caso extra `first_update` fallisce e restituisce **SUPPORTED** invece di PARTIAL_SUPPORT.

Questi risultati concordano con `agents/c4-test-mutazioni/mutation_log.txt`, dove le due mutazioni sopravvivono a 73 test.

## Dedotto dal codice
- compare.py:366 e :382 implementano correttamente il protocollo: le righe 16 e 18 del documento chiedono identita' dei locali e campi storici al passo 545. `classify` (righe 176-187) da' la precedenza ai locali, come prescrive la riga 23.
- Nessun altro controllo intercetta queste due regressioni. `test_cn_restart_compare.py:21-25` prova `classify` solo con valori booleani. Il parametrize e2e (`test_cn_restart_end_to_end.py:162-167`) non contiene un caso con locali diversi, uno senza scarto ridotto o uno col primo update diverso.
- Il run vero usa un checkout congelato di questo SHA, dove le righe sono corrette. La lacuna conta solo per una modifica futura di compare.py, quindi non e' un controesempio contro il run.

## Correzione minima
Aggiungere al parametrize e2e i casi `local_diff`, `no_discrepancy` e `first_update` presi da `agents/c4-test-mutazioni/test_c4_extra.py`, con l'helper `rewrite_rank`. Non blocca la release.
