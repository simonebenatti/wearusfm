# Verifica avversaria R6-1 (chiave verifica-r6-test-R6-1-1)

Esito: il rilievo REGGE nella sostanza ma va ridimensionato (gravita' corretta: minore).

## Eseguito (simulazione torch 2.2: shim agents/verifica-r6-test-R6-1-1/shim/r6v_shim.py, ValueError se mmap=True e f non str)
- Senza shim: tests/cpu/test_cn_restart_observe.py -> 8 passed.
- Con shim (-p r6v_shim): 1 failed (test positivo, ValueError mmap), 7 passed. Riprodotto.
- Variante con messaggio stampato e assert "mmap" not in msg (agents/verifica-r6-test-R6-1-1/test_r6v_observe_matched.py):
  senza shim 8 passed, messaggi: pin/rank "invalid observation receipt or vector pin"; name_collision/bucket_members
  "invalid parameter or bucket inventory"; slot "bucket/parameter mapping differs"; shape/gradient_hash
  "observed gradient differs from its receipt". Con shim: 6 failed, 2 passed (pin, rank).

## Ridimensionamento
1. I casi mascherati sono 5 su 7, non 7: `pin` e `rank` sono rifiutati dal controllo della ricevuta
   (observe.py:139-143) prima di torch.load (riga 144) e passano legittimamente anche su torch 2.2.
2. La suite NON e' cieca a B-1: il test positivo chiama load_observation nel processo principale e fallisce sotto shim.
   Il fixture (load_observation dentro i figli mp.spawn, dove il plugin non arriva) passa, ma non serve a nasconderlo.
3. Non cambia l'esito del run vero: il difetto che rende INCONCLUSIVE e' B-1 (gia' bloccante). R6-1 e' qualita' dei test
   (regressione futura), non un controesempio nuovo sul run.

## Correzione minima
- Codice (= B-1): torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True) in observe.py:144.
- Test: match= per caso, con le stringhe sopra (le quattro proposte dal revisore sono corrette, verificate).
- Facoltativo: test nel processo principale con torch.load avvolto (errore se mmap=True e f non str).
