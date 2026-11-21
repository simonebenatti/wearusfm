# Verifica avversaria di R6-3 (dichiarato: importante)

Esito: il rilievo regge solo come lacuna dei test. Gravita' corretta: **minore**. Nessun difetto del run.

## Che cosa ho eseguito
1. `agents/verifica-r6-test-R6-3-2/guards.py`: il vero `instrument_leg`/`dispatch` sotto DDP Gloo a 2 rank, con tre riscaldamenti o stati difettosi. Uscita:
   - riscaldamento che riscrive `opt.state[p]["exp_avg"]`: `ValueError: Null warm-up or boundary observer changed a checked state`, diff_keys=["optimizer"] su entrambi i rank;
   - gradiente residuo al fork: `ValueError: Residual gradients at common fork`;
   - prima chiamata con step 545: `ValueError: First backward did not start at the archived fork`.
   Le guardie L5, L6 e L8 del codice vero funzionano. Le mutazioni sopravvissute provano che i test non le coprono, non che il codice sia sbagliato.
2. Secondo livello in compare: `check_boundary` (compare.py:170) rifiuta da solo un boundary con before != after_warmup o con grad_none False (eseguito: "warmup changed state/input or RNG" in entrambi i casi). Anche senza L5/L6 il report uscirebbe INCONCLUSIVE, non con un esito attribuito al riduttore. Lo scenario «attribuito al riduttore» quindi non si verifica.
3. `agents/verifica-r6-test-R6-3-2/real_ddp.py`: `null_backward` con il **vero** `_accumulate_terms`, il modello e il runtime InfoNCE veri (montaggio di test_null_shadow), avvolto da `trainer._ddp_wrap` in DDP Gloo a 2 rank. Uscita: `rank0 diff=[] real_failure=None grads=106 pending=True` (uguale rank1). Il riscaldamento e' neutro sullo snapshot e il backward vero successivo passa. Limite: 106 parametri, CPU, senza osservatore e senza bf16.

## Dedotto dal codice
- L9: la guardia e' ridondante. `opt, = optimizers` (leg.py:149) solleva gia' se gli ottimizzatori non sono uno; run.py:743-744 solleva se i micro non sono uno con il contrastivo.
- L16 (count != 1): senza osservazione non c'e' boundary.rank*.json, compare lo legge e il report esce INCONCLUSIVE. Non c'e' rischio di attribuire male l'esito.
- L14/L15/L20: sono mutazioni del codice. Il codice vero include optimizer, runtime preparato e buffer nello snapshot (leg.py:64-70). Per l'optimizer la caduta e' provata nel punto 1.

## Correzione proposta (non bloccante)
Aggiungere a test_cn_restart_leg.py i tre casi del punto 1 (Gloo, con i messaggi attesi) e un'asserzione sull'insieme esatto delle chiavi di `boundary["before"]`. Facoltativo: il caso del punto 3 come test.
