# Verifica avversaria R6-3 (chiave verifica-r6-test-R6-3-1)

Esito: il rilievo regge solo come lacuna di copertura dei test. Lo scenario dannoso non si verifica nel codice al commit 9d34dc69.
Gravita' corretta: **minore** (dichiarata: importante).

## Eseguito
Ho esportato una copia con `git archive HEAD src scripts tests/...` in agents/verifica-r6-test-R6-3-1/tree. Il clone non e' stato toccato.
- Ho scritto tree/tests/cpu/test_r63_guards.py. E' un DDP Gloo reale a 4 rank con lo stesso fixture a 270 parametri, Adam gia' popolato
  e `instrument_leg("warm")` vero. Il riscaldamento sostitutivo:
  - altera opt.state exp_avg: tutti e 4 i rank sollevano "Null warm-up or boundary observer changed" (PASS);
  - altera un parametro: stessa eccezione (PASS);
  - lascia gradienti residui al fork: "Residual gradients at common fork" (PASS);
  - prima chiamata con step=543: "First backward did not start at the archived fork" (PASS);
  - altera un buffer persistente: NESSUNA eccezione. E' voluto: null_backward (leg.py:79,89-92) ripristina tutti i buffer
    prima dello snapshot "after", quindi il vero 545 li vede intatti. Non e' una guardia mancante.
- Mutazioni sulla copia (mut.py): con M5 (L164 disattivata), M6 (grad_none), M8 (First backward), M9 (L145) e M16 (count!=1),
  la suite originale passa sempre. Questo conferma la sopravvivenza dichiarata dal revisore.
  I nuovi test uccidono M5, M6 e M8. M9 e M16 restano scoperti anche dai miei test.
  Il fallimento di test_null_shadow nella copia e' un artefatto: manca test_pose_training.

## Dedotto dal codice
- Le guardie esistono e sono corrette: leg.py:139-152, 164-165 e 180-181. Lo snapshot (60-73) include optimizer, student e teacher
  state_dict, buffer, runtime serializzato e transitorio, ancore, RNG e digest dell'input.
  "Un riscaldamento che alteri lo stato di Adam non verrebbe fermato" e' falso: viene fermato (prova sopra). Il rischio vero e'
  solo una regressione futura non intercettata. Non riguarda il commit candidato, che e' congelato.
- Il riscaldamento non chiama opt.step (leg.py:76-93), quindi Adam non puo' cambiare. _ddp_wrap (run.py:430-438) usa
  find_unused_parameters solo con RVQ o keep, senza static_graph. Un secondo backward DDP sullo stesso modello e' ammesso.
- Il vero _accumulate_terms sotto DDP nel riscaldamento non e' provato sul Mac. E' un rischio di fedelta' coperto da altre
  verifiche (R1/R3). Il rilievo non fornisce un controesempio concreto.

## Correzione minima (facoltativa, non blocca la release)
Aggiungere al test Gloo i casi adam, param, residual e firststep di test_r63_guards.py e un'asserzione sulle chiavi di boundary.before.
Se possibile, aggiungere un caso per L145 e L180 (due chiamate al 544, nessuna chiamata).
