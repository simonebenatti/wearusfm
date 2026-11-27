# Verifica avversaria C4-1 (chiave verifica-c4-test-mutazioni-C4-1-2)

Commit esaminato: 8c72655c3d894c82fd4918ed523baf0ac9c69e23 (clone pulito, `git status` vuoto prima e dopo).

## Esito
Il rilievo regge come lacuna dei test, ma **non e' un difetto del codice**: va ridimensionato da importante a **minore**.

## Eseguito
1. Clone originale, 5 file `test_cn_restart_*`: `73 passed in 10.51s`.
2. Copia di lavoro in `agents/verifica-c4-test-mutazioni-C4-1-2/tree` (senza .git). Tre test di
   `test_cn_restart_package.py` falliscono gia' sul codice non mutato, con `git diff` exit 129, perche'
   la copia non ha .git. Sono un artefatto della mia copia e non dipendono dalle mutazioni.
3. Mutazione C61 (`any_local |= False`, compare.py:366): stessi 70 passati e 3 artefatti, quindi la
   mutazione sopravvive alla suite. `test_c4_extra` la uccide: `local_diff` riceve `SUPPORTED` invece di
   `HYPOTHESIS_UNSUPPORTED`.
4. Mutazione C64 (`d["step"] == FORK`, compare.py:382): sopravvive alla suite. `test_c4_extra` la uccide:
   `first_update` riceve `SUPPORTED` invece di `PARTIAL_SUPPORT`.
5. `test_c4_extra` sul codice non mutato: `6 passed`.

## Dedotto dal codice
- compare.py:366 confronta i gradienti locali fra le due gambe. Lo fa per ogni parametro e per ogni rank,
  ed e' corretto.
- compare.py:382 controlla il passo `FORK + 1 = 545`, perche' `FORK, END = 544, 576`
  (scripts/contrastive_target_resume.py:31). `rows()` (righe 213-219) impone i passi 545..576, quindi
  con la mutazione C64 il passo 544 non comparirebbe mai e `first_exact` sarebbe sempre vero.
  Il codice attuale e' corretto.
- Nessun altro controllo intercetterebbe le due regressioni. Se una differenza locale piccola resta sotto
  lo screen, `excessive` e' falso e la lettura finisce in `SUPPORTED`. Una differenza delle metriche al
  solo passo 545, con endpoint esatto, finisce anch'essa in `SUPPORTED`.
- `test_cn_restart_compare.py:21,25` prova `classify` solo con booleani. Nella tabella end-to-end
  (test_cn_restart_end_to_end.py:162-167) mancano tre esiti: `HYPOTHESIS_UNSUPPORTED` da gradiente locale,
  lo stesso esito senza scarto, e `PARTIAL_SUPPORT` dal primo update.

## Gravita'
Nessun controesempio rende il run vero INCONCLUSIVE, fallito o attribuito male: il codice congelato al SHA
e' corretto nelle due righe. Il rischio riguarda solo una regressione futura. La gravita' e' minore e non
blocca la release.

## Correzione minima
Aggiungere al parametrize di `test_complete_restart_readout_from_four_folders` i casi `local_diff`,
`no_discrepancy` e `first_update` di `agents/c4-test-mutazioni/test_c4_extra.py`, con i rispettivi esiti
attesi. Si puo' fare nel prossimo commit, senza un nuovo giro di review.
