# verify-r1-parita-R1-N6-1 — scettico indipendente n. 1 su R1-N6 (NOTA)

Inizio 20/11. Copia a 0efa4d45 (contenuti fbfde797), sola lettura. Script miei in rev_runtime/tmp/verify-r1-n6/scripts.

## Passi
- Diff ricorsivo dei config review_configs/{C0,CN,CR}: differiscono solo in contrastive.arm (1 chiave per coppia); nce_weight/rnc_weight null in tutti e tre. calibrated_configs (contrastive_prepare.py:394-408) riempie le stesse λ in tutti e tre: anche i config di produzione differiscono solo per arm.
- contrastive_objectives.py:102-105: unico uso di arm in src (grep) oltre alla validazione; attach() (227-354) calcola NCE e RnC in tutti i bracci e scrive weight=(wn,wr); backward_loss (jepa.py:270-275) somma weight*num per tutte le chiavi; _ddp_wrap (run.py:430-438) non dipende da arm.
- _measure_case (contrastive_target_parity.py:98-152): prepare + _accumulate_terms, nessun before_update/commit.

## Riproduzione (script mio: rev_runtime/tmp/verify-r1-n6/scripts/repro_n6.py, uscita repro_out.json)
- config_diffs C0/CN, C0/CR, CN/CR: ["contrastive.arm"] soltanto.
- weights() con fill/ramp di produzione (256/256) e 200 coppie λ casuali in [1e-4,1]: (0.0, 0.0), zero positivo, in tutti i bracci a update 128 e 256; a update 257 non nullo solo per il braccio attivo. wants_gradients(255) vero in tutti e tre.
- Sei probe CPU reali (_measure_case, stato/coda/batch come nel test), punti 128 e 256: 1 hash di gradiente, stessa firma del grafo autograd (1128 nodi, stesso multinsieme di tipi) nei 6 casi, pesi visti solo {0,0}; commit() 0 chiamate, before_update() 0 chiamate.
- Controllo positivo (in memoria, nessun file toccato): peso CN 1e-3 ai passi a peso zero -> entro braccio ancora identico (C0, CN, CR: True), fra bracci C0/CN diverso, C0/CR uguale. Quindi il confronto fra bracci ha una funzione propria: e' la verifica empirica che l'etichetta sia inerte (qui lo intercetterebbe anche assess_point:195-196 sui pesi; un ramo per arm a peso zero solo l'hash).
- tests/cpu/test_contrastive_target_parity.py: 29 passed in 4,23 s.

## Motivi cercati per confutare
- Prontuario 19/11 (righe 29-47) e plan.json numerical_amendment.neutrality chiedono "byte-identici entro e tra bracci" ma non dicono che fra bracci, a peso zero, equivale a ripetibilita' + etichetta inerte. Lo dicono in sostanza documenti precedenti: decisione faa4a8d7 riga 3 e 19 (fra bracci letto contro la variabilita' entro braccio), criteri storici contrastive_common_diagnostics_20261118.md:19 ("C0/CN ... differiscono mentre C0/C0 coincide ... sospetto sul percorso del braccio a peso zero"), diagnostica 20261118 riga 37 e review 94f5094 riga 70 (config diversi solo per arm). Il prontuario dichiara di prevalere sui prontuari operativi 16-18/11 e non riprende il punto.
- Nessuna sovrastima nel codice: il readout chiama il cancello "decisive zero-weight neutrality gate" (readout:342), coerente.
- Commit della coda: commit()/before_update() non leggono arm (contrastive_objectives.py:356-391; commento 269 "All branches update the identical queue policy"); non ci sono differenze fra bracci nell'aggiornamento della coda da esercitare. La frase del rilievo e' vera ma il limite e' vuoto per la coda; per la dose e' per costruzione (test a peso zero).

## Esito
Non confutato. Fatti riprodotti tutti. Gravita' reale NOTA (chiarezza della lettura, nessun difetto di codice, nessun effetto su cancelli o attribuzioni). Precisazione per la review: l'identita' fra bracci discende dalla ripetibilita' SOLO dato che l'etichetta e' inerte; il confronto fra bracci e' proprio la guardia empirica di questa inerzia (controllo positivo), non una prova ridondante ne' una prova sulla dose o sulla coda.
