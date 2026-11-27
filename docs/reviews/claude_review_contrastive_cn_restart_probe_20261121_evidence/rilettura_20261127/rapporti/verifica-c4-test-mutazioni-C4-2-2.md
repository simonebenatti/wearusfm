# Verifica avversaria C4-2 (chiave verifica-c4-test-mutazioni-C4-2-2)

Commit esaminato: 8c72655c (clone in sola lettura). Lavoro in `agents/verifica-c4-test-mutazioni-C4-2-2/`.

## Esito
Il rilievo regge **come lacuna dei test**, non come difetto del codice. Gravita' ridotta da *importante* a **minore**.

## Codice (letto)
`scripts/diagnostics/contrastive_cn_restart_compare.py:103-117`: l'esenzione di `cuda_rng.1-3` richiede
`same_shape and matches_ordinary`, con `matches_ordinary = identical(ordinary, measured)` (r.108). Se il
dormiente riscaldato differisce da continuo e ordinaria, il percorso resta in `scientific_invalid` e
`run()` restituisce INCONCLUSIVE con il percorso nel motivo (r.321-325). La regola di B-2 e' implementata correttamente.
`ordinary_cuda_rng` proviene dall'endpoint ordinario misurato (r.312), che a r.320 deve coincidere byte per byte
con l'ordinaria storica (SHA ENDPOINT e `exact_payload`). Il riferimento e' quindi la CN_A storica.

## Test (eseguito)
- Mutante N09 (r.108 -> `not identical(reference, measured)`), applicato a una copia dell'albero:
  compare, end_to_end e observe passano tutti. I fallimenti di `test_cn_restart_package.py` nella mia copia
  sono un artefatto: la copia non ha `.git` (`git diff ... exit 129`). Il revisore C4, eseguendo nel clone,
  riporta N09 sopravvissuto con 73 passati (`agents/c4-test-mutazioni/mutation_log.txt:80`). Confermato.
- Perche' sopravvive: in `test_process_rng_is_not_the_active_rng_on_any_rank` il dormiente modificato e' posto
  uguale al continuo (`warm["cuda_rng"][2] = continuous[...]`). Nell'end-to-end `dormant_not_ordinary` fa lo stesso
  (r.113-114). In entrambi i casi la condizione mutata «diverso dal continuo» da' lo stesso verdetto.
  Il caso `ordinary_dormant` (r.125-128) altera l'ordinaria. Si ferma prima, al require di r.320, e quindi non distingue N09.
- Nuovo test unitario `test_warm_both.py` (dormiente = [99,5,9], diverso da entrambi): passa sul codice vero e fallisce
  sul mutante. Anche `test_c4_extra.py::dormant_differs_both` del revisore passa sul vero (INCONCLUSIVE) e fallisce sul mutante.
  Suite vera con i due file in piu': 76 passati.

## Gravita'
Al commit 8c72655c il run vero classifica correttamente: nessun controesempio produce INCONCLUSIVE spurio o
un'attribuzione sbagliata. Il rischio riguarda solo una futura modifica di r.108, che la suite non fermerebbe.
Secondo le regole il rilievo non e' bloccante. Non e' nemmeno «importante», perche' non c'e' un difetto introdotto. E' **minore**:
la regressione proteggerebbe proprio la correzione di B-2.

## Correzione minima
Aggiungere a `tests/cpu/test_cn_restart_compare.py` un caso con il dormiente riscaldato diverso da continuo e ordinaria.
Esito atteso: `scientific_metadata_or_schema_different == ["cuda_rng.2"]` e `exact_scientific_payload` falso.
Facoltativamente, aggiungere il parametro end-to-end corrispondente, con INCONCLUSIVE e `cuda_rng.k` nel motivo.
Il caso e' il testo di `agents/verifica-c4-test-mutazioni-C4-2-2/test_warm_both.py`.
