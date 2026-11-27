# Verifica avversaria del rilievo C4-2 (commit 8c72655c)

Esito: il rilievo regge come lacuna dei test. Il codice corretto e' giusto. Gravita' giusta: **minore**, non importante.

## Eseguito
- Mutante N09 (`matches_ordinary = not identical(reference, measured)`, compare.py:108), messo in
  `agents/verifica-c4-test-mutazioni-C4-2-1/mutroot/scripts/diagnostics/` con link a results/docs/src, cosi' ROOT=parents[2] resta valido.
  Il mutante sopravvive a test_cn_restart_{compare,end_to_end,package,observe}: `69 passed` (n09_run.txt).
  (Senza la radice simulata, 11 fallimenti in e2e erano dovuti solo al percorso di ROOT: scartati.)
- Test nuovo `agents/.../test_both.py`: continuo seme 24, ordinaria seme 12, riscaldata = ordinaria tranne cuda_rng[2]=[99,5,9].
  Sul codice vero: passa (`scientific_metadata_or_schema_different == ["cuda_rng.2"]`, non esatto). Col mutante N09: fallisce
  (`assert not True`, cioe' la variazione presente solo nella gamba riscaldata verrebbe esentata).

## Dal codice
- test_cn_restart_compare.py:141-160 e end_to_end.py:113-114 (`dormant_not_ordinary`) mettono il dormiente riscaldato uguale al
  continuo: in quel caso "uguale all'ordinaria" e "diverso dal continuo" danno lo stesso verdetto, quindi la suite non distingue
  la regola di B-2 da quella ritirata.
- Nessun altro controllo in run() (compare.py:295-324) confronta cuda_rng.1-3 della riscaldata con l'ordinaria: l'unica guardia
  e' quella di riga 108/113.

## Gravita'
Il codice al commit e' corretto (test_both passa): il run vero non diventa INCONCLUSIVE ne' viene attribuito male. Manca solo
la protezione contro una regressione futura della correzione B-2. Minore.

## Correzione minima
Aggiungere a test_process_rng_is_not_the_active_rng_on_any_rank (o come test a parte) il caso di test_both.py; facoltativo
il gemello end-to-end con esito INCONCLUSIVE e "cuda_rng.2" nel motivo.
