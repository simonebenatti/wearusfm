# Benchmark posa: controllo indipendente, 17/11/2026

CONFORME. Accounting ricalcolato dai sacct grezzi: 64665960/64665968/64665970 COMPLETED 0:0, 27/24/23 s, totale 0,164444 core-ore. Picchi 1,652/1,915/1,248 GiB. Runtime numerico identico tra benchmark, 8 thread, 40 utenti e 4.813.620 elementi validi. Dimensioni 384/896/384 e supporto coerente con la validazione.

Il costruttore pose_budget è AST-identico al congelato df2de35: invocato solo in memoria, 304,84055 s di base, 577,26083 s con margine, derived_seconds=600 su cap 3600. L'allocazione congelata resta 8 CPU/30 GiB/3600 s. Nessun risultato scientifico letto, nessuna selezione degli iperparametri. Nessun rilascio anticipato dei fit: mancano i benchmark classificazione e i relativi budget. Inviluppo cumulativo prudenziale 343,597222444/351 core-ore.
