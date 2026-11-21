# Calibrazione GPU RMS100/fast125 — esecuzione operativa

Operatore 3-launcher, 21/11/2026. Istruzioni d9cd1bb8, esecuzione
e05c05826daa9a6feda47f1f75742ea329c84f5f. Nessuna modifica tecnica.
Piano rigenerato identico byte per byte; pin, submitter, ledger e ref deploy
verificati. Prima di ogni invio snapshot operativi e controlli congelati sul login.

Prima del checkout: saldo disponibile 653.928 ore locali; WORK libero circa
40,9 GiB, gate 30 GiB (12 coorte + 18 per altre scritture); nessun job debug
sbenatti, corsie UP. Reservation manutenzione 23/11 08:00 CET–25/11 08:00 CET
confermata. Calendario ampiamente sufficiente per questa coppia di allocazioni.
Gli impegni residui degli altri utenti sono conservati nello snapshot progetto.

before_checkout positivo: check-binding, check-audit 64937087, make-job/cmp,
build_command, ricetta e release --validate-only passati. Stage calibration,
job/release arm null, config C0/calibration_only=true, RMS100/fast125,
quattro rank, time-limit 1770. Primo job **65014846**, checkout 4 CPU/2 GiB/4 min.
La calibrazione attende sacct COMPLETED 0:0 e before_calibration sul checkout.
Evidenze: calibration_execution_20261121/ nel pacchetto della coorte.

Checkout **65014846 COMPLETED 0:0**, 39 s. before_calibration positivo sul
checkout effettivo: SHA, pulizia, permessi, binding/audit e ricetta verificati.
Calibrazione **65015116** sottomessa una volta via ledger comune, 4 GPU,
32 CPU, 240 GiB, 30 minuti. Ricevuta --export include TORCH_DIST_INIT_BARRIER=1.

Snapshot checkout 09:36:14 CET e calibrazione 09:43:04 CET: saldo 653.928 ore
locali; residuo degli altri job del progetto 3.901,357778 e 3.898,631111 ore
(proxy billing=CPU, coerente con le allocazioni storiche). Sottratto l'inviluppo
di questa fase 19,866667 ore, saldo netto prudenziale rispettivamente
650.006,775555 e 650.009,502222 ore. Nessuna duplicazione dello storico già
addebitato. Spazio oltre 40 GiB, corsie UP, nessun job dbg concorrente
dell'utente; reservation e calendario verificati prima di entrambi gli invii.
La raccolta del rapporto e dei consuntivi conclude questa consegna.
