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

## Esito raccolto

Checkout 65014846 COMPLETED 0:0, 39 s; calibrazione **65015116 COMPLETED 0:0,
891 s**. Profilo bf16_math_tf32_off attestato dai quattro rank. Batch globale
32 (8 per rank), sei worker per rank; **512 passi completi**, zero update
dell'optimizer e identico hash iniziale/finale di student e teacher.

Rapporto congelato **PASSED_FOR_REVIEW**, failure_reasons vuoto;
training_approved=false. I primi 32 batch congiuntamente informativi sono
257–288. Pesi misurati:

| Termine | Peso |
|---|---:|
| InfoNCE | 0,11847082160202581 |
| Rank-N-Contrast | 0,32885281420266826 |

Copertura positiva: nce_fraction 0,9984130859375 e rnc_active_fraction 1,0,
esenzione CapgMyo già firmata. Picco memoria allocata CUDA massimo fra i rank
**19,2730884552 GiB** (summary, riduzione MAX del trainer). Tempi t_step_s medi:
1,471273 sui 512 passi; 0,955804 nel fill; 1,986742 sul pannello 257–512;
1,990070 sui 32 batch selezionati. Elapsed trainer 859,556647 s, elapsed SLURM
891 s: le misure hanno perimetri distinti e non vanno scambiate.

Rapporto remoto: `$WORK/wearusfm_runs/results/contrastive_targets_20261118/calibration_report.json`;
SHA256 **20cb900522b62b46f17d1b65cbbe1ef579a4cdd5e935a7ab31b7a309ed516532**.
Verificati 14 artefatti originali contro gli hash remoti, inclusi config, summary,
metriche e quattro profili. Checkpoint 234.339.194 byte, SHA256
d0dd318302f278e08513ec7d5acde3c82b36a523f7c650829167a68864870ebc;
rimane nella cartella calibration sul cluster. Archivio locale completo in
calibration_execution_20261121/, senza checkpoint o segnali.

Il primo tentativo del solo lettore è fallito su import NumPy: l'assegnazione
PYTHONPATH=src:scripts del prontuario rimuove i percorsi dei moduli CINECA.
Preservati questi ultimi come nei login-check (`src:scripts${PYTHONPATH:+:$PYTHONPATH}`),
il medesimo lettore congelato e il medesimo interprete hanno prodotto il rapporto.
Entrambi i log sono conservati; nessun retry di training né modifica al codice.

| Risorse | Richieste | Effettive | Residuo richiesto | Inviluppo con margine SLURM |
|---|---:|---:|---:|---:|
| GPU-ore | 2 | **0,99** | 1,01 | 2,4 |
| Core-ore | 16,266667 | **7,963333** | 8,303334 | 19,866667 |

Billing osservato pari alle CPU allocate: 7,92 ore locali calibrazione più
0,043333 checkout. Nuova coorte, audit incluso: 0,99 GPU-ore / 16,953333 core-ore.
Consegna conclusa. Nessun job CUDA successivo o training inviato; la release
successiva richiede la lettura scientifica e l'incorporazione di queste misure.
