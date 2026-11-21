# Benchmark continui C0/CN/CR RMS100/fast125 — esecuzione operativa

Consegna ab7a69a6; codice eseguito 746112dd711903e5c92ade2efd90834264b2676a.
Compito di 3-launcher: checkout comune e tre continui a 576 update, poi raccolta.
Riprese, lettura CPU completa e training lunghi restano consegne successive.

Piano rigenerato byte-identico, pin/submitter/ledger verificati, ref di deploy
pubblicato e SHA verificato. Prima di ogni invio: snapshot saldo/impegni/quota,
reservation/slot/calendario, controlli login congelati e validazione del runner
per braccio. File originali in benchmark_execution_20261121/.

Before_checkout ha verificato i prerequisiti conclusi, le impronte fisiche e
i tre comandi effettivi, con config/release/job coerenti per C0/CN/CR,
RMS100/fast125, pesi calibrati comuni, dose 576, batch 32 e limite 1770 s.
Checkout **65017186 COMPLETED 0:0**, 19 s, 4 CPU/2 GiB/4 min.
I benchmark richiedono la verifica sul checkout completato e aggiornamento
operativo immediatamente prima del relativo invio. Un tentativo per ID.

Riserva 12 GiB per coorte più 18 GiB per altre scritture; libero WORK circa
39,6 GiB all'avvio. Saldo 653.928 ore locali; nessuna cella dbg sbenatti
prima della prima ondata. Corsie UP; manutenzione 23/11 08:00 CET–25/11 08:00 CET.
Si procede C0 e CN, poi CR al primo slot; massimo due debug dell'utente.

Before_C0 e before_CN passati sul checkout completato, con snapshot aggiornati.
Primi ID benchmark: **C0 65017240**, **CN 65017310**. Entrambi inviati una sola
volta via submitter, export TORCH_DIST_INIT_BARRIER=1 nelle ricevute.
CR resta da inviare al primo slot disponibile dopo ricontrollo completo.

C0 e CN conclusi COMPLETED 0:0 in 795 e 784 s, entrambi a 576 update.
Prima raccolta disponibile: parità C0 aggregata e quattro rapporti individuali;
gate PASSED per entrambi i punti 128/256, gradienti byte-identici entro e fra
i casi. Before_CR nuovamente positivo e slot verificato libero: **CR 65020151**
inviato una volta. Non è stato necessario accodare una terza cella agli altri due.

## Raccolta conclusiva

| Braccio | SLURM | Stato | Elapsed SLURM | Elapsed trainer | Picco CUDA massimo rank |
|---|---:|---|---:|---:|---:|
| C0 | 65017240 | COMPLETED 0:0 | 795 s | 762,049888 s | 19,209005 GiB |
| CN | 65017310 | COMPLETED 0:0 | 784 s | 753,035842 s | 19,209005 GiB |
| CR | 65020151 | COMPLETED 0:0 | 773 s | 737,857988 s | 19,209005 GiB |

Tutti: 576 update completi, stopped=max_steps; ricetta realizzata normalizzata
identica a quella congelata del braccio; codice, digest sorgenti e quattro
input coincidenti. Quattro profili bf16_math_tf32_off per run verificati.
Tempi t_step_s medi dopo i primi otto passi: C0 1,155707, CN 1,110840,
CR 1,108899. Non confondere questa media col walltime end-to-end.

Raccolti **60 artefatti originali**, tutti verificati per SHA e byte. Include
aggregato C0 e quattro rapporti individuali; le rispettive righe coincidono.
Gate ufficiale validate_parity_report ricalcolato: **PASSED ai punti 128 e 256**,
gradienti byte-identici entro e fra casi. Rapporti di parità con training_approved=false.
SHA aggregato: 78cdae11e2a385a5fc92b9fff50c9225d537865a397d7e4ddd971017ee42a976.

Mappe origin_sessions identiche nei tre bracci, SHA
aa837c099d43b4e52bba68b31aabc4e0d40a385c7291dffd2a0ebee600e4dc9d.
CN e CR hanno tutte le 32 tracce 545–576 per ognuno dei quattro rank;
hash batch/input coincidono fra i due bracci nell'intero supporto tracciato.
Non si estende questa attestazione ai passi privi di traccia dei tensori effettivi.

Mediane descrittive del rapporto fra gradiente contrastivo pesato e base,
64 update 513–576: **CN 0,21541209, CR 0,08278750**. Non viene applicato un
nuovo criterio di promozione e non si anticipa la lettura completa delle riprese.

SHA256 summary originali:
- C0 f07a87de9ae09200cfff0cc5f7ce54e1e7fa86af0f5bdcaa68d203594d1caff7;
- CN e7bc701dab54ca1b5d3fae8d6a0fb1393234933ebde083b35cc94b10a1c49d17;
- CR 69dc27315497faec486122af1fd24bf492889014bfe172c17ab76bbc420dadaa.

Checkpoint finali a 576 (checkpoint.pt, 463.190.144 byte ciascuno) e a 544
(checkpoint_step0000544.pt, 463.142.486 byte ciascuno) tutti conservati sul cluster.
CN 544 aa5952f0c4b7a8f07b5965b096877d568f3b5662bf08e6d8a4fc56f2a9e37ce6;
CN 576 4743abaff631d43c0f9267b886087fdcbefdfc156332bf34d43b668297668dec;
CR 544 a9a5b1deb69c46c0238480500605f24fe9415c41f2a964d91d0cd2f7f5608e1a;
CR 576 c57f9437f9a38b880f5c558a9e293ef5d7467a92956bc960b1ae76d44fc54ab2.
C0 finale 6813728af7b171b3c18bacd07050feb7df4700f69d40b5257507599e35707ffb.
Tutte le impronte, incluso C0 a 544, in checkpoints_remote.txt.

| Risorsa, quattro allocazioni | Richiesta | Effettiva | Residuo sul richiesto | Inviluppo con margine |
|---|---:|---:|---:|---:|
| GPU-ore | 6 | **2,613333** | 3,386667 | 7,2 |
| Core-ore | 48,266667 | **20,927778** | 27,338889 | 58,266667 |

Billing osservato uguale alle CPU allocate. Cumulativo effettivo nuova coorte:
3,747778 GPU-ore / 39,067778 core-ore. Prima di ciascun invio, sottratti dal saldo
misurato 653.928 gli impegni degli altri job e l'inviluppo di questa tranche una
sola volta (escluse dagli altri impegni le sue celle già in coda): saldo netto
prudenziale 649.994,482222 / 649.995,542222 / 649.996,342222 / 650.011,168889 ore
rispettivamente checkout/C0/CN/CR. Snapshot integrali conservati.

Archivio: results/infra/contrastive_targets_20261118/benchmark_execution_20261121/.
Remoto: `$WORK/wearusfm_runs/results/contrastive_targets_20261118/benchmark/{C0,CN,CR}`.
Consegna conclusa, nessun retry o modifica al codice. **Nessuna ripresa, lettura
CPU completa o training lungo avviato**; segue la lettura scientifica.
