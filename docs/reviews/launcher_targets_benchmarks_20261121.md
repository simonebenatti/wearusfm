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
