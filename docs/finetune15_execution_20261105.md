# EPN15 — esecuzione dopo convergenza

Review Claude `ceaf95b66519b93e4f2bcfdda2d3cafd2f9f5e24`, codice scientifico
identico a `90ffcd4`: source13de2f31…, ricetta db975bd1…. R2/R3 chiuse in
protocollo, piano e decisioni: tutti i regimi devono passare prima di qualsiasi
fit; Simone conferma il tetto304,5GPUh/2436,667coreh, retry esclusi.

Alle12:15CET: checkpoint teacher presente, SHA256 a31c8950… corretto;
namespace/output/checkoutFT15 assenti,27ID logici non presenti nel ledger.
WORK ha35,46GiB liberi sotto quota progetto, sopra i6GiB richiesti; inode
progetto senza limite impostato e filesystem con margine. Evidenza in
`results/infra/finetune15_20261105/preflight_20261105.json`.
Quota condivisa, da rileggere prima delle sottomissioni se la prova invecchia.

## Catena concreta

1. Pubblicare e verificare SHA su origin e ref deploy `finetune15-bench-20261105`.
2. `FT15-checkout-bench-20261105`: launcher di checkout standard passato su
   stdin al submitter,1CPU richiesta/2prudenziali,2GiB,10min,0GPU.
3. Verificare COMPLETED0:0, SHA, pulizia e codice in sola lettura. Non usare
   afterok verso un ID già rimosso dalla memoria SLURM: ricevuta completata
   verificata è prerequisito soddisfatto; non duplicare il checkout.
4. `FT15-BENCH-20261105`:1GPU/8CPU/100GiB,30min, tetto0,5GPUh. Checkout nuovo,
   OUT `WORK/wearusfm_runs/results/finetune15_20261105/benchmark`.
5. Verificare report completo, accounting0:0, pin, source/recipe/input/teacher,
   32update dopo4warm-up, memoria85%, proiezione ricalcolata per tutti i regimi.
   **Se un regime è blocked, non sottomettere alcun fit.**
6. Solo se tutti passano: pubblicare pin e preventivo misurato, poi secondo
   checkout `finetune15-fits-20261105` con sorgenti immutati. Quota fresca,
   verifica duplicati e24fit split secondo launch_plan; ogni C1 afterok al suoC0.

Submitter `scripts/submit_job.py`, ledger comune `~/wearusfm_local/jobs`, ownerGPT.
Nessun monitor automatico; nessuna nuova loss/dose/seme/griglia o nuovo dataset.
Continuazione lunga sui suoi ID/checkout invariati; non duplicare né confondere
questa matriceFT con le sonde a due dosi/profondità ancora da predisporre.

## Risorse

24fit:8frozen×6h e16FT/scratch×16h,1GPU e8CPU ciascuno. Tetto304GPUh/2432coreh.
Benchmark0,5GPUh/4coreh, due checkout massimo0,667coreh. Totale304,5GPUh e
2436,667coreh; proiezione prudenziale di riferimento265,038GPUh, atteso dal
consuntivo204,815GPUh per i fit. Misura nuova può alzare il preventivo, mai
ridurre dose/griglia o permettere una matrice parziale. Consuntivo da SLURM.
