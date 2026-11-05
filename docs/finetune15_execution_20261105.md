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

## Avvio del 05/11, ore 12:44 CET

Pubblicato e verificato su origin e ref deploy il commit
`02f3de8547e0e50ee8d66f6ea43a356518c18e70`. Checkout CPU
**63578395 COMPLETED 0:0**, 31 s, 2 CPU allocate, 0,017222 core-ore.
Verificati SHA, pulizia, sola lettura, sorgenti 13de2f31… e piano 76a6c87d….
Il controllo dell'aggregatore sullo snapshot Git reale è in
`published_snapshot_validation.json`: verifica di preparazione, non risultato dei fit.

**Benchmark 63579657 RUNNING** alle 12:44:47 CET, da 405 s su lrdn0001.
Ricevuta del submitter e snapshot in `submission/`; nessun fit sottomesso.
La copia pubblicata del ledger omette soltanto l'annuncio del certificato
SSH pubblico, con hash del file originale; il ledger canonico resta intatto.

Il primo controllo da login07 ha restituito un errore DNS nella risoluzione
del controller SLURM. La lettura da login05 ha funzionato e conferma il job
in esecuzione: non è un errore del benchmark, nessun job duplicato o rilanciato.

## Benchmark concluso e pin per i fit, ore 12:50 CET

**63579657 COMPLETED 0:0**, 453 s: 0,125833 GPU-ore e 1,006667 core-ore.
Report completo da WORK, SHA256 `772bc48777d7e898b6a7db38b297ec502db339ee47287ea3690798912e308b55`,
ora pinnato nel piano. Verificati codice 02f3de8, piano eseguito 76a6c87d…,
sorgenti, ricetta, teacher, input e memoria. Il nuovo piano con il pin ha
SHA256 8a07e975…; il checkout del benchmark conserva il proprio piano originale.

| Regime | Proiezione per candidato | Limite operativo | Candidati | Proiezione GPU-ore | Esito |
|---|---:|---:|---:|---:|---|
| Frozen | 17.551 s | 21.300 s | 8 | 39,0022 | split |
| Fine-tuning | 50.701 s | 57.300 s | 8 | 112,6689 | split |
| Scratch | 51.027 s | 57.300 s | 8 | 113,3933 | split |

Ricalcolo dell'owner e indipendente del critico concordanti: q=1 per tutti,
D=4,161389 s; tutti e tre ammessi prima di qualsiasi fit. Memoria proiettata
host 32,386 GiB, massimo riservato GPU 6,584 GiB. Le proiezioni diagnostiche
del microbenchmark non sostituiscono la formula pre-registrata sui job completi.

Preventivo prudenziale dei fit **265,064444 GPU-ore / 2.120,515556 core-ore**.
Con benchmark già consumato: 265,190278 GPU-ore; includendo anche checkout
iniziale reale e secondo checkout al tetto: 2.121,872778 core-ore. Tetto
complessivo invariato 304,5 / 2.436,667. Quota alle 12:48 CET: 35,27 GiB liberi.
Il runtime scientifico resta invariato: pubblicazione di questi pin e poi
checkout nuovo per i 24 fit, con controllo finale di duplicati e output.
