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

## Matrice interamente sottomessa, verifica delle 13:15 CET

Checkout fit **63581502 COMPLETED 0:0**, 26 s, 2 CPU allocate:
0,014444 core-ore. SHA `1359b924d25601f92e9035b8bd621c2c6a93f947`,
piano 8a07e975…, source 13de2f31…, ricevuta benchmark 772bc487…;
checkout pulito e non scrivibile, nessun output preesistente o symlink nei
genitori. Quota alle 12:59 CET: 35,23 GiB liberi, evidenza valida durante
tutti gli invii; 24 ID logici inizialmente assenti.

**24/24 fit sottomessi, 24 ID SLURM distinti, 12 dipendenze C1→proprio C0.**
Stato SLURM alle 13:15:17 CET: **12 RUNNING / 12 PENDING Dependency**.
Ogni job attivo ha 1 GPU, 8 CPU e 100 GiB; le 12 righe di accettazione del
gate compaiono nei log, senza traceback o errori di avvio osservati.
Queste sono verifiche operative, non esiti scientifici.

| Regime | Fold | C0: RUNNING | C1: PENDING Dependency |
|---|---:|---:|---:|
| frozen | 0 | 63581664 | 63581670 |
| frozen | 1 | 63581672 | 63581673 |
| frozen | 2 | 63581674 | 63581677 |
| frozen | 3 | 63581682 | 63581713 |
| finetune | 0 | 63581718 | 63581725 |
| finetune | 1 | 63581735 | 63581741 |
| finetune | 2 | 63581742 | 63581745 |
| finetune | 3 | 63581750 | 63581751 |
| scratch | 0 | 63581768 | 63581837 |
| scratch | 1 | 63581887 | 63582180 |
| scratch | 2 | 63582244 | 63582253 |
| scratch | 3 | 63582258 | 63582269 |

Ricevute in `submission/`, controllo completo del ledger in
`matrix_submission_verification.json`, stato/log in `matrix_status.json`.
Sottomissioni tutte tramite submitter e ledger comune. Nessun job duplicato,
annullato o risottomesso. Nessun monitor automatico.

Durante gli invii lo strumento ha richiesto ripetutamente un consenso già
acquisito e due comandi sono stati rifiutati prima di eseguire il submitter:
gli ID corrispondenti erano assenti nel ledger, nessun rifiuto di SLURM.
GPT ha dichiarato troppo presto risolta la causa delle richieste; la prima
ipotesi sulla forma dei comandi non era stata verificata. Simone ha ribadito:
«non voglio che fermi la sequenza, devi solo lanciare senza chiedrrmi e tutto corretto».
Ripresa dai soli ID assenti. Per gli ultimi cinque invii è stato usato il
percorso assoluto del submitter comune già presente fra i prefissi approvati,
nel worktree science: verificato byte-identico a quello di consolidamento
(SHA256 814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82).
Lo SHA scientifico proviene da EXPECTED_SHA, invariato; stesso ledger.
Non si deduce da questo una garanzia generale sul comportamento futuro dei
dialoghi dell'app: il fatto verificato è l'accettazione di tutti i 24 invii.

Consuntivo delle tre attività concluse (benchmark e due checkout):
**0,125833 GPU-ore / 1,038333 core-ore**. I fit sono ancora in corso,
quindi non esiste un consuntivo finale della matrice. Tetto autorizzato
304,5 GPU-ore / 2.436,667 core-ore invariato; nessuna estensione automatica.
