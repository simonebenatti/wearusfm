# Consolidamento — esecuzione dopo convergenza, 04/11/2026

Review Claude `0b2623a` su `2dbd498`, conservata integralmente nel ramo.
Continuazione, benchmark FT e regole F-R2/V-R1 convergenti. R-1 risolto con
la sola destinazione `fit_output_root` sotto WORK: anche i `.pt` restano lì,
circa 2,7 GB per 24 candidati. Ricetta/hash e sorgenti scientifici invariati.
N-2: quota di progetto 20147512 del 04/11 alle 17:04:12 UTC, testo originale
in `results/infra/consolidation_20261104/lfs_quota_work_20261104.txt`, pin nel
piano. Circa 40,42 GiB liberi, nessuna quota inode; rilettura remota alle 17:19
UTC ancora sopra 40 GiB. Non è una prenotazione: gate fresco a runtime.

Owner GPT, namespace CONSOL/FT, submitter e ledger comuni. L'autorizzazione di
Simone comprende i lanci dopo la review; non occorre un secondo consenso.
Nessun monitor automatico. Nessuna modifica alle altre linee o al main remoto.

## Prima tranche: checkout, continuazione e benchmark

Codice: commit che introduce questa nota, pubblicato e verificato su origin e
su `leonardo:refs/heads/deploy/consolidation-20261104`; SHA completo nella
ricevuta di sottomissione. Un unico checkout immutabile
`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/consolidation-20261104`.

| ID logico | Dipendenza afterok | GPU | CPU | RAM | Tempo massimo | GPU-ore massime | Core-ore allocate |
|---|---|---:|---:|---:|---:|---:|---:|
| CONSOL-checkout-20261104 | nessuna | 0 | 1 | 2 GiB | 10 min | 0 | 0,166667 |
| CONSOL-G-30000-20261104 | checkout | 4 | 32 | 480 GiB | 3 h | 12 | 96 |
| CONSOL-G-39237-20261104 | 30000 | 4 | 32 | 480 GiB | 3 h | 12 | 96 |
| CONSOL-G-49046-20261104 | 39237 | 4 | 32 | 480 GiB | 3 h | 12 | 96 |
| FT-BENCH-20261104 | checkout | 1 | 8 | 100 GiB | 30 min | 0,5 | 4 |

Totale prima tranche: **36,5 GPU-ore / 292,166667 core-ore massime**, incluse
quelle allocate nei job GPU. Stima della continuazione 25,3521 GPU-ore;
benchmark non ancora misurato. Checkout a 2 GiB come nel launcher, senza
variazioni al costo CPU. Le tre fasi preservano lo stesso SHA e lo stesso piano.
FT è indipendente dalla continuazione e usa solo il teacher originale a 19.618.

Submitter: `scripts/submit_job.py`, `--agente gpt`, campagna
`consolidation-20261104`, registro predefinito `~/wearusfm_local/jobs/`.
Per checkout passare `scripts/slurm/campaign_checkout.sbatch` su stdin con
`CAMPAIGN=consolidation-20261104,CAMPAIGN_SHA=<SHA>,EXPECTED_SHA=<SHA>`.
Per gli altri job esportare `CODE_DIR=<checkout>,EXPECTED_SHA=<SHA>`;
launcher `consolidation_continue.sbatch STEP` o
`consolidation_finetune.sbatch benchmark OUT`. Specificare QoS normal per GPU
e risorse della tabella; dipendenze tramite `--dopo ID` del submitter.

Benchmark OUT:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/finetuning/benchmark`.
Continuazione: run su SCRATCH, report e quattro copie verificate dei checkpoint
su WORK come nel piano. Prima di ogni invio verificare assenza di duplicati nel
ledger e chiamare sul Mac:

```python
check_quota_evidence(plan['retention']['quota_preflight'], plan['retention'],
                     submission_time=datetime.now(timezone.utc))
```

La verifica usa il codice revisionato, non è delegata al submitter generico.
Ricevuta e ID SLURM si raccolgono dopo ogni invio. Risposta ambigua: riconciliare,
mai ripetere alla cieca. Eventuale errore di fase blocca i successori afterok;
recupero solo secondo C-M1, senza consumare una seconda volta update completati.

## Passi successivi condizionati

Fit: report benchmark completo e hash verificato, stessa ricetta/sorgenti;
pubblicare pin, tempi, memoria, modalità joint/split e preventivo effettivo.
Nuovo checkout per il piano aggiornato, senza toccare quello della continuazione.
Tetti alternativi 48 GPU-ore con joint o fino a 96 con split, non cumulativi.
Se anche il singolo candidato supera 14.100 s o la memoria supera il gate,
fermare i fit del regime coinvolto: nessun cambio automatico della dose.

Matrice lunga e aggregatore OOF/IC non sono inclusi in questa tranche: servono
ancora implementazione, pin e preventivo. La convergenza sulle regole di lettura
non equivale a completamento dei loro controller. Un mancato scatto di F-R2
non dimostra la convergenza di scratch, come precisa N-3 nella rilettura.
