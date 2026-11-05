# T100-LQ: rilascio e training dopo review 6c0ce65

**Aggiornamento05/11:** 63532335 FAILED1:0 prima del training (21s);
63532370 CANCELLED senza avvio. Nessun checkpoint. I dettagli sotto sugli
invii04/11 sono storici. Diagnosi proposta e seguito in
`docs/logsquare_initialization_failure_20261105.md`; nessun rilancio.

Claude approva gli adattatori di c8f5c12, i gate di risorse e i prerequisiti
per il rilascio. Review archiviata testualmente in
`docs/reviews/claude_review_tokenizer_logsquare_adapters_20261104.md`.
L'autorizzazione operativa di Simone copre già la catena: GPT è l'unico
submitter della linea tramite `scripts/submit_job.py` e ledger comune, come
nel collaudo precedente. Nessun consenso aggiuntivo, nessun monitor automatico.

## Ordine dei commit e identità esecutiva

1. Registrare review, note A1–A5 e margini storici, senza cambiare ricetta,
   controller o sorgenti scientifici. Aggiornare lo stato descrittivo degli
   adattatori; il training rimane ancora `REVIEW_REQUIRED` in questa fase.
2. Commit dedicato minimale: solo stato `APPROVED_FOR_TRAINING` e pin
   `training_release` nel piano, più il JSON di release. Quest'ultimo attesta
   approvazione, epsilon 0,001, hash correnti di report/analisi/preventivo,
   controller/launcher invariati e review Claude pinnata.
3. A4, commit distinto: aggiornare prima `extraction_plan.training_plan`
   (hash e byte) e poi `inference_plan.extraction_plan`. Il discendente
   pubblicato, con questi pin aggiornati, è lo SHA esecutivo congelato;
   registrare separatamente lo SHA del commit minimale. Nessun cambiamento
   ai 102 sorgenti scientifici o ai 110 sorgenti di estrazione/inferenza.
4. Pubblicare e verificare su origin e su nuova ref deploy Leonardo. Creare
   il checkout tramite il job CPU versionato; verificare COMPLETED 0:0,
   SHA, pulizia e sola lettura prima del training.

Entrambe le fasi di training devono usare **lo stesso SHA esecutivo e gli
stessi byte del piano training**. I futuri commit con pin di checkpoint e
cache servono checkout di valutazione distinti; non aggiornano quello del
training e non cambiano la continuazione.

Commit dedicato di rilascio: **79546926886e566a14fab62abd3fd8272d857cb4**,
due soli file (stato/pin del piano e JSON release). Questa successiva consegna
sincronizza i pin A4; il suo SHA completo pubblicato sarà registrato nelle
ricevute di checkout e lancio come SHA esecutivo comune alle due fasi.
Audit: `results/infra/logsquare_training_20261104/release_validation.json`.

## Job di questa tranche

| ID logico | Risorse | Walltime esplicito | Tetto |
| --- | --- | --- | --- |
| LOGSQ-training-checkout-20261104 | seriale, 2 CPU, 4G | 00:10:00 | 0,3333 coreh, zero GPU |
| LOGSQ-T100-LQ-5000-20261104 | 1 nodo, 4 GPU, 32 CPU, 480G | 01:30:00 | 6 GPUh / 48 coreh |
| LOGSQ-T100-LQ-19618-20261104 | 1 nodo, 4 GPU, 32 CPU, 480G | 04:00:00 | 16 GPUh / 128 coreh |

Partizione training `boost_usr_prod`. Il secondo training ha `afterok` sul
primo ed esegue anche i controlli del controller sulla ricevuta e sul proprio
checkpoint completo. Non dipende dal benchmark o dalle estrazioni.
Stima training 16,7052 GPUh; tetto di questa tranche 22 GPUh e 176,3333 coreh
incluso checkout. I due time-limit interni lasciano 300 s di riserva.

Campagna e ref deploy nuove: `logsquare-training-20261104` e
`refs/heads/deploy/logsquare-training-20261104`.
Checkout: `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/logsquare-training-20261104`.
Piano: `results/infra/logsquare_training_20261104/training_plan.json`.
Run: `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/logsquare_20261104/T100-LQ`.
Ricevute: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/logsquare_training_20261104/T100-LQ/{5000,19618}`.

Prima del deploy: ledger senza questi ID; percorsi nuovi assenti; repo remoto
su main, nuova ref non esistente, nessun hook attivo. Evidenza iniziale in
`results/infra/logsquare_training_20261104/prelaunch_evidence.json`.
Non usare `sbatch` diretto, non modificare il checkout canonico o campagne
attive. Risposta ambigua: riconciliazione del submitter, mai duplicazione.
Un time_limit nella continuazione richiede revisione: nessun rilancio o
arretramento di checkpoint. Conservazione su SCRATCH e copia verificata
in WORK prima della purga, se necessaria, come N3.

## A1–A4: benchmark ed estrazioni successive

**A1 già controllata in sola lettura:** cache storica T100-L a 5.000,
SHA256 `c0bc0575d1d900f6e5d0d6d412073fab74fe30d6783d75fe5fb83d1d0e21f32a`;
meta di 1.907 finestre, registrazione 0 = **10**, registrazione 1 = **10**.
Selezione confermata, nessuna riselezione. Evidenza con hash del file e del
membro `meta_json.npy` archiviata; ricontrollare identità prima del benchmark.

Al checkpoint 5.000 completo: pin effettivi, nuovo checkout di valutazione,
benchmark train-only già approvato (0,5 GPUh / 4 coreh). Dopo il benchmark
pubblicare rho_infer, stime e tetti, prima di qualunque estrazione. La
continuazione a 19.618 non aspetta. I pin futuri restano nulli e non eseguibili
fino alla raccolta; nessun benchmark o estrazione viene sottomesso ora.

**A2:** il submitter passa `--time` uguale al tetto misurato di ciascuna cella
di estrazione, senza affidarsi al default 00:45:00 del launcher.
**A3:** tutti i 110 sorgenti elencati in `source_sha256` sono congelati dalla
ricevuta del benchmark. Una modifica successiva anche a un adattatore o
launcher presente nell'inventario invalida la ricevuta e richiede un nuovo
benchmark; nessuna sostituzione silenziosa degli hash. Aggiornamenti dei soli
pin futuri e delle allocazioni seguono il contesto stabile già implementato.
**A4:** sincronizzazione dei pin nel punto 3 sopra; il contesto del benchmark
si fissa dopo il rilascio, senza dipendenze circolari.

## A5: margini delle sonde, accettati da Claude

Fonte pinnata: `results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/resources_fit_aggregate.json`,
SHA256 `2b92e1fdbadd6ba2a86e32f0fbaccb295f1b3bf83fce7f02f4b9251f72da18fc`.
Massimi sui tre bracci storici, tutti COMPLETED 0:0:

| Shard | Elapsed massimo (s) | Tetto (s) | Rapporto esatto o arrotondato a sei cifre |
| --- | ---: | ---: | ---: |
| db2-01 | 1.782 | 3.600 | 0,495000 |
| db2-23 | 1.691 | 3.600 | 0,469722 |
| db3 | 596 | 1.800 | 0,331111 |
| db6 | 1.035 | 1.800 | 0,575000 |
| p2 | 1.628 | 3.600 | 0,452222 |

T100-L: 13,257778 coreh di fit; tetto delle sonde candidate, aggregazione e
confronto inclusi: 34,333333 coreh. RSS massimo storico 3,113056 GiB.
Contesto numerico `07b776ba…` e conteggi dei fold identici a T100-L sono già
richiesti dal runtime. Questi sono margini storici, non una garanzia: le
feature LQ possono cambiare le iterazioni di lbfgs.

Se uno shard raggiunge il walltime, la fase si ferma e si rivede il piano,
senza rilancio né aumento del tetto. Le eccezioni Python scrivono già
`failed.json`; un kill di SLURM può impedire quella scrittura. In tale caso
GPT archivia un `failed.json` **nella raccolta locale dei risultati**, con ID,
accounting `TIMEOUT`/exit code e provenienza esplicita del collector, senza
attribuirlo al processo terminato o modificare gli output originali. `sacct`
resta l'evidenza autorevole e non si dichiara completato lo shard.

Non si cambia alcuna loss, dose, seme, soglia o architettura. T100-LQ resta
lineare + log(0,001+x²), controllo principale T100-L a dose uguale e secondari
T25-L/T100-G. La contrastiva non appartiene a questa tranche.

## Invii effettivi

Codice esecutivo congelato **b2934b07f296778eefe97d4bdb8ca08f2b4e29be**,
pubblicato e verificato su origin e nuova ref deploy Leonardo. Release
minimale **7954692**. Audit indipendente favorevole: minimalità della release,
pin effettivi, controller reale e tutti i 102/110 sorgenti invariati.

| Job | Fase | Stato alla verifica del 04/11, circa 21:27 CET |
| --- | --- | --- |
| 63531256 | Checkout | COMPLETED 0:0, 22 s; SHA, pulizia e sola lettura verificati |
| 63532335 | T100-LQ a 5.000 | PENDING, Priority; 4 GPU, 32 CPU, 480G, 01:30:00 |
| 63532370 | Continuazione a 19.618 | PENDING, afterok:63532335; 4 GPU, 32 CPU, 480G, 04:00:00 |

La continuazione è già accodata e non richiede un invio successivo. La
ricevuta di checkout e le prenotazioni del submitter sono archiviate in
`results/infra/logsquare_training_20261104/`; scontrol conferma tempi,
risorse e dipendenza. Costo checkout reale **0,012222 coreh**, zero GPU.
Nessun tempo GPU consumato all'osservazione; 22 GPUh è il tetto prenotato.
Non confondere sottomissione con training in esecuzione o risultato.

Il primo tentativo locale di invio di checkout/5.000 è stato fermato dal
sandbox prima di creare la prenotazione o chiamare sbatch. Il submitter è
stato poi eseguito con accesso tecnico al ledger condiviso: un solo ID SLURM
per fase, nessun duplicato, nessun retry di un job scientifico.

Prossimo controllo: sacct e log dei due ID esistenti; nessuna risottomissione.
Dopo 5.000 COMPLETED 0:0, verificare ricevuta e checkpoint nominato, pin e
diagnostica W_log; preparare il benchmark sul checkout di valutazione nuovo.
Il training continua sul checkout b2934b0 indipendentemente. Nessuna
automazione in background attivata.
