# TCN frozen: consegna eseguibile del 19/11/2026

> **Aggiornamento del 20/11/2026.** Dopo il preflight 64931573 fallito c'è una seconda consegna:
> `docs/pose_temponet_execution_20261120.md`. Per i lanci valgono i suoi valori: ID con suffisso `-20261120`, checkout
> `pose_temponet_20261120`, piano in `results/infra/pose_temponet_execution_20261120/plan.json`. Dove questo documento scrive
> `20261119` per ID, checkout, piano o riferimento di deploy, vale il valore nuovo. Il resto resta valido.

## Stato, titolarità e identità

Il protocollo v3 ha la convergenza documentale effettiva di Claude
`ad9f5141711ba60ea1014bd9ba842af168ff0a37`. I tre ritocchi richiesti senza
un nuovo giro sul disegno sono applicati: elenco completo degli allarmi in
decisioni, limiti degli IC nella tabella §10, delta candidato meno comparatore.
Il JSON scientifico rivisto resta byte-identico, sha256
`171b52785f37f80e0a9f1b6194e90e29bd7620c4b7106c117267465bfc6c2906`.

**Questa consegna operativa è REVIEW_REQUIRED.** La review runtime di Claude
`527f8dae716398203f8f2d7413d93494f020e185` ha richiesto le correzioni documentate in
`docs/reviews/pose_temponet_runtime_response_20261119.md`; questa revisione non è ancora
un verdetto favorevole del revisore. Il codice è implementato e
testato sul Mac; benchmark A100, gate sui dati reali e confronti non sono ancora
eseguiti. Le misure locali su fixture non autorizzano stime di tempo su Leonardo.
Il pin della consegna e la ricevuta dei test stanno in
`results/infra/pose_temponet_execution_20261119/`.

Simone ha autorizzato la chiusura della review e il lancio del confronto sul 30M
congelato a 49.046. Nessun secondo consenso per questa catena entro i tetti e i
gate. Il contatto con Claude passa esclusivamente da Simone. L'unico operatore è
**2-TCN head launcher**: questa chat prepara scienza e codice, non crea un secondo
submitter. Le TEMPONet raw e le altre linee sono distinte.

Il pacchetto è isolato in `scripts/pose_temponet/`, con due nuovi launcher sotto
`scripts/slurm/`. Nessun sorgente del FM, default, target, checkpoint o programma
storico è modificato. Il codice non istanzia TEMPONet raw né ne riusa i pesi.

Tre identità hanno ruoli distinti:

1. Produttori storici delle cache: ricevute originali `2930d7d`, codice
   `7d276c98fc54d66dab1226226eb6c5e447422c69`. Le copie byte-identiche sono incluse
   per rendere verificabile il pin anche senza gli oggetti Git di quei rami.
2. Sorgenti del lettore: manifest esplicito di nuovo pacchetto, dipendenze Python
   storiche e launcher; SHA del commit effettivamente rivisto da Claude.
3. Checkout di esecuzione: SHA completo passato come EXPECTED_SHA, registrato
   nella ricevuta. Un discendente che aggiunge soltanto piani/pin/documenti è
   ammesso solo se tutti i sorgenti coincidono col commit rivisto. Ogni modifica
   ai sorgenti invalida il piano e richiede review; nessuna reinterpretazione
   della provenienza delle cache.

## Entry point e artefatti

Da un checkout del pacchetto, con `PYTHONPATH=src:scripts`:

```bash
python -m pose_temponet --help
python -m pose_temponet prepare-plan --output results/infra/pose_temponet_execution_20261119/plan.json
python -m pose_temponet inspect --plan results/infra/pose_temponet_execution_20261119/plan.json
```

`prepare-plan` crea un file nuovo, mai sovrascrive quello esistente. Quello
consegnato è già generato: per riprodurlo usare un percorso temporaneo distinto.
`inspect` valida sorgenti e pin senza accedere ai dati o sottomettere job.
Il piano registra tutti gli input, produttori, ricetta e 24 ID logici/output
univoci. I walltime dei sedici job neurali restano null finché non misurati.

Radice persistente:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117`.
Ricevute in `receipts/<ID>.json`; output di fase in `stage/world/head/fold/`
omettendo i campi non pertinenti. Ogni file completato è scritto atomicamente,
senza sovrascrittura. Un output già presente richiede diagnosi, non una ripresa
automatica. I file `.pt` e `.npz` rimangono su WORK; JSON, manifest, log e
contabilità si raccolgono sul Mac e si committano.

Gli otto input restano quelli della proposta. Nessun file nuovo di EMG e nessuna
estrazione. `pose_stats.json` e i due riferimenti B19618 su SCRATCH restano
prerequisiti con hash: assenza, scadenza o cambiamento fermano la fase.

## Matrice e risorse

| Fase | Job | Risorse per job | Condizione prima dell'invio |
|---|---:|---|---|
| Checkout e preflight R-1 | 1 | 2 CPU, 24 GiB, 12 min | review operativa, quota, ledger e manutenzioni |
| Benchmark CPU | 1 | 8 CPU, 24 GiB, 27 min | preflight COMPLETED 0:0 e R-1 |
| Benchmark GPU | 1 | 1 GPU, 8 CPU, 32 GiB, 27 min | stesso preflight |
| WLS e calibrazione, due direzioni | 2 | 8 CPU, 24 GiB, 3 h 57 min | gate globale risorse e ricevute pinnate |
| Controlli, 2 mondi × 3 teste | 6 | 1 GPU, 8 CPU, 32 GiB, tempo misurato | entrambe le calibrazioni e gate dei riferimenti |
| Selezioni reali, cinque teste | 5 | come sopra | tutti i sei controlli e relativo gate |
| Rifit, tre semi per testa | 5 | come sopra | tutte le cinque selezioni congelate |
| Inferenza bloccata, quindici celle | 1 | 1 GPU, 8 CPU, 32 GiB, 57 min | tutti i quindici rifit verificati |
| Metriche, parità R-1, bootstrap | 1 | 8 CPU, 24 GiB, 57 min | inferenza completa |
| Inventario e raccolta | 1 | 2 CPU, 24 GiB, 12 min | rapporto completo |

Sono **18 job GPU e 6 CPU-only**. Il checkout è incluso nel primo job, non è
una venticinquesima allocazione. I walltime richiesti sono arrotondati per eccesso al minuto, come SLURM;
il confronto dell’allocazione e il preventivo usano gli stessi minuti.
Ogni preventivo conteggia 180 s oltre al walltime
richiesto: i 3 h 57 min seriali diventano 4 h inclusive. Tetti immutati:
24 GPUh, 80 coreh CPU-only, 272 coreh totali; massimo 20 GPUh per controlli,
selezioni e rifit, 1 GPUh inferenza, 64 coreh WLS/calibrazione.

Benchmark e proiezioni comprendono sei LR e 300 epoche, tutte le estensioni,
132 traiettorie CV, 15 rifit e 40.128 valutazioni interne. La griglia realmente
raggiungibile può essere più piccola: il gate conserva il conteggio prudente.
Fattore due su tempi complessivi; p95 da dieci epoche dopo un warmup escluso.
Il benchmark CPU misura dimensione piena L13 e pattern di maschera, non deduce
il costo da una matrice piccola. La scansione delle 1.936 combinazioni per
direzione è inclusa. Se la memoria prevista supera il limite, si registra il
blocco prima di allocare il solver, senza inventare tempi.

Valutazione e training hanno batch 32; l'ultimo batch resta. Storage host FP16
dei latenti e cast FP32 per minibatch. Benchmark solo train: anche l'I/O dei
riferimenti finali è proiettato dai byte pinnati usando la cache train come proxy.
La memoria val si misura con un surrogato train della geometria prescritta.
La lettura I/O misurata è il primo caricamento applicativo del runner, non una seconda
lettura intenzionalmente calda. Lo stato della page cache del sistema operativo non è
controllato: la ricevuta lo dichiara e non chiama questa misura una prova di disco freddo.
La valutazione misurata usa lo stesso `workflow._score_model` dei fit. L'identità del
dataset e lo stato vengono riscaldati prima dei dieci campioni: snapshot in memoria e
scrittura/lettura degli artefatti finali hanno costi separati. Le scritture su disco non
sono moltiplicate per tutte le epoche come se il trainer le effettuasse.
La fase metrics include I/O proiettato, le due valutazioni dei riferimenti R-1 e batch 32.
Due caricamenti della cache finale permettono di trattenere una vista per volta.
Le predizioni salvate sono `[N,148,20]`; il preventivo usa prudentemente 160
istanti e contabilizza anche stati, curve e controlli. Limite persistente 8 GiB.

Un gate negativo ferma globalmente WLS/fit prima degli invii pertinenti. Nessun
aumento automatico, nuovo seme, LR fuori regola o retry. La riserva non autorizza
tentativi aggiuntivi. L'operatore rilegge la corsia seriale prima di ogni fase CPU
e conta le scritture pendenti delle altre linee nella soglia WORK di 12 GiB.
La misura usata da `resource-plan` deve essere successiva al completamento di entrambi
i benchmark, con data, fuso orario e prova originale pinnata. Non si ricicla quella del rilascio.
Prima del rilascio l'operatore confronta anche il modello CPU dei due nodi seriali:
il pin `cpu8` include il modello. Se differiscono, si prepara una regola per classe di nodo
prima di qualsiasi invio dipendente; non si elimina il campo per far passare il confronto.
Nessuna lettura dei nodi è stata fatta da questa consegna locale.

## Sequenza dell'operatore dopo la review

1. Leggere il verdetto effettivo di Claude sul pacchetto, non dedurlo dal verde
   dei test. Registrare il commit originale e il file della review. Preparare
   un JSON di evidenza con `observed_at`, `work_free_bytes`,
   `pending_write_bytes`, `no_duplicate_jobs`, `serial_lane_checked`,
   `reservations_checked`, `balance_checked`, più fonti/log delle letture.
2. Eseguire `release-plan` con piano, commit/path della review, SHA dei sorgenti
   rivisti e quell'evidenza. Il comando verifica i sorgenti del commit, copia
   testualmente la review e rilascia soltanto il perimetro già autorizzato.
   Esempio, con valori reali ricavati dai pin:

   ```bash
   python -m pose_temponet release-plan --plan results/infra/pose_temponet_execution_20261119/plan.json --review-commit "$TCN_REVIEW_SHA" --review-path "$TCN_REVIEW_PATH" --reviewed-code-sha "$TCN_CODE_SHA" --operator-checks "$TCN_OPERATOR_CHECKS" --output "$TCN_RELEASED_PLAN"
   ```

3. Committare e pushare la consegna di esecuzione, verificare SHA remoto. Solo
   l'operatore pubblica il riferimento nuovo `deploy/pose-temponet-20261119`;
   non sposta main né un checkout in uso. Checkout in
   `$WORK/wearusfm_campaigns/pose_temponet_20261119`, detached e non scrivibile.
   Prima di creare il checkout, il wrapper verifica SHA completo del piano, percorso
   relativo sicuro e hash del piano nel commit pubblicato. Solo allora usa
   `campaign_checkout.sbatch` preso dallo SHA congelato. Per gli stadi successivi,
   lo script su stdin verifica `EXPECTED_SHA` ed esegue il launcher nel checkout.
4. `commands` stampa le righe complete di `scripts/submit_job.py`; non le
   esegue. Specificare `--plan`, `--checkout-sha`, `--plan-commit`,
   `--plan-repository-path` e `--stage preflight`. I launcher materializzano il
   piano committato in `manifests/<sha256>.json` e verificano l'hash ricevuto.
   Invocare il submitter una sola volta per ID; registrare la risposta nel
   ledger comune. Nessun `sbatch` diretto.
5. Dopo COMPLETED 0:0 e ricevuta preflight verificata, generare/inviare i due
   benchmark con `--stage cpu_benchmark --stage gpu_benchmark`. Le dipendenze
   afterok restano per i job ancora noti a SLURM.
6. Raccogliere ricevute, log e `sacct -X -n -P --format=JobIDRaw,State,ExitCode`
   dei benchmark; verificare gli hash remoti e acquisire una nuova evidenza operatore
   con `observed_at` successivo ai due `details.finished_at`. Passarla con
   `--operator-checks <JSON>` a `resource-plan`, che legge le copie
   locali, ricalcola il gate completo e lega nel piano i pin delle due ricevute,
   i contesti numerici CPU/GPU e i tempi di tutti i sedici gruppi neurali. Se
   fallisce non pubblicare un piano eseguibile dei fit.
7. Committare/pushare il piano misurato con preventivo e ricevute, verificare
   origin e pubblicare un nuovo riferimento deploy di soli metadati. Lo stesso
   checkout immutabile può materializzare il nuovo piano; source identity
   invariata. Non servono una nuova ricetta o un nuovo checkout.
8. Procedere per le fasi della tabella. Il controller rilegge i predecessori,
   gli artefatti e i gate: una dipendenza SLURM soddisfatta da sola non basta.
   Prima di emettere i job dei controlli, fornire anche i pin delle due `calibration.json`
   nelle prove dei predecessori WLS, tramite campo `calibration`; le copie locali devono
   coincidere con gli artefatti pinnati dalle ricevute. Il gate combinato deve essere diagnostico. Per `commands --stage controls`
   è obbligatorio `--closed-reference-gate-output <NUOVO_JSON_LOCALE>`: il gate con i
   suoi pin viene salvato prima di qualsiasi comando, anche quando fallisce; il file
   non viene sovrascritto. Con gate negativo non viene emessa nessuna riga di invio.
   Nel runtime il gate è salvato e un esito negativo rimane un gate scientifico, non un guasto tecnico.
   Nessuna selezione reale prima di tutti i controlli; nessuna inferenza prima
   di tutte le selezioni e dei quindici pesi congelati.

Per precedenti già completati e rimossi dalla memoria SLURM, `commands` accetta
`--completed-prerequisites <JSON>`: mappa ID → `{receipt: pin, sacct: pin}` delle
copie locali committate. Il codice esige una ricevuta valida della stessa catena
e una sola riga `JobIDRaw|COMPLETED|0:0` nell'accounting, prima di omettere quella
dipendenza dalla riga sbatch. L'assenza da squeue non è prova sufficiente. Nel
piano e nelle ricevute la dipendenza scientifica resta e viene ricontrollata.
Gli ID ancora accodati/in esecuzione mantengono afterok. Un rifiuto esplicito del
submitter si tratta con la sua procedura; uno stato ambiguo si riconcilia, non si
ritenta con un ID nuovo.

## Fallimenti e chiusura

Il controller distingue `gate_failed` da `technical_failure`. Conserva gli
artefatti già scritti e l'errore originario; non genera contrasti con celle
incomplete. TIMEOUT, guasto, non finiti o determinismo non supportato fermano la
fase. Le traiettorie conservano optimizer/RNG per le estensioni interne approvate;
questa capacità non autorizza un rilancio di job o una terza dose.
Gli ID logici, il checkout e la radice degli output restano fissi. Un tentativo aggiuntivo
richiede una decisione e una nuova consegna con ID/checkout nuovi; non è stato introdotto
un meccanismo di retry né il riuso automatico di una destinazione esistente.
Non rinfrescare i file su SCRATCH mentre gira uno stadio TCNH: cambiare date/stato dei file
attiva le guardie. Un rinfresco fra stadi, a contenuti invariati, è compatibile con i pin.

Lettura finale: cinque IC99 appaiati prescritti, tre semi aggregati come
accumulatori e non ensemble, pannelli all/dynamic/static, tutti i giunti,
MAE/livello e compromesso S-2. Gli allarmi reali hanno la direzione prevista;
nessuna promozione automatica del FM o modifica della loss. Il rapporto rende
espliciti anche i diagnostici WLS e i controlli positivi train-only. La condizione assoluta
`E_dyn <= 0.7630564046643697` è verificata e registrata insieme alla candidatura;
la seconda parità R-1, prima dei contrasti, lascia un artefatto esplicito.

Alla chiusura raccogliere report, ricevute, inventario, log, stato unico per ID e
contabilità `sacct -X` per tutte le 24 allocazioni effettivamente eseguite.
Confrontare risorse richieste/consumate coi tetti, verificare i file su WORK,
pubblicare registro nelle due sezioni e decisioni. Nessun monitor in background
viene creato da questa consegna.
