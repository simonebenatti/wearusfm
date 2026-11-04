# Esecuzione delle valutazioni scaling a 5.000 — 4 novembre 2026

Review Claude `a988a72`: approvati lettori, S5, comparatore e S3, nessun blocco.
La memoria reale di S3 è un requisito separato; finché il job 63527446 non termina
con prove valide, il training scientifico non parte. Questa consegna definisce
le operazioni successive senza inventare checkpoint, cache o benchmark futuri.

## Training scientifico: rilascio separato

Il launcher `scripts/slurm/scaling_train.sbatch` accetta soltanto il candidato
fresco T100-G-300M a 5.000: un nodo, quattro GPU, 32 CPU, 480G, 5 h 30 min,
tetto **22 GPU-ore e 176 core-ore**. Il wrapper resta identico a quello di S3
e verifica quattro gate reali: preflight, memoria S3, preparazione completa delle
valutazioni e review legata agli hash finali. Il checkpoint tecnico non inizializza
il run scientifico; non si riduce il batch né si cambia lo schedule.

Prima dell’invio saranno pubblicati piano finale, pin del report S3 completo,
review e argv del submitter comune, poi il nuovo checkout congelato. Non si
compilano hash fittizi per anticipare un gate non ancora disponibile. I JSON
della readiness dei lettori attestano preparazione e inviluppo, non il successo
di S3 né un benchmark sul 300M a 5.000. Nessun monitor automatico.

## Preparazione e misure

Prima del training si congelano codice, matrici S5, popolazioni, formule,
launcher e tetti. Dopo il checkpoint completo si scrivono i piani di estrazione
con i due pin reali. Dopo le cache si scrivono i piani dei lettori, poi si misurano
i costi sul train. Solo il preventivo misurato abilita i fit entro l'inviluppo.
Non è necessario né possibile misurare una cache 300M a 5.000 prima del training.
S5 nel protocollo esplicita questo ordine; metriche e criteri restano invariati.

Tre viste: 30M naturale, 300M naturale, 300M proiettato. Le due cache di posa 30M
già esistenti si riusano. Cinque estrazioni producono 12 nuove cache: due posa e
dieci classificazione. Le cache totali sono 14; S5 non richiede estrazioni ulteriori.

## Estrazioni

Tutti i job usano una GPU, 8 CPU e 100G di RAM host, con teacher a 5.000.
Il controller valida già il checkpoint prima di estrarre: non è previsto un job
GPU aggiuntivo di sola validazione.

| Job | Durata massima | GPU-ore |
|---|---:|---:|
| Posa 300M, train e val | 150 min | 2,50 |
| P1 300M | 120 min | 2,00 |
| P2 300M | 150 min | 2,50 |
| P1 30M | 30 min | 0,50 |
| P2 30M | 45 min | 0,75 |
| Totale | | **8,25** |

Le CPU allocate con GPU ammontano a **66 core-ore**; sono separate dalle 180
core-ore dei lettori CPU. Lo storico T100-G richiese 28,82/14,92/28,13 minuti
per posa/P1/P2. Moltiplicare per il rapporto di training 3,90856 dà soltanto
una proxy: non è una misura dell'inferenza 300M. S3 non certifica i picchi di
questi estrattori: posa usa FP32/no_grad, classificazione bf16/no_grad. Si raccolgono
tempi e memoria reali; un errore ferma la fase, senza ridurre batch o precisione
e senza retry cieco. Fonti: `completed_19618/extraction_resources.json` sotto
`results/infra/tokenizer_evaluation_20261103/` e verifica del preflight 63526191.

## CPU: tetto completo prima del training

| Fase | Allocazione e durata massima | Core-ore |
|---|---|---:|
| Checkout comune, se necessario | 2 CPU, 4G, 10 min | 0,333333 |
| Validate delle quattro cache posa | 2 CPU, 30G, 30 min | 1 |
| Tre benchmark posa | ciascuno 8 CPU, 30G, 10 min | 4 |
| Quindici benchmark delle sonde | tre job di 8 CPU, 30G, 45/45/90 min | 24 |
| Ridge delle tre viste | 8 CPU, 30G, 60 min | 8 |
| Fit delle sonde | 8 CPU, 30G; ogni shard fino a 120 min | 136 cumulativi |
| Sei aggregati P1/P2 | ciascuno 2 CPU, 30G, 10 min | 2 |
| Confronto completo | 8 CPU, 30G, 30 min | 4 |
| Riserva operativa non prenotata | uso documentato, sempre entro il totale | 0,666667 |
| Totale | zero GPU | **180** |

Il confronto mantiene 30 minuti e otto thread come la ridge: non si riduce il
walltime a un quarto per compensare le CPU. Le tre core-ore aggiuntive usano
la riserva interna; il tetto totale resta 180, senza nuovi fit.

I benchmark delle sonde eseguono i sei C, senza score di validazione, per cinque
dataset e tre viste: 90 fit tecnici. I fit scientifici restano 420, con ricetta
invariata. Il raggruppamento è di cinque benchmark sequenziali per cella, con
report separati; 45 min per 30M naturale e 300M proiettato, 90 per 300M naturale.

Il precedente massimo provvisorio di dieci shard è sostituito da **quindici**,
sempre entro 136 core-ore e 120 minuti per shard. Non aggiunge fit: permette di
suddividere lo stesso lavoro. La proxy degli audit storici suggerisce 13 shard
e circa 130,53 core-ore con la nuova formula prudente; la fattibilità non è
garantita, perché larghezza e condizionamento possono cambiare le iterazioni.
Fonte: consuntivo e audit T100-G in `probe_cpu_19618/` sotto la cartella tokenizer.

Per ogni shard delle sonde si applica la formula già implementata:

`120 + 1,5 × somma_dataset[(somma dei sei tempi C + massimo tempo C) × somma_fold max(1, n_train_fold/n_train_fold0)]`.

La prenotazione arrotondata non scende sotto la stima; RAM almeno 1,3 volte RSS
del benchmark e non oltre 30 GiB. Si controllano sia ogni shard sia la somma di
tutte le prenotazioni. Se non rientra, stop prima dei fit.

Per la ridge si usa:

`ceil_60(1,5 × [2V + somma_3celle(11S_i + 13E_i)] + 120) <= 3600 secondi`.

V è ElapsedRaw verificato del validate completo; S ed E sono tempi delle statistiche
e dei tre eigensistemi nei benchmark reali sul train. Il fattore 11 comprende
train e costo prudenziale della val (18.548 contro 1.907 finestre); la formula
mantiene il margine storico e conta separatamente la proiezione. Si registra anche
RSS dalla contabilità SLURM del validate e dei tre benchmark, ciascuna pinnata
e associata al job completo 0:0 e alla sua allocazione. Questi picchi misurano
il benchmark, non garantiscono il picco successivo di fit/predizione.
Gate e ricevute devono corrispondere allo stesso piano,
codice, matrici e runtime. Nessuna scelta scientifica usa questi tempi.

## Vincoli operativi dopo la review

- F-1 corretto: il comparatore ignora solo `states` nel confronto fra ispezioni
  di fasi diverse. Hash, supporti e controlli contro modifiche durante una fase
  rimangono obbligatori. Quattro regressioni coprono touch, copia identica,
  corruzione dei byte e alterazione del supporto.
- F-2: lettori e confronto eseguiti nella stessa partizione seriale, sul nodo
  `login08`, dentro allocazioni SLURM; modulo cineca-ai/4.3.0 e venv storico.
  Fit e confronto usano otto thread. Il nome del nodo non autorizza calcolo
  tramite SSH: l'esecuzione avviene solo tramite sbatch. Un bootstrap comune
  precarica i moduli del comparatore in ogni processo CPU, quindi verifica
  il runtime corrente rispetto alla ricevuta della posa prima del confronto.
  La review del launcher ha riprodotto sul Mac un pool OpenMP aggiuntivo
  caricato dalle sonde: uniformare solo nodo e versioni non sarebbe sufficiente.
  Il precaricamento uniforma le librerie senza modificare i sei valutatori.
- F-3/F-4: congelati fino al lancio dei 5.000 i tre sorgenti S3/training e i sei
  valutatori elencati in `scientific_runtime_freeze.json`, oltre ai 103 sorgenti
  del contratto. Generatore e matrici S5 invariati; nessuna rigenerazione.
- F-5 differito: il preventivo comprende il costo delle due letture 300M;
  non si cambia adesso il lettore già approvato per risparmiare I/O.

I piani e i gate prodotti più tardi sono JSON pinnati, pubblicati e leggibili
da checkout successivi; il codice dei lettori resta in un unico checkout frozen.
Questo consente di usare benchmark, shard e manifest del confronto senza editare
un checkout attivo né mescolare le revisioni di esecuzione dei lettori.

Il controllo 30M e i suoi snapshot sono stati rihashati il 4 novembre: checkpoint
`49dd518b…daa80`, 454.381.728 byte; config/ricevuta/summary coincidono con i pin.
Orizzonte operativo per la prima catena: **11 novembre 2026, 00:00 UTC**; limite
conservativo dello scratch: 13 dicembre, come nel contratto. Il controllo viene
riverificato al lancio; un rinvio oltre l'orizzonte richiede aggiornare la verifica
o conservare una copia pinnata, senza usarlo se manca.

Rettifica materiale della review Claude §2.3: **22 GPU-ore sono il tetto del primo
training a 5.000**, ricavato dal preflight. Il costo della tranche aggiuntiva a
19.618 si calcola con la formula già fissata dopo aver misurato il training 300M
a 5.000; non è di 22 GPU-ore e non è incluso in questa autorizzazione.


## Interfacce dei launcher pubblicati

- Training: `scaling_train.sbatch <piano relativo al checkout> <ricevute nuove>`.
  Risorse fisse 4 GPU/32 CPU/480G/05:30:00; nessuna opzione di ricetta.
- Estrazione: `scaling_evaluation.sbatch <piano relativo> <braccio> <fase> <none|p1|p2> <output nuovo>`.
  Si imposta `--time` esattamente alla durata della cella; le sole cinque celle
  ammesse sono quelle della tabella, tutte a 5.000.
- CPU: `scaling_readers.sbatch <fase> <piano assoluto> <SHA256> <output nuovo> [opzioni]`.
  Piano e budget possono risiedere in un checkout successivo pubblicato; il
  codice dei lettori e il suo `EXPECTED_SHA` restano unici per tutta la catena.
  Ogni invio usa override espliciti di CPU e durata secondo la tabella.

| Fase CPU | Opzioni ammesse |
|---|---|
| `pose-validate` | nessuna |
| `pose-benchmark` | `--cell` |
| `pose-ridge` | `--pose-budget`, `--pose-budget-sha256` |
| `probe-benchmark` | `--cell`; cinque dataset in sequenza |
| `probe-fit` | `--cell`, `--task`, `--datasets`, `--folds`, `--resource-plan`, `--resource-sha256` |
| `probe-aggregate` | `--cell`, `--task`, `--shards` |
| `compare` | nessuna; il primo JSON è il manifest degli input |

Il gate posa contiene `validation_accounting` e `benchmark_accounting` per le
tre viste, con pin di JSON derivati da sacct. Ogni ricevuta associa `JobIDRaw`,
`State`, `ExitCode`, `ElapsedRaw`, `AllocCPUS`, `MaxRSS_bytes` al report corretto.
I benchmark non possono superare 600 s e i picchi registrati devono essere
positivi e non oltre 30 GiB. Il submitter/ledger impedisce invii duplicati;
il controllo delle allocazioni nel launcher non sostituisce quel registro.
