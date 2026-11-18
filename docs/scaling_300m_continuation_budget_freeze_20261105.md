# Tetto condizionale del seguito scaling a 19.618 — 5 novembre 2026

**Versione1 storica, superata prima degli score dalla
[versione2](scaling_300m_continuation_budget_freeze_20261105_v2.md).** La reviewClaude7
ha rilevato MinTRES incompatibile della QoSbprod e limitiCPU insufficienti.
Il JSONv1 resta inalterato per audit; per ogni piano futuro usare la v2.

Fissiamo prima degli score downstream a 5.000 il costo massimo dell'eventuale prosecuzione **300M da 5.000 a 19.618**, confrontata con il 30M già disponibile alla stessa dose. La regola scientifica S8 resta immutata. Questo documento e il [JSON dei tetti](../results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105.json) non rilasciano l'esecuzione e non autorizzano nuovi job.

| Voce aggiuntiva | Tetto GPU-ore | Core-ore allocate nei job GPU | Core-ore solo CPU |
|---|---:|---:|---:|
| Training 300M: altri 14.618 passi |59|472|0|
| Tre estrazioni 300M |7|56|0|
| Validazione, benchmark e lettori |0|0|180|
| Totale della possibile tranche |**66**|**528**|**180**|

Il totale CPU, sommando le due categorie, è708 core-ore. Non comprende nuovamente i costi dei primi5.000 già conclusi. I tetti sono limiti vincolanti: i piani operativi potranno prenotare meno, sulla base di misure tecniche, e dovranno fermarsi se una fase non rientra. Nessun aumento sarà motivato da uno score promettente.

## Calcolo del training e scheduler

Applichiamo la formula S8 già pubblicata mediante `scaling_reading.continuation_training_cap`, senza cambiarla:

```text
t5k =14665,05355668068 secondi, summary del training300M5000
u300=2,7666786608873997 secondi, media dei112 update misurati del collaudo
max(t5k/5000,u300)=2,9330107113361357 secondi/update
ceil_900(14618 × max(...) ×1,20 +1200)=53100 secondi
4 ×53100/3600=59 GPU-ore
```

Il [summary reale](../results/infra/scaling_300m_20261104/scientific5000_completion_20261105/summary.json) ha SHA `f0975c4de0f810b0b5782984e699b50c41c1c67ad930aa9dd0bd650979182af9`. Il [report del collaudo](../results/infra/scaling_300m_20261104/preflight_63526191/report.json) ha SHA `90254bb4bd8aa525730cdd43cec2fa6c98237387754145f0704037e3f5ec8831`; la media usa gli indici zero-based8–63 e72–127. La proiezione senza margine è42874,75 secondi; è una stima, non il tetto. **22 GPU-ore** erano il limite del primo training a5.000 e **49,40 GPU-ore** una vecchia proxy storica: nessuno dei due è il limite di questa prosecuzione.

Si fissa un solo segmento **14:45:00**, un nodo,4GPU,32CPU,480G, partizione `boost_usr_prod`, con **QoS `boost_qos_bprod` esplicita**. La [lettura dello scheduler prima degli score](../results/infra/scaling_300m_20261104/cpu_readers_20261105/scheduler_before_scores.txt), SHA `0848c80da72f527596fe091729b459e8eb3c09a49764a83ec4b9ee1baa604368`, riporta `MaxTime=1-00:00:00`, `MaxWall=1-00:00:00` per quella QoS e la sua presenza nell'associazione `iscrb_wearusfm/sbenatti`. Il segmento rientra nelle due soglie24h osservate; disponibilità e associazione saranno riverificate prima di un eventuale lancio. Non si aggiungono segmenti o overhead oltre53.100 secondi per adattare il piano dopo gli score.

La baseline di memoria è il **picco allocato37,751831GiB** del training scientifico5.000, non la proxy precedente33,52GiB. Il futuro guard dovrà registrare allocato e riservato per ciascuno dei quattro rank, conservando il limite tecnico sull'allocato≤90% del device e verificando riservato≥allocato. Il summary5.000 non contiene il riservato per tutti i rank e non viene presentato come tale. Strumentazione e guard richiedono review prima della prosecuzione.

Il launcher attuale consente soltanto fresh5.000: non si aggira cambiando il JSON. Una futura ripresa richiederà un wrapper revisionato e verifica completa di optimizer/momenti associati ai parametri, teacher, RNG, loader e schedule. Si conservano batch globale32,4 rank, accum1/micro8,6 worker per rank e lo schedule originale49.046/warmup1.962; nessuna nuova dose, loss, architettura o seme. Il tetto non include collaudi GPU aggiuntivi o retry automatici.

## Estrazioni e controllo30M

A19.618 sono già inventariate due cache posa e cinque cache classificazione del controllo30M T100-G, checkpoint `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`. Riferimenti: [piano posa storico](../results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json) e [piano sonde storico](../results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json). **L'inventario locale non certifica la conservazione remota attuale**: prima di un futuro lancio vanno riverificati checkpoint, snapshot, cache, ricevute, hash, supporti e orizzonte di conservazione.

Per300M si conservano i tetti delle estrazioni5.000, tutte con1GPU,8CPU,100G:

| Estrazione | Tempo osservato a5.000 | Tetto a19.618 | GPU-ore |
|---|---:|---:|---:|
| Posa train+val |7053s|150min|2,5|
| P1: tre dataset |1958s|120min|2,0|
| P2: due dataset |5905s|150min|2,5|
| Totale |14916s, circa4,143GPUh|420min|**7,0**|

Accounting e raw `sacct` sono pinnati nel JSON. Questi tempi reali5.000 giustificano i limiti tecnici, ma non sono misure dell'endpoint futuro. S5 riusa le cache e non richiede altre estrazioni. Non sono comprese nuove estrazioni o nuovo training30M: se il controllo manca, la fase si ferma per review, senza recuperi fuori tetto.

## Lettori:180 core-ore come inviluppo, non come gate già passato

Manteniamo le tre viste30M naturale,300M naturale e300M S5, con i supporti e le ricette congelati. Il limite comprende prudentemente anche una nuova lettura del controllo; non si anticipano risparmi dal riuso delle OOF senza validarne provenienza e compatibilità.

| Fase | Core-ore massime |
|---|---:|
| Validate posa,2CPU×30min |1|
| Tre benchmark posa,8CPU×10min |4|
| Tre benchmark sonde, cinque dataset per vista,8CPU×45/45/90min |24|
| Ridge,8CPU×60min |8|
| Fit sonde,8CPU,max120min per shard emax15 shard |136 cumulativi|
| Sei aggregati,2CPU×10min |2|
| Confronto,8CPU×30min |4|
| Checkout2CPU×10min |0,333333|
| Riserva operativa interna |0,666667|
| Totale |**180**|

Memoria massima30GiB per i lettori,4GiB per il checkout;8 thread per fit e confronto, stessa piattaforma seriale `lrd_all_serial/login08` dentro SLURM. Eventuali ulteriori checkout consumano la riserva interna; non si sommano oltre180. La formula ridge resta `ceil60(1,5×[2V+Σ(11S+13E)]+120)`. La formula sonde resta `120+1,5×Σ_dataset[(Σ6Ctempi+maxCtempo)×Σ_fold max(1,ntrain_fold/ntrain0)]`, con RAM almeno1,3×RSS misurato e≤30GiB.

I benchmark train-only5.000 possono informare la fattibilità tecnica, ma **prima dei fit a19.618 servono benchmark sulle cache effettive19.618**. Una rappresentazione diversa può richiedere più iterazioni. Se tempi, memoria o runtime non passano i guard, si ferma la fase: niente riduzioni della griglia, dei fold o delle iterazioni per far passare il preventivo. Restano O-6/O-7/O-8: verifica del runtime reale, accounting con raw pubblicati, controllo cumulativo e stop al primo errore/TIMEOUT, senza retry ciechi.

## Momento del freeze e decisione successiva

Il presente JSON e questo documento devono essere committati e pubblicati, con push verificato, **prima di aprire qualsiasi score downstream5.000**. Come confine operativo si usa l'invio del primo `pose-ridge` o `probe-fit`, che può già produrre score nei propri output: non si aspetta il solo confronto finale. Validate e benchmark train-only possono continuare perché non selezionanoC e non producono score val/held.

Il [criterio S8](../results/infra/scaling_300m_20261104/scientific_reading_policy.json) resta identico, SHA `da58ec3a7965da8db29e97a1089ae1d1d7e652dba4250f24dcb1b8e608602960`. Dopo il confronto stabilirà quale proposta presentare. Questo freeze non rilascia l'esecuzione a19.618: serviranno piano concreto, review e decisione di Simone. Non si chiede alcun nuovo consenso adesso e non si lancia nulla.

Se il freeze non precedesse l'apertura degli score, il protocollo consentirebbe comunque una lettura descrittiva, ma il seguito non potrebbe essere presentato come interamente preregistrato. Non si retrodata la pubblicazione.

Il digest operativo del JSON è `3ab57580a34a9300baac6efa10efe741a619221eecd538200cf217767d653a33`; le dimensioni effettive sono nel riepilogo di consegna.
