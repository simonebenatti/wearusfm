# Baseline encoder casuale — protocollo congelato

Stato al 14/10/2026: **verificato localmente, lancio autorizzato da Simone** («eh, allora muoviti e completalo»),
dopo la comunicazione del tetto di 3 GPU-ora + 64 core-ora CPU. ID `step1_random_downstream_4000_v1`.
Il checkout CPU aggiuntivo ha il limite descritto sotto; nessun retry o estensione implicita. E' la baseline non negoziabile di v10 §8,
non un nuovo A/B di training e non una scelta automatica del default.

## Unico cambiamento: encoder prima dell'apprendimento

Due inizializzazioni, seed modello 0 e 1, identiche a quelle dei quattro run gia' conclusi.
Architettura completa identica, inclusa la testa keep non usata dal readout; **zero update**, nessun optimizer,
forward di training o EMA. Non caricare state_dict appresi nel nuovo modello.

`initial_teacher` legge dal checkpoint A finale, verificato prima e dopo per SHA, SOLO la configurazione,
quattro buffer immutabili di calibrazione della testa keep e il guadagno fisso `tokenizer.fe_gain`.
Quest'ultimo viene calcolato una volta alla costruzione, non e' un parametro ottimizzato e l'EMA lo copia senza modificarlo.
I pesi della testa e dell'encoder sono
ricreati dal seed; il buffer non entra nel pooling P1/P2. Il controllo SHA sull'INTERO stato ricostruito
dimostra che tutti i valori coincidono con il teacher prima del primo update, compresi quei buffer:

| Seed | SHA256 teacher iniziale, da pairing.json originale |
|---|---|
| 0 | `fe4cfe34930e4f0836ab8107bf203f0e0ce95db8cfa0cdf1aaa239bcf9dacdb0` |
| 1 | `f02883fd0cc0ac8d744ee7ae9036b7b1346c16c686f08a6de3a714c1d3570446` |

I pairing A/B sono identici per seed, recuperati e verificati sul cluster. Checkpoint finali riferimento:
quelli congelati in `scripts/step1_downstream.py`. Il nuovo `.pt` e' un artefatto di sola inferenza,
marcato `untrained_random_encoder`, privo di student/optimizer per impedirne il resume come training.
CPU RNG ripristinato dopo costruzione; CUDA RNG non toccato.
La costruzione avviene con un thread CPU, come `step1_ab_train.sbatch` storico: il guadagno fisso del tokenizer e' sensibile
all'ambiente numerico oltre che all'ordine delle riduzioni float64. Il guadagno viene ripristinato dal valore storico verificato,
non ricalcolato con una tolleranza. Si ripristina poi il numero di thread precedente per l'estrazione. Lo SHA completo resta obbligatorio.

## Estrazione e sonde

Riusare **senza modificarli** `probe_p1.py`, `probe_p2.py`, filtro, scale, pooling QC-aware e linear probe.
Il wrapper verifica che file dell'encoder/sonde e AST di `_filter`/`pool_p1p2` siano identici al commit
storico `8dad7e82713ce4ed2741acc9c9d4fd9ac656d2f7`. Modifiche successive alle loss non sono esercitate.

- P1: DB2/3/6, split firmato, finestre 1 s, validation 10%, weighting soggetti, seed sonda 0.
- P2: EPN-612/UCI-EMG completi, tre split soggetto 0/1/2, seed sonda 0; nessuna riduzione opportunistica.
- Batch estrazione 64, CUDA bf16 come la campagna precedente; feature 768, stesse scale/crop/ordine/QC.
- Ogni task/seed ha una directory nuova fuori dal codice, checkpoint iniziale e provenienza propri.
  Report/cache/checkpoint protetti per hash prima e dopo ogni fase; nessuna sovrascrittura o retry.

Confrontare Random→A e Random→B con pesi P1 8:3:2 e P2 1:1, conteggi pooled per classe/soggetto,
bootstrap accoppiato 1.000 draw, seed 20261009, stesso trattamento degli split P2 sovrapposti.
Gli hash dei report appresi devono coincidere con la comparison originale
`7c93c1b700b7462ce46eeb27a3de65abd38d95caef38b6dc03c76520957527c1`.
Hash dei veri input e inventari etichette/soggetti devono essere identici al ramo appreso corrispondente.
Riportare aggregati, dataset, split e delta per seed; intervalli condizionati ai due seed, non stima completa
dell'incertezza di training. **Nessuna nuova soglia di adozione** scelta dopo aver visto A/B.

Successo tecnico: stato iniziale esatto, input accoppiati, feature/report finiti e provenienza concorde.
Se fallisce, STOP e analisi; niente rilancio automatico. Warning del solver registrati come nell'A/B,
senza tuning su test. UCI-EMG e DB3 non esclusi.

## Risorse e dipendenze autorizzate

Quattro extract (P1/P2 × seed0/1) e quattro probe CPU. Dal14/10, per il limite di dieci sottomissioni CPU per utente,
le quattro sonde sono raggruppate in **due job**, uno P1 e uno P2 (cap r3: 3 h 55 min): stessa sonda e stessi report,
due ore massime per seme tramite GNU timeout. Il job P1 dipende da entrambi gli extract P1; il job P2 dipende
da entrambi gli extract P2 e dal job P1, e il suo ultimo probe esegue anche il confronto. Stesso `submit_job.py` sotto lock,
ID logico distinto per task/seed/stage; nessuna submit diretta fuori meccanismo comune.

| Fase | Risorse/job | Walltime/job | Numero | Tetto |
|---|---|---|---:|---|
| Extract P1 | 1 A100, 8 CPU, 100 GB, boost_qos_lprod | 28 min (override r3) | 2 | 0,9333 GPUh |
| Extract P2 | 1 A100, 8 CPU, 100 GB, boost_qos_lprod | 1 h | 2 | 2 GPUh |
| Probe raggruppati per task + confronto nell'ultimo | 8 CPU, 30 GB, lrd_all_serial | 3 h 55 min (timeout 2 h per seed invariato) | 2 | 62,6667 core-ora CPU |

Tetto complessivo **3 GPUh = 24 ore locali GPU + 64 core-ora CPU**. Primo tentativo del14/10 fermato correttamente dal gate
iniziale, senza feature:65s totali =0,018056 GPUh. Dopo la correzione dell'ambiente di costruzione, cap nuovi extract
2 x29min +2 x60min =2,9667 GPUh, cumulato massimo2,9847 GPUh (entro3). Atteso dalle durate appena misurate
circa 1,86 GPUh + 24,27 core-ora CPU, escluso eventuale overhead di ricostruzione/hash.
Ultima stima locale del budget passo 6 dopo A/B downstream (12/10): ~33,08/100 GPUh; residuo ~66,92,
minimo ~63,92 dopo questo cap. Non e' un consuntivo aggiornato: prima del lancio ricontrollare il registro comune.
Budget infrastruttura 25 GPUh separato e non utilizzato dalla baseline.

Codice da un nuovo checkout congelato `CODE_DIR`, full SHA pulito e read-only. I due launcher sono
`step1_random_extract.sbatch` e `step1_random_probe.sbatch`; il raggruppamento usa `step1_random_probe_group.sbatch`.
P2 extract richiede override walltime 1 h. I due gruppi CPU piu' gli otto job2A rispettano il limite di dieci;
prima di accodarli verificare che il checkout e gli eventuali altri job CPU non occupino ancora uno slot.
Se occorre creare un checkout di campagna: prima job CPU `campaign_checkout.sbatch`, 1 CPU/2 GB/10 min,
**ulteriore cap 0,1667 core-ora CPU**, da includere nel preventivo autorizzato (totale CPU 64,1667).
Prima del deploy verificare hook/ref, disponibilita' dati/quote/scratch e i limiti QoS effettivi tramite OPS.
ID, stati e consuntivi del lancio sono registrati separatamente in `docs/claude_GPT_shared` e nel registro comune delle sottomissioni.

## Verifica prima del lancio (14/10/2026)

29 test CPU passati: `test_random_encoder`, `test_downstream_ab`, `test_p1_cross_subject`, `test_probe_p2`.
Sintassi dei due launcher valida; review indipendente senza blocker. Il confronto byte/AST con il codice storico passa.
Nessuna misura downstream dell'encoder casuale disponibile al congelamento del protocollo. Non sono state aggiunte soglie
o varianti dopo aver guardato risultati; il confronto resta descrittivo Random→A e Random→B.

### Correzione della ricostruzione (14/10/2026)

I quattro primi extract sono falliti prima delle feature per SHA iniziale diverso: il launcher usava otto thread, contro uno nel training
storico. Riprodotto sulla CPU per entrambi i semi: l'unica differenza e' `tokenizer.fe_gain` (circa2e-12), tutti i parametri identici.
Ripristinato il contesto di costruzione storico; poi thread e RNG vengono ripristinati. Nessun parametro appreso caricato, nessuna
tolleranza o sostituzione degli SHA. Nuovo ref/checkout/output `step1-random-20261014-r2` / `step1_random_20261014_r2`;
verifica locale della correzione prima della sottomissione, cap ridotti per rispettare il budget cumulativo. Nessun ritocco scientifico.

### Diagnosi residua e ricostruzione del buffer storico (14/10/2026)

Anche r2 e' fallito: il solo thread CPU non ricostruiva il buffer sul cluster. La diagnostica di Claude60561493, checkout977e767,
ha confrontato cinque processi nuovi per ciascuno dei due semi. Le varianti `ab_replica` e `gpt_like_env8_sklearn_first` ritrovano
esattamente i quattro SHA iniziali A/B student/teacher; le altre no. L'UNICO tensore diverso e' `tokenizer.fe_gain`, non un peso appreso.
Runtime Torch2.2.0a0+git6c8c5ad, Xeon8358. Non si attribuisce il risultato al solo numero di thread o alla CPU login/Booster.

La correzione recupera solo quel buffer immutabile dal checkpoint di riferimento protetto per hash, oltre alle quattro statistiche
keep gia' previste. Il modello continua a ricreare OGNI parametro dal seed e a verificare lo SHA dell'INTERO stato prima delle feature.
Schema del buffer (forma/dtype), finitezza e positivita' obbligatori; drift del valore o di qualunque parametro non puo' superare lo SHA
iniziale. Nessun caricamento di state_dict appreso, cambio di SHA, tolleranza o cambio dell'esperimento/probe.
Test nuovi: entrambi i semi con arrotondamento simulato diverso, nessun parametro appreso importato; rifiuto di cinque tipi di drift.
La correzione locale non equivale ancora al passaggio del gate nei nuovi extract reali: quello si registra separatamente.

### Tranche r3 dopo diagnosi e correzione verificata (14/10/2026)

Prosecuzione del controllo gia' autorizzato, richiesta corrente di Simone: «diagnosi della baseline (2) e review del trainer/monitor
(3-4). PROCEDI!». Nessuna nuova analisi/sonda o cambio di soglia. Checkout/output NUOVI `step1-random-20261014-r3` /
`step1_random_20261014_r3`; codice dal commit che pubblica questa correzione. Ledger e preflight freschi obbligatori, nessun retry cieco.

Consuntivi prima del r3: r1 65 s, r2 54 s = 119 s = 0,033056 GPUh. Diagnosi separata di Claude, autorizzata tramite lui: 48 s =
0,013333 GPUh. Contandola conservativamente nello stesso cap: 167 s gia' consumati. Quattro nuovi extract con i cap della tabella
(2x28min +2x60min = 176min) portano il massimo cumulativo a **2,979722 GPUh**, entro3; nessun aumento del budget richiesto.
Le quattro sonde restano identiche, raggruppate in due job3h55 (timeout per seme7200s invariato): massimo62,6667coreh; checkoutCPU
nuovo, cap reale2CPUx10min=0,3333coreh, piu' piccoli consuntivi precedenti: totale inferiore64coreh. Riserva di coda/timeout, non tuning.

Si sottomettono sotto lock il checkout, quattro extract e due gruppi afterok; P1group dopo i due extractP1, P2group dopo i due extractP2
e P1group, ultimo probe con confronto. Una risposta ambigua si riconcilia. Il gate completo e la nuova provenienza vengono verificati
prima di dichiarare la baseline tecnicamente valida; sonde/risultati non si dichiarano finiti dalla sola sottomissione.
