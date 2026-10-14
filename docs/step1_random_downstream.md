# Baseline encoder casuale — protocollo congelato

Stato al 14/10/2026: **verificato localmente, lancio autorizzato da Simone** («eh, allora muoviti e completalo»),
dopo la comunicazione del tetto di 3 GPU-ora + 64 core-ora CPU. ID `step1_random_downstream_4000_v1`.
Il checkout CPU aggiuntivo ha il limite descritto sotto; nessun retry o estensione implicita. E' la baseline non negoziabile di v10 §8,
non un nuovo A/B di training e non una scelta automatica del default.

## Unico cambiamento: encoder prima dell'apprendimento

Due inizializzazioni, seed modello 0 e 1, identiche a quelle dei quattro run gia' conclusi.
Architettura completa identica, inclusa la testa keep non usata dal readout; **zero update**, nessun optimizer,
forward di training o EMA. Non caricare state_dict appresi nel nuovo modello.

`initial_teacher` legge dal checkpoint A finale, verificato prima e dopo per SHA, SOLO la configurazione
e quattro buffer immutabili di calibrazione della testa keep. I pesi della testa e dell'encoder sono
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
le quattro sonde sono raggruppate in **due job da quattro ore**, uno P1 e uno P2: stessa sonda e stessi report,
due ore massime per seme tramite GNU timeout. Il job P1 dipende da entrambi gli extract P1; il job P2 dipende
da entrambi gli extract P2 e dal job P1, e il suo ultimo probe esegue anche il confronto. Stesso `submit_job.py` sotto lock,
ID logico distinto per task/seed/stage; nessuna submit diretta fuori meccanismo comune.

| Fase | Risorse/job | Walltime/job | Numero | Tetto |
|---|---|---|---:|---|
| Extract P1 | 1 A100, 8 CPU, 100 GB, boost_qos_lprod | 30 min | 2 | 1 GPUh |
| Extract P2 | 1 A100, 8 CPU, 100 GB, boost_qos_lprod | 1 h | 2 | 2 GPUh |
| Probe raggruppati per task + confronto nell'ultimo | 8 CPU, 30 GB, lrd_all_serial | 4 h (2 h per seed) | 2 | 64 core-ora CPU |

Totale massimo **3 GPUh = 24 ore locali GPU + 64 core-ora CPU**; atteso dalle durate appena misurate
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
