# Tokenizer: review indipendente della fase CPU a 19.618 passi — 04/11/2026

**Esito favorevole al lancio delle analisi già autorizzate. Nessun difetto
bloccante rilevato nei piani, nei pin raccolti o nei launcher.** La review non
anticipa risultati scientifici e non sostituisce il gate misurato prima della
ridge, né i controlli di completezza/convergenza dei rapporti finali.

Piani verificati:

- posa: `pose_readout_plan_19618.json`, SHA256
  `7a469c694637b870280ce6fcc8f9dcac1181e6b16644bc0042d1f942dc206e49`;
- sonde: `probe_plan_19618.json`, SHA256
  `dac83e0d3206da9f0c9c9cd3290cc1e7430fa40d496ba73bb01915cc0b573010`.

Entrambi sono in `results/infra/tokenizer_evaluation_20261103/`. I rispettivi
validatori scientifici esistenti sono stati eseguiti localmente e passano.
Non sono stati lanciati fit, estrazioni o nuovi test scientifici per questa
review; non è stato modificato il codice scientifico.

## Posa

Verificati tutti i sei pin train/val contro gli SHA256 remoti raccolti,
le sei provenienze originali e le tre ricevute complete disponibili in
`/private/tmp/wearusfm-pose19618-original-receipts/`. Ricalcolati localmente gli
hash dei byte originali delle ricevute e delle provenienze: coincidono con i
pin. I riepiloghi committabili non sono usati come ricevute dai lettori.

Dose 19.618, teacher, checkpoint distinti dei tre bracci e codice di estrazione
`453ce6c496745f2d73aa3007406721586e6dd767` coincidono. Conteggi, firme di target,
metadati e maschere sono comuni. Ricetta, sorgenti, statistiche, riferimenti e
fold utenti sono **identici** al piano a 5.000 passi. I riferimenti storici
fissano soltanto selezione/bersagli, non feature né predizioni.

Il wrapper e l'API mantengono crop6, media K64 in FP16, LayerNorm FP32, ridge
FP64 e selezione alpha esclusivamente sui due fold di train. Validate e
benchmark possono partire in parallelo; il benchmark legge soltanto T25-L/train.
La ridge richiede prima entrambi i report completi e il gate già fissato:

`ceil_60s(1.5 * (2*V + 3*((1+10)*S + 13*E)) + 120) <= 1800`.

V è ElapsedRaw di validate, S il tempo delle statistiche, E il tempo dei tre
eigensistemi; occorrono inoltre identità corrette e otto thread effettivi.
La formula resta una previsione empirica. Nessun job separato di confronto:
la ridge produce già il confronto appaiato dei tre bracci.

## Sonde

Verificata la griglia completa di 15 cache contro ricevute, provenienze e
metadati in `completed_19618/classification/collection_summary.json`, il cui
hash coincide con `probe_verification_19618.json`. Per ogni cella sono coerenti
cache/provenienza/ricevuta, checkpoint, dose, teacher effettivo, finestra e
attivazione. Gli hash NPZ raccolti indipendentemente coincidono con le ricevute.
La review non ha ripetuto una seconda lettura integrale dei 15 NPZ remoti:
ogni fit li rilegge, ricalcola gli hash e controlla finitezza e geometria.

Gli hash dei membri NPY di target, soggetti e lista soggetti coincidono
esattamente con i riferimenti storici e fra i tre bracci. Dtype nativi:
EPN612 `<U9`, gli altri dataset `<i8`. Nessuna conversione numerica delle
etichette testuali. Ricalcolati i ruoli dei fold dalle liste dei soggetti:
coincidono sia con il fingerprint delle provenienze sia con il digest del
piano, che usano funzioni hash differenti. Verificate anche le selezioni P2
interne a partire dalle liste complete dei soggetti.

| Dataset | Finestre | Soggetti |
|---|---:|---:|
| NinaPro DB2 | 38.544 | 32 |
| NinaPro DB3 | 9.834 | 8 |
| NinaPro DB6 | 33.961 | 8 |
| EPN612 | 47.100 | 314 |
| UCI EMG | 364 | 19 |

Il problema di trasporto documentato in `transport_repair.json` riguardava
l'output troncato della raccolta T100-L DB3/DB6. I metadati e l'hash mancanti
sono stati riletti, senza rilanciare estrazioni o rigenerare cache. Il piano
finale contiene tutte le celle e la verifica indipendente di parità passa.

La ricetta resta pool768 FP32 senza PCA, StandardScaler sul solo train del
fold, logistica bilanciata con griglia C congelata e 12.000 iterazioni.
L'aggregazione conserva la firma numerica dei fit a otto thread anche quando
gira su due CPU; richiede tutti i fold e gli audit, verifica selezione C,
identità OOF e copertura dei candidati/refit. Warning o limite di iterazioni
irrisolti invalidano la lettura di non inferiorità, senza eliminare fit o
selezionare bracci. Gli score restano descrittivi in quel caso.

## Risorse e sequenza operativa

Ricalcolata la somma dei 22 job sonde: 15 shard = 96 core-ore, sei aggregati
= 4, confronto = 1, totale **101 core-ore**. Tutti rispettano al massimo
8 CPU, 30G e 60 minuti; DB3/DB6 hanno 30 minuti, aggregati 2 CPU/20 minuti,
confronto 2 CPU/30 minuti. La posa prenota 7 core-ore. Il checkout nuovo comune
`tokenizer-endpointcpu-20261104` usa 2 CPU/4G/10 minuti, contato una sola volta.
Totale della fase **108,333333 core-ore e zero GPU-ore**: 106 già riservate,
2 per il secondo benchmark posa dichiarato aggiuntivo e 0,333333 per checkout.

La stima storica delle sonde è una base prudenziale pubblicata, non una garanzia
di runtime. L'API non ha un benchmark dedicato e non serve duplicare fit:
avviare i cinque shard T25-L pianificati in parallelo è coerente con la matrice.
Leggere identità, audit e timing del primo completato prima degli altri bracci
è un controllo operativo, senza uso degli score; il tempo di uno shard corto
non dimostra il tetto di un altro tipo. Rimangono i tetti per tipo e il divieto
di retry cieco. Rispettare il limite prudenziale di dieci job RUNNING+PENDING,
inclusa la posa, usando il submitter e il ledger comuni.

Prima del lancio resta necessario pubblicare commit/push, attendere l'esito
effettivo del push, verificare il riferimento remoto e il nuovo checkout
congelato. Non riusare né modificare il checkout GPU. Queste sono condizioni
operative già previste, non una nuova richiesta di consenso. Monitor in
background spento; nessuna estensione scientifica autorizzata da questa review.
