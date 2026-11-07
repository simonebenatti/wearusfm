# T100-LQ: esecuzione dei lettori e dei confronti

## Prerequisiti soddisfatti

Tutte le estrazioni sono COMPLETED 0:0: posa 5.000 (63615952, 1.866 s),
posa 19.618 (63772648, 1.845 s), P1 (63772649, 832 s), P2 (63772650, 1.719 s).
Ricalcolati SHA e byte delle nove cache e delle provenienze; archiviate le ricevute
originali senza perdita. Tutti i job hanno usato 1 GPU, 8 CPU e 100G entro i tetti.
Le quattro estrazioni costano 1,739444 GPU-ore / 13,915556 core-ore, checkout esclusi.
Le tre di oggi costano 1,221111 GPU-ore / 9,768889 core-ore.

Piani effettivi in `results/infra/logsquare_evaluation_20261107/cpu_readout/`.
Modifiche ai piani preparatori limitate a stato, pin effettivi, riferimenti delle
risorse e attestazione della release. Ricette, fold, confronti, fonti e ambiente
numerico restano quelli approvati nella review Claude 6c0ce65.
T100-L è il riferimento primario; T100-G e T25-L sono secondari.

## Risorse e ordine

Manifest operativo: `results/infra/logsquare_evaluation_20261107/cpu_job_manifest.json`.
Sedici job CPU scientifici e due checkout condivisi, tetto complessivo 51 core-ore,
zero GPU. Ogni job scientifico usa al massimo 8 CPU, 30G e 60 minuti.
Le due release di fit posa sono ancora nulle: nessuna ridge prima del benchmark.

| Famiglia | Job | CPU per job | Tetto per job | Checkout |
| --- | ---: | ---: | ---: | --- |
| Posa: verifica, due dosi | 2 | 2 | 30 min | A |
| Posa: benchmark train, due dosi | 2 | 8 | 15 min | A |
| Sonde: db2-01, db2-23, p2 | 3 | 8 | 60 min | A |
| Sonde: db3, db6 | 2 | 8 | 30 min | A |
| Sonde: aggregati P1/P2 | 2 | 2 | 20 min | A |
| Sonde: confronto appaiato | 1 | 2 | 30 min | A |
| Posa: ridge, due dosi | 2 | 8 | 30 min | B |
| Posa: confronto, due dosi | 2 | 2 | 30 min | B |
| Checkout A/B | 2 | 2 | 10 min, 4G | — |

Checkout A `logsquare-cpu-a-20261107`: quattro preparatori posa e cinque shard
indipendenti; gli aggregati dipendono dagli shard pertinenti e il confronto dai
due aggregati. Tutti gli otto job delle sonde mantengono piano e checkout A.
Output nuovi fuori dal codice, sotto
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/logsquare_cpu_20261107/`.

Dopo i quattro preparatori posa COMPLETED 0:0, si pubblicano insieme le due
fit_release, soltanto se ceil_a_60s[1,5 × (2V + 11S + 13E) + 120] ≤ 1.800 s
per ciascuna dose. V viene dall'accounting della verifica, S ed E dal benchmark
train a otto thread. Report e accounting devono avere gli stessi ID e pin.
Checkout B `logsquare-cpu-b-20261107` usa le nuove release. I confronti posa
partono solo dopo i fit, con SHA reale del rapporto candidato. Nessuna release
si applica retroattivamente ai preparatori o ai job sonde.

I gate storici delle sonde restano un proxy: max elapsed/tetto db2-01 0,495,
db2-23 0,470, db3 0,331, db6 0,575, p2 0,452; fit T100-L 13,2578 core-ore.
Con feature nuove LBFGS può richiedere altre iterazioni. Timeout, incongruenze o
superamento del tetto fermano la fase, senza retry o aumenti automatici.
Avvisi di convergenza sospendono l'interpretazione interessata, conservando i numeri.

Autorizzazione operativa di Simone già acquisita. Uso di submitter e ledger comuni,
SHA pubblicato e verificato, checkout nuovi e immutabili; nessun monitor in background.
Il launcher posa include la correzione dell'interprete già verificata con 43 test.
