# T100-LQ: seguito CPU delle valutazioni finali

Nessun nuovo esperimento: lettori e confronti già approvati da Claude in 6c0ce65.
Input necessari: tutte le nove cache verificate, due di posa a 5.000, due di posa
a 19.618 e cinque di classificazione a 19.618. Si preservano i pin originari di
estrazione e i controlli storici; nessun refit o nuova estrazione dei controlli.
I tre job finali 63772648/63772649/63772650 sono partiti il 07/11 alle 13:28:03 CET.

## Posa, separata per dose

Creare due piani effettivi dai piani preparatori pose_readout_plan_5000 e _19618,
con nove campi reali per ogni cache train/val: path, sha256, bytes, provenance_path,
provenance_sha256, receipt_path, receipt_sha256, checkpoint_sha256, extraction_code_sha.
Stato APPROVED_FOR_EVALUATION; fit_release nullo per validate e benchmark.

| Stadio per dose | CPU | Memoria | Walltime | Core-ore |
| --- | ---: | --- | --- | ---: |
| validate | 2 | 30G | 30 min | 1 |
| benchmark train | 8 | 30G | 15 min | 2 |
| ridge candidato | 8 | 30G | 30 min | 4 |
| compare | 2 | 30G | 30 min | 1 |

Dopo validate e benchmark COMPLETED 0:0, pubblicare la fit_release con report e
accounting pinnati. Richiesta: ceil_a_60s[1,5 × (2V + 11S + 13E) + 120] ≤ 1.800 s.
V è ElapsedRaw di validate; S statistics_seconds del benchmark; E three_eigensystems_seconds.
Contesto numerico a otto thread identico a quello storico. Il gate non si allenta.
La release cambia il piano: nuovo checkout congelato prima della ridge.
Il compare richiede percorso e SHA effettivo del rapporto candidato; si invia
soltanto dopo la ridge completa. Il checkout della ridge può essere riusato.

## Sonde di classificazione a 19.618

Popolare il piano probe_plan_19618 con i cinque pin completi delle cache candidate.
Riuso dei gate storici già approvato, non un nuovo benchmark LQ. Stessa ricetta,
contesto numerico, larghezza pool768, fold e conteggi del T100-L storico.
Margini pubblicati max elapsed/tetto: db2-01 0,495; db2-23 0,470; db3 0,331;
db6 0,575; p2 0,452. T100-L: 13,26 core-ore di fit su 34,33 di tetto della catena.
Le feature nuove possono cambiare le iterazioni di lbfgs: è un proxy, non una garanzia.

| Job | CPU | Memoria | Walltime |
| --- | ---: | --- | --- |
| db2-01 | 8 | 30G | 60 min |
| db2-23 | 8 | 30G | 60 min |
| db3 | 8 | 30G | 30 min |
| db6 | 8 | 30G | 30 min |
| p2 | 8 | 30G | 60 min |
| aggregate-p1 | 2 | 30G | 20 min |
| aggregate-p2 | 2 | 30G | 20 min |
| compare | 2 | 30G | 30 min |

Tutti gli otto job delle sonde conservano lo stesso checkout e SHA del piano,
anche dopo la pubblicazione delle release posa sul checkout successivo.
I cinque shard sono indipendenti. Aggregate-p1 dipende dai quattro shard P1;
aggregate-p2 dal solo shard p2; compare dai due aggregati. La catena usa lo stesso
piano e checkout. Totale 34,333333 core-ore, zero GPU. Se uno shard raggiunge il
walltime si conserva failed.json e si ferma la fase, senza retry o aumento del tetto.
Gli avvisi di convergenza mantengono i numeri descrittivi ma sospendono l'interpretazione
scientifica del solver interessato.

## Checkout, contabilità e controlli

Sono sufficienti due nuovi checkout CPU condivisi fra endpoint, se i relativi
piani sono pubblicati insieme: il primo per validate/benchmark posa e l'intera
catena sonde; il secondo per le release dei fit posa e i loro confronti.
Ogni checkout: 2 CPU, 4G, 10 min = 0,333333 core-ore. Tetto CPU complessivo:
16 + 34,333333 + 0,666667 = 51 core-ore; zero GPU. Conteggiare una sola volta
quelli condivisi. Nuovi pin, risorse e SHA vanno pubblicati prima degli invii.

Correzione del launcher posa e review completate: APPROVATO, 43 test verdi.
Il guard Python ora usa l'ambiente del progetto, attivato prima della verifica. Nessuna modifica ai cinque sorgenti numerici o ai 110 sorgenti delle
estrazioni; i checkout in esecuzione rimangono immutabili. Nessun job CPU ancora
sottomesso da questa consegna. Nessun monitor in background.
