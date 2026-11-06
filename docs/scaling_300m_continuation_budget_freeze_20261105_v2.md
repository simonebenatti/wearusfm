# S8: correzione tecnica del freeze prima degli score — versione2

**Nessuna continuazione a19.618 autorizzata. I tetti350CPUh/66,35GPUh restano una proposta futura, non un budget approvato da Simone.** Questa versione sostituisce i tetti
e la QoS della v1, prima del primo fit downstream a5.000, su sola evidenza tecnica
e reviewClaude7 B-3. Il JSONv1 resta inalterato come storico; il riferimento
aggiornato è `results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json`.
La regola scientifica S8 resta byte-identica, hash
`da58ec3a7965da8db29e97a1089ae1d1d7e652dba4250f24dcb1b8e608602960`.

## Correzione della QoS e parteGPU

La verifica diretta dello scheduler è in `cpu_recovery_20261105/scheduler_review7.txt`.
`boost_qos_bprod` impone `MinTRES=node=65,cpu=2080,gres/gpu=260,mem=31280G` con
`DenyOnLimit`: non è utilizzabile per il nostro singolo nodo. La vecchia verifica
di MaxWall e associazione era insufficiente. Per il piano condizionale fissiamo
**`--qos=normal`, partizione boost_usr_prod**, con un nodo,4GPU,32CPU,480G.
Normal è nell'associazione e non ha MinTRES/MaxWall aggiuntivi; la partizione
ammette24h. Non è una prova di sottomissione: prima di un futuro lancio va
riverificato lo scheduler e rivisto il wrapper di continuazione.

La formula S8 e i suoi input restano identici:14.618update aggiuntivi,
`ceil900(14618×max(14665,05355668068/5000,2,7666786608873997)×1,20+1200)`
=53.100s, ossia14h45. Training richiesto59GPUh, estrazioni7GPUh, totale**66GPUh**;
CPU associate alle allocazioniGPU528coreh. Il margine prudenziale di180s per job
aggiunge0,35GPUh e2,8coreh: **66,35GPUh /530,8coreh contabili**, senza aumentare
walltime richiesto, passi o dose. Il margine non è garanzia assoluta sui tempi
di terminazione; il consuntivo conserva gli elapsed effettivi.

Restano tutti i guard e i pin della v1: checkpoint completo, memoria allocata e
riservata per rank con baseline37,751831GiB, optimizer/teacher/RNG/loader/schedule
di ripresa integri, controllo30M19618 da riverificare, nessun recupero fuori piano.

## Lettori futuri:350core-ore, distinta campagna

Le misure a5.000 invalidano180/24/136 della v1. Congeliamo ora un inviluppo
**350core-ore CPU-only**, senza riportare i22,027778 già spesi nella campagna5.000.
Sono tetti condizionali, non misure di feature19618 e non autorizzazione a lanciarle.

| Fase futura | Tetto contabile core-ore, incluso180s/job |
|---|---:|
| Tre checkout metadata,2CPU/5min |0,8|
| Validazione posa,2CPU/30min |1,1|
| Tre benchmark posa,8CPU/10min |5,2|
| Quindici benchmark sonde per dataset,8CPU |76,666667|
| Fit:≤15shard,≤3h/shard,≤240coreh richieste |246|
| Ridge posa,8CPU/60min |8,4|
| Sei aggregati,2CPU/10min |2,6|
| Confronto,8CPU/30min |4,4|
| Totale |**345,166667**|
| Riserva entro350 |**4,833333**|

Tetti richiesti dei benchmark sonde, minuti per dataset:

| Vista | DB2 | DB3 | DB6 | EPN612 | UCI |
|---|---:|---:|---:|---:|---:|
|30M naturale|45|20|20|30|10|
|300M naturale|60|30|60|120|10|
|300M matched|45|20|20|30|10|

Sono530min su8CPU=70,666667coreh richieste, più6coreh di margine per15job.
Derivano da tempi tecnici e margini, non da score; non garantiscono successo
a19618. Si mantengono30GiB,login08,8thread, stessa ricetta e stessi fold; prima
dei fit servono tutti i benchmark reali dell'endpoint e la formula congelata.
Se non rientrano si ferma la fase: niente aumenti motivati da risultati promettenti.
Il dispatcher del recupero5.000 non abilita questa matrice futura: serve un
wrapper di endpoint e review separati, oltre ai checkpoint/pin effettivi.

Il totale condizionale CPU è878coreh contando528coreh nominali dei jobGPU
più l'inviluppoCPU350 (che già include i suoi margini); diventa **880,8coreh**
aggiungendo anche il margine scheduler dei jobGPU.
Il seguito richiederà piano concreto, review e decisione di Simone: questa
pubblicazione fissa i limiti prima degli score, non concede quel via.
