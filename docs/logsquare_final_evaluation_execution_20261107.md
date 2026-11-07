# T100-LQ: endpoint finale e valutazioni

## Training completato

63593128 COMPLETED 0:0 il 07/11/2026, dalle 09:06:13 alle 12:15:41 CET,
11.368 secondi, fine per max_steps a 19.618. Configurazione identica a quella
5.000 salvo max_steps; ripresa dal checkpoint 5.000 nominato, verificato invariato.
Ricevuta, configurazione, summary, launch e log originale archiviati con hash.
Checkpoint finale: 454.776.554 byte, SHA256
`316fa96fb10a941a1e3be230e2738b19bbfb83f691dd940e4d32e089ebb2c214`.
Gain storico attestato dalla ricevuta, nessun STOP.

Consuntivo training, incluso il tentativo iniziale fallito prima degli aggiornamenti:
17,065556 GPU-ore / 136,524444 core-ore, entro 22 / 176. Diagnostica, checkout,
benchmark ed estrazioni restano contabilizzati nelle rispettive tranche.

## Tre estrazioni finali

Il nuovo piano aggiunge il solo pin 19.618 alla ricetta di estrazione approvata;
checkpoint 5.000, contesto, 110 sorgenti, ricevuta del benchmark e allocazioni
rimangono identici. Non serve ripetere il benchmark di inferenza.
Piano: `results/infra/logsquare_evaluation_20261107/extraction_plan_19618.json`.
Risorse: `results/infra/logsquare_evaluation_20261107/resources_extraction.json`.

| Job | ID logico | Allocazione | Tetto |
| --- | --- | --- | --- |
| Checkout condiviso | LOGSQ-extract-checkout-20261107 | 2 CPU, 4G, 10 min | 0,333333 core-ore |
| Posa 19.618 | LOGSQ-pose-19618-20261107 | 1 GPU, 8 CPU, 100G, 50 min | 0,833333 GPU-ore / 6,666667 core-ore |
| P1 19.618 | LOGSQ-p1-19618-20261107 | 1 GPU, 8 CPU, 100G, 35 min | 0,583333 GPU-ore / 4,666667 core-ore |
| P2 19.618 | LOGSQ-p2-19618-20261107 | 1 GPU, 8 CPU, 100G, 50 min | 0,833333 GPU-ore / 6,666667 core-ore |

Totale nuovi job: 2,25 GPU-ore / 18,333333 core-ore, checkout incluso.
Le estrazioni sono indipendenti e possono essere in coda o in esecuzione insieme.
Stime proxy: posa 1.719 s, P1 891 s, P2 1.678 s; valgono i tetti pubblicati.

Checkout nuovo `logsquare-extract-19618-20261107`, SHA della presente consegna
pubblicato e verificato prima del deploy. Output fuori dal codice, sotto
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/logsquare_evaluation_20261107/19618/`,
sottocartelle pose, p1, p2. Submitter e ledger comuni; override del walltime esatto
per ogni cella. Nessuna ripetizione della posa 5.000 o modifica al training concluso.

## Letture scientifiche dopo le estrazioni

Come richiesto da Simone, si leggono insieme i due endpoint. Dopo verifica delle
nove cache e delle provenienze: posa 5.000 e 19.618 con validate, benchmark train,
release del fit, ridge e confronto; sonde di classificazione soltanto a 19.618,
cinque shard, due aggregati e confronto. T100-L è il controllo primario;
T25-L e T100-G restano secondari. Fit solo del candidato, controlli riusati.
I lettori sono lavoro CPU già approvato e non nuovo addestramento del FM.
I relativi pin e checkout si pubblicano dopo la disponibilità delle cache.
Nessuna contrastiva, cambio di loss, nuova dose o monitor in background.

Review indipendente di prontezza: APPROVATO. Pin, originali, contesto e risorse
verificati; stato APPROVED_FOR_EXTRACTION pubblicato nei due nuovi piani.

## Sottomissioni e avvio

Consegna esecutiva `c1963236a08690998c812c86278bdefbf745ccf1`, pubblicata e verificata
su origin. Checkout 63772400 COMPLETED 0:0 in 24 s, pulito e in sola lettura,
SHA/piano e 110 sorgenti ricontrollati. Le tre estrazioni sono state accettate al primo
invio e sono RUNNING dalle 13:28:03 CET del 07/11: posa 63772648, P1 63772649,
P2 63772650. Risorse e tempi osservati corrispondono al piano. Ricevute archiviate
sotto `results/infra/logsquare_evaluation_20261107/execution/`.
Nessun job CPU di lettura ancora sottomesso.
