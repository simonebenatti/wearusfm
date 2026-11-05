# T100-LQ: benchmark di inferenza dopo il checkpoint a 5.000

05/11/2026. Training 63593091 COMPLETED 0:0 in 3.970 s (15:31:30–16:37:40 CET),
5.000 passi, fine per max_steps. Ricevuta, quattro attestazioni dei rank, dose,
configurazione, piano/release e hash del checkpoint nominato verificati. Costo:
4,411111 GPU-ore / 35,288889 core-ore, entro i tetti di 6 / 48.
La continuazione 63593128 è già accodata e indipendente.

Endpoint `checkpoint_step0005000.pt`, 454.776.554 byte,
SHA256 `6233b0705e8db2caaf3c84068812053878ec2291887be88740302fe6c844b0a9`.
Ricevuta e configurazione originali conservate in
`results/infra/logsquare_training_20261105/completed_5000/`.
Nessun punteggio scientifico ancora prodotto da questo endpoint.

## Rilascio del benchmark già approvato

Adattatori invariati, approvati da Claude in 6c0ce65. Review indipendente di prontezza
favorevole: 110 sorgenti identici e pin incrociati verificati. Acquisiti i riferimenti
effettivi del checkpoint 5.000; stato APPROVED_FOR_BENCHMARK nei piani di estrazione
e inferenza, con nuovo pin dell'estrazione nel piano di inferenza. Endpoint 19.618
nullo, budget delle estrazioni vuoto e ricevuta di inferenza nulla: nessuna estrazione
scientifica può partire con questi piani.

A1 verificato sulla cache train B19618 esatta del piano: 1.907 finestre, 200 registrazioni.
Le registrazioni 0 e 1 hanno 10 finestre ciascuna e identità user/session/stage/side/stem
coerenti con la selezione di produzione ricostruita col seme congelato. SHA della
cache verificato. Nessuna riselezione, nessun cambio di epsilon o ricetta.

| Job | ID logico | Risorse | Tetto |
| --- | --- | --- | --- |
| Checkout | LOGSQ-infer-checkout-20261105 | 2 CPU, 4G, 10 min | 0,333333 core-ore |
| Benchmark | LOGSQ-infer-5000-20261105 | 1 GPU, 8 CPU, 100G, 30 min | 0,5 GPU-ore / 4 core-ore |

Checkout nuovo `logsquare-infer-5000-20261105`, commit pubblicato con questi pin;
output `$WORK/wearusfm_runs/results/logsquare_inference_20261105/5000`.
Submitter e ledger comuni, nessun duplicato. D1 resta invariato: massimo dei rapporti
per il tetto di ogni cella, media pesata sui conteggi storici per la stima puntuale.
Misura FP32 per la posa e BF16 per la classificazione con decoder, su input train appaiati.
Nessun fit, score o cache scientifica; il rapporto includerà le 64 norme W_log per
student/teacher e le famiglie, come lettura descrittiva R1-b.

Dopo il benchmark: verificare tempi, supporti e provenienze, pubblicare rho_infer e
tetti prima di qualunque estrazione. La continuazione a 19.618 non aspetta questo job.
Nessun monitor in background e nessuna nuova richiesta di consenso.
