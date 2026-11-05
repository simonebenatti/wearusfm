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

## Raccolta e passaggio alle estrazioni

Il benchmark produce `receipt.json`, che contiene anche il rapporto dei tempi;
non è previsto un file `report.json` distinto. Raccogliere ricevuta, log e accounting.
Verificare completezza, SHA di checkout/piano, checkpoint 5.000, controllo storico,
contesto e 110 sorgenti. Ricalcolare i sei gruppi dai sei confronti successivi ai due
di riscaldamento e verificare conteggi storici e `resource_proposal`.

Pubblicare le quattro celle pose_5000, pose_19618, p1_19618, p2_19618.
Il tetto temporale è ceil(walltime storico × max(1, rho_cap) / 300) × 300 secondi;
la stima usa il rapporto ponderato dai conteggi storici. Copiare integralmente
`resource_proposal.jobs[cell]` nelle allocazioni. Resta un proxy: rho misura il
forward, mentre le durate storiche comprendono anche le altre operazioni.

Conservare immutabile l'inference_plan eseguito. Preparare un nuovo extraction_plan
con stato APPROVED_FOR_EXTRACTION, ricevuta JSON pinnata, riferimento al piano
benchmark immutabile, inference_code_sha e allocazioni. Non aggiornare a ritroso
l'inference_plan: si creerebbe un ciclo e si invaliderebbe la ricevuta.
Le celle a 19.618 aspettano il rispettivo checkpoint completo; quella di posa a
5.000 può partire dopo verifica e pubblicazione dei tetti. Ogni estrazione riceve
il walltime esatto della cella. Qualunque errore ferma la fase, senza retry automatici.

## Invio del 05/11/2026

Codice congelato `cc814303e52d8e4067cf12fb1a291459a402c78c`, pubblicato su origin
e sul nuovo riferimento di deploy. Checkout 63614193 COMPLETED 0:0 in 28 secondi
(0,015556 core-ore), SHA, pulizia, 110 sorgenti, pin e sola lettura verificati.
Benchmark 63614514 sottomesso tramite submitter e ledger comuni; ricevute archiviate.
Nessuna dipendenza SLURM dal checkout concluso: il suo esito è verificato prima
dell'invio. La continuazione 63593128 mantiene il proprio checkout e piano.

Benchmark 63614514 COMPLETED 0:0 alle 19:08:50 CET, durata SLURM 110 secondi,
0,030556 GPU-ore / 0,244444 core-ore. Ricevuta originale da 198.900 byte raccolta,
SHA256 `7e6e665eb4fb321b81266d0f35806c61814958e758c236a8266baae945db4e9a`.
Il rapporto è ora sottoposto al ricalcolo indipendente dei tempi e dei tetti.
