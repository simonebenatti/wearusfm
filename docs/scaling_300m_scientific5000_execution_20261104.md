# Rilascio scientifico 300M a 5.000 — 4 novembre 2026

## Esito reale di S3

Job **63527446 COMPLETED 0:0**, 20:42:52–21:09:51 CET, 1.619 secondi.
500 aggiornamenti freschi, batch globale32, quattro processi, accum_steps1,
prefetch e diagnostiche periodiche intatti. Copie di15file verificate contro
hash remoti e pin del report; audit indipendente nel repo. Picchi allocati:

| Rank | GiB | Frazione della memoria effettiva |
|---|---:|---:|
| 0 | 29.610982 | 46.688% |
| 1 | 31.241646 | 49.259% |
| 2 | 29.710452 | 46.845% |
| 3 | 34.544032 | 54.466% |

Il massimo **34,544032GiB** supera il proxy33,522417GiB di circa1,022GiB,
ma resta sotto il limite **57,080402GiB** (90% del dispositivo rilevato).
Il proxy non era una misura né un tetto. Non si attiva il ripiego accum2.
Consuntivo **1,798889GPU-ore e14,391111core-ore**, entro4/32; checkoutS3
0,011667core-ore separato. Runtime Python/Torch/CUDA corrispondente al30M;
cuDNN8907 registrato, non confrontabile con lo storico che non lo riportava.

Fixed eval e pannello presenti ai passi0/500; gradienti e diagnostiche al500;
summary e quattro picchi riconciliati. Il producer sul compute ha eseguito
`inspect_evidence` e `validate_report` prima di scrivere il report. L'audit
indipendente sul Mac verifica le copie piccole, **non** scarica né carica il
checkpoint tecnico da4,67GB. Non pretende di sostituire il guard integrale:
il wrapper scientifico lo riesegue nella nuova allocazione, prima di torchrun.
Restano conservati vecchio checkoutS3, directory del run e ricevute originali.

Questa è fattibilità tecnica a500 passi, non evidenza di superioritàscientifica,
convergenza o generalizzazione del300M. Il nuovo run parte dazero; il checkpoint
S3 non viene usato per inizializzare o riprendere il training.

## Identità e gate

Sorgenti scientifici identici ad afb644e (103file). ProduttoriS3/training invariati
rispetto a0d5f0f9. ReviewClaude del preflight e dei lettori acquisite; correzioni
P1/P2 sul readout recepite senza alterare gli artefatti originali. Preparazione
operativa34b878a:571test passati, review interna e guard readiness effettivo.
Il piano finale lega i quattro gate agli hash reali; i tre antecedenti sono:

- preflight: `90254bb4bd8aa525730cdd43cec2fa6c98237387754145f0704037e3f5ec8831`;
- S3: `45acd95c81db3a550eb9bd1dabf5055dd5ae5d35e94a8cb0fabb39035144f7f9`;
- lettori: `7b30f54c8527e2fee251101b926d2dec4e9b99b5b32a777afb6554a1bf059b8b`.

La review finale interna `docs/reviews/scaling_scientific5000_gate_review_20261104.md`
è **APPROVATA** sui tre gate reali. Il quarto gate la lega a sorgenti e risorse.
È distinta dalla reviewClaude già acquisita e non le attribuisce
l'esame di codice successivo. Il controllo30M è il teacherT100-G a5000 del
contratto; file e ricevuta verificati, orizzonte11novembre. Il wrapper rilegge
integralmente checkpoint, stato strutturale, configurazione e gain sul compute.

## Un solo training, destinazioni nuove

- ID logico checkout: `SCALING-300M-checkout-scientific5000-20261104`.
- ID logico training: `SCALING-300M-train5000-20261104`.
- Campagna/deploy: `scaling_300m_scientific5000_20261104_v1`.
- Checkout: `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_scientific5000_20261104_v1`.
- Run: `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-5000`.
- Ricevute: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/scaling_300m_20261104/scientific5000_v1`.

| Job | Risorse | Durata massima | Tetto |
|---|---|---|---|
| Checkout | 2CPU,4G,lrd_all_serial |10min|0GPUh,0,333333coreh |
| Fresh300M5000 |1nodo,4GPU,32CPU,480G,boost_usr_prod|5h30|22GPUh,176coreh |

Il checkout del training è separato dall'inviluppo delle valutazioni. Somma
massima di questa coppia:22GPUh e176,333333coreh. Nessuna quota estrazione o fit
viene consumata da questi due job. Valutazioni giàpreparate:8,25GPUh,66coreh
allocate conGPU e180CPU-only; benchmark reali dopo cache e prima dei fit.

## Sequenza di rilascio

1. Pubblicare piano, gate, review e risorse; commit/push e SHA remoto verificato.
2. Pubblicare un nuovo riferimento deploy al commit esatto; non toccaremain o
   checkout attivi. Destinazioni e ID verificati assenti prima dell'invio.
3. Checkout via submitter comune, launcher su stdin e risorse sopra. Dopo
   COMPLETED0:0 verificare HEAD, tree pulito, hash e sola lettura.
4. Invio unico del training via `scripts/submit_job.py` e ledger comune. Gate
   sincrono dopo checkout; non usare afterok scaduto con MinJobAge30s.
5. Pubblicare ricevute/ID e stato osservato. Al completamento, checkpoint e
   ricevute reali abilitano le cinque estrazioni; nessun pin anticipato.

Le autorizzazioni giàespresse da Simone coprono questo piano. Nessun nuovo
consenso, nessun monitor automatico, nessuna continuazione a19618. Errori o
gate incoerenti fermano la fase, senza retry cieco o modifica della ricetta.
