# Fine-tuning EPN — esecuzione dell'emendamento approvato

Claude f70dc72 approva 2c2e857 senza residui, 133 test CPU passati. Frozen resta 4 h,
FT/scratch 7 h; N-2 facoltativo non adottato. Nessun cambio di codice runtime,
ricetta o benchmark rispetto alla versione revisionata; solo stati convergenti,
nuovi percorsi operativi e successivo pin della ricevuta.

Simone conferma: «qua semplicione non c'e piu, lanci sempre tu». Lanci diretti
GPT, stesso submitter/ledger, nessun nuovo consenso. Nessun monitor automatico.

## Benchmark nuovo

Piano `results/infra/consolidation_20261104/ft_v2_launch_plan.json`.
Nuovo ref/checkout `consolidation-ft-v2-20261104`, SHA pubblicato su origin e
leonardo prima del checkout; completamento 0:0, SHA/pulizia/sola lettura verificati.
OUT nuovo e persistente: WORK/wearusfm_runs/results/consolidation_20261104/
finetuning/benchmark_v2. Stato del piano e fits_status convergenti; pin ancora nullo.

| ID | Risorse | Durata massima | Tetto |
|---|---|---|---|
| FT-checkout-v2-20261104 | 1 CPU richiesta, 2 CPU prudenziali, 2 GiB | 10 min | 0 GPUh, 0,333333 coreh |
| FT-BENCH-v2-20261104 | 1 GPU, 8 CPU, 100 GiB | 30 min | 0,5 GPUh, 4 coreh |

Benchmark dipende afterok dal checkout. Source_identity deve coincidere con
walltime_amendment_validation.json, ricetta 2ba1b7f invariata. Verificare report
completo, SLURM 0:0, input/checkpoint/piano/SHA, policy/limiti, 32 update + 4 warm-up
per regime, tempi finiti e gate memoria. Ricalcolo joint/split con benchmark_packaging.

## Fit

Solo dopo gate: pubblicare ricevuta con SHA, modalità per regime, tempi e tetti;
nuovo checkout per il pin senza cambiare sorgenti. Quattro fold per regime,
due candidati, 5 epoche, batch 8, seed 0. Layout WORK/finetuning/epn/foldN/regime/
joint o candidate0/candidate1; nello split candidate1 dipende afterok da 0 e
verifica ricevuta/pesi prima di selezionare sulla validation e predire held.

Limiti frozen 14.100 s / 4 h, FT/scratch 24.900 s / 7 h, riserva 300 s. Policy invariata:
joint se entra la coppia; altrimenti split se entra il singolo; altrimenti stop
del regime. Tetto massimo 24 fit, 144 GPUh / 1.152 coreh, da ridurre se il nuovo
benchmark consente joint. Non sottomettere candidati ai quali il gate dice blocked.
Tutti gli ID prima degli invii nel piano misurato, submitter con prenotazione unica;
nessun retry ambiguo. Riportare stato reale distinguendo completamento e coda.

Continuazione 63525803 → 63525805 → 63525809 su 17dd455 intatta. Matrice lunga e
aggregatore OOF/IC sono consegne separate; questo lancio non li dichiara pronti.

## Invio del benchmark, 04/11 ore 19:12 CET

Codice congelato `ae283052da086c814890d8e8e5b57505395b37a1`, pubblicato e
verificato su origin e sul ref di deploy Leonardo.
Checkout **63527415 COMPLETED 0:0**, 18 s con 2 CPU allocate: **0,010 coreh**.
Verificati SHA, pulizia e codice in sola lettura.
Benchmark **63527476 RUNNING** alla lettura delle 18:13:23 UTC.
Ricevute del submitter in `results/infra/consolidation_20261104/submission_v2/`.

Alle 18:13:41 UTC WORK ha circa 39,8 GiB liberi nella quota di progetto;
nessun output EPN fit né checkout dei fit già esistente. Ledger senza ID fit
duplicati. Nessun monitor automatico attivato.

## Misura conclusiva e piano dei fit, prima degli invii

Benchmark **63527476 COMPLETED 0:0**, 450 s: **0,125 GPUh / 1,000 coreh**.
Ricevuta locale e remota identiche:
`c14b78b0d593a56f272192bd2c246003e657788551936c24c9dc1fba1eb0241d`.
Il pin è in finetune_plan.json; vecchia ricevuta conservata separatamente.
Verificate provenienze di codice, piano, sorgenti, ricetta, checkpoint e input,
policy e limiti, 4 warm-up / 32 update per regime, forward sul solo train.
Controllo indipendente favorevole: 85 verifiche positive, aritmetica ricostruita,
gate del controller ricalcolato. Nessun dato di qualità scientifica dal benchmark.

| Regime | Singolo candidato, proiezione | Gate | Modalità | Job | GPUh proiettate | Tetto GPUh |
|---|---:|---:|---|---:|---:|---:|
| frozen | 7.972,65 s | 14.100 s | split | 8 | 17,717 | 32 |
| finetune | 23.583,18 s | 24.900 s | split | 8 | 52,407 | 56 |
| scratch | 23.553,49 s | 24.900 s | split | 8 | 52,341 | 56 |
| Totale | | | | **24** | **122,465** | **144** |

Memoria host proiettata **32,385 GiB** su soglia 85 GiB; massimo CUDA riservato
**6,584 GiB**, sotto l'85% dei 63,423 GiB disponibili. Cap CPU dei fit:
**1.152 coreh**. Nuovo checkout CPU aggiuntivo: massimo **0,333333 coreh**.
La proiezione comprende margini conservativi; il consuntivo verrà da SLURM.

Piano concreto: `results/infra/consolidation_20261104/ft_v2_fit_launch_plan.json`.
Checkout nuovo `consolidation-ft-v2-fits-20261104`, con pin della ricevuta
e sorgenti invariati (aa857fb). Tutti i 24 ID, output, risorse e dipendenze
sono pubblicati prima del lancio. Per ciascuna delle 12 coppie, candidato 0
dopo il checkout, candidato 1 dopo il proprio candidato 0.
Ogni job usa 1 GPU, 8 CPU, 100 GiB, 4 h frozen oppure 7 h finetune/scratch.
Nessuna selezione sul held-out e nessuna variazione di ricetta.

## Rettifica operativa delle dipendenze, prima dei fit

Checkout fit **63527861 COMPLETED 0:0**, 16 s / 2 CPU, SHA a2eeb75, pulito
e in sola lettura. SHA del piano remoto identico al locale (2ce4927), ricevuta
benchmark c14b78b verificata nuovamente.

Il primo invio di FT-EPN-frozen-F0-20261104-C0 è stato **esplicitamente
rifiutato**, senza ID SLURM: `Job dependency problem`. Diagnosi verificata:
`MinJobAge = 30 sec`; il checkout completato alle 19:23:27 non era più nel
controller all'invio delle 19:24:56, pur restando COMPLETED 0:0 in sacct.
La dipendenza dal checkout è già soddisfatta e non serve un nuovo job.

Rettifica pubblicata nel piano operativo: C0 parte dopo verifica documentata
del checkout completato, senza afterok verso l'ID eliminato; **ogni C1 conserva
afterok verso il proprio C0**. Nessun cambio di sorgenti o piano scientifico:
il checkout runtime resta a2eeb755676a0ab780fc269f0679684eb41cd6f2.
Primo C0 con lo stesso ID logico e `--dopo-rifiuto`; gli altri ID sono nuovi.
La ricevuta del rifiuto è conservata in submission_v2/fits. Nessun retry cieco
e nessuna duplicazione del checkout.

## Sottomissioni completate — 04/11, ore 19:27 CET

Tutti i **24 fit sottomessi** via submitter e ledger comuni, dopo pubblicazione
e verifica dello SHA remoto del piano operativo ec6e102. Il codice eseguito
resta a2eeb75. Ricevute complete in `submission_v2/fits/`; verificati tutti
i nomi, 24 ID unici, risorse e dipendenze effettive SLURM.

| Regime | Fold | Candidato 0 | Candidato 1 |
|---|---:|---:|---:|
| frozen | 0 | 63527972 | 63527973 |
| frozen | 1 | 63527974 | 63527975 |
| frozen | 2 | 63527976 | 63527977 |
| frozen | 3 | 63527978 | 63527979 |
| finetune | 0 | 63527980 | 63527981 |
| finetune | 1 | 63527982 | 63527983 |
| finetune | 2 | 63527986 | 63527989 |
| finetune | 3 | 63527990 | 63527991 |
| scratch | 0 | 63527992 | 63527993 |
| scratch | 1 | 63527994 | 63527995 |
| scratch | 2 | 63527996 | 63527997 |
| scratch | 3 | 63527999 | 63528000 |

Alle 18:27:32 UTC: **10 RUNNING, 14 PENDING** (2 Priority e 12 Dependency).
Tutti i C1 dipendono dal C0 della stessa riga. Consuntivo fit ancora aperto.
Benchmark nuovo: 0,125 GPUh / 1 coreh; due checkout v2 conclusi:
0 GPUh / 0,018889 coreh complessive. Rifiuto iniziale: zero calcolo.
Continuazione 63525803 → 63525805 → 63525809 ancora PENDING, invariata.
Nessun monitor automatico creato. Matrice lunga e aggregatore OOF/IC restano
consegne separate.
