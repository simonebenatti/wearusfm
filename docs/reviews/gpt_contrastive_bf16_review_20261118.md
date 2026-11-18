# Review GPT del collaudo bf16 e lettura della ripresa — 18/11/2026

## Verdetto e lettura fissata prima del rilancio

Approvato l'emendamento tecnico di Claude in be21629: nelle tre norme diagnostiche bf16
il limite diventa max(limite precedente, 0,01). Rimangono identici FP32, perdite bf16,
gradiente effettivo, spostamenti studente/teacher, STOP, probe, dati, inizializzazione e λ.
È una modifica del criterio **post hoc**: 64730941 resta FAILED, non viene rietichettato
come superato. La nuova cella CUDA dovrà passare integralmente. Nessun training scientifico.

Il fallimento riguarda otto confronti diagnostici su 320: norma InfoNCE bf16 all'update 2,
nei due bracci sui quattro rank. I controlli FP32 superati sostengono la correttezza del
calcolo distribuito in questa fixture; non sono una dimostrazione per tutti i possibili
input. Scarti bf16 piccoli sono compatibili con arrotondamenti differenti, e la soglia
minima dell'1% riprende quella già usata per le perdite. Resta un criterio ingegneristico,
non una prova di efficacia delle loss. I limiti residui dipendenti dal rumore del riferimento
restano invariati: se falliscono, si conserva il rapporto e si diagnostica, senza retry cieco.

## Confronto continuo/ripreso: stesso limite, rapporto conservato

Prima del nuovo CUDA GPT adotta la proposta di Claude: uno scarto FINITO maggiore dell'1%
nei tensori floating dell'ottimizzatore o della coda rende `compare_resume.passed=false`,
con percorso del tensore, errore relativo, limite e motivo nel rapporto. Il lettore finale
propaga il motivo e produce `FAILED_REQUIRES_REVIEW`, senza perdere le altre misure.
**Non diventa accettabile uno scarto prima respinto; il limite resta 0,01.**

Schema, chiavi, forme, dtype, valori non finiti, step, RNG, metadati e buffer restano controlli
stretti. Differenze strutturali o esatte continuano a sollevare errore: non sono classificabili
come semplice rumore numerico. Il confronto studente/teacher mantiene la normalizzazione
sullo spostamento dallo stato iniziale autentico. Nessuna modifica al trainer o alla ripresa.

La sola modifica runtime aggiuntiva è in contrastive_benchmark_readout.py; test con checkpoint
sintetici completi verificano sia rapporto negativo serializzabile sia rifiuti esatti/nonfiniti.
Manifest e release rigenerati dopo questa decisione; il documento di Claude pinnato nei job
resta intatto. Si conservano i riferimenti storici nel commit originale e nei rapporti raccolti.

## Esecuzione autorizzata e condizioni

Simone ha già autorizzato correzioni, aggiornamento dei riferimenti, nuovo CUDA e cinque benchmark
se superato; nessuna nuova domanda operativa. Nuovo checkout e output, submitter/ledger comuni.
Il checkout CPU può attendere la quota della linea 1: CUDA è accodabile con afterok e launcher
immutabile byte-identico già esistente, CODE_DIR/EXPECTED_SHA sempre sul nuovo commit.
Il successo del checkout e il launcher verificano SHA, pulizia e sola lettura prima di eseguire.

I benchmark non sono accodati prima della lettura positiva del nuovo CUDA e del probe.
QoS boost_qos_dbg letta sul cluster: MaxJobsPU=2, MaxSubmitPU=4. Dopo il gate, inviare a gruppi
contando le altre linee; nessuna duplicazione o uso di rifiuti come tentativi di sondare la quota.
Le cinque celle restano tre run a 576 passi e due riprese 544+32, senza training lungo.

Consuntivo e tetti confermati dal sacct: consumato 1,075556 GPU-ore / 17,297778 core-ore;
massimo con sei celle GPU, due checkout e lettura finale 13,075556 / 121,831111, entro
14 / 129,333333. Nessuna automazione in background. Hash di checkpoint, audit, quattro input
e launcher ricontrollati sul cluster e archiviati in amendment_bf16_20261118/gpt_remote_precheck.txt.
Il pin documentale del manifest è associato al checkout immutabile; il validatore ne verifica
il contenuto. Il futuro job di lettura dovrà pinnare esattamente l'audit originale.
