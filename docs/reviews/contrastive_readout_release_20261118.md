# Lettura CPU dei cinque benchmark contrastivi — 18/11/2026

## Autorità e scopo

Simone ha autorizzato la lettura CPU e il 18/11 ha affidato tutti i nuovi lanci della
linea 3 alla chat **3-launcher**. Questa consegna prepara esclusivamente checkout e
lettura dei cinque benchmark già conclusi; nessun nuovo training, CUDA o retry.
GPT scientifico pubblica piani e riferimenti, 3-launcher sottomette con submitter e
ledger comuni. I criteri e i sorgenti restano congelati.

## Prerequisiti osservati

Alle 18:24 Europe/Rome, tutti COMPLETED 0:0: C0 64771457 (661 s), CN 64771478 (683 s),
CR 64771595 (769 s), ripresa CN 64771859 (775 s), ripresa CR 64772690 (759 s).
Cinque summary finali a 576, metadati e log raccolti con 25 hash remoti verificati in
`results/infra/contrastive_comparison_20261116/benchmarks_completed_bf16_20261118/`.
Metriche originali compresse senza modificarne i byte; hash degli originali conservati.
Checkpoint restano nei percorsi remoti e saranno autenticati dal lettore sul cluster.

La lettura preliminare sui soli metadati rileva già **parità negativa** delle norme
nei primi 256 update: CN 146/256 scarti oltre tolleranza, CR 135/256. La loss totale
passa 256/256; stream, origini e inizializzazione coincidono. I rapporti delle norme
attive passano: mediane CN 0,226064 e CR 0,083401; p95 0,431119 e 0,214031;
rapporto fra mediane 2,710568. Review indipendente concorde.
Non è un verdetto sulle riprese, né un motivo per alterare la soglia dopo i risultati.
La lettura CPU rimane necessaria per completare l'audit dei checkpoint e dei riavvii;
ci si attende almeno FAILED_REQUIRES_REVIEW per la parità. Nessun training rilasciato.

## Pacchetto e risorse

Job: `results/infra/contrastive_comparison_20261116/releases/readout_bf16_20261118/readout.job.json`.
Source SHA256 invariato: `46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.
Il job pinna review già accettata, cinque summary finali, coorte bf16, audit originale,
CUDA e ricevuta/summary della calibrazione. Il comando reale è stato costruito localmente
con gli undici pin verificati su copie originali e percorsi remoti preservati.
Il benchmark CPU verifica anche gli altri file, gli stati student/teacher, optimizer,
coda, RNG e metadati. Il probe CUDA rimane nel percorso originale registrato.

Checkout nuovo: `contrastive-v2-20261118-readout`, 4 CPU, 2 GiB, 240 s (0,266667 core-ore).
Lettura: 8 CPU, 30 GiB, 3.600 s (8 core-ore), **zero GPU**.
Costo già consumato: 5,284444 GPU-ore / 51,027778 core-ore.
Massimo compreso il residuo: 5,284444 GPU-ore / 59,294444 core-ore,
entro i tetti 14 / 129,333333. Valori esatti in `resources.json`.

ID nuovi: `CONTRAST-CHECKOUT-readout-20261118` e `CONTRAST-READOUT-bf16-20261118`.
Comandi con SHA congelato saranno in `submission_plan.json` accanto al job.
Il launcher CPU byte-identico già presente nel checkout bf16 può trasportare il nuovo
job con CODE_DIR/EXPECTED_SHA del checkout nuovo, dopo successo della preparazione.

## Regole per 3-launcher

Verificare che nessuno dei due ID sia prenotato e che output/checkout siano nuovi.
Pubblicare esclusivamente il ref deploy dedicato allo SHA fornito, senza modificare main
né checkout esistenti. Tutti gli invii tramite scripts/submit_job.py e ledger comuni.
Non rieseguire nessuno dei cinque benchmark. Conservare sorgenti, criteri, pin e risorse.
Se il lettore produce FAILED_REQUIRES_REVIEW con exit 2, raccogliere comunque il rapporto:
è un esito del controllo, non un invito a ritentare. Per eccezioni strutturali raccogliere
log e traceback. Nessun rilancio, aumento di soglia o di risorse. Restituire ID, stati,
rapporto e hash, contabilità e log a Simone; GPT scientifico completa l'interpretazione.
Nessun monitor in background o nuovo training.
