# Confronto CPU RMS100/fast125 — rapporto finale raccolto

**Stato corrente:** rapporto completo `FAILED_REQUIRES_REVIEW`, nessuna
ragione di inconclusività, `training_approved=false`. Checkout 65035600
COMPLETED 0:0 in 41 s; confronto 65035852 FAILED 2:0 dopo 107 s,
15:24:55–15:26:42 CET del 21/11. L'uscita 2 è quella prevista per il
verdetto negativo, non un arresto prima del rapporto. SHA effettivo
744ecfa1cccaa7ec650cc874fdc5799a83e0787e, consegna 2c3e16cf.
Le sezioni seguenti conservano lo storico del primo controllo e degli invii.

21/11/2026, 3-launcher. Consegna f68c753b, SHA di esecuzione
e0ddbac808c74f133cad619b8a61821e0afe18ea, ricevute delle riprese 358ede81.
Piano rigenerato byte-identico, ID checkout e lettura assenti dal ledger,
submitter persistente e pin verificati. Ref deploy pubblicato e verificato
sullo SHA atteso; repository remoto sul proprio main, non modificato.

Snapshot operativo alle 15:14:19 CET: saldo disponibile 653.928 ore locali,
quota Lustre del progetto 20147512 usata 989,7 GiB su 1 TiB, circa 34,3 GiB
liberi. Riserva della coorte 12 GiB più 18 GiB prudenziali per scritture
concorrenti. Snapshot completo con tutti gli impegni di progetto, tre job
msimonin attivi e relative code; nessun job seriale attivo di sbenatti.
Partizione seriale up, manutenzione confermata 23/11 08:00–25/11 08:00 CET.
Il massimo richiesto di questa fase resta 0 GPU-ore e 8,266667 core-ore,
inviluppo 9,466667 core-ore. Nessuna allocazione è stata richiesta.

Il comando SSH diretto del controllo checkout, su connessione nuova senza
multiplexing, è terminato con **exit 1**. Sono passati accounting dei
prerequisiti, impronte storiche e ricevute binding/audit. Il validatore
fisico arriva alla chiamata simulata della CLI e solleva:

`ValueError: Frozen reader parsed different arm/receipt paths`.

La causa è identificabile senza rilanciare: `reader.main.mapping()` restituisce
dizionari con valori stringa per full e resumed. Il sentinel di
`readout_handoff_20261121/validate_readout.py` confronta invece quei valori
con dizionari di Path in expected_args. A parità di testo del percorso,
stringa e Path non sono uguali. Non è una misura sui modelli e non è il
verdetto scientifico del preflight. Nessun confronto fra tensori eseguito.

**STOPPED_BEFORE_SUBMISSION**: nessun checkout e nessuna lettura CPU
sottomessi, nessun ID SLURM e nessun costo di allocazione di questa fase.
Log combinato stdout/stderr, snapshot, meta e verifica deploy conservati
in results/infra/contrastive_targets_20261118/readout_execution_20261121/.
Il launcher non modifica validatore, soglie, pin o piani. Problema restituito
alla chat scientifica per correzione del validatore, regressione e nuovo
pacchetto pubblicato prima del deploy. Autorizzazione operativa già acquisita;
nessun retry del pacchetto fallito e nessun nuovo consenso richiesto.

## Seguito corretto e invii

Il pacchetto 744ecfa1 corregge solo il validatore dei percorsi e ne fissa
la regressione (consegna scientifica: 167 test passati); fra i 29 pin cambia
solo il verificatore. Piano rigenerato byte-identico, ID ancora liberi nel
ledger e submitter persistente identico. Ref deploy avanzato fast-forward
senza alcun checkout attivo sul riferimento precedente.

Controllo checkout corretto exit 0, ricevuta reader_argv_valid=true e dieci
checkpoint fisici verificati. Checkout inviato una volta, 65035600:
4 CPU, 2 GiB, 4 min; COMPLETED 0:0, 15:21:32–15:22:13 CET.
Verificati SHA effettivo, pulizia e protezione del checkout. Controllo
before_readout successivo exit 0, stesse dieci ricevute e parser validi.
Snapshot operativo rinnovato: quota 989,7 GiB / 1 TiB, saldo 653.928 ore
locali, riserva 12+18 GiB, seriale up e manutenzione confermata. Il proxy
dei restanti impegni di progetto dallo snapshot è circa 3.763,7 core-ore;
non è un consuntivo né una nuova imputazione dei tempi già trascorsi.

Confronto inviato una volta come 65035852, 8 CPU / 30 GiB / 60 min,
0 GPU. Sacct e squeue ne attestano l'avvio effettivo alle 15:24:55 CET su
login08; richiesta e allocazione con billing=8, cpu=8. I log e le ricevute
di questo seguito sono distinti dal primo controllo. Nessun training
lungo avviato; il rapporto finale va raccolto anche con uscita 2.

## Esito finale, raccolta e consuntivo

| Braccio | Coppia | Esito del criterio firmato |
|---|---|---|
| CN | continuo / A | FAILED: quattro momenti Adam oltre 1% |
| CN | continuo / B | FAILED: gli stessi quattro momenti oltre 1% |
| CN | A / B | PASSED |
| CR | continuo / A | PASSED |
| CR | continuo / B | PASSED |
| CR | A / B | PASSED |

Unico motivo globale: `CN: common-checkpoint historical bounds or invariants failed`.
Il contesto numerico dei 28 rank dei sette run è PASSED (profilo
bf16_math_tf32_off, TF32 spento, SDP math attivo, flash ed efficient spenti),
così come la parità a peso zero. Nessun disallineamento del confine a 544,
supporto attivo mancante, differenza di input/RNG/ordine della coda nelle sei coppie.
Nessuna ragione di inconclusività. Le due riprese indipendenti entro ciascun
braccio sono identiche anche nello stato finale confrontato dal lettore.

L'ordine della coda va distinto dai metadati DDP: confronto descrittivo
separato dei file trace-rank.ddp.jsonl, in ddp_trace_readout.json. Per CN
e CR, continuo contro ciascuna ripresa differisce nei campi
has_rebuilt_buckets, rebuilt_bucket_sizes e rebuilt_per_bucket_param_indices
agli aggiornamenti 545 e 546, in tutti i quattro rank; dal 547 quei campi
coincidono. Fra A e B nessuna differenza nei metadati considerati. Il
contatore locale iteration è escluso dal confronto descrittivo. Questo
rilievo non cambia il verdetto congelato e non dimostra da solo una causa.

I soli superamenti CN sono optimizer.state.266–269.exp_avg: rispettivamente
1,221616%, 1,034195%, 1,316470% e 1,108411%, contro limite storico 1%.
Sono uguali in continuo/A e continuo/B. Questa è la localizzazione del
fallimento nel rapporto, non una diagnosi causale o una revisione della soglia.

Le distanze modello continuo/A sono le stesse di continuo/B:

| Braccio | Studente / spostamento iniziale | Teacher / spostamento iniziale | Studente / movimento 544–576 | Teacher / movimento 544–576 |
|---|---:|---:|---:|---:|
| CN | 0,089749% | 0,008214% | 0,645438% | 0,081902% |
| CR | 0,080816% | 0,007269% | 0,616371% | 0,073844% |

Il criterio firmato usa il denominatore dall'inizializzazione; le due colonne
dal fork sono esclusivamente descrittive. Le A/B hanno distanza zero.
Gli interventi attivi restano nei limiti firmati: CN mediana 0,215412 e
p95 0,366483; CR mediana 0,082787 e p95 0,210753; rapporto mediane 2,601988.
Nessun altro limite modificato o reinterpretato dal launcher.

JSON originale su WORK:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/preflight_readout.json`,
3.081.874 byte, SHA-256
`d4300a2e31c85b39f1f36b8f3146886811d70fe9e24c2d0a9d07e683e84cd4b8`.
Archiviato come preflight_readout.json.gz (234.289 byte) in
readout_execution_20261121, con decompressione verificata byte per byte e
per hash. Log SLURM originale 65035852.out, 2.948 byte, raccolto e verificato;
ricevute del submitter, accounting allocazioni e dettagli batch conservati.
Sintesi leggibile in execution_summary.json; il rapporto completo resta
immutato e non viene rigenerato sul Mac.

Consumo di questa fase: **0 GPU-ore e 0,283333 core-ore**, checkout incluso,
contro richiesta 8,266667 e inviluppo 9,466667 core-ore. Batch MaxRSS
4.678.792 KiB, sotto la richiesta di 30 GiB. Nuova coorte complessiva,
usando il consuntivo precedente: circa 4,740000 GPU-ore / 47,332222 core-ore.
Nessun ulteriore job o retry. Risultati e riferimenti restituiti alla chat
scientifica; il seguito richiede la sua lettura e la decisione di Simone.
