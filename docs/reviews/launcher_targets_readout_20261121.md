# Confronto CPU RMS100/fast125 — esecuzione avviata

**Stato corrente:** controllo corretto superato, checkout 65035600
COMPLETED 0:0 in 41 s; confronto 65035852 RUNNING dalle 15:24:55 CET
del 21/11. SHA effettivo 744ecfa1cccaa7ec650cc874fdc5799a83e0787e,
consegna 2c3e16cf. La sezione seguente conserva lo storico del primo
controllo, fermato prima di qualsiasi invio.

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
