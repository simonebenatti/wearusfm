# Confronto CPU RMS100/fast125 — controllo preliminare fermato

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
