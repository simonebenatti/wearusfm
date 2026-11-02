# Review prima dell'esecuzione — lettore temporale e PCA, 02/11/2026

Disegno: `docs/pose_temporal_pca_protocol_20261102.md`, seguito approvato di `63ba143`.
Implementazioni: `scripts/pose_temporal_readout.py` e `scripts/pca_existing_oof_compare.py`.

Il subagent critico ha riletto indipendentemente disegno, codice e test: **nessun rilievo scientifico/algoritmico bloccante**. La review non equivale a esecuzione su Leonardo. GPT ha rieseguito i due file di test insieme: **35 passati in 0,99 s**, Python del venv wearusfm-tok, PYTHONPATH=src e OMP/OpenBLAS/MKL=1. `bash -n` del launcher e `git diff --check` passano.

Controlli sostanziali del temporale: corrispondenza con ridge diretta su un problema sintetico, lag senza attraversare finestre, padding da `meta.n_patches`, indipendenza degli ingressi dalle annotazioni dei vicini, normalizzazione per patch, identico supporto centrale, CV per utenti esclusivamente sul train, minimo di alpha al bordo rifiutato, ricampionamento appaiato e differenza dei miglioramenti. Benchmark senza lettura della val; la soglia delle finestre dinamiche resta quella archiviata. Le cache non preservano gli slot interi e il backbone è bidirezionale: limiti dichiarati.

Controlli sostanziali PCA: hash originali, righe riordinate per identità, confronto esatto di target/soggetti/fold, esclusione di duplicati/mancanti, classi lecite, etichette EPN612 testuali e altre numeriche, stessi pesi bootstrap e aggregati P1/P2. Validazione dei dati reali completata prima del bootstrap: quattro OOF e dieci shard coincidono sui supporti (DB2 38.544/32 utenti, DB3 9.834/8, DB6 33.961/8, EPN612 47.100/314, UCI364/19). Nessun nuovo fit. Il confronto non identifica causalmente la sola PCA perché precisione e thread differiscono.

In questa consegna il risultato temporale non esiste ancora. Restano timing train-only e verifica runtime/cache prima del run. Nessun esito PCA sceglie parametri del lettore temporale; nessuna soglia di adozione o autorizzazione automatica a un nuovo training del FM.
