# Review unica: target125 e compatibilità storica

18/11/2026. Ramo Claude `claude/target-windows-fix-20261118`,
commit `0d91297776a5ee63ccd3f3d10b48cf63163172c9`, base `b4b0963c7da763b0eefb1d078b78894a096de1a6`.
Mandato diretto di Simone: «fai tutto tu», dopo aver distinto questa consegna dalla TCN.

## Blocchi

Nessun difetto bloccante individuato nella review numerica e della compatibilità storica.
Integrazione approvata: tutti i 3.380 casi della suite CPU sono passati. Ricevuta: `results/infra/target_windows_integration_20261118/verification.json`.
Nessun lancio, deploy o modifica dei checkout congelati in questa verifica.

## Prove indipendenti

- Oracolo NumPy float64 su 12 frequenze incluse 200, 1926, 2148 e 10240 Hz: maschere identiche,
  errore massimo delle log-potenze 1,2584936026627247e-6, sotto 3e-5.
  A 2148 Hz supporto di 268 campioni, arrotondamento half-even verificato indipendentemente.
  Bin 2/3/4/8/14/23 alle 11 frequenze alte; 200 Hz conserva le prime 3 bande.
  Script e ricevuta: `results/infra/target_windows_integration_20261118/numerical_oracle.*`.
- Uguaglianza esatta ricostruita sui due config reali T100-LQ (`config_T100-LQ.json`
  e `completed_5000/config.json`) dopo i soli campi 25/50: versione v1 conservata.
- `historical_args` conserva argomenti espliciti, abbreviazioni ed uguale; la normalizzazione
  dei config aggiunge soltanto due campi, non accetta altri campi mancanti per tolleranza.
- `consolidation_continue.py` identico a 17dd455: blob
  `ba5832ceaa85634dc67d6ea599deee35244b7cb3`.
- Il fix riguarda target di pretraining. `consolidation_long_pose` legge target/mask/lat
  dalle cache: non rigenera i bersagli. Pin, ritaglio e metriche della TCN restano invariati.
  Non applicare retroattivamente l'identità dei nuovi sorgenti alle ricevute dei produttori storici.

## Note non bloccanti, nessun secondo giro richiesto

N1. Nell'oracolo del test `test_ms_fast_target_window.py` la lunghezza usa half-up,
mentre la produzione usa half-even; i quattro fs parametrizzati evitano il caso limite.
Suggerimento a Claude per il suo ramo: aggiungere 1926/2148 e usare half-even indipendente
(es. divmod(fs,8)). La prova indipendente su 2148 è già positiva: non è un bug di produzione.

N2. `with_historical_targets` imposta sempre 25/50, diversamente dagli altri helper che
preservano i valori presenti. Il solo impiego attuale in step1_ab è corretto; non usarlo
come parser generico di config arbitrari.

L'uguaglianza dei conteggi FFT risolve la discontinuità indicata, non dimostra identità
fra tutti gli spettri né superiorità scientifica della finestra di 125 ms. Nessun rilievo di Claude
è stato dichiarato sbagliato senza prova; la sua distinzione sul limite LQ prima del 04/11 è corretta.

## Perimetro restante

C1, guardia WSD per piani nuovi: rinviata, manca uno schema esplicito di migrazione dei piani.
La nuova linea 3 usa config JSON completi verificati prima del runtime, e riporta anche le due CLI.
C2, campi LQ mancanti negli artefatti prima del 04/11: limite preesistente conservato e dichiarato.
Non si allentano confronti esatti né si riscrivono config/checkpoint storici.

Per la linea 3: worktree separato `wearusfm-contrastive-targets`, nuova coorte 100/125,
nuovi audit/calibrazione/λ/benchmark obbligatori. Prima degli invii costi da presentare e
problemi bf16/ripresa da risolvere; ricevute vecchie non riutilizzabili. TCN e linea 1 invariati.

## Suite completa e incidenti locali

Sorgenti e test del merge `7364f7c` identici a Claude `0d91297`. Copertura completa:
869 + 751 + 892 + 862 casi, più 6 DDP eseguiti separatamente = **3.380 passati**.
Non è una singola invocazione pulita: il tentativo in sandbox aveva restrizioni multiprocesso;
il primo giro esterno ha esaurito il disco, mentre tre gruppi e DDP erano passati. Il gruppo
interrotto è stato recuperato con partizione verificata, 307 + 562 casi distinti. Un test di
supervisione dei processi si è bloccato in una chiamata di attesa: interrotto, è passato da solo
in 3,27 s e nel recupero finale, senza cambiare sorgenti. La causa transitoria non è dimostrata.
Log originali, XML e partizione dei casi sono conservati; nessun errore irrisolto.

Su richiesta di Simone sono stati rimossi solo temporanei pytest non più attivi:
8,43 GiB recuperati, circa 17 GiB disponibili immediatamente dopo. Dataset, checkpoint,
worktree e cache dei pacchetti conservati. Ricevuta `mac_cleanup.json`.
