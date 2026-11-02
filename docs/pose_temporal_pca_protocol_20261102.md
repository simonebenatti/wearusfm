# Lettore temporale e controllo PCA — protocollo del 02/11/2026

**Esecuzione conclusa:** [risultati temporali](reviews/pose_temporal_results_20261102.md) e
[confronto appaiato PCA](reviews/pca_existing_results_20261102.md). Catena temporale0,55389core-ore,
0GPU, quattro job COMPLETED0:0. Il disegno scientifico sotto resta quello congelato prima dei fit.

**Aggiornamento operativo, senza modifica scientifica:** Simone autorizza «il lancio diretto di questa catena». GPT può quindi sottomettere il temporale attraverso submitter/ledger, in eccezione limitata alla regola di semplicione. Restano identici codice scientifico congelato, criteri e tetto di 20,5 core-ore/0 GPU.

Simone approva il [piano scientifico](piano_scientifico_20261102.md), commit `63ba143`, e chiede di eseguire i due lavori in parallelo: «va bene procedi, puoi fare tutto tu? in paralello? lettore temporale e PCA?». GPT implementa, verifica, pubblica e analizza; la nuova parte scientifica passa dal critico prima dell'esecuzione. Sottomissioni Leonardo tramite semplicione e submitter/ledger, senza nuovo consenso per questo piano. Nessun training del FM, estrazione GPU, modifica alle loss o lancio di R3 completo è incluso.

## A. Lettore temporale della posa

**Domanda e attesa a due lati.** Sui latenti congelati, il contesto a ±100 ms può ridurre l'errore delle variazioni della posa rispetto alla lettura del solo istante. Se lo riduce, esiste un beneficio leggibile da questa testa; se non lo riduce, il risultato riguarda questa vista e questa scala. Nessuno dei due esiti dimostra che l'encoder debba essere riaddestrato o che l'informazione temporale sia interamente assente/presente. La nuova testa ha più parametri e il backbone è già bidirezionale: nessuna attribuzione esclusiva al meccanismo dell'encoder o al funzionamento online.

Ricetta fissata prima dei nuovi fit:

- B e R3c, **student a 19.618 passi**, cache di posa già estratte con media K64 e latenti T160×384 FP16. Quattro file con SHA/byte/sidecar/ricevute originali; nessuna nuova estrazione prevista. `scripts/pose_temporal_readout.py` genera `results/infra/pose_temporal_20261102/plan.json` dai pin archiviati.
- Due lettori per modello: istantaneo D384 e temporale D1152 con concatenazione **t−4, t, t+4** patch da 25 ms. LayerNorm stateless FP32, eps 1e−5, applicata indipendentemente a ogni istante; nessuno scaler o PCA. Regressione ridge FP64 con intercetta non penalizzata, bersagli standardizzati originali.
- Supporto centrale identico fra tutte le celle: centro t≥4 e t+4<n_patches della finestra; nessun accesso a padding o altre finestre/registrazioni. La maschera della posa governa il bersaglio centrale, **non** l'utilizzabilità dell'EMG ai vicini. Contesti reali non finiti sono errori, non righe da escludere dopo aver visto l'esito.
- Train originale: 40 utenti, 1.907 finestre/200 registrazioni. Stessi due gruppi ottenuti alternando gli utenti ordinati; selezione di alpha mediante SSE standardizzato mascherato pooled sui due fold. Griglia/politica di estensione di E3: potenze di dieci da 10^-2 a 10^4, fino a tre estensioni per lato se il minimo è al bordo; pareggio all'alpha minore. Minimo ancora al bordo dopo le estensioni: segnalazione e arresto prima della lettura scientifica sulla val, nessun tuning ad hoc. Refit sui 40 utenti.
- Val originale: 15 utenti disgiunti, 18.548 finestre/1.950 registrazioni. È già stata osservata: nuova lettura **esplorativa**, non test intatto o nuovo criterio firmato. Iperparametri, supporti e metrica non cambiano secondo il suo risultato.
- Finestre dinamiche definite dalla `pose_var` originale dell'intera finestra rispetto alla soglia già osservata **0,2574231028556824**. Non ricalcolare la soglia o le etichette dopo il taglio dei bordi. Il codice conserva la convenzione di confronto della precedente analisi.
- Primario: **SSE temporale centrata / somma della varianza dei bersagli** sul supporto comune dinamico. Affiancare MAE in radianti, correlazione centrata, varianza predetta/target e conteggi. Ricomputare la ridge istantanea sullo stesso supporto del temporale; il vecchio E3 a finestra intera è soltanto contesto.
- Contrasti fissati: temporale−istantaneo per B, temporale−istantaneo per R3c, differenza dei due miglioramenti. IC percentile95 con **2.000 ricampionamenti degli utenti, seme 20261102**, stessi pesi per tutte le celle. Il segno negativo dell'errore indica miglioramento. Riportare anche per utente; nessuna soglia post hoc di promozione, nessuna equivalenza dedotta da IC che comprende zero.

Tre fasi: `validate` controlla hash/header/config/identità senza metriche scientifiche; `benchmark` legge **solo il train B**, misurando entrambi i lettori; `run` produce le quattro celle e i contrasti. SHA del codice, file di piano e sorgenti controllati prima/dopo; output nuovi, nessuna sovrascrittura. Il test controlla anche la validità dei contesti con posa non annotata, padding e assenza di contaminazione della selezione dalla val.

## B. PCA: prima analisi appaiata degli output già prodotti

**Domanda.** Quanto differiscono le due pipeline esistenti, pool768 senza PCA e pool128 di PSD-3, sugli stessi esempi? Gli score puntuali sono già noti: questo non è un contrasto pre-registrato rispetto alla loro prima lettura. Qui si fissano abbinamento, ricampionamento e reporting prima del nuovo calcolo degli intervalli.

- Quattro NPZ OOF delle sonde fase B a 12.000 iterazioni, originali e immutati, più dieci shard pool/logistica di PSD-3. SHA e riferimenti in `results/infra/piano_scientifico_20261102/comparison.json`. Recupero in sola lettura, array fuori dal repository; nel repo soltanto report aggregati/provenienze.
- Stessi B/R3c teacher19618 e cinque dataset interni. Verificare identità di riga, bersaglio, soggetto, fold e checkpoint, con gestione esplicita delle etichette testuali di EPN612. Riordinare per identità originale; righe mancanti/duplicate, classi spurie o pin errati fermano il confronto. Non usare score di report come sostituti delle predizioni.
- Balanced accuracy OOF; bootstrap appaiato per soggetto entro dataset, **2.000 repliche, seme 20261102**. Stessi campionamenti per entrambe le pipeline e i modelli. Pesi congelati: P1=8/13 DB2+3/13 DB3+2/13 DB6, P2=1/2 EPN612+1/2 UCI.
- Contrasti: pool128−pool768 per modello/dataset e aggregato, B−R3c in ciascuna pipeline, differenza fra questi due contrasti. IC percentile95 descrittivi; non coprono refit/pretraining indipendenti e non correggono la molteplicità. Nessun nuovo fit, nessun MLP, nessuna scelta degli iperparametri.
- **Non è l'effetto isolato della PCA:** cambiano anche precisione del preprocessing e thread. La griglia C è invece identica. La lettura può motivare un successivo controllo con implementazione identica che salti soltanto la PCA; tale rifit non è parte automatica di questa prima analisi.

Attesa a due lati: una perdita consistente invita a limitare le conclusioni tratte dalle viste compresse; un contrasto incerto/non sfavorevole non dimostra che la PCA sia ininfluente per tutti i lettori o per la vista decoder. Nessun risultato di questo confronto seleziona i parametri del lettore temporale.

## Risorse, parallelismo e condizioni operative

Preparazione, test e letture dei due rami possono procedere in parallelo perché non condividono iperparametri o output. Il parallelismo reale dei job dipende da code e QoS: non si promettono due allocazioni simultanee. Nessun rilancio se l'ID logico è già prenotato o sottomesso.

| Attività Leonardo | CPU | Memoria | Walltime massimo | Tetto core-ore |
|---|---:|---:|---:|---:|
| Nuovo checkout congelato | 1 | 2 GB | 10 min | 0,167 |
| Validazione quattro cache temporali | 2 | 4 GB | 10 min | 0,333 |
| Timing sul train B, entrambi i lettori | 8 | 30 GB | 30 min | 4,000 |
| Confronto temporale completo, **dopo il timing** | 8 | 30 GB | fino a 2 h | fino a 16,000 |
| **Tetto della catena temporale** | | | | **20,500** |

**GPU: zero.** Questi sono tetti prudenziali di allocazione, non una previsione dei tempi reali. Il precedente rapporto ridge istantaneo B/R3c costava 0,411 core-ore, ma non misura la nuova larghezza1152. Dopo benchmark, scegliere walltime e memoria con margine esplicito entro i tetti; se la proiezione non vi rientra, non accodare il run e registrare il problema prima di ridefinire le risorse. L'avvio dipende soltanto da integrità, costo e review, non dal segno dei risultati. Il benchmark non legge metriche val.

La PCA si analizza localmente sugli output, a un thread, senza fit o GPU. Se fosse necessaria l'esecuzione del bootstrap su Leonardo, preparare un job CPU separato e contabilizzato prima di lanciarlo; non eseguire calcolo pesante sul login node. Il residuo della campagna dopo sonde/PSD-3 era circa320 core-ore e64,45 GPU-ore: verificare il ledger prima di prenotare. Nessuna spesa già effettuata viene ricontata.

Per la catena: SHA completo pubblicato, checkout di campagna dedicato, submitter unico e ledger, output sotto `$WORK/wearusfm_runs/results/pose_temporal_20261102/`, log esterni. La chat semplicione non era raggiungibile all'avvio; Simone ha poi autorizzato direttamente GPT per questa sola catena, come registrato sopra. Nessun agente omonimo creato. Tutte le sottomissioni sono concluse: non duplicarle. Il recupero in sola lettura dei quattro OOF PCA è riuscito e i loro SHA coincidono con quelli originali.
