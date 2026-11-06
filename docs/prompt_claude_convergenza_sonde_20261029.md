# Prompt per Claude — controllo della convergenza delle sonde B19618/R3c

Simone ha approvato il controllo delle sonde e ti affida l'implementazione. GPT mantiene la responsabilità scientifica e rivede la consegna; le esecuzioni su Leonardo spettano a semplicione. **Prepara codice, test, protocollo eseguibile e preventivo; non sottomettere job.** Il controllo è già autorizzato: dopo la review tecnica GPT lo affiderà a semplicione senza chiedere un secondo consenso per lo stesso perimetro.

Lavora nel tuo worktree/ramo, preserva i cambi altrui, committa e pusha la consegna su origin. Questa assegnazione tecnica è specifica del controllo delle sonde, non un cambio generale della responsabilità scientifica. Non modificare trainer, loss, pooling o checkpoint.

## Contesto da leggere

Riferimento dei risultati: **f7433f441d6bbdc3cc97da47df253c8bf9b24152**, pubblicato su `origin/codex/r3-long-collaudo-20261028`.

- `docs/reviews/b19618_results_20261029.md` e `docs/continuazione_b19618_20261028.md`.
- `results/infra/pilot_posa/b19618_20261029/final/`: rapporti, provenienze, log originali e `accounting_summary.json`.
- `results/infra/pilot_posa/b19618_20261029/independent_reviews/`.
- `src/wearusfm/harness/probe.py`, `scripts/channel_readout.py`, comparatori appaiati e `scripts/pose_b19618_eval_guard.py`, con i relativi test.
- `CLAUDE.md` e le ultime voci di `docs/claude_GPT_shared`.

Le sonde hanno completato i fit: P1 B−R3c = −0,556 punti, P2 = −1,233. I log contengono **18 emissioni di ConvergenceWarning** (R3c-P1/P2: 5/4; B-P1/P2: 4/5), senza C/fold/tipo di fit. Non sappiamo quali siano candidati e quali refit finali. La domanda è: **la conclusione sul trasferimento è sensibile all'arresto del solver?** Non cerchiamo una ricetta che faccia vincere B.

## 1. Conservare la misura di riferimento

Riusare le cache già estratte dei teacher B e R3c a 19.618, strato `pool`, larghezza 768, nessuna PCA. Stessi cinque dataset, quattro fold, soggetti/finestre, classi ammesse, ordinamento, preprocessing, scaler stimato solo sul train, griglia C, semi, pesi e bootstrap. Test ufficiali esclusi come prima. Nessuna estrazione GPU o riaddestramento del backbone.

Ogni fit attuale usa `LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)`. La griglia è `(1e-3, 1e-2, 1e-1, 1, 10, 100)`: sei candidati sul train, scelta sul val interno (a parità vince C minore), poi un nuovo fit sul **solo train**. Non unire train e val nel refit. In totale: **280 fit = 2 encoder × 5 dataset × 4 fold × 7**.

Registrare i parametri effettivi con `get_params()` nell'ambiente originale, senza dedurli dai default del Mac o aggiornare sklearn. Il freeze archiviato riporta numpy 1.26.4, scipy 1.12.0, sklearn 1.4.1.post1: confermare l'ambiente effettivo delle sonde. Conservare anche dtype, BLAS e limiti dei thread. Con questi default sono attesi `lbfgs`, L2, `tol=1e-4`, `multi_class=auto`, `warm_start=False`.

## 2. Strumentazione per singolo fit

Aggiungere un audit opzionale, con default scientifici invariati e senza duplicare la logica della sonda in un secondo programma divergente. La soluzione concreta può essere un risultato diagnostico o un callback; mantieni compatibili i chiamanti esistenti.

Per ciascun fit salvare in forma strutturata:

- encoder/checkpoint/cache e relative firme, dataset, fold, seme, fase `candidate` oppure `refit`, C;
- ricetta completa del solver, dimensioni e classi del train, firma dei ruoli e dello scaler;
- `n_iter_`, tempo trascorso, warning con categoria e messaggio, raggiungimento del limite, eccezione eventuale;
- score sul val per ogni candidato, C selezionato e regola di tie-break; score e predizioni finali con le provenienze già previste.

Catturare i warning **attorno alla singola chiamata `.fit`**, senza confonderli con lo scoring. Conservare separatamente i warning di metrica: i 978 messaggi di classi assenti della fase DB3 non sono 978 fit falliti. Mantenere visibilità degli avvisi nei log o in un riepilogo esplicito; niente soppressione silenziosa. Una cattura non deve trasformare un'eccezione in successo.

`n_iter_ < max_iter` e assenza di warning sono diagnostiche del criterio configurato, non una prova matematica di ottimalità. Segnalare anche arresti anomali con iterazioni sotto il limite. Non eliminare candidati non convergenti dalla selezione e non cambiare la metrica.

## 3. Due fasi fissate prima dell'esecuzione

**Fase A — riproduzione strumentata, 3000 iterazioni.** Ricalcolare tutte e quattro le sonde B/R3c × P1/P2 con i parametri originali. Attribuire i warning ai 280 fit. Confrontare con gli originali: identità di y/soggetti/righe/fold, C selezionati, predizioni OOF e punteggi. Con gli stessi input e ambiente ci si attende uguaglianza delle predizioni; salvare anche conteggi esatti delle differenze. Se cambia un risultato, trattarlo prima come problema di riproducibilità: non alzare le tolleranze dopo aver visto i numeri per farlo passare.

Per la parità OOF confrontare gli array e il loro ordine: un nuovo archivio ZIP/NPZ può avere byte diversi pur contenendo gli stessi array. Gli hash degli input originali, invece, devono restare identici.

**Fase B — robustezza simmetrica, massimo 12000 iterazioni.** Dopo la verifica della fase A, eseguire una sola variante fissata in anticipo, cambiando **solo `max_iter: 3000 → 12000`**, con la stessa tolleranza originale. Rifare l'intera griglia e il refit da zero per entrambi gli encoder e tutti i cinque dataset, anche se il warning è comparso in un solo braccio. La maggiore convergenza di un candidato può cambiare C: non basta rifare soltanto il modello finale al vecchio C. Nessun warm start, cambio di solver o aumento simultaneo della tolleranza.

La fase B è una verifica della misura, non una ricerca di iperparametri guidata dagli score held-out. Se a 12000 restano warning, dichiarare il limite irrisolto; nessun retry automatico o nuova scala di iterazioni. Non scegliere a posteriori fra 3000 e 12000 in base al vantaggio di B. Gli esiti storici restano intatti.

Preparare fasi distinte e un controllo automatico dei prerequisiti: non avviare B se A è incompleta, non riproduce la baseline o presenta un'incongruenza di provenienza.

## 4. Confronti e provenienza

Per ogni fase produrre P1/P2, dettaglio per dataset/fold, C selezionati, numero e posizione dei fit con warning. Calcolare:

1. B−R3c a 3000 e a 12000, con il bootstrap utenti appaiato già congelato.
2. Variazione 12000−3000 per ciascun encoder.
3. Variazione del confronto: `(B−R3c)_12000 − (B−R3c)_3000`, usando gli stessi ricampionamenti dei soggetti e i pesi originali, non sottraendo gli estremi di due IC separati.

Riportare anche frazione di predizioni cambiate e spostamenti dei C. Nessuna nuova soglia di adozione: risultato descrittivo di robustezza, un solo seme di pretraining. Non usare `not_harmful` legacy come prova di assenza di danno. Un delta stabile pur con warning residui significa insensibilità osservata a questo aumento di iterazioni, non convergenza certificata.

Nuovi percorsi di output; originali read-only. Verificare SHA delle cache e relative provenienze prima e dopo, checkpoint teacher a 19.618 e firme di dati/split. Aggiungere una firma della ricetta del solver all'audit e alla guardia del confronto: i vecchi `PROBE_KEYS` non bastano. All'interno di ciascuna fase B e R3c devono avere la stessa ricetta; fra fasi è ammessa soltanto la differenza di `max_iter`. Gestire esplicitamente report legacy privi di questi nuovi campi, senza attribuire loro parametri non verificati.

Registrare separatamente il codice che ha prodotto le cache e il nuovo codice di analisi. Eseguire quest'ultimo da uno SHA pubblicato e congelato; non dichiarare che appartenga al vecchio checkout scientifico e non modificare quel checkout.

Conservare tutti i 280 record attesi per fase e rifiutare output parziali. Il validatore deve supportare etichette testuali EPN612 e numeriche: `isfinite` sulle feature e sui punteggi numerici, non indiscriminatamente sulle etichette. Stesse esclusioni dichiarate, inclusa la singola riga UCI malformata; nessuna intersezione opportunistica per rendere confrontabili due risultati diversi.

## 5. Test CPU necessari

- Audit acceso/spento: identici scaler, selezione C, predizioni e score con parametri invariati; etichette numeriche e stringhe.
- Warning indotto: attribuzione al candidato/refit corretto, record per tutti i fit e distinzione dai warning di scoring/bootstrap. Eccezioni propagate e report incompleto riconosciuto.
- Assenza di leakage: scaler/fit sul solo train, C dal solo val; perturbare gli ingressi held-out non cambia scaler o C. Fold disgiunti per soggetto e refit con gli indici train originali.
- Propagazione dei parametri a tutti i candidati e al refit; tie-break e default invariati. Nessun candidato scartato perché non convergente.
- Guardia di identità e ricetta: rifiuto di fase/parametri/hash/ordine incongruenti, inclusi casi EPN612; permanenza della correzione del precedente errore `np.isfinite`.
- Piccolo esempio controllato per i delta appaiati, compreso il delta dei delta e i pesi degli aggregati.

Eseguire test mirati e regressioni pertinenti con `PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; dichiarare ambiente e comandi. Non simulare un successo su cache reali non disponibili localmente.

## 6. Preventivo e consegna a GPT

I quattro job originari delle sonde hanno consumato **26,517778 core-ore allocate** (8 CPU): R3c-P1 10,542222; R3c-P2 2,735556; B-P1 10,108889; B-P2 3,131111. Comprende la guardia finale del P2 storico fallito dopo il fit. Sono tempi osservati, non un tetto per il nuovo codice.

Preventivare separatamente fase A, fase B, preflight/checkout e confronti, includendo CPU, memoria, walltime, thread effettivi e output. Il fattore quattro di `max_iter` non significa quattro volte il tempo reale né garantisce un limite temporale. GPU richieste: zero. Residuo registrato dopo B19618: **432,14 core-ore CPU-only / 64,448056 GPU-ore**.

Per A, riferimento iniziale: due P1 da 2 ore e due P2 da 45 minuti, a 8 core, tetto 44 core-ore più preflight/rapporto. Per B la proiezione grezza di quattro volte i tempi storici è 106,07 core-ore (132,59 includendo una replica ai tempi storici), non un preventivo garantito. Usare i tempi per fit della fase A per affinare e pubblicare risorse e tetto di B prima di eseguirla, senza modificare la ricetta numerica fissata qui in funzione dei risultati scientifici.

Rispetta il limite della partizione seriale: se un job aggregato rischia il walltime, prepara job per dataset e un aggregatore deterministico, preservando identici fold e semi. Nessuna duplicazione di fit o perdita della provenienza fra shard. Non proporre semplicemente un job più lungo del limite della coda.

Consegna protocollo, implementazione, test, schema degli audit, comandi e preventivo riproducibile. Elenca eventuali obiezioni scientifiche prima di cambiare il disegno. Pubblica commit e aggiorna entrambe le sezioni del registro e `docs/decisioni.md`. GPT rivedrà codice e preventivo; semplicione userà submitter/ledger comune per i lanci, senza modificare checkout attivi.

**Fuori perimetro:** nuovi training, loss, PSD, latenti non mediati, tuning sul test, upgrade delle librerie o correzioni opportunistiche di un solo encoder. Questi sono filoni da discutere separatamente.
