# Review indipendente offline — benchmark sonde scaling, 05/11/2026

**Verdetto: evidenze coerenti; STOP confermato. Il tetto dei fit di 136 core-ore è già incompatibile con la formula congelata, anche senza completare i due benchmark mancanti.** Nessun job, SSH, modifica del repository o score downstream effettuato da questo revisore.

## Verifiche eseguite

Dossier: `results/infra/scaling_300m_20261104/cpu_readers_20261105/probe_benchmark/` nella worktree scaling-300m.

- Ricalcolati indipendentemente dimensione e SHA-256 di tutti i **30 file** nel manifest; nessuna discrepanza. Verificati anche i **13 sorgenti** pinnati in `probe_plan.json` contro i byte locali.
- Verificati i **13 report completi**: identità cella/dataset/arm, codice `aef665ba7ea66fc80291846487907bb3ef9107df`, piano, ricetta, teacher/step 5000, fold 0 solo train, nessuna selezione, dimensione 768/1792 e sei C esatti `[0.001, 0.01, 0.1, 1, 10, 100]`. Ogni JSONL completo coincide con la lista `fits` del report, e la sintesi di convergenza coincide.
- Rigenerati i quattro ruoli per ciascuno dei cinque dataset dal piano con `calibration_probe.fold_roles`; coincidono sia il pin dell'intera lista sia gli hash fold0 nei report. Tra le viste coincidono numero di esempi train, conteggi dei quattro fold, classi e frequenze. Questa è una verifica dei metadati: le cache NPZ non sono state ricaricate in locale e non si certificano nuovamente qui tutti i soggetti/label dei file remoti.
- **78 fit nei report completi + 5 fit nell'audit parziale EPN = 83 fit terminati**, tutti dichiarati convergenti dal criterio del solver, senza warning, eccezione o raggiungimento di 12.000 iterazioni. Verificati solver LBFGS, tolleranza 1e-4, class_weight balanced, seed 0 e StandardScaler identificato nello stesso fold. Nessuno score held-out è contenuto o calcolato nel benchmark.
- I **13 report completi hanno contesto numerico identico**, hash `1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c`: Python 3.11.6, NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.1.post1, float32 e cinque threadpool tutti a 8 thread. L'audit EPN parziale non è promosso a report completo e non possiede un proprio certificato conclusivo del runtime.
- Riletti accounting e `sacct.txt` originali, verificati hash, stato parent, CPU, nodo login08, durata, RSS derivato dai record step e costo. Eseguito inoltre `audit_collected.py.main()` in sola lettura: riproduce esattamente `diagnosis.json`, e il suo lower bound coincide col ricalcolo indipendente.

Pin principali: piano `d2cc7da1cf08473067d96b52bff532463453ff5ce633814743b1a53dda5bbb63`; hash canonico piano `9954c47fc03c2ccde86991f1eac226b9a76d4cc8b8ce17fe86757578805ac65b`; auditor `bb76b45a9776486bc01628b362f1f718ab28cfb2ca8e24caa474237f15d6342c`; diagnosis `c34cb70ee44db4b743fc299d83e8bb0b96ef6d90fb90d015b998e7e2d838ae97`.

## Esito operativo e costo osservato

| Cella | Job | Stato parent | Secondi parent | Core-ore osservate | MaxRSS GiB |
|---|---:|---|---:|---:|---:|
| 30M naturale | 63593262 | COMPLETED 0:0 | 2018 | 4.484444 | 2.633930 |
| 300M naturale | 63593508 | TIMEOUT 0:0 | 5584 | 12.408889 | 3.838757 |
| 300M matched | 63593593 | COMPLETED 0:0 | 2194 | 4.875556 | 2.669174 |

Somma dei soli tre benchmark sonde: **21.768889 core-ore**. Non è il consuntivo dell'intera campagna.

Il parent TIMEOUT resta fallito nonostante ExitCode `0:0`; il batch è CANCELLED `0:15`. Il log riporta esplicitamente la cancellazione per limite temporale. Il walltime prenotato era 5400 s; l'accounting parent registra 5584 s, quindi il costo effettivo è 12.408889 core-ore, non 12. Non sommare parent e step e non troncare il consuntivo al limite richiesto: i 184 s aggiuntivi sono osservati, non un'autorizzazione a più calcolo. Non attribuisco un meccanismo preciso del ritardo di terminazione senza evidenza di scheduler.

Sono completi 5/5 dataset controllo, 3/5 naturale300M e 5/5 matched. Nell'EPN naturale sono persistiti cinque C, fino a10; manca il report EPN conclusivo e manca UCI. Non si può identificare con certezza l'istruzione interrotta dal solo audit persistito dopo ogni fit. Nessuna evidenza di OOM: RSS batch circa3.84GiB contro30GiB. Nessuna evidenza di un fallimento della convergenza nei fit terminati; questo non dice ancora nulla sul vantaggio scientifico del modello.

## Budget: ostacolo già dimostrato dai 13 completi

La formula effettiva in `scripts/scaling_probe.py:validate_resources` è, per shard:

`120 + 1.5 * somma_dataset[(somma_6C t_C + max_6C t_C) * somma_fold max(1, n_train_fold/n_train_fold0)]` secondi.

Con8CPU, i soli termini variabili per la copertura dei quattro fold producono:

| Cella | Dataset completi | Core-ore della formula, senza overhead |
|---|---:|---:|
| 30M naturale | 5 | 38.70085668746337 |
| 300M naturale | 3 | 67.72077000828517 |
| 300M matched | 5 | 41.713241859130534 |
| **Totale noto** | **13** | **148.1348685548791** |

Sono già **12.134869 core-ore sopra il tetto136**, prima dei120s per shard, di ogni arrotondamento e dei due dataset mancanti. Cambiare partizione dei dataset/fold non riduce questi termini additivi; introduce semmai più overhead. Il limite136 è quindi impossibile secondo la formula pubblicata per qualunque suddivisione che conservi tutta la copertura. È un lower bound del preventivo prescritto, non una prova che l'effettivo tempo futuro sarà almeno altrettanto lungo.

Per completezza diagnostica, i cinque C EPN già terminati sommano1422.494821681641s e hanno massimo674.8787507489324s. Con tutti i rapporti di supporto posti prudenzialmente a1, danno già ulteriori **27.964980965740978 core-ore** di termini per quattro fold; totale noto + questo lower bound parziale **176.0998495206201 core-ore**. Mancano ancora sestoC, UCI e overhead. Questo calcolo non rende EPN un benchmark valido e non costituisce un preventivo finale.

Il gate dei **7200s per shard** non è già impossibile a livello di singolo fold nei13 report: il massimo `120+termine_fold` è **3830.1352136802816s**, DB2 naturale300M fold1. Alcuni dataset con tutti e quattro i fold in uno shard superano7200s (DB2 nelle tre viste e DB6 naturale300M), ma questa è una necessità di partizione, non ancora un'impossibilità del limite2h. Mancano i dati completi per certificare EPN naturale e l'intera copertura in non più di15shard.

## Conseguenza e alternativa lecita da preparare

O-6/O-8 prescrivono STOP al primo errore/TIMEOUT e nessun retry; quindi **nessun nuovo fit, compreso quello della posa**, anche se il suo budget660s è già disponibile. O-7 richiede mantenere i sacct originali e contabilizzare anche questo tentativo fallito. I tre report completi prodotti nel parent TIMEOUT sono evidenze diagnostiche utilizzabili per questa stima, non una sanatoria automatica del job né un'autorizzazione implicita al riuso operativo.

Si può preparare offline un emendamento operativo che mantenga immutati ricetta, C, tolleranza, fold, feature, runtime e thread: eventuale recupero mirato dei soli benchmark mancanti, output nuovi, riuso esplicito dei completi verificati, nuovo costo cumulativo e budget fit coerente. Separare EPN/UCI dal pacchetto sequenziale evita di rifare i tredici report, ma **da solo non risolve il cap136**. Il launcher congelato esegue i cinque dataset del benchmark; un eventuale dispatcher mirato va documentato e revisionato, non aggirato ad hoc. Non si riducono C, fold o dati e non si modifica la formula per far rientrare il costo.

Prima di nuovi invii serve risolvere e pubblicare l'emendamento del budget e del recupero, con la decisione prevista dal protocollo dopo STOP. Nessun rilancio è proposto come già pronto o autorizzato da questa review.
