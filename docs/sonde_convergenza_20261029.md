# Controllo della convergenza delle sonde P1/P2 di B19618 contro R3c — protocollo eseguibile e preventivo (29/10/2026)

**Stato: ESEGUITO il 29/10 (fase A 14:03-17:31, fase B 17:47-23:42, 30 job + checkout, tutti COMPLETED 0:0; risultati nel §8; rapporti,
audit e confronto in `results/infra/pilot_posa/sonde_convergenza_20261029/{A,B,confronto}/`). Era: in esecuzione dal 29/10 pomeriggio (checkout `pilot-posa-sonde-20261029` a f6c89cc, blocco 1 sottomesso da semplicione); consegna
rivista dopo la review di GPT (de06bca: disegno approvato, tre correzioni R1-R3 fatte) e dopo la review del `critico` del 29/10 (vincolo
sull'hash di `probe.py` spostato con decisione, nessun cambio al codice delle sonde).**
Richiesta: `docs/prompt_claude_convergenza_sonde_20261029.md` (GPT, commit b1c1bda), su mandato di Simone («stavolta devi scrivere tu il
codice, poi gpt lo revisionerà ... Non sottomettere job: GPT farà la review e semplicione li eseguirà»). Il controllo e' gia' autorizzato nel
perimetro del prompt; dopo la review tecnica GPT lo affida a semplicione. Nessuna modifica a trainer, loss, pooling o checkpoint; nessun
training GPU. Correzioni della review: R1 parita' completa su tutti i campi scientifici confrontabili; R2 audit salvato fit per fit anche in
caso di errore, chiavi `experiment_phase` (A/B) e `fit_phase` (candidate/refit) distinte; R3 ricetta e contesto numerico verificati prima dei
fit della fase B, rapporti attestati dal cancello ricontrollati.

## 1. Domanda e disegno (fissati prima dell'esecuzione)

Le sonde di B19618 (`docs/reviews/b19618_results_20261029.md`, risultati f7433f4) hanno dato P1 B−R3c = −0,556 punti [−1,104; −0,026] e
P2 = −1,233 [−2,997; +0,576], con **18 emissioni di ConvergenceWarning** di lbfgs nei log (R3c-P1/P2 5/4, B-P1/P2 4/5) non attribuite a C, fold
o tipo di fit. La domanda e' una sola: **la conclusione sul trasferimento e' sensibile all'arresto del solver?** Non si cerca una ricetta che
faccia vincere B. Due fasi, nell'ordine, con un cancello automatico fra le due:

| Fase | Che cosa | Parametri | Esito atteso |
|---|---|---|---|
| **A** | riproduzione strumentata delle quattro sonde (B/R3c x P1/P2) | identici agli originali: `LogisticRegression(C, max_iter=3000, class_weight="balanced", random_state=0)`, lbfgs, L2, `tol=1e-4`, griglia C 10^-3..10^2, scaler sul solo train del fold, C sulla validazione, refit sul solo train | **uguaglianza** di righe, soggetti, ruoli dei fold, C scelti, predizioni OOF e S con gli originali; i 280 fit (2 encoder x 5 dataset x 4 fold x 7) con il loro record di audit |
| **B** | la sola variante fissata: `max_iter` 3000 -> **12000** | tutto il resto identico (stessa tolleranza, stessa griglia, candidati e refit rifatti da zero per ENTRAMBI gli encoder e tutti i dataset, anche dove il warning e' comparso in un solo braccio) | B−R3c a 12000, 12000−3000 per encoder, differenza dei due B−R3c, sugli stessi ricampionamenti dei soggetti; frazione di predizioni cambiate e spostamenti dei C |

Se a 12000 restano warning, il limite si dichiara irrisolto: nessun retry, nessuna nuova scala di iterazioni, nessuna scelta a posteriori
fra 3000 e 12000. Gli esiti storici restano intatti; questa e' una lettura descrittiva di robustezza, con un solo seme di pretraining e
nessuna nuova soglia di adozione. Un IC che contiene 0 e' inconclusivo, non equivalenza; un delta stabile con warning residui e'
insensibilita' osservata a questo aumento di iterazioni, non convergenza certificata. `n_iter_ < max_iter` e assenza di warning sono
diagnostiche del criterio configurato, non una prova di ottimalita'.

## 2. Che cosa cambia nel codice (default scientifici invariati)

- **`src/wearusfm/harness/probe.py`.** `linear_probe(..., max_iter=3000, audit=None, context=None)`: con `audit` (una lista) ogni `.fit`,
  i sei candidati e il riaddestramento, lascia un record; il percorso di calcolo e' identico a prima (stesse chiamate, stesso ordine).
  I warning sono catturati **attorno alla singola chiamata `.fit`** (`warnings.catch_warnings(record=True)` + `simplefilter("always")`) e
  **riemessi subito dopo** con i filtri dell'utente: nessuna soppressione. I warning della metrica (`balanced_accuracy_score` sulla
  validazione: classi predette assenti) e del bootstrap dell'errore standard sono contati a parte (`scoring_warnings`, `bootstrap_warnings`):
  i 978 messaggi di classi assenti della fase DB3 non sono 978 fit falliti. Un'eccezione dentro il fit si registra (`exception`) e si rilancia.
  `ProbeResult` guadagna `val_scores` (punteggio di ogni candidato), `solver_params` (`get_params()` del refit) e `bootstrap_warnings`.
  `solver_recipe(max_iter)` e' la firma della ricetta del solver (`get_params()` senza C, griglia, regola di scelta, refit, scala, versioni di
  sklearn/numpy/scipy) registrata nell'ambiente in cui gira: i confronti la esigono uguale fra B e R3c dentro una fase. Se l'oggetto `audit`
  ha un metodo `persist(record)`, la libreria lo chiama appena un record e' definitivo (candidato: dopo il punteggio di validazione;
  riaddestramento: dopo il fit; eccezione: prima del rilancio): e' il gancio del salvataggio fit per fit (R2).
- **`scripts/channel_readout.py`.** `oof_predictions(..., max_iter=3000, audit=None, folds=None, context=None)`: passa i parametri a
  `linear_probe`, aggiunge ai record per fold (solo in audit) `fold`, `max_iter`, `val_scores`, `tie_break`, conteggi dei fit al limite e con
  warning, firme dei ruoli e dello scaler; `folds` calcola un sottoinsieme di fold (uno shard) con le stesse righe e lo stesso ordine della
  sonda intera. La lettura e il controllo della cache sono in `load_cache` (prima dentro `probe()`, testo dei controlli identico).
  `check_readout_pairing`: se uno dei due rapporti porta `solver_recipe_sha256`, devono averlo entrambi uguale; **legacy contro nuovo e'
  rifiutato** (i parametri del legacy non sono verificati), **due rapporti legacy continuano ad accoppiarsi** come prima (nessuna proibizione
  dei confronti storici): la parita' con gli storici si fa nella fase `parity`, non nel confronto ufficiale.
- **`scripts/probe_convergence.py`** (nuovo): le fasi `probe` (uno shard), `aggregate` (rapporto nel formato di `channel_readout.py --stage
  probe`, OOF e audit uniti: 7 fit per fold, **28 per dataset, 84 per P1 e 56 per P2**, 280 per fase; con `--original` anche la parita'),
  `parity`, `gate`, `compare-phases` (bootstrap congiunto) e `plan`. Ogni uscita e' nuova, assoluta e fuori dal codice; le cache si verificano
  per SHA-256 contro `pin_ingressi.json` (dalle provenienze di GPT in f7433f4): file, provenienza e **ricevuta dell'estrazione (anche il suo
  SHA, `extraction_receipt_sha256`)**, **prima e dopo** i fit; il codice di analisi (`WEARUSFM_CODE_SHA` del checkout congelato) e' registrato
  separatamente dal codice che ha prodotto le cache (`cache_code_sha` 24139ab). Lo shard salva l'audit **fit per fit** in
  `audit_progress.jsonl` (`AuditLog.persist`); un errore lascia il record del fallimento e `shard_failed.json` (`complete: false`, eccezione,
  traceback, dataset completati) e **mai** `shard.json`: l'aggregatore rifiuta uno shard interrotto. A shard completo si scrivono `audit.jsonl`
  (con `selected`), `shard.oof.npz` e `shard.json` (`complete: true`). Nessun resume o retry automatico; dopo un SIGKILL restano i fit salvati.
  **Prima dei fit della fase B** (`gate_preflight`): cancello superato, firma della ricetta corrente riportata a `max_iter` 3000 uguale a
  quella attestata dal cancello, ricetta per esteso uguale, contesto numerico uguale (dtype delle feature, backend BLAS/OpenMP con versione e
  strato di threading, limiti dei thread; niente job ID, host, orari o percorsi), i quattro rapporti attestati ancora presenti con gli SHA
  attestati. Un seme diverso da riga di comando o un ambiente diverso fermano il job senza spendere un fit.
- **`scripts/slurm/probe_convergence.sbatch`** (nuovo): CPU seriale, `CODE_DIR` ed `EXPECTED_SHA` obbligatori (`lib_code_dir.sh`), uscita
  assoluta e fuori dal codice, thread BLAS = core allocati (come le sonde originali, a 8 core).

### Schema dell'audit (un record JSON per fit: `audit_progress.jsonl` fit per fit, `audit.jsonl` a shard completo, poi `<rapporto>.audit.jsonl`)

Contesto: `protocol`, `experiment_phase` (A/B), `encoder`, `task`, `dataset`, `fold`, `seed`, `checkpoint_sha256`, `cache_sha256`,
`cache_code_sha`, `analysis_code_sha`, `solver_recipe_sha256`, `numeric_sha256`, `roles_sha256` (train/val/held del fold), `reduced`; poi per
il fit: `fit_phase` (`candidate` | `refit`), `C`, `solver_params` (`get_params()` effettivo), `max_iter`, `n_train`, `n_features`, `dtype`,
`classes`, `class_counts`, `scale`, `scaler_sha256` (media e scala dello StandardScaler), `n_iter`, `n_iter_all`, `elapsed_s`, `warnings`
[{categoria, messaggio, file, riga}], `convergence_warnings`, `reached_limit`, `converged_by_criterion`, `anomalous_stop` (warning con
iterazioni sotto il limite), `exception`, `val_bacc` e `scoring_warnings` (candidati), `selected` (il candidato scelto con la regola «val_bacc
massima, a parita' C minore»; nel file progressivo e' `null`, perche' si sa solo a fine fold). Nei rapporti aggregati: `audit_summary` (totali
e posizioni dei fit al limite o con warning), `solver_recipe`, `numeric_context` (+ `numeric_sha256`), `environment` (versioni dei fit, ambiente
di ogni shard e dell'aggregatore: l'aggregatore gira a 2 core e non si confronta coi fit). Nessun candidato non convergente e' escluso dalla
selezione. Il cancello (`gate.json`) attesta ricetta per esteso e firma, contesto numerico e firma, i quattro rapporti con SHA e i 280 record.

### Parita' con il rapporto storico (fase A): campi confrontati, tutti esatti

Rapporto: `k_folds`, `fold_seed`, `val_frac`, `probe_seed`, `window_s`, `task`, `splits_sha256`, `weights`, `extraction_recipe`, `feature_set`,
`checkpoint_sha256`, `S`, `S_se`. Per dataset: `input_sha256`, `oof_rows_sha256`, `subjects`, `s`, `n_classes`, `n_windows`, `n_oof_windows`,
`readout`. Per fold: `train`, `val`, `held`, `c`, `held_bacc`, `val_bacc`, `train_bacc`, `n_train`, `n_val`, `n_held`, `dropped_val_windows`,
`dropped_held_windows`, `k`, `k_capped`, `explained_variance`. Piu' gli array OOF (y, soggetti, righe, predizioni). Ogni differenza finisce in
`problems` e in `differences`; `identical` solo con zero differenze. Fuori dal confronto: tempi, campi diagnostici nuovi, byte dei file.

## 3. Procedura (cinque blocchi, operatore semplicione, submitter e ledger comuni)

Prerequisito: checkout di campagna **`pilot-posa-sonde-20261029`** dal commit di questa consegna dopo la review di GPT (`git push leonardo
deploy/pilot-posa-sonde-20261029` dal commit, poi `campaign_checkout.sbatch`: 1 core, 10 min); il piano e i comandi si rigenerano con lo SHA
definitivo (`python3 scripts/probe_convergence.py --stage plan --code-sha <sha> --pins
results/infra/pilot_posa/sonde_convergenza_20261029/pin_ingressi.json`), e stanno in `results/infra/pilot_posa/sonde_convergenza_20261029/`
(`piano.json`, `comandi.txt`). Le cache restano quelle di B19618 su $SCRATCH (`.../B19618_20261028/{B,R3c}/{p1,p2}`), gli originali su $WORK
(`.../B19618_20261028/{B,R3c}_{p1,p2}_pool.json`); le uscite in `$WORK/wearusfm_runs/results/pilot_posa/sonde_convergenza_20261029/`.

| Blocco | Job | Risorse | Che cosa controlla da solo |
|---|---|---|---|
| 1 | fase A, 10 shard (2 encoder x {DB2 fold 0-1, DB2 fold 2-3, DB3, DB6, P2}) | 8 core, 30 GB, tetti 45/45/20/30/40 min | SHA delle cache, ricevute e provenienze; 7 fit per fold; cache invariate alla fine |
| 2 | fase A, 4 aggregati + parita' con gli originali | 2 core, 16 GB, 20 min | ogni (dataset, fold) una volta sola; stessa ricetta e stesse cache fra gli shard; 84/56 record; parita' esatta (SHA degli originali dal pin) |
| 3 | cancello | 1 core, 4 GB, 10 min | 280 record, quattro parita' identiche, una sola ricetta del solver e un solo contesto numerico; scrive `gate.json` con ricetta, contesto e SHA dei rapporti |
| 4 | fase B, 10 shard a 12000 | 8 core, 30 GB, tetti 120/120/40/60/90 min | come il blocco 1, piu' il preflight del cancello **prima dei fit**: cancello superato, ricetta (a max_iter 3000) e contesto numerico uguali a quelli attestati, rapporti attestati intatti; altrimenti esce senza fit |
| 5 | fase B, 4 aggregati + confronto fra le fasi (afterok sui quattro) | 2 core, 16 GB, 20/30 min | guardie della ricetta (uguale fra encoder, solo `max_iter` fra le fasi), identita' delle righe, cancello; bootstrap congiunto e controllo incrociato col confronto ufficiale |

Ogni blocco si sottomette quando il precedente e' interamente COMPLETED 0:0 (`leonardo_watch.py --mode all`); un job che trova un
prerequisito mancante esce con errore senza spendere; nessun retry automatico: un job fallito si porta a Claude, che dal 29/10 conduce
l'esecuzione (prompt GPT 73b6fba; il testo «si porta a GPT» nell'intestazione di `comandi.txt` viene dal codice congelato f6c89cc e vale come
«a chi conduce»). Al piu' 10 job per blocco
(limite della coda seriale). DB2 e' diviso per fold per restare sotto le 4 ore della seriale anche nel caso peggiore della fase B.

## 4. Confronti e lettura (fase `compare-phases`)

Sugli OOF dei quattro rapporti di un compito (R3c e B a 3000, R3c e B a 12000), con **gli stessi ricampionamenti dei soggetti** di
`arms_compare.s_comparison` (seme 20261009, 1.000 repliche, dataset in ordine alfabetico, una estrazione per replica condivisa dai quattro
rapporti) e i pesi originali (P1 per soggetti di test: DB2 0,615, DB3 0,231, DB6 0,154; P2 0,5/0,5):

1. `B_minus_R3c_A` e `B_minus_R3c_B`: B−R3c a 3000 e a 12000 (il primo deve coincidere col confronto ufficiale, verificato nel job);
2. `R3c_B_minus_A` e `B_B_minus_A`: variazione 12000−3000 di ciascun encoder;
3. `delta_of_deltas`: (B−R3c)_12000 − (B−R3c)_3000, stima e IC95 dalle stesse repliche, mai dagli estremi di due intervalli separati;

piu' il dettaglio per dataset, la frazione di predizioni cambiate per encoder e dataset, i C scelti per fold nelle due fasi, i conteggi e le
posizioni dei fit al limite o con warning in ciascuna fase e i warning residui a 12000. Il campo legacy `not_harmful` non e' una prova di
assenza di danno e non si usa. La lettura la facciamo insieme (Simone, GPT, Claude) sui JSON di `confronto/`.

## 5. Test CPU (tutti verdi sul Mac; ambiente e comandi sotto)

`tests/cpu/test_probe_convergence.py` (23 test): audit acceso/spento identico (etichette numeriche e testuali), warning indotti attribuiti al
fit giusto e separati da quelli della metrica, eccezione propagata con record, nessuna fuga di informazione (scaler e C dal solo train e
validazione; fold disgiunti per soggetto; shard di fold con le stesse righe della sonda intera), `max_iter` propagato a tutti i fit e tie-break
invariato, shard -> aggregato uguale alla sonda intera di `channel_readout.py` e parita' esatta con un «originale» (poi incompleto, doppio,
manomesso rifiutati), cancello che passa e che chiude la fase B, fase B e confronto fra le fasi con controllo ufficiale incrociato, guardie
della ricetta (legacy, max_iter, tol, checkpoint, contesto numerico; EPN612 con etichette testuali; feature non finite), bootstrap congiunto
uguale a quello ufficiale e differenza dei delta, piano (30 job, copertura di ogni shard una volta per fase, blocchi <= 10, dipendenze, tetti,
checkout), launcher con la regola dei checkout. Regressioni della review GPT (equivalenti alle sue tre prove, adattate ai nuovi prerequisiti):
R1 una differenza sola alla volta in undici campi scientifici (val_bacc, train_bacc, held_bacc, c, n_held, ruoli, soggetti, s, S_se, pesi,
fold_seed) chiude la parita' e una parita' falsa chiude il cancello; R2 un fit fallito lascia su disco i record finiti e quello del fallimento
con `shard_failed.json`, nessun `shard.json`, aggregatore che rifiuta, chiavi `experiment_phase`/`fit_phase` distinte; R3 la fase B si ferma
prima di qualunque fit con seme diverso, firma o contenuto della ricetta diversi, contesto numerico diverso, rapporti attestati assenti o
cambiati, cancello chiuso. Le suite esistenti `test_harness_probe.py` e `test_channel_readout.py` restano verdi (default invariati).

Comando: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -c "import sys; sys.path.insert(0, 'src'); import pytest;
sys.exit(pytest.main(['-q', 'tests/cpu/test_probe_convergence.py']))"` (Python 3.11, sklearn 1.7.2 / numpy 2.3.5 sul Mac; su Leonardo
l'ambiente delle sonde e' sklearn 1.4.1.post1 / numpy 1.26.4 / scipy 1.12.0, `pip_freeze_62582916.txt`: l'audit registra le versioni effettive).
Non e' stato simulato nessun successo su cache reali: le cache non sono sul Mac.

## 6. Preventivo (0 GPU-ora; tetti CPU allocati, 8 core per shard come le sonde originali)

Tempi osservati nelle sonde del 28-29/10 (8 core, `TotalCPU` ~7,7 core occupati): P1 76-79 min (DB2 ~56, DB3 ~7, DB6 ~12), P2 20-23 min;
le quattro sonde hanno consumato 26,52 core-ore allocate.

| Blocco | Job | Tetto core-ore | Atteso core-ore |
|---|---|---:|---:|
| 1. fase A, shard | 10 | 48,00 | 26,4 (consuntivo del 29/10) |
| 2. fase A, aggregati + parita' | 4 | 2,67 | ~0,4 |
| 3. cancello | 1 | 0,17 | ~0,02 |
| **Fase A** | 15 | **50,83** | **~26** |
| 4. fase B, shard a 12000 | 10 | 57,33 (tetti dai tempi della fase A, 29/10; prima 114,67) | ~36 |
| 5. fase B, aggregati + confronto | 5 | 3,67 | ~0,6 |
| **Fase B** | 15 | **61,00** (prima 118,33) | **~36** |
| **Totale scientifico** | 30 | **111,83** (prima 169,17) | **~62** |
| Checkout di campagna (prerequisito) | 1 | 0,17 | |
| **Totale con il checkout** | 31 | **112,00** (prima 169,33) | **~62** |

Il fattore quattro di `max_iter` non e' un tempo e non e' un limite rigoroso: vale solo per i fit che toccano il limite (5/4/4/5 emissioni di
warning nei log, su 84 fit per P1 e 56 per P2, cioe' 28 per dataset). I tempi e le iterazioni per fit della fase A (`elapsed_s`, `n_iter` nei
record) servono ad affinare il tetto della fase B **prima** di eseguirla, con margine e overhead, senza toccare la ricetta numerica; i tetti
sono allocazioni che fermano il job, non garanzie che il lavoro finisca. Residuo registrato dopo B19618: 432,14 core-ore CPU-only / 64,45
GPU-ore; questo controllo ne impegna al massimo 112,00 core-ore dopo l'affinamento della fase B del 29/10 (169,33 prima; residuo teorico al tetto
completo 320,14) e nessuna GPU-ora.

## 7. Obiezioni e scelte dichiarate

- **Nessuna obiezione scientifica al disegno del prompt.** Due scelte tecniche: (a) gli shard per dataset e, per DB2, per coppie di fold, con
  un aggregatore esatto (ogni fold e' indipendente: scaler, C e refit dentro il fold; le righe OOF si concatenano nell'ordine dei fold), per
  rispettare le 4 ore della seriale senza proporre un job piu' lungo del limite; (b) i prerequisiti fra i blocchi sono verificati dai job
  stessi (ricevute, parita', cancello) e non da `afterok` fra blocchi, perche' SLURM scarta le dipendenze su job gia' usciti dalla coda (lezione
  del 25/10); dentro il blocco 5 il confronto resta in `afterok` sui quattro aggregati.
- **Categoria del ledger `--agente gpt`**, operatore semplicione, come nel piano di recupero di GPT (`submit_job.py` ammette solo claude, gpt,
  simone). Se GPT preferisce `simone` o `claude`, basta rigenerare il piano con `--agente`.
- **Parita' senza tolleranza.** Stesse cache, stesso ambiente (cineca-ai 4.3.0 + venv), stesso seme: ci si attende identita' di predizioni,
  C, punteggi e S. Se la fase A non riproduce gli originali, anche per un ulp, il cancello resta chiuso e la discrepanza si studia prima
  (ambiente, thread, ordine): non si alza una tolleranza dopo aver visto i numeri. La tolleranza 1e-12 fra il bootstrap congiunto e il
  confronto ufficiale e' un controllo numerico fra due implementazioni, separato dalla parita'.
- **Ordine delle chiamate nella sonda.** Nella prima consegna il bootstrap dell'errore standard precedeva il calcolo di `test_bacc`; ora
  l'ordine e' quello di sempre (rilievo 1 della review: nessuna conseguenza numerica, il bootstrap ha un generatore locale).
- **Coda seriale (leonardo-ops, 29/10 14:11):** con un job da 8 core in esecuzione gli altri nove restano in attesa per `QOSMaxCpuPerUserLimit`:
  gli shard girano uno alla volta. Tempo di parete atteso per la fase A circa 3 ore, per la fase B fra 3 e 13; i tetti per job non cambiano.
- **Fuori perimetro**, come nel prompt: nuovi training, loss, PSD, latenti non mediati, tuning sul test, aggiornamenti delle librerie,
  correzioni di un solo encoder.

## 8. Risultati (29/10/2026, sera; lettura descrittiva, nessuna nuova soglia)

**Riproducibilita' (fase A).** I quattro rapporti a 3.000 iterazioni riproducono gli originali di B19618 alla lettera: parita' IDENTICA su
predizioni, C, punteggi per fold e dataset, righe, soggetti, S e S_se (nessuna tolleranza); cancello `passed`, 280 record, una sola ricetta del
solver e un solo contesto numerico (sklearn 1.4.1.post1, numpy 1.26.4, scipy 1.12.0, OpenBLAS pthreads, 8 thread, float32).

**Convergenza (fase B, `max_iter` 12.000).** 280 record, **0 fit al limite, 0 ConvergenceWarning** (a 3.000 erano 18: tutti candidati C = 100 in
DB6 ed EPN612, piu' i due riaddestramenti dove C = 100 era stato scelto). Quei 18 fit convergono per criterio fra 3.160 e 4.823 iterazioni
(DB6 3.334-4.265, EPN612 3.160-4.823). I punteggi di validazione dei candidati cambiano al massimo di 0,0049 (B DB6), 0,0039 (B EPN612), 0,0029
(R3c DB6), 0,0019 (R3c EPN612); in DB2, DB3 e UCI zero fit cambiano iterazioni e nessun numero cambia.

**Selezioni e predizioni.** Cambia il C scelto in due fold su 40: EPN612 B fold 3 e R3c fold 1, da 10 a 100 (il candidato C = 100, ora
convergente, vince la validazione per pochi millesimi). In DB6 nessun cambio di C; R3c DB6 fold 3 mantiene C = 100 e il suo riaddestramento
converge a 4.244 iterazioni invece di fermarsi a 3.000. Predizioni OOF cambiate: R3c DB6 255 su 33.961 (0,75 %), B EPN612 666 su 47.100
(1,41 %), R3c EPN612 358 su 47.100 (0,76 %); zero in DB2, DB3, UCI e in B DB6.

| | 3.000 (A) | 12.000 (B) |
|---|---:|---:|
| S P1 B | 16,995 | 16,995 |
| S P1 R3c | 17,551 | 17,547 |
| S P2 B | 79,604 | 79,613 |
| S P2 R3c | 80,837 | 80,785 |
| P1 B−R3c (IC95, bootstrap congiunto per soggetto) | −0,556 [−1,104; −0,026] | −0,553 [−1,105; −0,017] |
| P2 B−R3c | −1,233 [−2,997; +0,576] | −1,172 [−2,956; +0,651] |

Per dataset: DB6 R3c 34,70 -> 34,68; EPN612 B 81,82 -> 81,84, R3c 84,40 -> 84,30; gli altri identici. **Differenza dei delta (12.000 − 3.000):
P1 +0,004 [−0,002; +0,013] (IC con lo zero); P2 +0,061 [+0,001; +0,120]** (tutta da EPN612, +0,121 [+0,002; +0,240]): per P2 l'IC esclude lo zero,
cioe' l'effetto della convergenza sul contrasto e' rilevabile dal bootstrap appaiato, ma vale un ventesimo dell'ampiezza dell'IC del contrasto
stesso (±1,8 punti) e va nella direzione di ridurre lo svantaggio di B. Segno di B−R3c invariato in entrambi i compiti; l'etichetta del lettore
del pilot resta «regola non superata» a 3.000 e a 12.000; il confronto ufficiale (`arms_compare`) e il bootstrap congiunto coincidono (1e−12).

**Lettura.** La lettura congiunta di B19618 (29/10) resta valida con le sonde convergenti: P1 calo piccolo con IC sotto lo zero, P2 inconclusivo
con cali possibili di 3 punti. I warning storici erano candidati C = 100 non selezionati (salvo due riaddestramenti) e il loro completamento
sposta S di al piu' 0,05 punti. Regole del prompt PSD-3 (§1): punto 2, punteggi cambiati ma segno invariato, conclusione aggiornata nel senso di
confermata; punto 3, nessun warning residuo a 12.000, quindi nessun controllo mirato e nessun cambio di `max_iter` per PSD-3; il segno delle
sonde non seleziona nulla di PSD-3, gia' in esecuzione come preparata.

**Avvisi di scoring.** DB3: 24 «y_pred contains classes not in y_true» per fase, identici in A e B (soggetti tenuti fuori privi di alcune classi:
le classi predette assenti contano come errori nella BAcc). Nessun arresto anomalo, nessuna eccezione.

**Consuntivo.** Fase A: shard 26,4 core-ore (10 job, 8 core, 198 min di parete), aggregati e cancello < 1. Fase B: shard 27,0 core-ore (10 job,
202,5 min: DB2 27-29 min per meta', DB3 10-11, DB6 12-14, P2 20-26), aggregati e confronto 0,6. **Sonde in tutto ~55 core-ore** (tetto 112,0 con
il checkout), 0 GPU-ora; 7 h 40 di parete per la fase B compreso un guasto di `slurmdbd` (sacct muto dalle ~21:00 alle 23:30: i job non ne
hanno risentito, il completamento si e' verificato dai file `shard.json` e dai log).
