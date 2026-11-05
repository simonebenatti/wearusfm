# Review di Claude — valutazione lunga T100-G 30M: pacchetto scientifico ed esecutivo di GPT (19c3166..e9d9d7b, consegna 6f5d909) — 05/11/2026

Richiesta: prompt di GPT incollato da Simone il 05/11 (due volte, testo identico): «Revisiona il pacchetto scientifico ed
esecutivo della valutazione lunga, senza lanciare job.» Solo review: nessun job, nessuna misura CUDA, nessuna modifica al
worktree di GPT o ai checkout in uso. Su Leonardo soltanto letture.

**Aggiornamento delle 20:00:** la seconda lettura, sulle correzioni `e2f19ee`, e' in fondo al file, con i verdetti aggiornati.

**Aggiornamento delle 21:00:** la terza lettura, sulla correzione `f8cb53f`, e' in fondo al file: convergenza su codice e
piano, lancio fermo su C-1.

Ramo `codex/consolidation-20261104`. Codice e piano `e9d9d7b`, consegna `6f5d909`, base del diff `19c3166`. A review in
corso GPT ha pubblicato `857d6f9` e `c66148b` (punta alle 19:14): toccano solo `tests/cpu/test_pose_b19618_launcher.py` e il
registro. A `c66148b` `source_identity()` vale ancora `2ab63f44…`, uguale al pin del piano. I test sono rieseguiti anche li'.

## Verdetti

| Punto | Verdetto | In breve |
|---|---|---|
| 1. Estrazioni e provenienze | **APPROVATO** | Nessun difetto trovato. Una nota (N-1). |
| 2. Classificazione e riuso storico | **DA CORREGGERE** (R-1) | Ricetta, riuso e lettura di `declared_historical_cache` corretti. Manca una dimensione: il lettore storico ha girato con 8 thread, le celle nuove con 1, e nessun controllo lo vede. |
| 3. Posa, aggregazione e lettura | **APPROVATO** | Logica coerente con V-R1 a due lati. Due note (N-2, N-3). |
| 4. Risorse e gate | **DA CORREGGERE** (R-1, R-2) | Conti e tetti esatti. Il dimensionamento pero' poggia su tempi a 8 thread e l'esecuzione e' a 1: tolleranza del gate 1,25. Il percorso di emendamento scritto nel documento non e' eseguibile senza rifare le estrazioni. |
| 5. Esecuzione e completezza | **APPROVATO con due condizioni** (C-1, C-2) | Struttura solida. Quota a 1,59 GiB dalla soglia e in calo; sorgenti del ramo da tenere fermi. |

Non e' ancora una convergenza: il mio commit non va pinnato come `claude_review_converged`. Una sola modifica chiude R-1 su
entrambi i punti. Va fatta **prima del checkout delle estrazioni**, per il motivo scritto in R-2. Il secondo passaggio e' sul
solo diff incrementale.

## R-1 — Otto thread nello storico, uno nel pacchetto (punti 2 e 4)

**Fatti.**

- Le sonde storiche a 19.618 hanno girato con 8 thread. `p1.json` (`4ded85b9…`) e `p2.json` (`ef53b8ab…`) riportano lo stesso
  `numeric_context`, hash `07b776ba…`: openblas 0.3.21.dev e 0.3.23.dev con `num_threads: 8`, libgomp 8, SkylakeX.
  Launcher storico: `scripts/slurm/tokenizer_probe.sbatch:98`, che esporta `$CPUS_EXPECTED` sulle tre variabili.
- Il pacchetto impone 1 thread a tutte le fasi: `scripts/slurm/consolidation_long.sbatch:22`;
  `consolidation_long_classification_runtime.py:44-50` (`_threads()`); `BENCHMARK_POLICY["blas_threads"] = 1` (riga 29).
- Nessun documento del pacchetto lo dichiara: `grep -i thread` su documento esecutivo, review interna e protocollo non trova
  nulla sul tema.

**Conseguenza scientifica (punto 2).** Il contrasto primario 49k−19k sul pool confronta OOF storiche ottenute con 8 thread
e OOF nuove ottenute con 1. Il codice tratta il contesto numerico come parte della confrontabilita' fra le celle nuove
(`consolidation_long_classification.py:637`, `_runtime.py:154` e `:202`), ma per i report storici controlla solo la coerenza
interna (`:537-538`, `_runtime.py:76`). Nessun confronto fra i due contesti, nessun campo che lo registri. L'effetto non e'
misurato. La mia stima, non una misura: piccolo rispetto a 1 pp, perche' il solver converge per criterio. Resta una
differenza non dichiarata proprio sul riferimento del contrasto primario.

**Conseguenza sulle risorse (punto 4).** Tutto il dimensionamento usa tempi a 8 thread: il pavimento storico dentro
`resource_gate` (`old["fit_seconds"]*work/pool_work`), le «114,24 coreh», la divisione di DB2 in coppie, il test
`test_archived_db2_floor_blocks_one4h_job_and_fits_registered_pairs`. Ho ricalcolato il gate con la funzione di GPT, usando
come benchmark i tempi archiviati del fold 0 moltiplicati per k (rapporto fra benchmark nuovo e archivio a 8 thread).

| k | Shard falliti, pre/post costosi come il pool | Shard falliti, pre/post a meta' costo |
|---:|---|---|
| 1,00 | 0 su 48 | 0 su 48 |
| 1,25 | 0 su 48 | 0 su 48 |
| 1,30 | 8 su 48 (DB2) | 2 su 48 (DB2 pool) |
| 1,35 | 24 su 48 (DB2, EPN) | 6 su 48 (DB2 ed EPN pool) |
| 3,00 | 32 su 48 (anche DB6) | 26 su 48 |

| Shard pool a k = 1 | Proiezione | Quota del gate 13.800 s | k massimo |
|---|---:|---:|---:|
| DB2 fold 0-1 | 11.153 s | 80,8% | 1,251 |
| DB2 fold 2-3 | 10.619 s | 76,9% | 1,318 |
| EPN, 4 fold | 10.532 s | 76,3% | 1,330 |
| DB6, 4 fold | 6.221 s | 45,1% | 2,353 |
| DB3, 4 fold | 3.548 s | 25,7% | 4,501 |

Il gate e' tutto o niente (`_runtime.py:290`): un benchmark piu' lento del 25% rispetto all'archivio a 8 thread blocca tutti
i 48 fit. Il rallentamento a un thread non e' misurato. Il fit e' fatto di prodotti di matrici, cioe' della parte che i
thread BLAS accelerano. Il rischio non e' il timeout: a k = 1 lo shard peggiore ha un tempo atteso di 1.570 s in un job da
4 ore, perche' la proiezione include margini 3×1,5 e il refit al costo del candidato piu' lento. Il rischio e' un gate che
ferma una matrice che girerebbe. In piu', con 1 thread su 8 CPU allocate non c'e' parallelismo ne' dentro ne' fra i job:
`lrd_all_serial` concede all'utente 8 CPU in tutto (letto alle 19:00: `MaxTRESPU cpu=8,mem=30800M`, 2 job, 10 sottomissioni).

**Correzione richiesta, una delle due.**

- **(a), che raccomando.** Thread per fase uguali al launcher storico della stessa fase: sonde di classificazione 8
  (`tokenizer_probe.sbatch:98`), lettore di posa 8 (`tokenizer_pose_readout.sbatch:83`), estrazioni 1
  (`tokenizer_evaluation.sbatch:59`). `_threads()` lo impone. Il confronto richiede che `numeric_sha256` dei fit nuovi sia
  uguale a quello dei report storici (`07b776ba…`) e lo registra nel report; se differisce si ferma. Risorse e tetti restano
  quelli. La ricetta non cambia hash; cambia `source_sha256`.
- **(b).** Tenere 1 thread. Allora: la policy e il report finale dichiarano che il riferimento 19k/pool e' stato stimato con
  8 thread; shard e walltime si rifanno con una regola scritta che non poggi sui tempi a 8 thread; un refit del 19k/pool a
  1 thread dalla nuova estrazione (6 job), dichiarato ora, misura l'effetto del contesto.

## R-2 — Un emendamento dopo le estrazioni le invalida (punto 4, tocca il 5)

**Fatti.** `verified_extraction` (`consolidation_long_common.py:235`) e la lettura delle cache di posa
(`consolidation_long_pose_runtime.py:83`) esigono che il `source_sha256` della ricevuta sia quello corrente.
`source_identity()` copre ogni `src/**/*.py`, ogni `scripts/*.py`, il launcher e `lib_code_dir.sh`. Policy dei benchmark,
shard e thread stanno in file coperti. Il documento dice: «Cambi del codice richiedono review pertinente e benchmark nuovo
se cambia la fonte della misura». Il gate dice: «reviewed resource amendment required». Dopo un emendamento del genere le
12 ricevute di estrazione non passano piu' la verifica. Il percorso scritto non e' eseguibile senza rifare le estrazioni:
fino a 9 GPU-ore, cioe' l'intero tetto GPU, quindi una nuova decisione di Simone.

**Correzione richiesta.** Scrivere prima delle estrazioni quale delle due vale: (i) ogni cambio di sorgente dopo le
estrazioni comporta la loro ripetizione, con costo e autorizzazione dichiarati; (ii) un meccanismo rivisto che le conserva,
per esempio ricevute legate a un'identita' dei soli sorgenti di estrazione, o hash accettati elencati nel piano pinnato.

Con (a) il margine torna quello di progetto, che resta 1,25 su DB2 fold 0-1 e 1,33 su EPN per costruzione del pavimento.
Opzione a scelta di GPT, non richiesta: fissare ora shard piu' fini. Dal mio ricalcolo DB2 per singolo fold tollera 2,40 ed
EPN in coppie 2,65; i job di fit salgono da 48 a 72 e il tetto di allocazione da 1.536 a 2.304 coreh, a consumo atteso
invariato.

## Condizioni per il punto 5

- **C-1 — Quota.** La soglia prima della prima estrazione e' 32 GiB liberi. Letture del progetto: 34,48 GiB alle 16:11
  (evidenza di GPT), 33,77 alle 19:00, 33,59 alle 19:22. Calo medio 0,28 GiB/ora, margine 1,59 GiB: a questo ritmo la soglia
  si supera in circa sei ore. Le soglie del piano non sono lette da nessun codice (`grep` vuoto in script, launcher e
  `submit_job.py`): e' un controllo dell'owner. Va rifatto subito prima del primo invio; se non passa ci si ferma, come da
  piano. Opzione: le cache grandi (mia stima 21,6 GB su 24 GiB preventivati) su `$SCRATCH`, dove stanno quelle storiche.
  Cambia `OUTPUT_ROOT`, quindi va deciso insieme a R-1.
- **C-2 — Sorgenti fermi sul ramo.** Fra una fase e l'altra nessun commit del ramo puo' toccare `src/` o `scripts/*.py`,
  nemmeno per FT15 che vive sullo stesso ramo: il checkout della fase successiva fallirebbe la verifica dei sorgenti. Il
  fallimento e' chiuso, quindi sicuro, ma ferma la catena.

## Cosa ho verificato, punto per punto

**1. Estrazioni.** Un solo forward con hook temporaneo su `model.pool` per pre, post e pool; ordine di soggetti e sessioni,
scale e impronte uguali agli estrattori storici per costruzione. Posa dal loader originale tramite adattatore, un forward
FP32 senza maschera; firme di bersagli, maschere, std e varianza uguali alle cache B19618; selezioni 1907/40/200 e
18548/15/1950. `load_teacher` ricostruisce la configurazione dall'argv validato e confronta ogni tensore con `torch.equal`,
buffer `fe_gain` compreso. `finish_output` rilegge l'hash del checkpoint e dello stato del teacher e controlla STOP, piano
e sorgenti. `validate_cache` si ferma se `key_time_valid` differisce dal supporto storico, senza toccare le maschere.
Su Leonardo, alle 19:00: i tre training COMPLETED 0:0 con 7994/7186/7575 s (25,2833 GPU-ore); SHA256 dei tre checkpoint
ricalcolati e uguali al piano; hash delle ricevute uguali ai pin; nessun STOP; radice di output inesistente.
*N-1:* la vista pre della posa e' salvata in FP16 e controllata solo per finitezza; a differenza della post non passa dal
LayerNorm finale. Facoltativo: registrare nella ricevuta intervallo dei valori ed errore di arrotondamento FP16.

**2. Classificazione.** `fit_fold` segue lo stesso percorso numerico di `TP.fit_fold`; griglia, scaler, selezione, refit,
fold e seme uguali alla ricetta storica; bootstrap appaiato uguale a `TP.compare_arrays`; pesi P1 e P2 corretti; logica V-R1
riletta sul testo e coerente. **Lettura di `declared_historical_cache`: approvata.** Il riferimento 19k/pool resta la cache e
il lettore storico nominati; la parita' entro tolleranza non e' identita' delle predizioni; l'uguaglianza esatta va
riportata. Va completata con R-1. Con (a) non serve alcun controllo aggiuntivo.

**3. Posa.** Miglioramento relativo `100*(1 − E_dyn_new/E_dyn_ref)` ricalcolato a ogni estrazione appaiata per utente;
soglie ±2% con il lato dell'intervallo corretto; riferimento rinviato su regressione chiara di livello, correlazione o gap
E2; le nove celle sono tutte richieste; ridge allenata prima di aprire il held; E2 dichiarata diversa da quella storica.
*N-2:* due pannelli primari vanno bene se la lettura finale li riporta sempre affiancati e un disaccordo resta un disaccordo.
*N-3:* 15 utenti held rendono grossolani gli intervalli percentili; e' ereditato dal lettore convergente.
Tempi e memoria della posa non sono a rischio: il lettore storico a tre celle ha impiegato 142 s con picco 4,67 GB
(63495299). La proiezione di memoria moltiplica per 9,73 tutta la RSS del benchmark, quota fissa compresa: serve un picco
train sotto 2,62 GiB; lo storico a una profondita' era 1,38 GB (63494999), la mia stima a due profondita' e' circa 1,7 GB.

**4. Risorse.** 82 ID e output unici; 12 + 15 + 1 + 48 + 3 + 2 + 1; tetti ricalcolati uguali: 9 GPU-ore e 1711,666667
coreh. Tempi storici dei fit per cella pool: 6.360,9 s, cioe' 14,14 coreh a 8 CPU (GPT scrive 14,28: differenza irrilevante). Launcher: controlli
in-job di walltime, CPU, RAM, nodo e GPU contro `plan['resources']`, `--no-requeue`, guardia su `SLURM_RESTART_COUNT`.

**5. Esecuzione.** Pin futuri `null` bloccano le fasi successive; output solo nuovi sotto `OUTPUT_ROOT`, niente symlink;
il fit richiede il gate e lo stesso contesto del benchmark; il confronto richiede esattamente 48 ricevute; la sintesi
richiede i due report finali completi, con stessi sorgenti e ricetta. Coda dell'utente alle 19:00: 57 job su
`boost_usr_prod`, 18 in esecuzione e 39 in attesa, nessuno sulla seriale; alle 19:22 gli stessi totali.
*Autorizzazione:* GPT cita in `decisioni.md` Simone tramite coordinamento: «ok, fai completare alla chat 2 la valutazione
del lungo, falla revisionare a claude e poi lanciatela appena pronta». Non l'ho verificata alla fonte. Mia lettura: nomina
questa valutazione, che e' quanto serve per estrazioni sopra 5 GPU-ore; i tetti 9 GPU-ore e 1711,67 coreh sono scritti nel
pacchetto dopo quella frase, e li riporto a Simone nella chiusura.

## Cosa ho eseguito

- Checkout temporaneo di `6f5d909`, poi di `c66148b`. I cinque file nuovi: 107 test passati. Con le regressioni collegate:
  346 passati. A `c66148b`, cinque file nuovi piu' `test_pose_b19618_launcher.py`: 122 passati. `bash -n` del launcher: OK.
- `source_identity()` e `recipe_identity(plan)` ricalcolati: uguali ai pin del piano (`2ab63f44…`, `b2356a8e…`).
- Gate ricalcolato con `resource_gate` di GPT sui tempi archiviati (`p1.audit.jsonl` `05419151…`, `p2.audit.jsonl`
  `c2d505c1…`), per k da 1 a 4 e due ipotesi sul costo di pre e post.
- Letture su Leonardo alle 19:00 e alle 19:22: `sacct`, `sha256sum`, `ls`, `lfs quota`, `sacctmgr`, `scontrol`, `squeue` e
  l'intestazione di un log storico.

## Limiti

- Il rapporto di tempo fra 1 e 8 thread non e' misurato: nessun job e nessuna misura sul login node. R-1 dimostra la
  differenza di contesto e la tolleranza del gate, non che il gate fallira'.
- L'effetto dei thread sulle predizioni di sonda non e' misurato.
- L'equivalenza delle estrazioni con gli estrattori storici e' verificata sul codice e sui test, non su un'estrazione reale.
- Le stime di memoria e di spazio sono mie e approssimate.
- Una sola lettura della quota non dice chi sta scrivendo: il calo va attribuito dal coordinamento.

---

# Seconda lettura — correzioni di GPT `e2f19ee`, consegna `ef5589f` — 05/11/2026, ore 20:00

Richiesta: prompt di GPT incollato da Simone, che rimanda a `docs/prompt_claude_consolidation_long_rereview_20261105.md`.
Simone, a rilettura in corso: «se convergiamo io poi farei lanciare». Ho riletto solo il diff `c66148b..e2f19ee`.
Nessun job, nessun benchmark; su Leonardo una lettura di quota e coda alle 19:57.

**Esito: non convergiamo ancora.** Tutto il resto e' chiuso. Resta un difetto solo, **R-1b**, e blocca: cosi' com'e' la
catena si fermerebbe al primo benchmark di classificazione, cioe' dopo le 12 estrazioni. Il commit di questa rilettura non
va pinnato come `claude_review_converged`.

| Punto | Verdetto dopo la seconda lettura |
|---|---|
| 1. Estrazioni e provenienze | **APPROVATO** |
| 2. Classificazione e riuso storico | **DA CORREGGERE** (R-1b) |
| 3. Posa, aggregazione e lettura | **APPROVATO** |
| 4. Risorse e gate | **DA CORREGGERE**, solo perche' `resource_gate` contiene lo stesso confronto di R-1b. R-2 chiuso. |
| 5. Esecuzione e completezza | **APPROVATO con C-1 e C-2**. C-1 e' piu' stretta di prima. |

## Chiusi

- **R-1, thread.** Launcher: 1 thread per le tre estrazioni e la sintesi, 8 per benchmark, fit e confronti di
  classificazione e posa, fissati prima di ogni import scientifico e uguali ai launcher storici di ogni fase. `_threads()`
  di classificazione e di posa li impongono su variabili d'ambiente e pool attivi. Un valore di `--stage` non registrato
  ferma il launcher.
- **R-1, risorse.** Misura e pavimento storico ora sono entrambi a 8 thread. Il margine 1,25 su DB2 fold 0-1 e 1,33 su EPN
  e' dichiarato, con la regola che un gate fallito ferma la fase. Scelta legittima di GPT: 48 shard, tetti invariati.
- **R-2.** Scelta la regola (i), scritta in piano, piano di lancio, documento e `decisioni.md`: ogni cambio a un sorgente
  coperto invalida le 12 estrazioni, la catena si ferma, la ripetizione richiede review, nuovi ID, preventivo e una nuova
  decisione di Simone; nessuna ripetizione automatica. C'e' il test che lo fissa.
- **C-2** adottata: sorgenti fermi sul ramo, FT15 su ramo separato. **N-2** adottata nel documento. **N-1** non adottata:
  va bene, era facoltativa.
- Identita' ricalcolate a `ef5589f`: `source_sha256` `010352c6…`, ricetta invariata `b2356a8e…`, `plan.json` `712d3a80…`,
  `launch_plan.json` `cf03b78e…`, uguali al file di validazione di GPT. Gli 82 job e i tetti non cambiano: nel diff del piano
  di lancio cambiano solo stato, hash dei sorgenti e i blocchi di policy aggiunti in coda. `validate_plan` rifiuta ancora il
  piano senza convergenza.

## R-1b — L'uguaglianza esatta del contesto numerico non e' raggiungibile

L'uguaglianza di `numeric_sha256` l'avevo raccomandata io in R-1(a), senza verificare che fosse raggiungibile con gli
import del codice nuovo. Non lo e'.

**Fatti.**

1. Il contesto storico (`07b776ba…`) ha tre pool: openblas 0.3.21.dev, openblas 0.3.23.dev, libgomp. Lo ha scritto
   `tokenizer_probe.py`, in un processo senza torch: nessun modulo della sua catena di import carica torch al livello di
   modulo, e quei file non sono cambiati da `b315dadf`.
2. Il processo nuovo carica torch all'import: `consolidation_long_common.py:16` e `consolidation_long_classification.py:16`.
3. Su Leonardo ogni contesto archiviato di un processo con torch elenca anche il runtime `libomp`: 14 registrazioni su 14.
   Le tre con anche scipy caricato hanno quattro pool (`results/infra/pose_slots_20261102/execution/R3c/receipt.json`, `B` e
   `preflight`: openblas 0.3.23.dev, libgomp, libomp, openblas 0.3.21.dev, tutti a 8 thread; stesso python 3.11.6, numpy
   1.26.4, torch 2.2.0a0). Le 34 registrazioni senza torch ne hanno tre.
4. `_numeric_match` di GPT, alimentata con il contesto storico piu' quel pool `libomp`, rifiuta: «new/historical
   classification numeric environment differs». Script: `docs/reviews/claude_review_consolidation_long_20261105_numeric_match.py`.
5. Stesso meccanismo in locale: `TP.numeric_context()` da' due pool in un processo che importa solo `tokenizer_probe`, tre in
   uno che importa `consolidation_long_classification_runtime`.
6. I test nuovi usano contesti finti (`{"threads": 8}`): nessuno confronta il contesto reale con quello storico.

**Conseguenza.** Il confronto e' in `benchmark-classification`, `fit-classification`, `compare-classification`, in
`resource_gate` e in `compare_reports`: tutti dopo le estrazioni, che non lo eseguono. Le 12 estrazioni partirebbero e
finirebbero; il primo benchmark di classificazione si fermerebbe. Con R-2(i) correggere a quel punto invalida le estrazioni:
fino a 9 GPU-ore perse e una nuova decisione di Simone.

**Correzione richiesta, una delle due.**

- **(B), che raccomando.** Sostituire l'uguaglianza dell'hash con una regola scritta: versioni di python, numpy, scipy e
  sklearn e `features_dtype` uguali allo storico; ogni pool storico presente nel processo nuovo con gli stessi sei campi;
  pool in piu' ammessi solo se sono runtime OpenMP a 8 thread, elencati nel report; `exact_match` registrato com'e'. La
  lettura dichiara: stesse librerie numeriche, stesse versioni, stesso numero di thread; nel processo nuovo e' caricato
  anche torch. Resta l'obbligo, gia' presente, di un solo `numeric_sha256` fra tutti i benchmark e i fit nuovi.
- **(A').** Tenere l'uguaglianza esatta e non caricare torch nelle fasi CPU di classificazione. Piu' invasivo.

**In entrambi i casi, prima del checkout delle estrazioni:** verificare nell'ambiente reale (moduli e venv del launcher,
8 thread) che il contesto del processo nuovo soddisfi la regola, e registrare l'uscita nel file di validazione. Minimo: due
comandi di pochi secondi sul login node, con e senza `import torch`, che stampano `threadpool_info()`. Meglio: un job CPU di
preflight su `lrd_all_serial` che precede le estrazioni. Con R-2(i) ogni controllo che puo' fallire per ragioni d'ambiente
va provato prima della spesa GPU, non dopo.

Facoltativo, senza job in piu': registrare nelle ricevute dei benchmark 19k/pool il confronto di `n_iter` con l'audit
storico del fold 0. Se la cache nuova e' identica alla storica, valori uguali indicano che la numerica coincide.

## Condizioni

- **C-1, piu' stretta.** Alle 19:57 i GiB liberi sono 33,15: margine 1,15 sulla soglia di 32. Nell'ultima mezz'ora il calo e'
  stato di 0,75 GiB/ora (33,59 alle 19:22). A quel ritmo la soglia si supera verso le 21:30. La regola di GPT e' corretta e
  ora e' scritta nel piano: rilettura subito prima del primo invio, sotto 32 GiB nessuna estrazione. Va trovato chi sta
  scrivendo, o liberato spazio, prima del lancio.
- **C-2** invariata.

## Cosa ho eseguito

- Checkout temporaneo di `ef5589f`. I dieci file di GPT: 236 test passati. I cinque file nuovi: 138 passati. `bash -n` del
  launcher e `git diff --check`: puliti.
- Identita' di sorgenti e ricetta e hash dei due piani ricalcolati.
- Censimento di tutti i `threadpools` archiviati in `results/`: 48 registrazioni, 6 firme distinte.
- Lo script indicato al punto 4 e il confronto locale del punto 5.

## Limiti

- Il contesto reale del processo nuovo su Leonardo non e' misurato: e' dedotto da 14 ricevute archiviate e dal grafo degli
  import. Non ho eseguito Python sul cluster: la verifica diretta e' il comando chiesto a GPT.
- Che torch caricato non cambi i risultati del fit e' un ragionamento, non una misura: sklearn, numpy e scipy usano le
  proprie librerie.

# Terza lettura — correzione di GPT `f8cb53f`, consegna `51b4574` — 05/11/2026, ore 21:00

Richiesta: prompt di GPT incollato da Simone, che rimanda a `docs/prompt_claude_consolidation_long_third_review_20261105.md`.
Ho riletto solo il diff `e2f19ee..f8cb53f`; `f8cb53f..51b4574` tocca soltanto registro e prompt. Nessun job. Su Leonardo
letture, piu' quattro processi di soli import sul login node (13,3 secondi di CPU in tutto), dello stesso tipo di quelli di GPT.

**Esito: sul codice e sul piano convergiamo.** R-1b e' chiuso e non resta alcun difetto aperto. Il commit che contiene questa
terza lettura si puo' pinnare come `claude_review_converged`. **Il lancio pero' resta fermo su C-1:** e' una condizione
operativa, separata dalla convergenza, e a questa lettura non e' soddisfatta.

| Punto | Verdetto dopo la terza lettura |
|---|---|
| 1. Estrazioni e provenienze | **APPROVATO** |
| 2. Classificazione e riuso storico | **APPROVATO**. R-1b chiuso. |
| 3. Posa, aggregazione e lettura | **APPROVATO** |
| 4. Risorse e gate | **APPROVATO**. R-1b e R-2 chiusi. |
| 5. Esecuzione e completezza | **APPROVATO con C-1 e C-2**. C-1 oggi non passa: mancano 0,59 GiB. |

La convergenza vale per `51b4574`: sorgenti `a2ff1608…`, ricetta `b2356a8e…`, 82 job e tetti invariati. Per pinnarla cambiano
solo il blocco `review` di `plan.json` e lo stato di `launch_plan.json`. Finche' non cambia altro non serve un'altra mia lettura.

## R-1b chiuso

**La regola.** `numeric_compatibility` (`scripts/consolidation_long_classification.py`) e' la regola B che avevo chiesto:
python, numpy, scipy, sklearn e `features_dtype` uguali allo storico; ogni pool storico presente con gli stessi sei campi e
almeno la stessa molteplicita'; pool in piu' ammessi solo se runtime OpenMP a 8 thread, elencati per intero; `exact_match`
registrato per quello che e'. Nessun campo viene normalizzato. Il testo della regola nel piano e' uguale alla costante del codice.

**Dove si applica.** In tutti i punti che prima chiedevano l'hash esatto: `benchmark-classification` e `fit-classification`,
prima di creare l'uscita e di fare un fit; `compare-classification`, su ognuna delle 48 ricevute; `resource_gate`, che
ricalcola l'evidenza invece di fidarsi di quella scritta; `compare_reports` e il report finale. Resta l'obbligo di un solo
`numeric_sha256` fra tutti i benchmark e i fit nuovi, e il controllo che il contesto non cambi durante un benchmark o un fit.

**Contesti reali.** Stessi moduli e venv del launcher, 8 thread.

| Processo | Nodo | Pool | Hash | Compatibile | Esatto | Pool in piu' |
|---|---|---|---|---|---|---|
| GPT, senza torch | login07 | 3 | `07b776ba…` | si' | si' | nessuno |
| GPT, con torch, import minimi | login07 | 5 | `1ac8481e…` | si' | no | libgomp, libomp |
| Mio, senza torch | login05 | 3 | `07b776ba…` | si' | si' | nessuno |
| Mio, con torch, import minimi | login05 | 5 | `1ac8481e…` | si' | no | libgomp, libomp |
| Mio, tutti gli import dell'eseguibile | login05 | 5 | `1ac8481e…` | si' | no | libgomp, libomp |
| Mio, anche gli import interni alle funzioni | login05 | 5 | `1ac8481e…` | si' | no | libgomp, libomp |

- La prova di GPT usava import minimi. Ho ricavato l'elenco completo dai 65 file di progetto che l'eseguibile carica. In piu'
  ci sono h5py, `scipy.signal`, `scipy.fft`, `sklearn.preprocessing`, `sklearn.metrics`, `torch.nn`, `torch.nn.functional` e
  `torch.utils.checkpoint`; dentro le funzioni, `sklearn.decomposition` e `torch.nn.parallel`. Nessuno aggiunge un pool:
  l'hash e' lo stesso di GPT.
- I tre pool storici sono le librerie che numpy, scipy e sklearn portano con se' (`numpy.libs`, `scipy.libs`,
  `scikit_learn.libs`). I due in piu' arrivano con torch: il `libgomp` di GCC 12.2 e `/usr/lib64/libomp.so`. Le sonde quindi
  calcolano con le stesse librerie dello storico.
- `lrd_all_serial` e' servita da `login08` e `login13`. La sonda storica p2 (job 63504862) ha girato su `login13` con 8 CPU, e
  il processo senza torch su `login05` e `login07` ne riproduce l'hash esatto: nodo e allocazione SLURM non hanno alterato i
  tre pool storici.
- Script e uscite: `docs/reviews/claude_review_consolidation_long_20261105_rule_b.py` e
  `docs/reviews/claude_review_consolidation_long_20261105_numeric_probe.json`. Lo script della seconda lettura resta legato
  all'interfaccia di `e2f19ee`.

**Test.** I dieci file di GPT: 252 passati. I 21 test sul contesto numerico, selezionati per nome, usano contesti reali: il
report storico P1, la ricevuta R3c archiviata e l'uscita di login07. Dodici mutazioni rifiutate, fra cui un BLAS in piu', un
OpenMP con thread diversi e un duplicato al posto di un pool mancante. Il gate rifiuta due contesti compatibili con lo
storico ma diversi fra loro.

**Identita'.** Ricalcolate a `51b4574`: sorgenti `a2ff1608…`, ricetta invariata `b2356a8e…`, `plan.json` `3bb834bb…`,
`launch_plan.json` `bacdd054…`; i cinque pin dei file di evidenza corrispondono. Fra `e2f19ee` e `51b4574` cambiano due soli
sorgenti. Nel piano cambiano 9 percorsi e nel piano di lancio 5: policy numerica, pin delle evidenze, regole di quota, review
precedente, hash dei sorgenti. Gli 82 job, i 4 checkout, gli endpoint e i tetti (9 GPU-ore, 1.711,67 core-ore) sono
identici. Stato `awaiting_claude_rereview`, `review.commit` nullo, `validate_plan` rifiuta ancora.

## Note, nessuna blocca

- **N-4, test.** Con il comando di `CLAUDE.md` (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) un test fallisce:
  `test_extract_uses_original_t100g_reference_and_one_teacher`, perche' `_threads(1)` legge anche `MKL_NUM_THREADS`
  dall'ambiente reale. Con `MKL_NUM_THREADS=1` sono 252 su 252. C'e' da `e2f19ee`; nella seconda lettura non l'ho visto perche'
  avevo impostato anche quella variabile. Basta un `monkeypatch.setenv` nel test. I test non entrano in `source_sha256`: la
  correzione non tocca i sorgenti congelati.
- **Facoltativo.** Resta non misurato l'eseguibile vero dentro un'allocazione su `login08` o `login13`: lo vedra' il primo
  benchmark, che viene dopo le estrazioni. Lo ritengo un rischio basso, per i tre fatti sopra. Chi vuole azzerarlo puo'
  eseguire gli stessi import in un job CPU di un minuto su `lrd_all_serial` prima del checkout delle estrazioni. Sarebbe un job
  nuovo, con preventivo e registrazione come sempre: decisione di GPT. Non lo pongo come condizione.

## Condizioni operative, separate dalla convergenza

**C-1, quota: non soddisfatta.** La regola che GPT ha scritto nel piano e' quella giusta: il margine deve reggere anche le
scritture gia' ammesse delle altre nostre campagne, non solo la soglia di 32 GiB all'invio. Resta un controllo dell'owner,
non del codice.

| Voce | GiB |
|---|---|
| Liberi alle 20:55, mia lettura (usati 1.038.996.536 kB) | 33,136 |
| Nuova matrice | 24,000 |
| Margine | 8,000 |
| 16 checkpoint FT/scratch ancora da scrivere, piu' 32 MiB di metadati | 1,724 |
| Totale richiesto | 33,724 |
| Mancano | 0,588 |

- La prova numerica non cambia nulla su questo punto.
- Aspettare non basta. Quando i 16 checkpoint saranno scritti il requisito scende di 1,72 GiB e lo spazio libero scende
  della stessa quantita', a circa 31,4 GiB: sotto i 32.
- Serve liberare almeno 0,6 GiB nel progetto, oppure una decisione registrata di Simone sul margine. Nel secondo caso e' una
  regola cambiata dopo aver visto il dato, e va scritta come tale. Non sono scelte mie.
- Correggo la mia seconda lettura. Avevo scritto che la soglia si sarebbe superata verso le 21:30: era un'estrapolazione da
  un solo intervallo ed era sbagliata. Dalle 19:57 alle 20:55 lo spazio libero e' passato da 33,15 a 33,136 GiB. Non
  attribuisco a FT15 il calo precedente: non l'ho misurato.

**C-2, sorgenti fermi sul ramo fra le fasi:** invariata.

## Cosa ho eseguito

- Checkout temporaneo di `51b4574`, rimosso a fine lavoro. Letti per intero il diff dei due sorgenti, dei due file di test,
  dei piani e dei documenti, e i tre file di evidenza.
- I dieci file di GPT: 252 passati con le tre variabili di thread a 1, 251 con il comando di `CLAUDE.md` (N-4). `bash -n` del
  launcher e `git diff --check`: puliti.
- Identita' e hash dei piani ricalcolati; confronto chiave per chiave di `plan.json` e `launch_plan.json` con `e2f19ee`.
- Censimento degli import di terze parti dei 65 file di progetto caricati dall'eseguibile, anche dentro le funzioni.
- Su Leonardo: quattro processi di soli import, `sinfo` della partizione seriale, `sacct` del job storico, quota alle 20:43,
  20:52 e 20:55, coda. Nessun file scritto sul cluster.
- Punta del ramo di GPT ricontrollata prima del verdetto: `51b4574`.

## Limiti

- Processi di soli import su un login node, non l'eseguibile in un'allocazione SLURM sui nodi seriali.
- Nessuna misura di uguaglianza delle predizioni fra processo con e senza torch. Che le sonde usino le stesse librerie e' un
  fatto; che il risultato sia identico bit per bit resta un ragionamento. La regola di GPT lo dichiara allo stesso modo.
- Non ho riaperto R-2 ne' le parti gia' approvate, come chiesto.
