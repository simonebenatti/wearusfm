# Review di Claude — T100-LQ dopo la correzione dei timeout: diagnostica, rilascio e invii di GPT (1dc6ded..54d7a3b, piu' cc81430) — 05/11/2026

Richiesta di Simone, 05/11 verso le 18:55. Alla domanda di Claude «Vuoi che riveda i commit di GPT dopo la mia consegna?» ha
risposto «Sì, rivedili». A review in corso ha aggiunto: «non so che casino ha fato gpt...». Solo review: nessun job, nessuna
modifica a codice o piani, su Leonardo soltanto letture.

Ramo `codex/tokenizer-logcontrast-20261104`. I sei commit chiesti vanno da 1dc6ded a 54d7a3b, sopra la mia consegna ff08483.
Alle 19:01 GPT ha pubblicato cc81430: l'ho letto per l'esito del training a 5.000 passi e ho rilanciato i test anche li'. Il
rilascio del benchmark d'inferenza che contiene non e' rivisto a fondo.

**Verdetto: APPROVATO. Nessun problema bloccante.** La catena e' ordinata e si ricostruisce per intero: review chiusa,
diagnostica eseguita una volta, lettura corretta, stessa ricetta rilasciata, invii uguali al piano. Un solo rilievo da
correggere, che non tocca i job: **R-1**, un test di GPT rotto dal suo ultimo commit. Quattro note minori, da N-1 a N-4.

| Commit | Cosa fa | Esito |
|---|---|---|
| 1dc6ded | Chiude R-TIMEOUT; piano e risorse della diagnostica `APPROVED_FOR_DIAGNOSTIC`; due precisazioni al mio documento | **CONFORME** |
| 4f8291a | Checkout 63590929 e diagnostica 63591184; primo `sbatch` rifiutato e risottomesso con lo stesso ID | **CONFORME** |
| 04bd994 | Raccolta e lettura della diagnostica, decisione di rilascio, bozza del piano del 05/11 | **CONFERMATA** dal mio ricalcolo |
| 2293b94 | Release `APPROVED_FOR_TRAINING`, due file | **CONFORME**: ricetta identica a quella approvata il 04/11 |
| 26cf367 | Pin A4 dei piani valutativi e piano esecutivo; e' il codice eseguito dal training | **CONFORME**, 365 test su 365 |
| 54d7a3b | Checkout 63592939, invii 63593091 (5.000) e 63593128 (19.618) | **CONFORME** a piano, ledger e `sacct` |
| cc81430 | Endpoint 5.000 verificato; rilascio del solo benchmark d'inferenza | Esito del training **CONFERMATO**; **R-1** |

## Cosa ho eseguito

- **Codice.** `git diff ff08483 cc81430 -- scripts src tests configs` e' vuoto: dopo la mia consegna GPT non ha toccato codice.
  Fra il codice del tentativo fallito (b2934b0) e quello rilanciato (26cf367) cambiano solo i nove file della correzione gia'
  rivista da GPT: quattro script, il launcher della diagnostica, quattro file di test. `src/` e' identico.
- **Lettura della diagnostica ricalcolata da zero** con `docs/reviews/claude_review_logsquare_relaunch_20261105_reading.py`:
  legge i campi grezzi e gli hash per tensore dell'archivio, non importa il progetto e non chiama `reading()`.
- **Pin ricalcolati** con `docs/reviews/claude_review_logsquare_relaunch_20261105_pins.py` sul contenuto dei commit 2293b94,
  26cf367 e cc81430: piano di training 114 pin, release 11, proposta di risorse 1, piano di estrazione 113, piano d'inferenza
  115. Tutti uguali ai file. Piano della diagnostica a 1dc6ded: 6 pin, guard: 4 pin, tutti uguali.
- **Piano del 04/11 contro piano del 05/11**, foglia per foglia: 254 foglie su 263 identiche. Le 9 diverse sono ID logici (2),
  ricevute di fase (2), cartella del run (1), proposta di risorse (2), release (2). Le 28 aggiunte sono il guard (3), la nota
  esecutiva sul guard (1), il contesto del rilancio (22) e due dimensioni in byte. Fra le foglie identiche: endpoint 5.000 e
  19.618, hash dei 102 sorgenti scientifici, piano e rapporto del preflight, riferimento iniziale `38cb4cce…`, controllo
  T100-L, passi aggiuntivi (5.000 e 14.618), risorse e tempi delle due fasi. Nella release epsilon resta 0,001.
- **Test** in un checkout temporaneo: i 12 file LQ di `tests/cpu` (undici `test_logsquare_*` e `test_tokenizer_logsquare.py`)
  piu' le due regressioni di GPT. A 26cf367 **365 passati**; a cc81430 **364 passati, 1 fallito** (R-1). `bash -n` dei sette
  launcher LQ passa. Nessun processo di test rimasto.
- **Leonardo, sola lettura, un comando alle 19:08:** `sacct` dei sei job, `squeue`, `ls` e `sha256sum` delle ricevute.
  Stati, risorse, tempi, nomi e commenti coincidono con le ricevute pubblicate. Il rapporto della diagnostica sul cluster ha
  SHA256 `270ef513…`, uguale all'originale archiviato; le sette dimensioni dei JSON coincidono. Ricevuta e quattro record dei
  rank del training coincidono con le copie archiviate. Il checkpoint a 5.000 ha 454.776.554 byte come nel pin.
- **Ledger comune** (`~/wearusfm_local/jobs/ledger.tsv`): righe 1148-1153 (diagnostica), 1162-1163 e 1166-1169 (training),
  dalla 1176 (benchmark), coerenti con le ricevute; un solo agente (`gpt`), un solo job SLURM per ID logico.
  `scripts/submit_job.py` ha lo stesso SHA256 `814da4bc…` su `main`, sul ramo di GPT e sul mio; `--dopo-rifiuto` e' l'opzione
  prevista dallo strumento per un ID rifiutato da `sbatch`. Il rifiuto era esplicito («Job dependency problem»): il checkout
  gia' concluso era uscito dalla tabella attiva di SLURM, nessun job creato, nessun doppione.

## La diagnostica 63591184: cosa dice davvero

Criteri fissati nel piano prima dell'esecuzione (ff08483, approvati a 1dc6ded alle 14:51; job partito alle 14:56). Il rapporto
porta lo SHA del piano `f7f0fc3e…`, uguale al file del commit congelato. Sei contesti, tutti con ritorno 0 e senza errori.

| Controllo ricalcolato | Esito |
|---|---|
| Dieci costruzioni (cinque contesti, due bracci), 269 tensori ciascuna | Differisce un solo tensore: `tokenizer.fe_gain` |
| Gain del contesto preflight | 548.6939187567023 (`0x1.1258d2546f0f0p+9`), hash `38cb4cce…`: il riferimento |
| Gain di controller e trainer | 548.6939187567021 (`0x1.1258d2546f0efp+9`): il double subito sotto, differenza −1,1368683772161603e−13 |
| Sostituendo il solo gain su una copia | Torna l'hash di riferimento in 10 costruzioni su 10 |
| T100-L contro T100-LQ nello stesso contesto | Stato condiviso e RNG CPU e CUDA identici in 5 contesti su 5 |
| Passaggio su CUDA | Stato conservato in 10 su 10 |
| Guard | Accetta controller, controller a un thread e i due contesti trainer; rifiuta il preflight, come previsto |
| Quattro rank reali sotto `torchrun --verify-only` | Un solo stato giudicato, `0b6abc83…`, quello del contesto trainer; W_log nullo (384x64) |
| Aggiornamenti | Nessuno: `trainer_returned` falso su ogni rank, cartella del run vuota, ritorno 0 |
| Supervisione dei processi | 7 su 7 senza timeout, segnali o superstiti; tabella dei processi disponibile sul nodo |

I tre enunciati del piano risultano veri anche nel mio ricalcolo: causa riprodotta nei contesti di oggi, il trainer
costruisce lo stato registrato, la prova a secco della correzione passa. La lettura di GPT in
`docs/reviews/logsquare_initialization_results_20261105.md` e' corretta in ogni numero che ho controllato. Corretto anche
il suo limite dichiarato: `0b6abc83…` e' un hash osservato oggi, non un riferimento storico; lo stato accettato resta una
ricostruzione registrata.

## Il training rilanciato

| Job | Cosa | Stato alle 19:08 | Durata | GPU-ore | core-ore |
|---|---|---|---|---|---|
| 63532335 | Tentativo del 04/11, fermato dal controllo | FAILED in 21 s, nessun update | 21 s | 0,0233 | 0,1867 |
| 63591184 | Diagnostica (tetto proprio 1 GPU-ora) | COMPLETED 0:0 | 56 s | 0,0622 | 0,4978 |
| 63593091 | Training fino a 5.000 passi | COMPLETED 0:0, fine per `max_steps` | 1 h 06 min 10 s | 4,4111 | 35,2889 |
| 63593128 | Continuazione fino a 19.618 | PENDING, motivo Priority | tetto 4 h | tetto 16 | tetto 128 |
| 63614514 | Benchmark d'inferenza (cc81430, tetto 0,5 GPU-ore) | sottomesso alle 19:06, stato non letto | tetto 30 min | tetto 0,5 | tetto 4 |

Nel training vero i quattro rank hanno attestato lo stesso stato dello studente, `0b6abc83…`, con il guard `9cc3c37e…`:
classificazione `historical_training_state`, W_log nullo, gli altri 268 tensori uguali al riferimento. La ricevuta e'
completa, porta codice 26cf367, piano `d90b55ee…`, release `ee08185d…`, e dichiara il gain del checkpoint uguale a quello
storico. Nel log nessun errore. La correzione ha quindi funzionato anche in produzione, non solo nella prova a secco.

## Autorizzazione e budget

- GPT ha rilanciato senza chiedere un nuovo consenso. La base che registra e' la frase di Simone del 04/11 riportata in
  `resource_proposal.json`: «si ma dovra fare altro immagino, lo autorizzo gia a desso», e la regola di `CLAUDE.md`:
  «L'approvazione di un test comprende il lancio gia' autorizzato [...] non autorizza nuovi esperimenti o retry ciechi».
  Lettura mia: regge. Il tentativo del 04/11 non ha addestrato nulla, il rilancio e' lo stesso test con la stessa ricetta, e
  non e' cieco: causa diagnosticata, correzione rivista, diagnostica letta prima del rilascio. La voce di GPT delle 15:19
  dice che Simone ha chiesto dell'attesa del rapporto e ha avuto la spiegazione: quella conversazione non la posso verificare.
- Tetto della tranche: 22 GPU-ore e 176 core-ore. Massimo ormai possibile: 0,0233 + 4,4111 + 16 = **20,4344 GPU-ore** e
  0,1867 + 35,2889 + 128 = **163,4756 core-ore**. Dentro il tetto. Diagnostica e benchmark hanno tetti propri sotto 5 GPU-ore.

## Rilievi

- **R-1, da correggere, non blocca i job.** A cc81430 fallisce
  `tests/cpu/test_logsquare_infer_benchmark.py::test_review_plan_freezes_selection_aggregation_and_null_candidate_pins`.
  Il test pretende lo stato di review del piano committato: `status == 'REVIEW_REQUIRED'` e pin dei checkpoint nulli. Il
  rilascio ha portato lo stato a `APPROVED_FOR_BENCHMARK` e ha riempito il pin del 5.000. I vincoli congelati che il test
  protegge valgono ancora: selezione, aggregazione, tetto 0,5 GPU-ore, 8 CPU, pin del 19.618 nullo, ricevuta d'inferenza
  nulla. Il codice chiede proprio `APPROVED_FOR_BENCHMARK` (`logsquare_infer_benchmark.py:315`, `logsquare_evaluation.py:205`),
  quindi il benchmark in coda non ne risente. Va corretto il test prima dell'integrazione su `main`: deve accettare i due
  stati e tenere fermi i vincoli. E' lo stesso tipo di rottura di D-5 dell'audit di allineamento.
- **N-1, contabilita'.** `resource_proposal.json` ripete il tetto intero di 22 GPU-ore per il rilancio e tiene a parte le
  0,0233 del tentativo fallito. Con la fase a 5.000 chiusa a 4,4111 il totale non puo' piu' superare il tetto. Nel consuntivo
  finale la somma va fatta su tentativo fallito, 5.000 e 19.618 insieme.
- **N-2, processo.** Il piano del 05/11 conserva la frase «Claude review before deployment remains required». Per il rilancio
  GPT ha considerato la condizione soddisfatta dalla mia review del 04/11 sulla ricetta (pin `097bc5b1…`) e dalle sue due
  review sulla correzione, che e' codice mio. Release, checkout e invii sono partiti in circa venti minuti dalla fine della
  diagnostica, senza passare da me. Con i ruoli del 05/11 una mia correzione e' soggetta alla review di GPT, quindi la lettura
  e' sostenibile, e questa review a posteriori non trova nulla che una review preventiva avrebbe fermato. Resta di Simone
  dire se un rilancio dopo un arresto tecnico deve ripassare da Claude prima del deploy.
- **N-3, limite dell'evidenza sulla mia correzione.** Sul cluster la pulizia forzata non e' mai scattata: e' provato solo il
  percorso normale. Le attese di 75 e 285 s erano larghe: l'intera diagnostica e' durata 56 s. Il supervisore e' usato solo
  dalla diagnostica, non dal training.
- **N-4, forma.** Nelle voci di GPT fino a 54d7a3b, nel registro e in `docs/decisioni.md`, molti numeri sono attaccati alle
  parole («Diagnostico63591184», «COMPLETED0:0/56s»). Il contenuto e' giusto, la lettura e la ricerca ne soffrono. In cc81430
  le voci sono gia' scritte con gli spazi.
- **Accettate** le due precisazioni di GPT a `docs/logsquare_initialization_fix_20261105.md`: senza tabella dei processi il
  segnale va al gruppo della radice, quindi «mai a un gruppo» era troppo; gli intervalli di 0,5 e 0,1 s sono nominali.

## Cosa non ho verificato

- Lo SHA256 del checkpoint a 5.000 (455 MB): ho controllato solo la dimensione.
- Lo SHA dei due checkout sul cluster letto direttamente: lo conosco dalla prima riga dei log dei job, scritta dal launcher
  che rifiuta uno SHA diverso.
- Il controllo A1 sulla selezione del benchmark e il resto del rilascio d'inferenza di cc81430.
- Il ricalcolo interno del `critico` di GPT e gli altri 24 file dell'archivio oltre al rapporto, per SHA sul cluster.

## Prompt per GPT

> Review di Claude su 1dc6ded..54d7a3b e, per il solo esito a 5.000 e i test, su cc81430: **APPROVATO, nessun blocco**.
> Lettura della diagnostica 63591184 ricalcolata da zero e confermata; ricetta del 05/11 identica a quella del 04/11 (254
> foglie su 263, le 9 diverse sono ID, percorsi e pin); pin tutti uguali ai file; invii uguali a piano, ledger e `sacct`;
> quattro rank del training vero sullo stato `0b6abc83…`. Review: `docs/reviews/claude_review_logsquare_relaunch_20261105.md`
> sul ramo `worktree-claude-training-engine`.
> **R-1, da correggere prima dell'integrazione su `main`:** a cc81430 fallisce
> `tests/cpu/test_logsquare_infer_benchmark.py::test_review_plan_freezes_selection_aggregation_and_null_candidate_pins`
> perche' pretende `REVIEW_REQUIRED` e pin nulli. Rendi il test valido nei due stati tenendo fermi selezione, aggregazione,
> tetto 0,5 GPU-ore e 8 CPU, e riesegui i 12 file LQ. Il benchmark 63614514 non ne risente: non fermarlo e non rilanciarlo.
> **N-1:** nel consuntivo finale somma tentativo fallito, 5.000 e 19.618 contro il tetto di 22 GPU-ore (massimo oggi 20,4344).
> **N-2:** il piano dice ancora «Claude review before deployment remains required»; per il rilancio hai usato la review del
> 04/11 e le tue due sulla correzione. Lettura sostenibile, segnalata a Simone, che decide se vuole altro per i prossimi rilanci.
> **N-4:** nelle voci di registro lascia gli spazi fra parole e numeri.
> Nessuna modifica richiesta a guard, controller, piano o release. Nessun job da parte di Claude.
