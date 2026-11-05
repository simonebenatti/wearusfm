# T100-LQ: causa dell'arresto all'inizializzazione e correzione

Consegna di Claude del 05/11/2026 sul compito scritto da GPT e inoltrato da Simone
(«Ti affido diagnosi e correzione del blocco di inizializzazione T100-LQ»).
Ramo `claude/logsquare-init-fix-20261105`, partito da `4d5a3ef` di GPT.
Nessun job sottomesso, nessun training, nessun retry. Gli artefatti del fallimento,
il piano e la release del 04/11 non sono stati toccati.

**Aggiornamento dopo la review di GPT** (`92fa557`,
`docs/reviews/gpt_review_logsquare_initialization_fix_20261105.md`). Stato accettato e
ingresso dei rank approvati. Lettura e risorse della diagnostica approvate. Corretto il
rilievo bloccante sui timeout: sezione 7. Rettificate le formulazioni indicate da GPT,
nei punti segnati «Rettifica». Il piano resta `REVIEW_REQUIRED`.

## In breve

- L'hash atteso `38cb4cce…` descrive il modello costruito dal **preflight**. I checkpoint
  del training storico di T100-L attestano un gain che differisce da quello del preflight
  per **un ULP**, nel buffer `tokenizer.fe_gain`. Il controllo del 04/11 chiedeva quindi a
  T100-LQ uno stato che il training storico non ha mai avuto.
- **Rettifica.** I checkpoint storici attestano il gain, non l'hash completo dello stato
  iniziale. Lo stato accettato è una **ricostruzione registrata dell'inizializzazione
  storica**: tensori del riferimento più gain storico. Non è un hash iniziale storico
  osservato.
- La correzione verifica lo stato **sul modello che ogni rank costruisce e addestra**, non
  su un modello temporaneo. Accetta un solo stato: i tensori del riferimento revisionato,
  il gain salvato dal training storico di T100-L, `W_log` nullo.
- Nessun sorgente scientifico cambia. Niente ripristini, tolleranze o hash nuovi.
- Resta una verifica sul cluster, già pronta: la diagnostica di GPT completata con la
  prova a secco della correzione. La sottomette GPT dopo la sua review.

## 1. Causa

### Dimostrato dai registri del repository, senza job

| Braccio | Preflight | Checkpoint del training storico (5000 e 19618) | Scarto |
| --- | --- | --- | --- |
| T100-L | 548.6939187567023 `0x1.1258d2546f0f0p+9` | 548.6939187567021 `0x1.1258d2546f0efp+9` | 1 ULP |
| T100-G | 548.6939187567023 | 548.6939187567021 | 1 ULP |
| T25-L | 982.425066637557 `0x1.eb3668956576dp+9` | 982.4250666375573 `0x1.eb3668956576fp+9` | 2 ULP |

Fonti: `results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json`
(codice ef8016a, i tre bracci), `results/infra/logsquare_execution_20261104/report.json.gz`
(codice 43cad1f, lo stesso valore per T100-L e T100-LQ),
`results/infra/tokenizer_evaluation_20261103/completed_5000/*/receipt_summary.json` e
`completed_19618/pose/*/receipt_summary.json` (training col codice afb644e).

1. `38cb4cce…` è l'hash dello stato intero di T100-L nel preflight del 03/11, gain
   compreso. Il preflight LQ del 04/11 lo ritrova identico su entrambi i bracci.
2. Il gain è un buffer calcolato nel costruttore. Nessun aggiornamento lo modifica, e
   l'EMA copia nel teacher quello dello studente. Il valore salvato nei checkpoint è
   quindi quello di partenza. **Il training storico di T100-L non ha mai avuto lo stato
   `38cb4cce…`.**
3. I due contesti differiscono per l'ordine di import delle librerie numeriche. I preflight
   caricano NumPy prima di Torch. Controller, trainer e il nuovo ingresso dei rank
   caricano Torch prima. Lo misura un test con un registratore degli import.
4. **Rettifica.** L'ordine degli import è la spiegazione proposta, da verificare con i
   contesti della diagnostica. Non è una misura del processo del 04/11, che non ha
   registrato né il gain né i tensori. La sostiene il precedente del 14/10 (job 60561493,
   `results/passo6/diag_initial_state_20261014.json`): sul cluster l'ordine degli import
   cambiava solo `tokenizer.fe_gain`. I due valori di allora per T25 sono esattamente
   quelli che oggi separano preflight e checkpoint di T25-L.

### Non ancora osservato: serve il cluster

- Il gain e i tensori del modello temporaneo nel job 63532335. Il vecchio messaggio non
  li stampava. Atteso: tensori del riferimento e gain del training.
- Che il trainer di oggi costruisca ancora il gain dei checkpoint storici.
- Che nel contesto del trainer gli altri tensori di T100 siano quelli del riferimento. Per
  T25 lo ha misurato il precedente del 14/10. I registri del training storico salvano il
  gain, non l'hash dello stato di partenza: quello stato è ricostruito da riferimento e
  gain, e i contesti `trainer` della diagnostica lo confermano o lo smentiscono.
- Il precedente del 14/10 mostra che contano anche i thread: con otto thread usciva un
  terzo valore. I launcher del training storico, di T100-LQ e della diagnostica esportano
  tutti un thread e caricano lo stesso ambiente.
- Sul Mac lo stato del cluster non si riproduce: differiscono anche altri tensori.

## 2. Correzione

### La regola: un solo stato accettato

`scripts/logsquare_initial_state.py`, usata da controller, rank e diagnostica.

1. Lo stato osservato senza `W_log`, con il **solo gain sostituito su una copia** dal
   valore del preflight, ha hash `38cb4cce…`. Ogni altro tensore è quindi quello revisionato.
2. Il gain osservato è **uguale bit per bit** a 548.6939187567021, quello dei checkpoint
   storici di T100-L.
3. `W_log` ha la forma attesa ed è esattamente nullo.

I valori stanno in `results/infra/logsquare_initialization_20261105/initial_state_guard.json`,
ognuno con il file storico da cui viene. A ogni esecuzione il controller li rilegge da
quei file e rifiuta un guard che non coincide.

Non c'è ripristino: nessun valore viene scritto nel modello. Non c'è tolleranza: i
confronti sono su bit. Non c'è un secondo hash registrato: il riferimento resta
`38cb4cce…`, il gain viene dai registri storici e non dal processo osservato.

### Dove si applica: sul modello realmente addestrato

Il trainer costruisce lo studente con il nome di modulo `run.WearUsFM` (`run.py:1337`),
dopo `torch.manual_seed` (`run.py:1333`). Il nuovo ingresso `scripts/logsquare_train_rank.py`,
che torchrun avvia al posto di `train_jepa.py` nella sola fase fresca:

- importa per primo il trainer di produzione, quindi ha il suo ordine di import;
- sostituisce `run.WearUsFM` con un costruttore che costruisce il modello come prima,
  lo verifica e lo restituisce **senza modificarlo**;
- se lo stato è rifiutato solleva dentro il costruttore: il trainer non riceve nessun
  modello, quindi non esistono teacher, optimizer, DDP né aggiornamenti;
- scrive per ogni rank un record con hash, gain e classificazione.

I tre punti sollevati da GPT:

- **Modello temporaneo.** Il controllo del controller resta, ma è dichiarato pre-controllo.
  La garanzia è nei rank. La ricevuta della fase richiede le quattro attestazioni.
- **`count_parameters` sul rank 0.** Costruisce un modello prima del seed, mai addestrato.
  Viene registrato con il suo gain e non giudicato. Ogni altro modello è giudicato come
  studente. Il controller pretende i ruoli esatti: conteggio e studente sul rank 0, un
  solo studente sugli altri.
- **DDP.** DDP diffonde i buffer dal rank 0 a ogni forward. Con un solo stato accettato
  tutti i rank partono identici bit per bit, quindi la diffusione non cambia nulla. Il
  controller verifica inoltre che il gain salvato nei checkpoint, studente e teacher, sia
  quello registrato.

### Perché l'esperimento approvato resta lo stesso

- I 102 sorgenti scientifici del piano sono identici byte per byte: lo verifica la
  diagnostica contro il piano del 04/11. Epsilon, dati, loss, seed, schedule, dose e
  architettura non cambiano. La stringa di protocollo non cambia.
- T100-LQ parte dalla ricostruzione registrata della partenza di T100-L, cioè tensori
  del riferimento e gain dei suoi checkpoint, più `W_log` nullo. I registri attestano il
  gain, non gli altri tensori di allora. È il confronto approvato: T100-L più
  `log(0,001 + x²)`.
- Dietro l'ingresso il trainer addestra lo stesso stato. Con quattro rank reali su CPU,
  gruppo di processi e DDP compresi, due passi danno studente, teacher, optimizer e
  generatori identici bit per bit al trainer normale.

### Punto deciso nella review

**Lo stato esatto del preflight viene rifiutato**, con classificazione
`preflight_reference_state`. Avevo chiesto a GPT di confermare che questa lettura rispetta
«Non adottare un nuovo hash atteso». **GPT ha approvato** nella review `92fa557`, punto A:
«È un emendamento esplicito del riferimento accettato, motivato da evidenze storiche
indipendenti dal modello osservato oggi. Non è l'adozione di un nuovo hash misurato dopo
il fallimento, né una tolleranza o una scrittura nel modello. Lo stato esatto del
preflight è correttamente rifiutato.»

Alternativa non implementata: scrivere nel modello addestrato il gain del preflight.
Recupererebbe l'hash `38cb4cce…`, ma modificherebbe lo stato che il trainer costruisce e
allontanerebbe T100-LQ di un ULP da T100-L. Cambiare soltanto l'ordine degli import del
controller farebbe passare il controllo lasciando diverso il modello addestrato.

### Messaggio diagnostico

Un rifiuto ora riporta: hash atteso, hash osservato, hash con il solo gain sostituito,
esito sugli altri tensori, gain in decimale ed esadecimale con la distanza in ULP dai
due valori registrati, quale gruppo di tensori differisce e una riga di lettura. Il
record JSON contiene l'hash di ogni tensore. Le quattro classificazioni sono
`historical_training_state`, `preflight_reference_state`,
`reference_tensors_with_unregistered_gain`, `tensor_difference_beyond_gain`.

## 3. Test

Ambiente `~/.venvs/wearusfm-tok`, un thread, sul Mac. Nessuna prova su CUDA.

- **Regressione prima e dopo.** `test_training_context_gain_passes_the_exact_initialization_check`
  costruisce il modello reale con il gain del training. Sul controller di `4d5a3ef`
  (SHA `cb3ff445…`, quello della release) fallisce con l'errore del job:
  `ValueError: fresh shared initialization differs from historical T100-L`. Sul controller
  corretto passa.
- **Una vera modifica dei parametri resta rifiutata.** Un ULP su un parametro reale è
  rifiutato prima e dopo la correzione. Sono rifiutati anche un altro seed, un gain non
  registrato, lo stato del preflight e un `W_log` non nullo.
- **Modello realmente addestrato.** Un passo sul trainer reale: stato identico con e senza
  ingresso. Quattro rank reali: stato identico dopo due passi, quattro attestazioni, gain
  dei checkpoint uguale a quello verificato. Uno studente rifiutato ferma ogni rank prima
  di qualsiasi aggiornamento, senza checkpoint né metriche.
- **Conteggi.** I quattro file LQ: 154 test passati alla consegna, 176 dopo la correzione dei
  timeout descritta nella sezione 7. Suite `tests/cpu` sul commit `b8689e1`: 2446 passati su 2447.
  L'unico fallito è `test_pose_b19618_launcher.py::test_exact_B_recipe_only_maxsteps_changes_schedule_identical`:
  fallisce allo stesso modo su `4d5a3ef` e questo ramo non tocca i suoi file. Su `4d5a3ef` i test
  erano 2349: i 98 in più sono tutti nei quattro file LQ.
- Il launcher della diagnostica passa `bash -n`.

## 4. Verifica ancora necessaria sul cluster

Una sola: la diagnostica di GPT, completata senza strumenti nuovi
(`scripts/logsquare_initialization_diagnostic.py`, piano e risorse in
`results/infra/logsquare_initialization_20261105/`, stato **REVIEW_REQUIRED**).

- I cinque contesti di GPT restano. Ogni modello T100-LQ riceve anche il verdetto del
  guard; ogni riga riporta la distanza in ULP dai due gain registrati.
- Un sesto contesto, `rank_entry_torchrun`, ripete il lancio corretto fino alla
  costruzione dello studente: pre-controllo del controller, poi il comando di produzione
  della fase fresca sotto torchrun con `--verify-only`. Quattro rank reali costruiscono e
  verificano lo studente e si fermano dentro il costruttore. Nessun modello arriva al
  trainer: niente teacher, optimizer, DDP, aggiornamenti o checkpoint.
- **Rettifica.** I rank leggono manifest, indici e scale come il trainer. La formulazione
  esatta è: nessun batch di training e nessun forward sui segnali. Non è assenza di
  letture dei dati.
- La lettura è fissata nel piano prima dell'esecuzione, su confronti esatti:
  `cause_demonstrated`, `trainer_builds_the_registered_training_state`,
  `fix_dry_run_passed`. Un contesto fallito non dà lettura. Il rapporto non autorizza mai
  il training. `cause_demonstrated` è la riproduzione operativa della discrepanza nei
  contesti di oggi, con i limiti storici detti sopra.
- Ogni contesto ha un limite che comprende la pulizia dei suoi processi: 90 secondi per
  le costruzioni, 300 per la prova a secco. Dettagli nella sezione 7.

| Job | Risorse | Durata massima | Tetto |
| --- | --- | --- | --- |
| Checkout | 2 CPU, 4G | 10 min | 0,33 core-ore |
| Diagnostica | 4 GPU, 32 CPU, 240G, `boost_qos_dbg` | 15 min | 1,0 GPU-ore, 8,0 core-ore |

Prima era 10 minuti e 0,67 GPU-ore. Sul Mac torchrun non parte, quindi il sesto contesto
è provato in locale solo nei suoi pezzi: comando, limite di tempo, lettura dei record e
quattro rank reali avviati senza il launcher di torchrun.

## 5. Pin e release da aggiornare prima del rilancio

A cura di GPT, dopo la review.

1. **Diagnostica.** Stato del piano a `APPROVED_FOR_DIAGNOSTIC`, checkout congelato nuovo
   dal commit che contiene questa consegna e la correzione dei timeout, sottomissione.
   Il piano fissa già il guard (SHA `9cc3c37e…`) e i tre limiti di tempo: 75, 285 e 15
   secondi. Il launcher rifiuta un piano con valori diversi.
2. **Release nuova.** `controller_sha256` deve coprire quattro file: `scripts/logsquare_train.py`,
   `scripts/slurm/logsquare_train.sbatch`, `scripts/logsquare_train_rank.py`,
   `scripts/logsquare_initial_state.py`. Serve il pin della review di questa correzione.
3. **Piano di training nuovo**, in un file nuovo: quello del 04/11 è fissato dalla
   diagnostica e fa parte delle evidenze. Aggiungere `initial_state_guard` con percorso e
   SHA del guard, e il pin della release nuova. `scientific_source_sha256` resta
   identico. Il controller rifiuta un piano senza guard.
4. **Valutazione.** I piani che fissano piano di training e release vanno riallineati.
5. **Manifest di lancio e checkout.** Checkout congelato nuovo e ID logici nuovi nel
   ledger. Le cartelle del run e delle ricevute non esistono sul cluster: lo registra il
   riepilogo del fallimento raccolto da GPT. I percorsi del piano restano quindi liberi.
6. Un rifiuto dei rank dopo il lancio lascia la cartella del run e quella della ricevuta.
   Un secondo lancio richiede allora una decisione esplicita, non un retry.

## 6. Correzione a una mia nota

Nella review della diagnostica (`9a46d7c`, nota M1) avevo scritto che il controller
fallito non interrogava CUDA prima del controllo. È sbagliato: `command_for` costruisce
il parser del trainer, il cui default di `--device` chiama `torch.cuda.is_available()`,
prima di `verify_initialization`. La nota M1 cade. La nota M2 è coperta dal sesto
contesto, che passa dal trainer reale.

## 7. Dopo la review di GPT: nessun processo resta dopo un contesto (R-TIMEOUT)

Review di GPT `92fa557`: regola dello stato e ingresso dei rank approvati; lettura e
risorse approvate; un rilievo bloccante sui timeout della diagnostica. Questa sezione
descrive la correzione, nei commit `88fce4c`, `8bbc3e1` e `0d264f7`. Guard, ingresso dei rank,
controller di training, criterio scientifico, epsilon, ricetta e sorgenti scientifici non
sono cambiati. Il piano resta `REVIEW_REQUIRED`. Nessun job, training, retry o release.

### Difetto

- `run_bounded` uccideva il solo gruppo di processi del launcher. I worker di torchrun
  stanno in sessioni proprie e restavano attivi.
- Il watchdog di `execute` terminava il solo figlio diretto.
- I limiti annidati 300 e 270 secondi non garantivano che scadesse prima quello interno.
- Riprodotto: sul codice revisionato i due test di GPT falliscono. Dopo la correzione
  passano senza modifiche.

### Meccanismo

- Ogni comando parte in una sessione propria. I suoi processi sono i membri dei gruppi di
  processi che l'invocazione ha creato: il gruppo della radice e il gruppo di ogni figlio
  di un membro. Si ricavano dalla tabella dei processi (`/bin/ps`), letta ogni 0,5 s
  durante l'attesa e ogni 0,1 s durante la pulizia.
- L'appartenenza a un gruppo sopravvive alla morte del padre. Un worker resta tracciato
  anche quando il suo launcher è terminato.
- I segnali vanno per PID, solo a membri elencati nella tabella corrente. Mai a un
  gruppo, a un identificatore memorizzato o a un nome di programma. Un gruppo vuoto
  viene dimenticato subito: il suo numero non può indicare un processo successivo.
- Lo stesso `run_bounded` serve i due livelli. Se scade prima il limite esterno, il
  genitore chiude figlio, launcher e worker.
- Senza tabella dei processi non parte nessun comando. Una lettura fallita durante la
  pulizia viene ripetuta fino alla scadenza. Se la tabella non torna, viene segnalato il
  solo gruppo della radice e il rapporto dice che i discendenti non sono stati verificati.

La pulizia ha una scadenza assoluta di 15 secondi e procede per passi.

| Passo | Durata massima | Azione |
| --- | --- | --- |
| Attesa spontanea | 2 s | Solo se il comando è tornato da sé: chi resta può finire da solo |
| Chiusura ordinata | 5 s | SIGTERM al solo launcher, cioè ai membri senza un padre attivo fra i membri |
| Richiesta a tutti | 3 s | SIGTERM a ogni membro ancora attivo, anche in sessioni separate |
| Terminazione | 5 s | SIGKILL a chi è ancora attivo, poi attesa |

Dopo un timeout la somma è 13 secondi, dopo un ritorno spontaneo 15. Chi sopravvive
viene riportato con PID e nome e non viene atteso oltre. Un membro rimasto senza padre
diventa a sua volta destinatario della chiusura ordinata.

Il launcher va avvisato da solo. Con il contesto elastic reale di torch, provato in
locale con torch 2.14, SIGTERM simultaneo a launcher e worker interrompe la chiusura del
launcher con un'eccezione. Avvisato da solo, il launcher chiude i worker e termina con
il proprio stato.

### Limiti di tempo

| Contesto | Attesa massima | Pulizia massima | Limite dichiarato |
| --- | --- | --- | --- |
| Cinque costruzioni, ciascuna | 75 s | 15 s | 90 s |
| Prova a secco con torchrun | 285 s | 15 s | 300 s |
| Sei contesti in tutto | | | 750 s |
| Margine per launcher, I/O e provenienza | | | 150 s |
| Allocazione | | | 900 s, 15 minuti |

- I limiti dichiarati e il tetto di 15 minuti sono quelli di prima. Per farci stare la
  pulizia, l'attesa scende da 90 a 75 secondi e da 300 a 285.
- Dentro la prova a secco torchrun riceve 285 secondi meno 15 di pulizia, meno 20
  riservati al rapporto, meno il tempo già speso nella preparazione. Il suo limite
  scade quindi prima di quello del genitore. Con meno di 30 secondi torchrun non parte
  e il contesto riporta l'errore.
- Il piano fissa `child_timeout_seconds` 75, `rank_entry_timeout_seconds` 285 e
  `cleanup_bound_seconds` 15. Il validatore e il launcher rifiutano valori diversi.
- Risorse invariate: 1,0 GPU-ore e 8,0 core-ore per la diagnostica, 0,33 core-ore per
  il checkout.

### Che cosa resta nel rapporto

- stdout e stderr del figlio e di torchrun restano nei file. Le loro code entrano nel
  rapporto di ogni contesto fallito.
- Ogni riga riporta il registro della supervisione: processi osservati, destinatari di
  SIGTERM e di SIGKILL, sopravvissuti, durata della pulizia.
- Un contesto riporta errore, e il rapporto non dà lettura, anche quando dei processi
  sono rimasti dopo un comando tornato da sé o sono sopravvissuti alla pulizia.

### Verifiche in sola lettura su Leonardo

- `/bin/ps` sul nodo di login è procps-ng 3.3.15 e stampa le cinque colonne usate.
- Torch 2.2.0a0 di `cineca-ai/4.3.0`, cartella `torch/distributed/elastic/`:
  `multiprocessing/api.py` registra l'handler di SIGTERM (righe 224-230), avvia i worker
  in sessioni proprie (566) e ne segnala il gruppo (584). `agent/server/api.py` (732-734)
  e `agent/server/local_elastic_agent.py` (306-311) chiudono i worker con il segnale
  ricevuto. `_close` (691-718) attende fino a 30 secondi e poi usa SIGKILL.
- SIGTERM al solo launcher è quindi la sua chiusura ordinata anche su Leonardo. La
  nostra scadenza prevale sui 30 secondi di torch.

### Test

`tests/cpu/test_logsquare_initialization_diagnostic.py`, 88 test. Ogni scenario usa
processi reali. Si controlla che il worker smetta di lavorare, non che sparisca un PID.
Ogni processo scrive il proprio PID e una fixture lo termina anche se il test fallisce.

| Scenario | Esito verificato |
| --- | --- |
| Limite interno, worker in sessione propria: primo scenario di GPT | Worker fermo; SIGTERM al launcher e poi al worker |
| Limite esterno con figlio, launcher e worker annidati: secondo scenario di GPT, due varianti | Tutti fermi; output ed errore nel rapporto |
| Ritorno normale | Stato e output conservati, nessun segnale |
| Errore del figlio | Stato ed errore conservati; worker rimasto chiuso, contesto in errore |
| Launcher con handler di SIGTERM | Chiude da sé il worker; segnalato il solo launcher |
| Contesto elastic reale del torch installato, 2.14 | Il launcher chiude i due worker; segnalato il solo launcher |
| Worker che ignora SIGTERM, launcher che lo ignora, entrambi | Passi nell'ordine dichiarato, SIGKILL solo a chi serve |
| Segnali senza effetto | Ritorno alla scadenza, sopravvissuti nominati |
| Processi estranei dello stesso utente e programma | Mai segnalati |
| Supervisore interrotto | Pulizia eseguita |
| Tabella dei processi assente o persa | Nessun avvio; pulizia non verificata dichiarata |
| Lettura della tabella fallita e poi riuscita | Pulizia completata e verificata |

I due test di GPT in `docs/reviews/test_logsquare_detached_timeout_20261105.py` passano
senza modifiche.

| Esecuzione | Esito |
| --- | --- |
| Quattro file LQ | 176 passati: 27, 9, 52 e 88 |
| Due test di GPT | 2 passati |
| `bash -n` del launcher | passa |

Prove di mutazione sul solo script, 18 test selezionati: quelli della supervisione più i
due di GPT. Dopo ogni prova lo script è stato ripristinato e il suo hash verificato.

| Mutazione | Test falliti |
| --- | --- |
| Nessuna | 0 su 18 |
| Tracciato il solo gruppo della radice, come nel codice revisionato | 17 su 18, compresi i due di GPT |
| Nessun SIGKILL | 5 su 18 |
| SIGTERM a tutti insieme, non prima al launcher | 3 su 18, compreso il contesto elastic reale |
| Il nome di un processo terminato sostituisce quello registrato | 2 su 18 |
| Una lettura fallita della tabella chiude la pulizia | 2 su 18 |

La suite `tests/cpu` completa non è stata rieseguita dopo questa correzione. I file
cambiati sono letti solo da tre file di test LQ, tutti rieseguiti.

### Smoke locale

Diagnostica reale dal checkout pulito del commit `0d264f7`, su una copia del piano fuori
dal repository: stato approvato, CUDA non richiesta, limite della prova a secco 120
secondi. Durata 97 secondi, uscita 1 come atteso in locale.

- I cinque contesti di costruzione tornano con stato 0. Un processo osservato ciascuno,
  nessun segnale, pulizia di circa 0,03 secondi.
- Nella prova a secco torchrun non parte su questo Mac. Scade il suo limite di
  83,8 secondi. SIGTERM al solo launcher basta: nessun SIGKILL, nessun
  sopravvissuto, pulizia in 0,22 secondi.
- Il rapporto riporta l'errore, le code degli output e nessuna lettura.
- A fine smoke nessun processo residuo e checkout pulito. Dettagli in
  `results/infra/logsquare_initialization_20261105/local_smoke.json`.

### Limiti

- Non provato in locale: torchrun reale con quattro rank e NCCL, e la supervisione su
  Linux. Il codice è lo stesso. Il formato di `ps` è stato letto sul nodo di login, non
  su un nodo di calcolo. Se `/bin/ps` mancasse, la diagnostica si fermerebbe prima di
  avviare qualunque contesto.
- Il test con il contesto elastic reale usa torch 2.14 e viene saltato con un'altra
  firma. Per torch 2.2 di Leonardo il comportamento è stato letto nel sorgente.
- Limite residuo: non viene tracciato un discendente che passa in una sessione propria e
  perde il padre prima che una tabella lo mostri come figlio di un membro. La finestra è
  al più 0,5 secondi in attesa e 0,1 in pulizia. I worker di torchrun restano figli
  dell'agente per tutta la loro vita. Walltime e cgroup di SLURM restano la protezione
  esterna.
- I 75 secondi di attesa di un contesto di costruzione non hanno una durata misurata
  sul cluster alle spalle. Un contesto che scade non dà lettura e non viene ripetuto.
- Regola severa, voluta: un processo ancora attivo 2 secondi dopo il ritorno del suo
  comando viene chiuso e rende il contesto un errore, quindi nessuna lettura.
- Nella prova a secco il conteggio dei processi osservati dal genitore comprende i
  lettori della tabella lanciati dal figlio: non è un numero di worker.
