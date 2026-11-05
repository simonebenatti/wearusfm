# T100-LQ: causa dell'arresto all'inizializzazione e correzione

Consegna di Claude del 05/11/2026 sul compito scritto da GPT e inoltrato da Simone
(«Ti affido diagnosi e correzione del blocco di inizializzazione T100-LQ»).
Ramo `claude/logsquare-init-fix-20261105`, partito da `4d5a3ef` di GPT.
Nessun job sottomesso, nessun training, nessun retry. Gli artefatti del fallimento,
il piano e la release del 04/11 non sono stati toccati.

## In breve

- L'hash atteso `38cb4cce…` descrive il modello costruito dal **preflight**. Il training
  storico di T100-L è partito da uno stato che differisce da quello per **un ULP del solo
  buffer `tokenizer.fe_gain`**. Il controllo del 04/11 chiedeva quindi a T100-LQ uno stato
  da cui nessun training T100 è mai partito.
- La correzione verifica lo stato **sul modello che ogni rank costruisce e addestra**, non
  su un modello temporaneo. Accetta un solo stato: i tensori del riferimento revisionato,
  il gain salvato dal training storico di T100-L, `W_log` nullo.
- Nessun sorgente scientifico cambia. Niente ripristini, tolleranze o hash nuovi.
- Resta una verifica sul cluster, già pronta: la diagnostica di GPT completata con la
  prova a secco della correzione. La sottomette GPT dopo la sua review.

## 1. Causa

### Dimostrato dai registri del repository, senza job

| Braccio | Preflight (03/11 e 04/11) | Checkpoint del training storico (5000 e 19618) | Scarto |
| --- | --- | --- | --- |
| T100-L | 548.6939187567023 `0x1.1258d2546f0f0p+9` | 548.6939187567021 `0x1.1258d2546f0efp+9` | 1 ULP |
| T100-G | 548.6939187567023 | 548.6939187567021 | 1 ULP |
| T25-L | 982.425066637557 `0x1.eb3668956576dp+9` | 982.4250666375573 `0x1.eb3668956576fp+9` | 2 ULP |

Fonti: `results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json`
(codice ef8016a), `results/infra/logsquare_execution_20261104/report.json.gz` (codice 43cad1f),
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
4. Il meccanismo è quello del precedente del 14/10 (job 60561493,
   `results/passo6/diag_initial_state_20261014.json`): sul cluster l'ordine degli import
   cambiava solo `tokenizer.fe_gain`. I due valori di allora per T25 sono esattamente
   quelli che oggi separano preflight e checkpoint di T25-L.

### Non ancora osservato: serve il cluster

- Il gain e i tensori del modello temporaneo nel job 63532335. Il vecchio messaggio non
  li stampava. Atteso: tensori del riferimento e gain del training.
- Che il trainer di oggi costruisca ancora il gain dei checkpoint storici.
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
- T100-LQ parte dallo stato da cui è partito T100-L, più `W_log` nullo. È il confronto
  approvato: T100-L più `log(0,001 + x²)`.
- Dietro l'ingresso il trainer addestra lo stesso stato. Con quattro rank reali su CPU,
  gruppo di processi e DDP compresi, due passi danno studente, teacher, optimizer e
  generatori identici bit per bit al trainer normale.

### Punto da decidere nella review

**Lo stato esatto del preflight ora viene rifiutato**, con classificazione
`preflight_reference_state`. È una scelta: nessun training T100 è partito da lì, e
accettarlo farebbe partire T100-LQ a un ULP dal confronto. Chiedo a GPT di confermare che
questa lettura rispetta «Non adottare un nuovo hash atteso»: l'hash registrato resta uno
solo, ma lo stato accettato è quello del training storico e non quello del preflight.

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
- **Conteggi.** I quattro file LQ: 154 test passati. Suite `tests/cpu` sul commit `b8689e1`: 2446 passati su 2447.
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
  trainer: niente teacher, optimizer, DDP, dati, aggiornamenti o checkpoint.
- La lettura è fissata nel piano prima dell'esecuzione, su confronti esatti:
  `cause_demonstrated`, `trainer_builds_the_registered_training_state`,
  `fix_dry_run_passed`. Un contesto fallito non dà lettura. Il rapporto non autorizza mai
  il training.

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
   dal commit che contiene questa consegna, sottomissione. Il piano fissa già il guard
   (SHA `9cc3c37e…`).
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
