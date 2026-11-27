# d2 — Codice che girerà sul cluster (passi 4, 5, 6)

Revisione del 27/11/2026, verso le 15:40. Fonti: worktree di release `b58a66f6` (pulito, verificato con `git status --porcelain`), sequenza
`docs/contrastive_cn_restart_probe_launch_20261127.md` (worktree claude-training-engine), bozza del brief, letture di Leonardo del 27/11, ledger comune.

## Sintesi

Il codice dei passi 4-6 è coerente con la sequenza. Tutti i commit che il codice legge con git sul cluster sono antenati del commit
di release. **C'è però un fatto nuovo che blocca il lancio così come è scritto.** Alle 15:33:41-42 il ledger comune vero ha ricevuto
una prenotazione per `CONTRAST-CHECKOUT-CN-RESTART545-20261121`, con stato `ambiguo` e stdout `finto-stdout`. La scrittura viene dalla prova
del verificatore d1 con un ssh finto, fatta senza un registro separato. Con il ledger in questo stato il passo 1 non dà 0 e il passo 4
viene rifiutato dal submitter.

## Rilievi

### B-1 (bloccante) Il ledger comune vero contiene ora l'ID del checkout in stato `ambiguo`
- Dove: sequenza righe 38 e 42-43 («nessuna voce nel ledger per i due ID», «Il ledger vero non è stato toccato»), passi 1 e 4; brief, punto 1.
- Prova: `grep -c "CN-RESTART545" ~/wearusfm_local/jobs/ledger.tsv` restituisce `2`. Le due righe sono `prenotato` (15:33:41) e `ambiguo`
  (15:33:42), con `codice_sha` b58a66f6. Il file `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json` contiene
  `"stato": "ambiguo"`, `"stdout": "finto-stdout\n"` e `"stderr": "finto-stderr\n"`. Lo ssh finto
  `agents2/d1-comandi/finto_ssh_bin/ssh` è stato creato alle 15:33:41 e stampa esattamente quelle stringhe. Per lo stato `ambiguo`,
  `Registry.reserve` (`scripts/submit_job.py`, righe 153-161) esce con «Non si risottomette: usare --riconcilia». L'ID del job GPU non è stato toccato.
- Correzione: non consegnare finché Simone non decide come ripulire. Le due strade possibili sono queste.
  1. Seguire la procedura del submitter: `--riconcilia` due volte, la prima almeno 10 minuti dopo le 15:33:41 e la seconda almeno 10 minuti
     dopo la prima, fino allo stato `assente`. Poi il passo 4 va aggiornato con `--dopo-assenza-verificata`.
  2. Fare una pulizia registrata da Simone.

  In entrambi i casi vanno corretti il passo 1 (il conteggio atteso non sarà 0), le righe 38 e 42-43 della sequenza e il punto 1 del brief.

### I-1 (importante) Il validatore non controlla i file storici su WORK che il confronto legge
- Dove: passo 5 e sezione «Che cosa non si può verificare prima» della sequenza.
- Prova: `contrastive_cn_restart_compare.py` legge i file delle cartelle storiche su WORK e li vuole identici, byte per byte, alle copie
  archiviate nel repository:
  - `history()`, righe 190-203: `checkpoint.pt` e `checkpoint_step0000544.pt`, coperti dal validatore;
  - `branch_traces()`, righe 205-223: `trace-{0..3}.jsonl` e `numeric_profile.rank{0..3}.json` di `reloads/CN_A` e di `benchmark/CN`, contro
    `ARCHIVE` (`reloads_execution_20261121/CN_A` e `benchmark_execution_20261121/CN`);
  - riga 264: `metrics.jsonl`;
  - riga 356: `trace-{0..3}.ddp.jsonl` di `benchmark/CN`.

  `commands()` (`contrastive_cn_restart.py`, righe 105-137) controlla solo le impronte dei tre checkpoint. Le letture del 27/11 e del 21/11
  hanno provato checkpoint, `config`, `summary` e `start`, non tracce, profili e metriche. Se uno di questi file manca o è diverso,
  `compare` cattura `ValueError`/`OSError` e scrive `INCONCLUSIVE`: il job finisce `COMPLETED 0:0` dopo aver speso l'unico tentativo GPU.
- Correzione: aggiungere al passo 3 o al passo 5 un confronto in sola lettura degli sha256 di quei file, fra
  `$WORK/wearusfm_runs/results/contrastive_targets_20261118/{reloads/CN_A,benchmark/CN}` e le copie in
  `results/infra/contrastive_targets_20261118/{reloads_execution_20261121/CN_A,benchmark_execution_20261121/CN}` del commit di release.
  Se il confronto non si aggiunge, la sequenza deve almeno dire che questa verifica si poteva fare e non è stata fatta.

### M-1 (minore) Gli esiti del job sono descritti in modo approssimato
- Dove: sequenza, righe 161-167.
- Prova: `run()` (righe 166-185) chiude ogni fase 25 secondi prima della fine dell'allocazione, entro 875 secondi dall'avvio. Una fase
  troppo lunga quindi non dà `TIMEOUT`. Dà `FAILED` con `execution_failure.json`, che porta la causa `TimeoutExpired` o `TimeoutError`.
  `TIMEOUT` resta possibile, anche con `execution_failure.json` già scritto, se la chiusura delle fasi (fino a 20 secondi, righe 35-48 di
  `contrastive_split_benchmark.py`) sfora. Un errore dopo `root.mkdir` e prima delle fasi lascia la cartella di uscita con al massimo
  `allocation.json` e senza `execution_failure.json`: per esempio un errore di `nvidia-smi` o di `allocation.json`, alle righe 158-164.
  Le guardie `[[ ... ]]` di `cn_restart_probe.sbatch` (righe 16, 17 e 21), sotto `set -e`, escono con 1 senza alcun messaggio nei log.
- Correzione: aggiungere una o due frasi.
  - Una fase interrotta per tempo compare come `FAILED` con `execution_failure.json`.
  - Un `FAILED` con i log vuoti, o con la sola riga `codice: …`, viene da una guardia dello script SLURM.
  - Una cartella di uscita con solo `allocation.json` viene da un errore prima delle fasi.

### M-2 (minore) Il passo 6 non ripete la cartella da cui lanciare
- Dove: sequenza, passo 6 (righe 149-156).
- Prova: `--script-stdin scripts/slurm/cn_restart_probe.sbatch` è un percorso relativo. Il file manca sia nel main locale (`c47e6557`)
  sia nel worktree claude-training-engine: `ls` risponde «No such file». Da un'altra cartella il submitter esce prima della prenotazione
  (`read_text`, riga 219, prima di `reserve`, riga 222): l'ID non si brucia, ma il passo fallisce.
- Correzione: ripetere al passo 6 «Dalla radice del worktree di release», come al passo 4, oppure mettere un `cd` esplicito.

### N-1 (nota) La validazione sul login fa più letture di quelle dichiarate
- Dove: sequenza, righe 140-142.
- Prova: oltre ai tre checkpoint di `commands()`, `contrastive_run.py --validate-only` fa altre letture.
  - `validate_target_fork` e `validate_fork` calcolano altre due volte l'impronta del fork, lo aprono con `mmap` e calcolano l'impronta
    dell'intero stato.
  - `validate_cuda_prerequisite` calcola l'impronta di `calibration/checkpoint.pt`.
  - Si leggono anche `audit.json` (20 MB), il manifest e la ricevuta del rank-exit.

  Il totale è di qualche GB di letture e qualche decina di secondi di CPU, comunque molto sotto i 10 minuti.

  Lo stdout conterrà anche, in quest'ordine:
  - la riga `codice: … @ b58a66f6…` e le due date;
  - il JSON del runner, che contiene anch'esso `"execution_started": false`;
  - per ultimo il JSON del driver, con `new_binding_valid`.
- Correzione: facoltativa. Precisare che il JSON da guardare è l'ultimo.

### N-2 (nota) Il confronto con WORK usa il percorso risolto da una parte sola
- Dove: `contrastive_cn_restart.py`, righe 112-115.
- Prova: `root.parent` viene dal piano e non è risolto; viene confrontato con `Path(os.environ["WORK"]).resolve()`. Se `$WORK` fosse un
  link simbolico, il passo 5 fallirebbe. L'errore arriverebbe prima del job GPU, quindi senza bruciare l'ID. `contrastive_job.py`
  (righe 82-86) usa la stessa risoluzione e i job storici sono passati, quindi il caso è improbabile. Non ho una lettura di `realpath $WORK`.
- Correzione: nessuna. Al più, aggiungere `readlink -f "$WORK"` alle letture del passo 3.

### N-3 (nota) La presenza del fork su WORK non è stata letta il 27/11
- Dove: letture del 27/11 (`stato_pre_lancio.sh`, riga 9).
- Prova: lo script ha letto `fork/CN/checkpoint.pt`, che non esiste («No such file»), e non il fork del piano
  (`benchmark/CN/checkpoint_step0000544.pt`). Il fork resta comunque controllato dal passo 5 con `checked_pin(plan["fork"])`.
- Correzione: nessuna. Al passo 3 conviene leggere il percorso giusto.

### N-4 (nota) `grep -c` esce con 1 quando conta zero righe
- Dove: sequenza, passo 1, riga 72.
- Prova: con zero corrispondenze `grep -c` stampa 0 ed esce con 1. Un operatore che guarda il codice d'uscita può leggerlo come un errore.
- Correzione: scrivere «stampa 0; l'uscita 1 di grep è normale». La riga va comunque rivista per B-1.

## Verifiche positive fatte davvero

- **Commit antenati.** Ho eseguito `git merge-base --is-ancestor <commit> HEAD` nel worktree di release su ogni commit che il codice
  legge con git sul cluster:
  - `f23b7cca` (`evidence_commit`, usato da `git diff --quiet` in `package()`): antenato;
  - `8c72655c` (`reviewed_commit` del blocco, usato da `git show` in `review_fields()`): antenato;
  - `fbfde797` (`reviewed_commit` della review storica, usato da `git show` in `verify_reviewed_sources()` e nei due piani di
    `validate_review_binding`): antenato;
  - `8dad7e82` (il main del cluster): antenato, quindi il push deve trasferire solo gli oggetti mancanti.

  `469b312c` e `76f5d0e7` non sono antenati, ma il codice dei passi 5 e 6 non li legge con git: `bind_review` e `REVIEW_REF` non vengono
  eseguiti, e `validate_review_binding` non fa `git show` del `review_commit`.
- **Validatori eseguiti sul Mac.** `require_binding()`, `checked_pin(historical_job, repository=True)` e
  `validate_review_binding(job["package_review"])` con `verify_git=True` sono passati sul Mac, dal worktree di release, con
  `python -I -B`. Dopo l'esecuzione `git status` è risultato vuoto. Il job storico è CN, ripetizione `a`, con il fork del piano.
  La release storica è `APPROVED_FOR_PREPARATION`, `benchmark`, `resume_a`, `time_limit_s` 1770 e `world_size` 4.
- **Script inviati e submitter.** Gli sha256 degli script sono `eaafc432…` per il checkout e `a8c8b8b4…` per il job GPU. Coincidono
  con lo stdin della prova a secco, e lo script del checkout ha gli stessi byte nel main locale. Il submitter condiviso e quello del
  commit hanno lo stesso sha256 (`814da4bc…`). La riga di `sbatch` non ha argomenti dopo lo script, quindi la guardia `"$#" == 0` passa.
- **Allocazione.** `validate_allocation("gpu")` vuole 1 nodo, 1 task, 32 CPU e `SLURM_MEM_PER_NODE` = 245.760, cioè `--mem=240G`:
  sono i valori dello script e del comando. `TimeLimit=00:15:00` corrisponde a `--time=00:15:00`. `TORCH_DIST_INIT_BARRIER=1` è
  richiesto sia dallo script sia dal driver ed è nel comando. Serve anche a tenere fermi tutti i rank fino a `init_process_group`
  prima che il rank 0 crei la cartella della gamba: altrimenti il controllo «cartella fresca» di `validate_argv` e di
  `contrastive_run` sugli altri rank fallirebbe.
- **Scritture nel checkout.** In nessun percorso i passi 5 e 6 scrivono nel checkout in sola lettura: `PYTHONDONTWRITEBYTECODE=1`; le
  uscite vanno solo sotto `plan["output"]`; `git diff` e `git status` toccano solo l'indice in `$WORK/wearusfm/.git/worktrees/…`, che è
  scrivibile. Le operazioni git dei quattro rank in parallelo sono sicure: il refresh dell'indice salta il lock senza errori.
- **Ordine dei controlli preliminari.** Il preambolo SLURM `set -euo pipefail` → `code_dir_setup` → `scontrol` → `module` → `activate`
  ha lo stesso schema di `contrastive_preflight_gpu.sbatch`, già usato con successo per la linea 3.
- **Esito `INCONCLUSIVE`.** `compare` cattura le eccezioni previste, scrive sempre `report.json` ed esce con 0; ogni suo `return`
  contiene `reason`. Un esito `INCONCLUSIVE` dà quindi `COMPLETED 0:0`, come dice la sequenza.
- **Lettura dei vettori.** `torch.load` usa `str(...)` con `mmap` in tutti i lettori usati: observe, target_resume e benchmark_readout.
  È il formato che la lettura del 27/11 ha confermato sul torch 2.2 del cluster.

## Punti non verificati

- La parte GPU: l'hook NCCL, `libcuda`, i sette gruppi e i tempi reali, cioè la stima di 450-620 secondi con la validazione ripetuta su
  quattro rank per ciascuna gamba.
- L'uguaglianza fra i file storici su WORK e l'archivio del repository (I-1).
- `realpath $WORK` (N-2) e la presenza di `SLURM_JOB_END_TIME` sui nodi; lo script la ricava comunque da `scontrol`.
- Se nel job GPU il controllo `ps` di `validate_receipt` scatta: succede solo se il nodo è quello del probe rank-exit, e anche allora
  confronta PID e ora di avvio.
- Lo spazio effettivo che occuperanno le gambe: vettori (circa 1,83 GB) più le copie dei checkpoint. La quota è soft a 1 T, senza
  hard limit.
