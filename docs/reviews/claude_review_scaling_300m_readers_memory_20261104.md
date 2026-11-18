# Review Claude — lettori scaling, proiezione S5, comparatore e controllo S3 (GPT 0d5f0f9, stato fino a 3cc3e09) — 4 novembre 2026

**VERDETTO: APPROVATO per questa consegna** (lettore della posa a due taglie e tre viste, sonde P1/P2, proiezioni S5,
comparatore, wrapper S3 e precisazioni S8). **Nessun blocco.** Cinque facoltativi, tutti non bloccanti.
La review **non attesta** la memoria reale di produzione (S3 `63527446` non ancora eseguito al momento della lettura) né la
readiness del training scientifico a 5.000: restano i gate operativi che GPT stesso dichiara.

## 0. Perimetro e metodo

- Richiesta di Simone del 04/11/2026 con il prompt di GPT `docs/prompt_claude_review_scaling_readers_memory_20261104.md`
  (commit `ec2f093`): review del codice `0d5f0f90074bbaf1142be86d22490e360b632c63` su `origin/codex/scaling-300m-20261104`,
  stato operativo pubblicato fino a `3cc3e091255154b129e478069d62ae65cd1de884`. La review offline `5e4eb8a` resta acquisita;
  qui si guarda il diff `ab2eb4a..0d5f0f9` (27 file, +3.807/−33; codice e test 14 file) e lo stato successivo senza confonderlo col perimetro.
- Solo review: checkout detached temporaneo di `3cc3e09` nello scratchpad (rimosso alla fine), worktree di GPT non toccato,
  nessun job, nessuna allocazione, nessuna modifica al codice. Su Leonardo solo comandi di lettura (`sacct`, `squeue`,
  `sinfo`, `ls`, `git rev-parse`/`status` nel checkout di campagna, lettura di JSON/JSONL del run storico).
- Metodo: lettura del codice nuovo (`scaling_projection.py`, `scaling_pose_readout.py`, `scaling_probe.py`,
  `scaling_compare.py`, `scaling_production_memory.py`, diff di `scaling_train.py`, `scaling_production_memory.sbatch`) e
  delle primitive storiche riusate (`tokenizer_pose_readout.py`, `tokenizer_probe.py`, `pose_temporal_readout.py`,
  `downstream_ab.py`, `spectral_step1.pool_p1p2`, `training/run.py`, `train_jepa.py`, `lib_code_dir.sh`); confronto dello
  schema atteso dal verificatore S3 con il produttore reale **e con gli artefatti reali** del run storico T100-G su Leonardo;
  rilettura dei sei file di test nuovi e del diff di `test_scaling_train.py`; esecuzione dei test; verifica delle ricevute.

## 1. Risposte alle sei domande del prompt

### 1.1 Posa (`scripts/scaling_pose_readout.py`) — conforme

- **Supporti, crop e LayerNorm invariati.** Le viste naturali chiamano le stesse funzioni del lettore storico:
  `Q.T.normalized_inputs` (zero sul padding reale, LayerNorm FP32 stateless, `eps` storico), `OFFSET` 6, `central_mask`,
  `E.sufficient`/`merge_stats`/`cross_validate`/`prepare_solver`, `Q.summarize`. `normalized_inputs` è indipendente dalla
  larghezza (LN sull'ultimo asse), quindi D896 passa per lo stesso percorso di D384.
- **Selezione alpha solo sui 40 utenti train.** `fit_reader` (l.175–188): statistiche per utente, due fold alternati sui
  40 utenti ordinati (controllo `folds == expected_folds` dal piano), `cross_validate` sulle sole statistiche train, refit
  sull'unione; `boundary_unresolved` blocca il report. **Separazione dai 15 val:** `run` (l.277) richiede utenti train e val
  disgiunti prima di aprire la val; `main` (l.323–325) lo richiede anche sui riferimenti; la val viene letta solo dopo il fit
  (test `test_runner_fits_train_before_reading_val_and_writes_all_three_cells`).
- **Riuso del controllo 30M.** `CELLS` ha una sola cella 30M (naturale); il 30M «matched» non esiste come fit (l.148: per
  `ARMS[0]` la vista matched restituisce gli input naturali; test `test_30m_matched_control_is_exactly_unchanged…`). Il
  checkpoint 30M è vincolato a `reference_checkpoint_sha256` del contratto (l.318). La provenienza storica (`legacy`,
  protocollo della valutazione tokenizer, arm `T100-G`) è accettata esplicitamente (l.102–113) e ho verificato che il
  `model_config` dei receipt storici della posa contiene tutte le chiavi richieste da `V.model_recipe` (dim, n_heads,
  local/backbone/decoder_layers, k_latents, patch_ms, window_ms, attivazione, pose_head, grad_checkpoint): il riuso del
  cache storico 30M è davvero possibile, non solo previsto.
- **Bootstrap appaiato.** `paired_comparison` (l.214–266): stesse `draws` (seed 20261102, 2.000 estrazioni) per le tre
  celle e i tre contrasti dentro ciascun gruppo, `support_sha256` e `targets_sha256` uguali a quelli della cella 30M,
  conteggi e denominatore per utente identici; `delta_ratio`, IC percentile 2,5/97,5 e `relative_error_reduction` con la
  stessa formula storica; `signal_2percent_and_ci_below_zero` coerente con la policy S8. Il test confronta le estrazioni
  e l'IC del contrasto primario con l'oracolo diretto e con `tokenizer_pose_readout.paired_comparison`.
- **Regressione D384.** `test_natural_fit_and_predictions_match_frozen_reader_at_both_widths`: pesi, bias, selezione,
  `fit_support_sha256` e predizioni **uguali bit a bit** a quelli del lettore storico, a 384 e a 896. Le viste naturali non
  hanno un percorso numerico proprio.
- **Vista proiettata.** `normalized_inputs` (l.150–156) azzera il padding reale, converte in FP32, applica la Q del
  backbone (896→384) e **poi** la LayerNorm; il test verifica l'uguaglianza con la LN applicata alla cache già proiettata e
  che l'ordine inverso dia numeri diversi. Dimensione 384, `n_features` 384 nel report, `projection_manifest_sha256`
  obbligatorio solo per la cella matched.

### 1.2 Classificazione (`scripts/scaling_probe.py`) — conforme

- **Scaler e C nel train del fold.** `fit_fold` è *lo stesso oggetto* di `tokenizer_probe.fit_fold` (`M.fit_fold is
  M.T.fit_fold`, testato): `StandardScaler` stimato solo sul train del fold, griglia C storica (1e-3…100), `max_iter`
  12.000, scelta su `val_bacc` con pareggio al C minore, refit sul solo train, predizioni held-out. `validate_fold`
  (l.361–401) rideriva la scelta dalla griglia e confronta `held_bacc` con l'OOF.
- **Split e OOF storici.** `CP.fold_roles`, `P1.keep_train_classes`, `validate_oof`, `scored` (balanced accuracy di
  sklearn) sono le primitive di sempre; per P1 i soggetti test ufficiali devono essere esclusi (`validate_plan`, l.92).
- **Etichette testuali EPN.** `labels()` accetta dtype `U/S`; gli OOF salvano/rileggono array unicode con
  `allow_pickle=False`; `subject_counts` e `balanced_accuracy_score` lavorano su stringhe. I test usano `noGesture`/`fist`
  da `cell_features` fino ad `aggregate` e `compare_arrays`.
- **Convergenza.** Ogni fit registra `n_iter`, `reached_limit`, `convergence_warnings`, `converged_by_criterion`;
  `validate_fold` controlla la coerenza interna; `aggregate` scrive `reading_status` `solver_unresolved` se un solo fit
  non converge; il comparatore porta il flag in `quality.solver_converged` e la lettura S8 restituisce
  `repair_evaluation_no_scientific_reading` (test `test_unresolved_probe_keeps_descriptive_metrics_but_disables_reading`).
- **Benchmark senza selezione.** `benchmark` (l.176–231) fitta i sei C sul solo train del fold 0 con `selected: None`,
  nessuno score di validazione (il test rende `predict` proibito); `validate_resources` richiede che il benchmark sia
  identico per solver/supporto e che l'allocazione pubblicata copra `1,5 × (Σ tempi + max) × Σ max(1, n_f/n_0) + 120 s` e
  `1,3 ×` il picco RSS.

### 1.3 Proiezioni S5 (`scripts/scaling_projection.py`) — conformi al protocollo

- Generazione FP64 con `Generator(PCG64(seed))`, semi 2026110401/2026110402, QR ridotta con diagonale di R resa positiva e
  segni delle colonne di Q coerenti, salvataggio FP32 C-contiguo, nessuna riscalatura; nessun ingresso dati e nessun seme
  configurabile (`generate()` senza parametri; il test verifica anche che il RNG globale non venga consumato). Manifest con
  algoritmo, semi, versioni, SHA dei file e dei byte degli array, hash del sorgente generatore; `load` rifiuta seme,
  algoritmo, sorgente, byte, scala o scambio dei file diversi, e verifica l'ortonormalità.
- **Ordine dei blocchi verificato nel produttore.** `spectral_step1.pool_p1p2` restituisce `torch.cat([z, local_pooled])`:
  prima il backbone (media su latenti e tempo), poi il locale. `project_pool` applica la Q backbone a `[:, :896]` e la Q
  locale a `[:, 896:]`, come dichiara `pool_block_order`. La stessa Q backbone serve la posa (`project_pose`).
- **Prima di LN e scaler, senza contaminazione.** Posa: proiezione prima della LayerNorm (1.1). Classificazione:
  `cell_features` proietta e solo dopo `fit_fold` stima lo scaler sul train del fold. Nessuna matrice dipende dai dati.
- **Non sostituisce i primari.** `scaling_compare.compose_summary` legge per EPN/P1/P2/posa il solo contrasto
  `natural_300m_minus_30m`; i contrasti matched stanno sotto `secondary`; `scaling_reading.read_comparison` non li vede.
  Il manifest pubblicato in `results/infra/scaling_300m_20261104/projection_s5/` ha SHA
  `74d1a0f2…9b3d` e il suo `generator_source_sha256` coincide con l'hash di `scaling_projection.py` registrato in
  `readers_memory_verification.json` (`a8c0f788…e88d`).

### 1.4 Comparatore (`scripts/scaling_compare.py`) — conforme

- **Provenienza.** I piani di sonde e posa devono condividere proiezione e contratto (l.67–69); i checkpoint per braccio
  devono coincidere fra posa e classificazione e il 30M deve essere il riferimento del contratto (l.72–77); sei report
  aggregati distinti (3 celle × 2 task); codice e contesto numerico coerenti (`runtime_coherence`).
- **OOF e shard.** Per ogni aggregato si rileggono gli shard sorgente, si confrontano gli audit (`shard_records ==
  records`) e gli array OOF riga per riga (l.189–206); cache, receipt, provenance, split e riferimenti vengono rihashati.
- **Predizioni e coefficienti.** Coefficienti ricaricati (forma `(dim, 20)`, finiti, hash uguale a `coefficients_sha256`),
  predizioni ricaricate (FP64, finite anche sul padding), `analysis` e `paired` **ricalcolati** e confrontati col report
  (l.159–174).
- **Intervalli e delta a precisione piena.** `compare_arrays` ricalcola punti e IC dagli OOF; la posa usa `cell_ratio` e
  `delta_ratio` del bootstrap appaiato e richiede `candidate − control` coincidente con `delta_ratio` (rtol 1e-10);
  `delta_E_dyn` è scritto come differenza dei punti; `read_comparison` ripete il controllo. Il mio F-d è recepito.
- **Niente raccomandazioni su artefatti corrotti o convergenza irrisolta.** Ogni incoerenza alza `ValueError` prima della
  scrittura (test: il file di uscita non esiste); con `solver_converged` falso la lettura S8 è disabilitata.

### 1.5 Collaudo S3 (`scripts/scaling_production_memory.py`, `.sbatch`, diff di `scaling_train.py`) — conforme

- **I 500 passi esercitano davvero il percorso.** Le cadenze del config T100-G (ereditate dal 300M) sono tutte 500:
  `eval_every`, `ckpt_every`, `pose_grad_every`, `fixed_eval_every`, `pose_panel_every`; `prefetch` True, batch 32,
  `accum_steps` 1. Al passo 500 il trainer emette insieme erank/collasso (`step % eval_every == 0`), `shared_grad_*`
  (`pose_grad_every`), valutazione fissa e pannello della posa, con AdamW e prefetch residenti dal passo 1. Un collaudo di
  64 o 128 passi non li toccherebbe: `inspect_evidence` richiede metriche contigue 1…500 con `windows` 32, le sette
  diagnostiche finite alla riga 500, `fixed_eval` esattamente a [0, 500] con 8 batch, pannello a [0, 500] sul supporto
  completo (256 finestre, 32 utenti), `stopped == "max_steps"`.
- **Schema dei log verificato contro il produttore reale e contro artefatti reali.** Oltre alla lettura di `run.py`
  (`_fixed_eval_set` l.879, `_pose_panel_set` l.959, `_pose_panel_errors` l.1044, `panel_sha256` l.1490,
  `_merge_records`, chiavi del summary l.1715, `_environment` l.179), ho letto in sola lettura gli artefatti del run
  storico T100-G in `$SCRATCH/wearusfm_runs/runs/tokenizer_20261103/T100-G`: `metrics.jsonl` riga 500 con `total`,
  `pose`, `grad_norm`, `lr`, `t_step_s`, `shared_grad_pose/anchors/cos`, `student/teacher_erank`,
  `student/teacher_collapse_ratio` tutti float e `windows` sempre 32; `fixed_eval.jsonl` ai passi 0, 500, 1000… con
  `batches` 8 e `total`; `fixed_eval_set.json` con `seed [0, 5151]`, **256 finestre piatte di 4 valori** e 8 hash;
  `pose_panel_set.json` con 32 utenti, 64 recording, 64 hash, 256 finestre e 256 `window_users`; `pose_panel.jsonl` con
  `panel_sha256` uguale all'hash canonico del set, `n_windows == n_windows_valid == 256`, `n_users` 32 e `per_window`
  allineato al set (`window == w[1:]`); `summary.json` con `batch_per_process`, `micro_batch`, `prefetch`, `accum_steps`,
  `world_size`, `parameters`, `gpu_max_memory_gib`, `elapsed_s`, `code.sha`, `pose_stats_sha256`, `environment`
  (`python` 3.11.6, `torch` 2.2.0a0+git6c8c5ad, `cuda` 12.1, niente cuDNN). Tutto coincide con le attese del verificatore.
  Il `config.json` reale del run storico (19.618 passi) coincide, tranne `max_steps`, con il pin `config_30m` del contratto
  (`tokenizer_review_20261103/config_T100-G.json`, 5.000 passi), e `check_source_contract` (eseguito dai test) impone che il config
  costruito dagli argv del contratto sia uguale a quel pin; `out_dir` e `time_limit_s` non entrano in `config_to_dict`: il confronto
  `config == expected` del verificatore (stesso meccanismo, `max_steps` 500) terrà.
- **Quattro misure reali e riconciliazione.** Ogni worker azzera i picchi, chiama `train_jepa.main` invariato (ritorna 0),
  sincronizza e scrive `peak_allocated`, `peak_reserved`, `device_memory` del proprio rank; il gate richiede quattro file
  distinti, `returncode` 0, picco ≤ 90 % della memoria rilevata e `max(peaks)` entro 1 MiB da `gpu_max_memory_gib` del
  summary, che il trainer riduce con MAX fra i rank in FP32 (risoluzione ≈ 2 KB a 28 GiB: la tolleranza è adeguata).
- **Runtime storico.** `runtime_comparison` confronta python/torch/cuda con `environment` del summary storico (stringhe
  presenti) e registra cuDNN senza dedurlo; eseguito nel supervisore prima di creare output e in ogni worker.
- **Output freschi, report per ultimo.** `execute` richiede `run_dir` e `receipt_dir` inesistenti e fuori dal codice,
  scrive `TECHNICAL_ONLY.json` prima del trainer (il trainer accetta la cartella: `mkdir(exist_ok=True)`, nessun
  checkpoint né STOP), `report.json` solo dopo `inspect_evidence` e `validate_report`.
- **Trainer e 103 sorgenti intatti.** `git diff ab2eb4a 0d5f0f9` su `src/`, `train_jepa.py`, `tokenizer_*.py`,
  `scaling_preflight.py`, `scaling_evaluation.py`, `scaling_reading.py`, `lib_code_dir.sh`, `tokenizer_train.sbatch`:
  vuoto. Il contratto (103 hash) è ricontrollato in `load_plan`.
- **Checkpoint tecnico escluso dal seguito.** A 500 passi il trainer scrive solo `checkpoint.pt` (`keep_checkpoints`
  = [5000, 19618], nessun decay); `inspect_evidence` rifiuta `checkpoint_step*`/`*decay*`/STOP; `recipe()` vieta
  `--init-from/--branch-from/--resume`; il wrapper dei 5.000 richiede output nuovi: il checkpoint S3 non può entrare nel
  percorso registrato.
- **Tetto coerente.** 1 nodo × 4 A100 × 60 min = 4 GPU-ore, 32 core × 1 h = 32 core-ore; checkout 2 CPU × 10 min =
  0,3333 core-ore. Formula `ceil_900(500 × 3,035413 × 1,20 + 1200) = 3600 s` dal collaudo reale (`seconds_per_step` 3,035413 in
  `production_memory_resource_policy.json` = 3883,03/5000 × 3,9086: il proxy storico domina la media misurata degli update 300M). Sotto le 5 GPU-ore dell'autonomia di Simone.
  Limite operativo del trainer 3.300 s contro ≈ 1.520 s di update più avvio e diagnostiche: margine sufficiente.
- **Launcher.** `scaling_production_memory.sbatch`: 1 nodo, `--gres=gpu:4`, 32 CPU, 480G, 01:00:00; `code_dir_setup`
  verifica SHA, pulizia e sola lettura del checkout, fa `cd`, mette `src/` del checkout in testa a `PYTHONPATH` ed esporta
  `WEARUSFM_CODE_SHA`; il piano è obbligatoriamente relativo al checkout e il checkout deve stare in `wearusfm_campaigns`.
  `scaling_train.verify_allocation` legge `plan["allocation"]["walltime_seconds"]` (3600 = `01:00:00`) e `AllocTRES`, formato già
  verificato su SLURM 25.05.7 nella review precedente.
- **Stato osservato (sola lettura).** Checkout `63527364` COMPLETED 0:0 in 21 s (2 CPU); checkout di campagna
  `scaling_300m_memory_20261104_v1` a `0d5f0f9`, pulito. S3 `63527446` **PENDING (Priority)**, nessun output in
  `wearusfm_runs/results/scaling_300m_20261104/production_memory_v1` né nella cartella scratch del run. Niente da leggere:
  nessuna memoria reale è attestata da questa review.

### 1.6 Precisazioni S8 — recepite senza spostare soglie né endpoint

Il diff del protocollo aggiunge due paragrafi: la review della pausa pesa entità e dose «senza cambiare la riga
applicata dalla tabella e senza introdurre una nuova soglia»; la tranche a 19.618 risponde alla dose e non misura la
variabilità fra training; nessuna probabilità numerica di falsa pausa. `scaling_reading.py` (hash `13231e82…` in
`readers_memory_verification.json`) e `scientific_reading_policy.json` non compaiono nel diff. I miei tre punti di `5e4eb8a` sono così trattati: (1) l'asimmetria
migliore/peggiore resta nel codice, ma la riga di pausa si chiama esplicitamente «review before 19618» e il protocollo
chiede ora di pesare entità e dose in quella review: accettabile, purché la review sia registrata prima di decidere;
(2) la molteplicità è dichiarata senza inventare un numero: corretto, il mio ~10 % era un ordine di grandezza
illustrativo, non una stima; (3) la tranche non misura la variabilità fra run: dichiarato. Implementazione approvabile,
memoria da misurare e gate operativi sono distinti sia nel documento sia in `readers_memory_verification.json`
(`production_memory_measured: false`, `scientific_5000_ready: false`, elenco `remaining`).

### 1.7 La regressione sul produttore reale intercetta il difetto

`test_fixed_set_real_producer_flat_schema_is_accepted` chiama il vero `R._fixed_eval_set` con un loader finto da 8 batch
× 32 finestre e pretende che il verificatore accetti le 256 righe piatte prodotte; con il vecchio verificatore (8 liste da
32) questo test fallirebbe, e `test_old_nested_fixed_set_schema_is_rejected` fissa il rifiuto dello schema annidato. La
lettura degli artefatti reali (1.5) conferma che il produttore scrive 256 righe piatte di 4 valori.

## 2. Esito

### 2.1 Blocchi

Nessuno.

### 2.2 Facoltativi (non bloccanti; a discrezione di GPT)

- **F-1 — Il comparatore confronta anche inode/mtime/ctime delle cache.** `scaling_compare.validate_pose_outputs`
  l.139–143 ricalcola `validate_artifact` e richiede `report["inspections"] == inspections`; ogni inspection contiene
  `states` (`[st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns]`, `scaling_pose_readout.py` l.141). Un `touch` (per
  esempio il rinfresco periodico dello scratch), una copia o un ripristino delle cache fra la fase `ridge` e il confronto
  fa rifiutare il confronto benché gli hash coincidano. Impatto: falso rifiuto, nessun errore scientifico. Correzione
  minima: confrontare le inspection senza la chiave `states` (gli hash sono già imposti da `_pin`).
- **F-2 — Uguaglianza esatta di `analysis` e `paired` ricalcolati.** `scaling_compare.py` l.169–174: le somme di
  `summarize` e i percentili del bootstrap devono coincidere bit a bit con quelli del lettore; `runtime_coherence` vincola
  posa e sonde fra loro ma non l'ambiente del comparatore. Eseguire il comparatore sullo stesso tipo di nodo e con lo
  stesso numpy dei lettori (nodo di calcolo, non login), altrimenti un ulp di differenza nelle riduzioni SIMD produce un
  rifiuto spurio. Nessuna modifica necessaria se il piano operativo lo prescrive.
- **F-3 — Le evidenze S3 e la readiness sono legate ai byte vivi del checkout di training.**
  `scaling_production_memory.validate_report` l.297–299 rihasha `scripts/scaling_train.py`, `scaling_preflight.py` e
  `scaling_production_memory.py` **del checkout corrente** contro i pin del piano S3; `scaling_train.check_evaluation_readiness`
  l.271 fa lo stesso per i sei valutatori. Qualunque modifica a quei file prima del lancio dei 5.000 (anche F-g sullo
  `scontrol --json`, o una correzione nei lettori) invalida l'evidenza S3 (4 GPU-ore) o la readiness. Va bene come scelta
  conservativa, ma deve essere esplicita nel piano: da ora quei nove file sono congelati fino al lancio, oppure si
  accetta di rieseguire S3. Alternativa minima: confrontare i pin del piano con gli hash registrati nel report S3 al momento
  della produzione, non con i file vivi.
- **F-4 — Il manifest S5 è legato ai byte del generatore.** `scaling_projection.py` l.98 e l.134:
  `generator_source_sha256 == sha(__file__)`. Qualunque modifica al file, anche a un commento, obbliga a rigenerare il
  manifest (stessi numeri, nuovo hash) e a rifare i piani che lo pinnano. Coerente con F-3; basta saperlo.
- **F-5 — Doppia lettura delle cache 300M nella fase `ridge`.** `scaling_pose_readout.run` legge train e val del braccio
  300M per la cella naturale e di nuovo per la cella matched (val D896 ≈ 5,3 GB FP16). Riusare le letture per le due viste
  dimezza I/O e tempo della fase; opzionale, da valutare nel preventivo CPU.

### 2.3 Lavoro futuro già dichiarato (non difetti)

Risultato reale S3 con verifica delle quattro misure; piani/launcher operativi di estrazione, lettura e confronto con
risorse misurate (i benchmark delle sonde e della posa precedono i fit); verifica del controllo 30M e della retention;
review della catena legata ai pin finali; formula dei 19.618 già fissata (22 GPU-ore di tetto, non una sottomissione).

## 3. Verifiche eseguite

- **Test mirati** (comando del prompt, checkout `3cc3e09`, `PYTHONPATH=src:tests/cpu`, un thread): **425 passati in 35,43 s**
  (log `~/wearusfm_local/logs/scaling_readers_tests13_3cc3e09.txt`). Suite intera `tests/cpu` sullo stesso checkout: **2.456 passati, 1 fallito in 25 min 34 s**; l'unico fallimento è quello
  preesistente di `test_pose_b19618_launcher.py` (gia' registrato il 04/11), estraneo alla consegna (log
  `~/wearusfm_local/logs/scaling_readers_tests_full_3cc3e09.txt`).
- **Produttore reale.** `run.py`: cadenze, `keep_checkpoints`, `_merge_records` (somma di `windows`), riduzione MAX della
  memoria, `_environment`, hash canonico del pannello; `train_jepa.main` ritorna `int`; `config.json` del run storico uguale
  al config degli argv (`config_to_dict`, senza `out_dir`/`time_limit_s`); `pool_p1p2` concatena backbone poi locale.
- **Artefatti reali su Leonardo** (sola lettura): run storico T100-G come in 1.5; summary storico con `environment`
  completo; checkout di campagna `scaling_300m_memory_20261104_v1` a `0d5f0f9` e pulito; `sacct` di `63527364`/`63527446`;
  `squeue`; assenza di output S3.
- **Identità.** Diff dei sorgenti congelati vuoto fra `ab2eb4a` e `0d5f0f9`; hash in `readers_memory_verification.json`
  coerenti con il manifest S5; piano S3 (`production_memory_plan.json`) coerente con il launcher e con la ricevuta del
  submitter (`jobs/SCALING-300M-memory-20261104/state.json`: `--export=ALL,CODE_DIR=…,EXPECTED_SHA=0d5f0f9…`, piano relativo).

## 4. Limiti

Non ho eseguito lettori, sonde o S3 su dati reali: la regressione D384/D896 è dimostrata per identità di codice e per test
sintetici bit a bit, lo schema S3 per confronto col produttore e con gli artefatti storici. La memoria reale a 500 passi
con prefetch, pannelli e AdamW residenti la dirà solo `63527446`. Nessuna attestazione di readiness dei 5.000.
