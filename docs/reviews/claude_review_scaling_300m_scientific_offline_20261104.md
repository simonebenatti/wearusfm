# Review di Claude — preparazione offline dello screening scientifico 300M (GPT 5a7b384, prompt d2691c7) — 04/11/2026

**VERDETTO: APPROVATO per la consegna offline.** Nessun difetto del codice consegnato che richieda una correzione prima
dell'uso; i gate del training a 5.000 restano chiusi come dichiarato (lettori, S5, aggregatore, memoria di produzione,
benchmark, launcher, pin reali, review della catena). Sette miglioramenti facoltativi (F-a … F-g) e una valutazione critica
della regola S8 (§2.3) sono da decidere prima delle estrazioni, non blocchi. Nessun job, nessuna allocazione, nessuna
modifica ai checkout di GPT.

## 0) Perimetro e metodo

- Richiesta di Simone (04/11, sera), con il prompt di GPT `docs/prompt_claude_review_scaling_scientific_20261104.md`: codice
  `5a7b384659ec38a3c59797fc238a3d478316ebb3` su `origin/codex/scaling-300m-20261104` (base `6d80e8d`), prompt in `d2691c7`
  (il diff 5a7b384 → d2691c7 è il solo file del prompt, 40 righe). Checkout detached temporaneo di d2691c7 nello scratchpad,
  rimosso alla fine; worktree di GPT non toccato.
- Letti per intero: `docs/scaling_300m_offline_preparation_20261104.md`, `docs/scaling_300m_protocol_20261104.md` (e il suo
  diff), `scripts/scaling_train.py` (455 righe), `scripts/scaling_evaluation.py` (416), `scripts/scaling_reading.py` (112),
  `training_source_contract.json`, `scientific_reading_policy.json`, `offline_verification.json`, i quattro file di test nuovi o
  modificati, le tre review interne di GPT, le voci del registro e di `decisioni.md` della consegna. Riletti i riusi: validatori
  storici di `tokenizer_evaluation.py`, `tokenizer_train.py` (`checkpoint_identity`, `validate_optimizer`, `validate_rng`,
  `receipt_for`, `command_for`), `tokenizer_preflight.py` (`file_identity`, `pinned_file`, `code_identity`), `psd3_inputs._labels`,
  `fm_features.load_model`, il report del collaudo in `scaling_preflight.py` (05d4bf0) e il summary del trainer in `run.py`.
- Verifiche indipendenti (dettaglio in §3): identità dei 103 sorgenti rispetto ad `afb644e` e inventario completo; cinque
  primitive rispetto a `f2061b8` e `b315dad`; diff vuoto rispetto a 05d4bf0 sui percorsi del collaudo; nove hash di
  `offline_verification.json`; derivazione del comando 300M; compatibilità degli schemi (report del collaudo, summary del trainer,
  `scontrol`) con gli artefatti reali; controlli in sola lettura su Leonardo del controllo 30M; suite di test rieseguita.
- Il collaudo approvato resta congelato a 05d4bf0: confermato (diff vuoto su `src/wearusfm`, `train_jepa.py`,
  `scaling_preflight.py`, `scaling_preflight.sbatch`, `preflight_plan.json`, `resource_policy.json`).

## 1) Risposte ai punti della review

**1. Identità del confronto (`scaling_train.py`, `training_source_contract.json`).** Sì. Il contratto elenca 103 file: i 98 di
`src/wearusfm` (inventario a `afb644e` e a d2691c7: 98 file, nessuno fuori dal contratto in nessuna direzione) più
`train_jepa.py`, `tokenizer_train.py`, `tokenizer_preflight.py`, `tokenizer_train.sbatch`, `lib_code_dir.sh`. Tutti i 103 hash
coincidono con `git show afb644e:<path>` e con i byte del checkout; `git diff afb644e d2691c7` su `src/` e sui cinque script è
vuoto (in `scripts/` solo aggiunte: i quattro script scaling e tre launcher tokenizer). `check_source_contract` (l.84–145)
confronta l'inventario con i file reali su disco (`rglob` di `src/wearusfm`, esclusi i `.pyc`), non con un sottoinsieme
dichiarato; richiede preset `sanity`, `--distributed`, 6 worker, `max_steps` 5000, i due root, i quattro input assoluti e la
ricevuta storica (`afb644e`, checkpoint `49dd518b…`). `derive_recipe` (l.148–163) sostituisce solo `--preset`, `--out-dir`,
`--max-steps` e aggiunge `--time-limit-s`; verificato con un piano sintetico: la CLI derivata è l'argv validato del T100-G con
`--preset 300m`, stesso manifest/scales/windows/pose-root, `--seed 0`, WSD 49.046, warmup 1.962, batch 32, `--erank-min 38.4`,
`--keep-checkpoints 5000:19618`, più `--time-limit-s 21300`. `configurations(base)["T100-G-30M"]` è identica alla
configurazione base; la 300M differisce solo in `model.dim/n_heads/local_layers/backbone_layers` (`decoder_layers` resta 2):
quattro campi del preset LADDER, come dichiarato. I pin delle cinque primitive di estrazione (`EXTRACTION_SOURCE_SHA256`)
coincidono con `f2061b8` (cache di posa storica) e `b315dad` (classificazione finale) e con il checkout; il test
`test_extraction_primitive_bytes_match_historical_baseline…` li ricontrolla con `git show`.

**2. Validità delle valutazioni (`scaling_evaluation.py`).** Sì, senza aggirare le guardie storiche. `validate_checkpoint`
(l.179) rifiuta ogni dose diversa da 5000 (anche `5000.`), impone l'intero preset LADDER più GELU/100 ms/25 ms/64 slot/testa di
posa (`model_recipe`, l.67) e `max_steps == 5000`; `load_teacher_endpoint` (l.120) richiede il checkpoint nominato
`checkpoint_step0005000.pt`, assenza di STOP, config pinnata uguale alla ricetta completa, summary con `stopped == max_steps` e
`steps == 5000` (il summary mobile dello scratch, oggi a 19.618, è rifiutato per contenuto, non per percorso), ricevuta col
protocollo giusto per braccio (storica `tokenizer-training-20261103-v1` con `code_sha == afb644e`; nuova
`scaling-training-20261104-v1` legata a `training_source_contract_sha256` e all'intero dizionario dei sorgenti), stato completo
(step, config, world 4, stream 6/1200/4, pose_stats, data_signature, RNG dei 4 rank, AdamW, contatori, pesi finiti, stessa
struttura student/teacher), caricamento stretto del teacher con `fe_gain` uguale al buffer salvato e positivo, config del modello
caricato uguale. Posa: `verify_pose_cache` (l.186) è il validatore D384 storico con la larghezza esplicita per braccio
(`lat [n,160,dim] <f2`, `pred [n,160,20] <f4`, `dim` nel cache) e gli stessi controlli di target/maschere/std/pose_var, selezione,
ricetta (teacher, 10 finestre, 4 s, seed 20261026) e finitezza sul tempo reale; il test
`test_d384_pose_validator_regresses_exactly…` confronta l'intero dizionario restituito con quello del validatore storico.
Classificazione: `verify_classification_cache` (l.220) conserva righe, etichette, soggetti e ordine (`np.array_equal` con dtype
uguale), split ufficiali, selezione interna P2, pool `[n, 2·dim] <f4` finito, fold deterministici (`fold_roles` +
`validate_roles`); il test D896 verifica che `ordered_rows_sha256` e `fold_roles_sha256` siano uguali a quelli D384. Etichette
testuali EPN: `psd3_inputs._labels` (l.111–120) applica `isfinite` solo ai dtype numerici e, per `U/S`, richiede etichette non
vuote; il test `test_classification_rejects_corruption_without_isfinite_on_text` copre etichette invertite, NaN, dtype object,
larghezza e precisione. Il percorso di estrazione è dimensionale: `channel_readout.py` scrive `encoder_dim = model.cfg.dim` e il
pool è `2·dim` (`fm_features.py` l.122), `pose_eval.py` scrive `dim = model.cfg.dim` (l.143, 198); l'unico 384 cablato resta nel
lettore storico `tokenizer_pose_readout.py`, che infatti è sostituito dal richiesto `scaling_pose_readout.py` (ancora assente).
`load_context` (l.273) lega piano, confronto, quattro input, checkpoint storico e configurazioni complete dei due bracci al
contratto verificato da `scaling_train.check_source_contract`; `execute` scrive la ricevuta per ultima, dopo `check_unchanged`,
ricontrollo dei sorgenti e assenza di STOP.

**3. Gate del training.** Sì, i gate chiudono davvero. `validate_plan` (l.166) esige quattro gate con pin di schema (il
segnaposto `0`×64 è rifiutato da `digest`, ma un hash qualsiasi passa lo schema: la sostanza è verificata a runtime), allocazione
1 nodo/4 GPU/32 CPU/480G, controllo uguale al T100-G 5000 congelato (checkpoint `49dd518b…`, ricevuta `d0a50fe…`), orizzonte
`required_until_utc ≤ retention_verified_until_utc`. `check_runtime_gates` (l.302) legge e ri-hasha ogni gate: collaudo
(`code_sha == 05d4bf0`, 128 batch, 64/64, 4 rank, 4096 finestre, input uguali al contratto, 128 tempi positivi per braccio, media
sui 112 misurati, memoria ≤ 90 %, parametri 28.362.013/292.054.847, proxy S3 `M300 × max(1, 15,47 GiB / M30)` ≤ 90 % della
memoria rilevata, walltime del piano ≥ `ceil900(5000 × max(3883,03/5000 × max(1, rapporto), u300) × 1,2 + 1200)`); memoria di
produzione (prefetch, valutazione fissa, pannello posa, `accum_steps` 1, stessa ricetta a meno di `max_steps`, quattro picchi per
rank ≤ 90 %, job SLURM reale); readiness delle valutazioni (`REQUIRED_EVALUATORS` = `scaling_evaluation.py`,
`scaling_pose_readout.py`, `scaling_probe.py`, `scaling_compare.py`: gli ultimi tre non esistono nel checkout e
`_contained_source(…, strict=True)` solleva `FileNotFoundError`; capacità esatte incluse `dimension_matched_secondary` e
`paired_comparison`; test legati agli stessi hash; piano di costi con tetti GPU/CPU, estrazioni 30M e secondario; verdetto
APPROVATO con pin del testo; orizzonte del controllo uguale a quello del piano); review scientifica legata agli hash degli altri
tre gate, ai due sorgenti di esecuzione e all'allocazione. La conservazione del controllo copre estrazioni e confronto nel senso
che l'orizzonte dichiarato deve superare `now + walltime`, coincidere fra piano e readiness ed essere coperto dalla retention
verificata, con il tetto `2026-12-13T00:00Z` per i percorsi in scratch; l'adeguatezza della data dichiarata resta materia della
review del gate, non del codice. Il summary 30M: la fonte è il pin del piano di estrazione, accettato solo con `steps == 5000`
(§1.2); su Leonardo lo snapshot `$WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/summary.json` riporta
`steps 5000, stopped max_steps, parameters 28.362.013, world_size 4, accum_steps 1`, quello mobile in scratch `steps 19618`.
Completamento: `verify_phase_outputs` (l.368) richiede `stopped == max_steps`, 5000 passi, world 4, 292.054.847 parametri,
`accum_steps` 1, niente STOP, config finale uguale, i due checkpoint con `checkpoint_identity` (step, config, stream, RNG, AdamW
con `0 < step ≤ 5000` e `lr ≤ 1e-4`, firma dei dati, pose_stats, pesi finiti) e `trained_state_sha256` uguale. Nessun launcher,
piano di esecuzione o ricevuta di readiness per il 5000 è stato creato (19 file nel commit, nessuno di questi).

**4. F5/F8 (`tests/cpu/test_scaling_preflight.py`).** Corretti e indipendenti. F5: `assert got["lr"] == cfg.lr` nella fase
stress (l.72–73 e 118–119), non più la sola ripetizione di `_lr_at`. F8: nel test della transizione (l.95–129) l'oracolo parte da
una copia dello stato iniziale, poi `reference_rng` viene ricatturato dallo stato globale **dopo l'aggiornamento dell'oracolo**
(l.115) e prima di `update_arm`; al passo 64 l'oracolo ripristina il proprio stato, non `arm["rng"]`; l'impronta
`rng_after_sha256` del wrapper deve coincidere con quella del riferimento (l.117). Nello stesso ciclo si confrontano i
`param_groups`, i momenti AdamW tensor per tensor e i pesi di student e teacher (EMA) dopo ciascun passo, e `step == 2` alla fine.
Con `muscle_dropout` 0,4 il consumo di RNG è effettivo (un wrapper che non ripristinasse lo stato fallirebbe). I limiti sono
dichiarati correttamente: i controlli strutturali (`validate_optimizer`: identità dei parametri come insieme, momenti finiti,
step) non legano ogni momento al suo parametro; la futura ripresa a 19.618 userà, come già nel wrapper tokenizer, la ripresa dal
proprio `checkpoint.pt` con `trained_state_sha256` uguale al nominato, e andrà verificata a sé.

**5. S8 (`scaling_reading.py`, policy, protocollo).** Coerenti. EPN: `delta_pp = acc300 − acc30`, migliore se `L > 0`, peggiore
se `U < 0`, altrimenti irrisolto, limiti uguali a zero irrisolti (l.65–66). Posa: `delta_E_dyn = E300 − E30`, migliore se
`U < 0`, peggiore se `L > 0` (l.72); il punto e il delta devono riconciliarsi; la rilevanza pratica è un'etichetta separata,
`(E30 − E300)/E30 ≥ 0,02` **e** `U < 0` (l.86), che non entra nella tabella e non crea un margine di danno. P1/P2: `L > −1`
conservazione, `U < −1` perdita oltre margine, altrimenti irrisolto, uguaglianze irrisolte (l.77); nessun margine su EPN
(`epn_margin_pp` 0). Tabella (l.78–83): qualunque primario peggiore o perdita P1/P2 oltre margine → pausa; altrimenti almeno un
primario migliore → proposta di conferma a 19.618; altrimenti → proposta per la domanda sulla dose. I gate di qualità
(`artifact_integrity`, `paired_support`, `solver_converged`) precedono la lettura; `continuation_authorized` è sempre `False`;
booleani, NaN e infiniti sono rifiutati da `number`. `scientific_reading_policy.json` è uguale a `POLICY` (test). I test coprono
confini (−1 e 0 inclusi), discordanza, 2 % come etichetta, fit non validi. Valutazione critica della scelta operativa in §2.3.

**6. Costi e interpretazione.** La formula della seconda tranche è fissata nel protocollo, nella policy e in
`continuation_training_cap` (l.91–97: `ceil900(14618 × max(t5k/5000, u300) × 1,2 + 1200)`, 4 GPU, `submission_authorized`
`False`, estrazioni e CPU esclusi); l'esempio del test (t5k 15.000 s, u300 4 s) dà 72.000 s, 80 GPU-ore. La pubblicazione dei
tetti numerici prima di aprire gli score è un impegno procedurale del protocollo, non eseguibile dal codice: va fatta nel piano
e nel registro. La pausa è dichiarata come pausa della ricetta, non bocciatura dello scaling (protocollo e `scope` del risultato).
Rettifica di S7: accolta. Una sola coppia storica (T25-L − B19618, stesso seme nominale) è un'osservazione, non una stima della
distribuzione fra run: non definisce un pavimento né una soglia; `historical_pair_is_noise_floor` è `False`. Resta legittimo
citarla come contesto descrittivo (una replica a seme uguale è differita fino a 0,86 pp su DB6 e 0,51 su P1), senza sottrazioni.

## 2) Rilievi

### 2.1 Blocchi

Nessuno. Non ho trovato nel codice consegnato un difetto che permetta un avvio anticipato, una lettura errata o una perdita di
provenienza; le incompatibilità di schema temute (report del collaudo, summary del trainer, output di `scontrol`) sono state
verificate contro gli artefatti reali (§3).

### 2.2 Facoltativi (da decidere prima dei gate, non blocchi)

- **F-a — Identità del codice nella misura di memoria.** `check_production_memory` (`scaling_train.py` l.251) lega il report al
  contratto dei sorgenti ma non a un `code_sha` né alla versione di torch/CUDA, a differenza di `check_preflight`. Correzione
  minima: richiedere `code_sha` (40 esadecimali) e `runtime.torch/cuda` nel report e registrarli nella ricevuta.
- **F-b — Identità dell'ambiente software fra controllo e candidato.** Il venv (`$WORK/venv_wearusfm`) è fuori dal repo; lo
  snapshot 30M ha le chiavi `environment` e `code` nel summary. `verify_phase_outputs` potrebbe richiedere che le versioni di
  torch/CUDA/cuDNN del summary 300M coincidano con quelle dello snapshot 30M pinnato (o registrare la differenza nella
  ricevuta): «stessa ricetta» include lo stack numerico.
- **F-c — Caso positivo di `check_evaluation_readiness` non testato.** Nella fixture `runtime` la funzione è sostituita da un
  no-op (`test_scaling_train.py` l.175), quindi solo il caso «evaluators missing» è esercitato. Quando esisteranno i quattro
  evaluator, aggiungere un test con un report completo (capacità, verifica, risorse, review) che passa, e uno per ciascun campo
  mancante.
- **F-d — Contratto numerico per l'aggregatore.** `read_comparison` richiede `delta_E_dyn == candidate − control` con
  `rel_tol 1e-10` (l.71): l'aggregatore futuro deve scrivere i punti a piena precisione e calcolare il delta dagli stessi due
  numeri (nessun arrotondamento nel JSON). Da scrivere nella specifica dell'aggregatore.
- **F-e — Doppio caricamento del checkpoint.** `load_teacher_endpoint` carica il file con `torch.load` e poi di nuovo in
  `FF.load_model` (l.154, 157): per il 300M (~4,7 GB con AdamW) sono ~10 GB di RAM transitoria. Accettabile su un nodo di
  calcolo; lo stage `validate` non va eseguito sul login node (limite di 10 minuti di CPU). Eventuale correzione: costruire il
  modello dallo `state` già caricato.
- **F-f — Dimensioni delle cache D896 nel piano dei lettori.** La cache di posa `val` a 896 pesa ≈ 20.000 × 160 × 896 × 2 B ≈
  5,7 GB in FP16 (2,4 GB a 384) e 11,5 GB in FP32; `verify_pose_cache` la legge per intero per il controllo di finitezza. Il
  gate dei lettori (ridge FP64, proiezione S5) deve contare questa memoria nei tetti CPU; l'inviluppo «fino a 30G» del
  protocollo non è automaticamente sufficiente.
- **F-g — Parsing di `scontrol`.** `verify_allocation` (l.359) spezza l'output `-o` sugli spazi: un valore con spazi e `=` (per
  esempio `Comment`) potrebbe iniettare chiavi. Rischio basso con gli ID logici del submitter; SLURM 25.05 offre
  `scontrol show job --json`, più robusto. Formato attuale verificato compatibile (§3).

### 2.3 Valutazione critica della regola S8 (scelta operativa)

La regola è coerente, congelata prima dei risultati e conservativa; è una raccomandazione, non un'autorizzazione. Tre punti da
decidere ora, prima delle estrazioni, perché dopo non si potranno cambiare:

1. **Asimmetria fra «migliore» e «peggiore».** Il miglioramento ha un'etichetta di rilevanza pratica (2 % sulla posa), il
   peggioramento no: una posa «peggiore» con IC `[+0,1 %, +0,4 %]` relativo mette in pausa la ricetta quanto una da +5 %. Con un
   solo seme di pretraining e variabilità fra run non misurata, un delta risolto ma minuscolo non è interpretabile come effetto
   della taglia; a 5.000 passi di uno schedule da 49.046 può anche essere un effetto di dose (il modello grande è più indietro
   nel proprio schedule). La pausa costa solo una review, quindi la scelta è accettabile; ma il protocollo dovrebbe dire che
   la review della pausa pesa l'entità del peggioramento e la dose, non solo il segno, oppure dare al «peggiore» un'etichetta di
   rilevanza simmetrica. Decisione di GPT e Simone, da fissare prima degli score.
2. **Molteplicità.** Quattro controlli unilaterali (EPN peggiore, posa peggiore, P1 e P2 oltre margine) portano la probabilità
   di una pausa infondata, sotto l'ipotesi di nessuna differenza, fino a circa il 10 % se indipendenti (meno per correlazione).
   Il protocollo lo dichiara («nessuna garanzia congiunta»); è il prezzo della conservatività e va detto così nella lettura.
3. **Che cosa non risolve la tranche a 19.618.** Le righe 1 e 2 della tabella differiscono solo nella formulazione della
   proposta (entrambe propongono 19.618; solo la pausa cambia l'uso delle risorse). In entrambi i casi la tranche risponde alla
   domanda sulla dose, non a quella sulla variabilità fra run, che S7 (rettificato) lascia non misurata: un esito «irrisolto» può
   dipendere dal rumore delle sonde e nessuna continuazione lo chiarisce. La proposta che uscirà dalla tabella dovrebbe dichiarare
   esplicitamente questo limite; se misurare la variabilità fra run interessa, la scelta spetta a Simone, fuori da questa regola.

### 2.4 Lavoro futuro già dichiarato (nessuna azione in questa review)

Lettori a due taglie (`scaling_pose_readout.py`, `scaling_probe.py`, `scaling_compare.py`), proiezioni S5 con manifest e SHA,
aggregatore appaiato con bootstrap 2.000/seed 20261102 e audit di convergenza, misura della memoria di produzione, benchmark e
tetti numerici (training, estrazioni, CPU), launcher e piano del 5000, ricevute di readiness, verifica della retention (S2),
review della catena completa; F1–F4 e F6 differiti; S4 già nel protocollo prima del collaudo. Coerente con `remaining_gates` in
`offline_verification.json`.

## 3) Verifiche eseguite

- **Identità.** Script indipendente (`identity_check.py`, nello scratchpad): 103 voci, 0 discrepanze fra contratto, `git show
  afb644e:<path>` e byte del checkout; `src/wearusfm` ha 98 file sia ad `afb644e` sia a d2691c7, tutti nel contratto, nessuna voce
  del contratto fuori da `src/` oltre ai cinque script; `git diff --stat afb644e d2691c7 -- src/wearusfm scripts/train_jepa.py
  scripts/tokenizer_train.py scripts/tokenizer_preflight.py scripts/slurm/tokenizer_train.sbatch scripts/slurm/lib_code_dir.sh`
  vuoto; cinque primitive uguali a `f2061b8`, `b315dad` e al checkout. Diff vuoto rispetto a 05d4bf0 sui sei percorsi del
  collaudo. Nove hash di `offline_verification.json` e hash della review del critico uguali ai file.
- **Ricetta.** `configurations(base)["T100-G-30M"] == base` (dizionari di configurazione uguali); 300M diversa solo in
  `model.dim 384→896, n_heads 6→14, local_layers 2→3, backbone_layers 8→18`; `validate_plan` su un piano sintetico restituisce il
  comando riportato in §1.1; il test `test_only_ladder_dimensions_change…` conferma `erank_min` 38,4 invariato.
- **Schemi reali.** Report del collaudo (`scaling_preflight.py` 05d4bf0, l.259–273): tutte le chiavi lette da `check_preflight`
  sono scritte (`paired_data_masks_queries_targets`, `one_model_resident`, `scientific_checkpoints_written`, `batches_per_arm`,
  `native_warmup_batches`, `peak_stress_batches`, `windows_per_arm` = 128 × 32, `world_size`, `inputs` = stati pinnati con
  `manifest/scales/windows/pose_stats`, per braccio `normal_update_rank_max_seconds` (128 valori), `peak_allocated_bytes`,
  `device_memory_bytes`, `memory_gate_passed`, `initial.parameters`). Summary del trainer (`run.py` l.1714–1725): `stopped`,
  `steps`, `parameters = sum(p.numel() for p in student.parameters())`, `world_size`, `accum_steps`; sullo snapshot reale 30M
  `parameters` = 28.362.013, come atteso dalla stessa formula per il 300M (292.054.847). `scontrol show job -o` su Leonardo
  (SLURM 25.05.7, job 63524541 di un altro utente, sola lettura) stampa `AllocTRES=cpu=…,mem=…G,node=…,billing=…,gres/gpu=…`,
  `NumNodes`, `NumCPUs`, `TimeLimit`, `JobState`: compatibile con `check_allocation_fields`; `sacct` del job tokenizer 63386403
  riporta `mem=480G`, formato accettato.
- **Controllo 30M su Leonardo (sola lettura, 04/11 sera).** `$WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/`:
  `config.json` (sha `dd68e900…`, uguale a `config_T100-G.json` del repo), `receipt.json` (sha `d0a50fe7…`, uguale alla copia nel
  repo pinnata dal contratto), `summary.json` con `steps 5000, stopped max_steps`. In scratch: `checkpoint_step0005000.pt`
  454.381.728 byte, mtime 03/11 12:34:36 CET, non toccato dalla continuazione (`checkpoint.pt` e `checkpoint_step0019618.pt` del
  03/11 23:27); 40 giorni dal mtime cadono il 13/12 12:34 CET, il tetto `2026-12-13T00:00Z` del contratto è conservativo; la
  purge per atime o mtime resta da confermare con CINECA (S2). Il summary mobile in scratch riporta `steps 19618`. Nessun
  checkout di campagna scaling ancora in `$WORK/wearusfm_campaigns/`.
- **Test.** Comando esatto del critico (sette file, `PYTHONPATH=src:tests/cpu`, thread a 1) sul checkout d2691c7: **248 passati in
  20,93 s**, come dichiarato. Intera `tests/cpu` sullo stesso checkout: 2.275 passati, 1 fallito, in 26 minuti. Il fallito è
  `test_pose_b19618_launcher.py::test_exact_B_recipe_only_maxsteps_changes_schedule_identical` («continuation recipe differs beyond
  max_steps»): fallisce già alla base 6d80e8d, non riguarda alcun file toccato dalla consegna e passa su `main` (78268d6) e sul
  ramo di Claude (14cdfb4). Dipende da `scripts/train_jepa.py` del ramo tokenizer (84408bf, lo stesso di afb644e, non ancora su
  `main`), che costruisce dalla CLI B una configurazione diversa dalla ricetta congelata in
  `results/infra/pilot_posa/collaudo_b_20261028/config.json` oltre `max_steps`. Problema separato del ramo, registrato nella
  sezione 2 del registro; non è un difetto di questa consegna.
- **Estrazione dimensionale.** `grep` di 384/768 nel percorso di estrazione: nessun valore cablato in `channel_readout.py`,
  `pose_eval.py`, `calibration_probe.py`, `fm_features.py`, `spectral_step1.py`; `encoder_dim` e `dim` vengono dal modello.

## 4) Limiti di questa review

Review su CPU e in sola lettura: nessuna misura di memoria, tempo o correttezza GPU del 300M; nessuna estrazione o lettura
eseguita; il comportamento del wrapper sul cluster (allocazione reale, tempi di verifica post-training dei checkpoint da ~4,7 GB
entro il margine di 300 s di `--time-limit-s`, che conta solo nel caso di arresto per tempo, già perdente per protocollo) resta da
osservare al primo uso. Le fixture dei test sono sintetiche e non costruiscono un modello 300M, come dichiarato. Nessun giudizio
sui risultati scientifici, che non esistono ancora.
