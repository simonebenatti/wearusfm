# Review di Claude — collaudo train-only 30M/300M (consegna 05d4bf0, ramo `codex/scaling-300m-20261104`) — 04/11/2026

**VERDETTO: APPROVATO** per il collaudo tecnico. Nessun blocco. Seguono miglioramenti facoltativi (F1–F7) e requisiti per il seguito
scientifico (S1–S7), che non condizionano la sottomissione del collaudo. Solo review: nessun esperimento, sottomissione, allocazione o
modifica al codice. Prompt di GPT: `docs/prompt_claude_scaling_review_20261104.md` sul suo ramo (commit 31f6d03).

## 0. Perimetro e metodo

- Commit rivisto `05d4bf0663481522a56d96c0b07847924cf5bd26` (base `f35d5bf`), preso da `origin` in un checkout temporaneo detached, fuori dai
  worktree di lavoro. Diff della consegna: 10 file, 977 righe aggiunte, nessuna riga tolta.
- Letti per intero: `docs/scaling_300m_protocol_20261104.md`, `scripts/scaling_preflight.py`, `scripts/slurm/scaling_preflight.sbatch`,
  `tests/cpu/test_scaling_preflight.py`, `results/infra/scaling_300m_20261104/{preflight_plan,resource_policy,verification}.json`,
  `docs/reviews/scaling_300m_preflight_critic_20261104.md`, le voci di registro e di `decisioni.md`; `scripts/tokenizer_preflight.py`
  (tutte le funzioni riusate: `load_plan`, `pinned_file`, `code_identity`, `update_arm`, `support_audit`, `collect_features`,
  `selected_parameters`, `rng_state`/`restore_rng`, `SourceGuard`, `check_unchanged`, `write_json`); in `src/wearusfm/training/run.py`
  `LADDER`, `_ddp_wrap`, `DistContext`, `_any_flag`, `_accumulate_terms`, `_lr_at`, `AlarmRule`/`AlarmMonitor`, la costruzione di autocast e
  AdamW e il passo di produzione (lr → `train()` → `zero_grad` → `_accumulate_terms` → `clip_grad_norm_` → `step` → `ema_update`);
  `scripts/slurm/lib_code_dir.sh` (`code_dir_setup`); `scripts/train_jepa.py` per la ricostruzione della ricetta dall'argv validato.
- Rieseguiti i test nel checkout temporaneo con il comando della consegna: **23 passati in 13,53 s** (15 di `test_scaling_preflight.py`
  + 8 storici di `test_tokenizer_preflight.py`); `bash -n` del launcher pulito. SHA256 del piano sorgente pinnato
  (`results/infra/tokenizer_execution_20261103/preflight_plan.json` = 13ed48b6…) ricalcolato e coincidente.
- Verifica numerica dello schedule con la configurazione T100-G reale (argv validato di `comparison_configurations.json`): LR nativo
  5,10e-8 al passo 0, 3,26e-6 al 63, 6,52e-6 al 127, 1e-4 dal 1961; con `warmup_steps=0` 1e-4 a tutti i passi 0–127 (fase stabile fino al
  39.237). Parametri 28.362.013 / 292.054.847, `LADDER["300m"]` = dim 896, 14 teste, 3/18/2 blocchi: tutto coerente col protocollo.
- Identità del codice di training fra il checkout del 30M a 5.000 (afb644e) e 05d4bf0: `git diff --stat afb644e 05d4bf0 -- src/wearusfm
  scripts/train_jepa.py scripts/slurm/tokenizer_train.sbatch scripts/slurm/lib_code_dir.sh` è **vuoto**.

## 1. Risposte alle sei domande

**1. Sequenza nativa/stress, stati di AdamW/teacher/RNG, oracolo a due passi.** Corretta. `update_arm` del collaudo sostituisce solo
`warmup_steps=0` dentro `arm["cfg"]` per la durata della chiamata (`try/finally`, riga 119–124) e delega a `P.update_arm`, che è il passo
di produzione: LR da `R._lr_at` sullo stesso `optimizer` (riga 313–314), stesso `AdamW(student.parameters(), lr, weight_decay)` di
`run.py:1357` (nessun parametro congelato, quindi la lista coincide), autocast bf16 come `run.py:1307`, `_accumulate_terms` con il wrapper
DDP, `clip_grad_norm_(student.parameters(), grad_clip)`, `optimizer.step()`, `ema_update(teacher, student, momentum)` costante come nel ramo
`else` di `run.py:1618`. Il RNG del modello viene ripristinato da `arm["rng"]` prima dell'update e salvato dopo (righe 305 e 345): la
traiettoria continua fra i passi e le diagnostiche non la consumano. L'oracolo a due passi (test `..._transition_preserves_adamw_moments_...`)
esegue 63 (nativo) e 64 (stress) sullo stesso braccio e confronta pesi, teacher, `param_groups` e tutti i momenti, con `step == 2`: copre
la continuità, non solo le fasi singole. Il LR della fase stress è 1e-4 esatto (verifica numerica sopra); il test lo confronta con
`_lr_at(step, cfg_warmup0)`, non col valore 1e-4 (F5).

**2. Pairing dei dati e RNG del modello; guardie sui report troncati.** Corretti. Ogni taglia riceve un DataLoader nuovo con lo stesso
seme e lo stesso `generator` (riga 203–206), quindi lo stesso flusso: le impronte confrontate (`batch_sha256`, segnali, maschere, target,
origini, riga 216–219) sono calcolate sui dati CPU prima dello staging e confrontate fra le taglie in `verify_pairing`, insieme a
`support` (supporto di query/posa atteso e osservato nel decoder, conteggi delle loss, finestre) e al LR per passo. Il RNG del modello
resta libero per costruzione (dropout su forme diverse) e il test lo dichiara. Report troncati: 128 record esatti per braccio, passi
consecutivi 1..128, sequenza di fasi 64/64 (riga 136–142), `rank{r}.json` scritto con `complete: true` solo a fine ciclo, `report.json`
solo dopo la verifica di tutti i rank; test negativi su record mancanti, passo e fase errati. Niente ripresa: `open("x")` e directory
nuova obbligatoria su tutti i rank.

**3. Memoria di un modello residente e tempi.** Corretti. `P.update_arm` azzera le statistiche di picco prima di ogni update
(`reset_peak_memory_stats`, riga 321) e registra baseline e picco per passo: il picco del 300M è misurato con il solo 300M residente,
purché il 30M sia stato liberato. Ho cercato riferimenti residui al primo modello: la chiusura `capture` è l'unico riferimento esterno
allo studente e viene spezzata (`decode = None`); DDP, teacher, ottimizzatore, gruppi diagnostici e `observed` (svuotato a ogni update)
vivono nel dizionario `arm`, cancellato con `del` e `gc.collect()`; i record conservati sono float e stringhe (serializzati in JSON a ogni
passo, quindi senza tensori); l'autocast svuota la propria cache all'uscita del contesto; `empty_cache()` precede la creazione del 300M.
Non ho trovato riferimenti che impediscano la liberazione. Tempi: `forward_backward_s` e `clip_optimizer_ema_s` sono delimitati da
`synchronize` e da barriere che tengono fuori le diagnostiche degli altri rank; il riassunto somma i massimi fra rank delle due componenti
e scarta i primi 8 passi di ogni fase. Esclusi, come dichiarato: staging del batch, loader, pannelli e checkpoint. Lavoro diagnostico
incluso per errore nei tempi: nessuno. Lavoro diagnostico fuori dai tempi ma inutile per questo collaudo: `collect_features` (F4).

**4. Gradiente globale, fattore di clip, update relativi, monitor QKV/MLP.** Informativi e correttamente delimitati. `grad_norm_before_clip` è
la norma globale restituita da `clip_grad_norm_` (uguale su tutti i rank); `clip_factor = min(1, clip/(norma+1e-6))` riproduce la formula
di PyTorch; gli update relativi sono ‖Δθ‖/‖θ‖ per gruppo con la divisione per zero omessa (testa di posa a zero). I nuovi gruppi puntano
davvero alle matrici `time_attn.qkv.weight` e `mlp.0.weight` del primo e dell'ultimo blocco (i test costruiscono il modello reale in
miniatura: un nome sbagliato darebbe `KeyError`). Il report e il protocollo dicono esplicitamente che lo stress non è una prova di
ottimalità del LR. Controllo tecnico indispensabile ora: nessuno. Due aggiunte utili ma non necessarie: la stabilità della loss nella
fase stress (F1) e la copertura delle forme del corpus (F3).

**5. Gate, scadenza, provenienza, output, pin, launcher, risorse.** Coerenti. Piano congelato (dose 64+64, world 4, batch 32, 6 worker,
gate 0,9 e 1.500 s imposti in `load_plan`, con test negativi); `code_identity` (HEAD atteso e `src`/`scripts` puliti) all'inizio e alla
fine; `check_unchanged` su piano sorgente, piano del collaudo, manifest, scale, finestre, statistiche di posa e configurazioni;
`SourceGuard` rifiuta righe non di pretraining ed EPN/UCI e verifica gli stat delle sorgenti prima e dopo. Launcher: `boost_qos_dbg`,
TimeLimit ≤ 1.800 s verificato da `scontrol`, un nodo e 32 CPU, checkout di campagna obbligatorio (`code_dir_setup`: HEAD = EXPECTED_SHA,
tree pulito anche dai file non tracciati, `cd` e `src` in testa al PYTHONPATH), piano dentro il checkout, output assoluto, nuovo e fuori
dal codice (controllato sia nel launcher sia da tutti i rank con `_any_flag`), 4 GPU visibili. Un `require` fallito fa uscire torchrun con
errore: nessun retry, nessuna riduzione del batch, nessuna sovrascrittura. Risorse: 4 GPU × 30 min = 2 GPU-ore e 16 core-ore allocate,
checkout 0,333 core-ore, come nel JSON. La scadenza interna di 1.500 s parte dopo il caricamento dell'indice: con 0,78 s/passo del 30M e
anche 4× per il 300M si resta sotto i 10 minuti di update, più diagnostiche e avvio dei worker.

**6. Confronto scientifico successivo.** Ben delimitato: 300M a 5.000 contro il 30M a 5.000 già disponibile (`checkpoint_step0005000.pt`
di T100-G, con teacher), stesso tokenizer, obiettivo, dati, schedule e seme, teacher EMA, stessi split/fold, lettori D896/pool1792 da
scrivere con regressioni contro D384, un solo seme e dati di sviluppo dichiarati, nessuna scelta del LR sulla val, 5.000 negativo non è un
veto. Il protocollo riconosce che la larghezza del readout fa parte della prova. I requisiti S1–S7 riguardano questo seguito.

## 2. Rilievi

### Bloccanti del collaudo
Nessuno.

### Miglioramenti facoltativi (non condizionano il lancio)

- **F1 — Stabilità della loss nel riassunto.** `summarize` (righe 153–177) riassume gradiente, clip e update relativi ma non le loss, che
  sono nei record (`losses`). Aggiungere per fase media e massimo di `losses["total"]` sui primi 8 e sugli ultimi 8 passi misurati: con LR
  al picco su pesi giovani una divergenza si vede prima nella loss che nel clipping. Impatto: lettura più completa dello stress; nessun
  cambiamento delle misure.
- **F2 — Prova esplicita del modello unico residente.** Riportare nel riassunto `memory_baseline_allocated_bytes` del primo update di ogni
  taglia e verificare che per il 300M sia sotto 1,2 × 2 × 4 byte × parametri (studente + teacher prima dei momenti di AdamW): un 30M non
  liberato (≈ 0,45 GB con i momenti) si vedrebbe. Oggi il report dichiara `one_model_resident: true` per costruzione, non per misura.
- **F3 — Copertura delle forme del corpus.** 128 batch non coprono le forme peggiori (canali × patch). Registrare per passo il massimo di
  canali e di `n_patches` (già in `support.windows`) e confrontarlo nel report con i massimi dell'indice: dice se il gate di memoria al 90 %
  è stato misurato vicino al caso peggiore o no. Niente batch aggiuntivi.
- **F4 — Lavoro diagnostico inutilizzato.** `P.update_arm` chiama `collect_features` a ogni passo (doppio passaggio del frontend su
  segnali grezzi e mascherati, momenti FP64 64×64 per dataset/frequenza), ma il collaudo non riporta `arm["features"]`. Fuori dai tempi,
  quindi innocuo per le misure; costa qualche decina di secondi sulla scadenza. O si riporta il riassunto (utile come controllo del punto
  di lavoro della GELU a 896, identico per costruzione) o si passa un oggetto con `add` nullo. Scelta di GPT.
- **F5 — Asserzione leggibile sul LR di stress.** Nel test della fase stress aggiungere `assert got["lr"] == cfg.lr`: oggi il confronto è
  con la stessa funzione `_lr_at`, che ho verificato dare 1e-4, ma il test non lo dice.
- **F6 — Blocco `runtime` come nel preflight del tokenizer.** Il report del collaudo registra torch/cuda/gpu/job; quello del tokenizer
  anche TF32, algoritmi deterministici, moduli e pacchetti. Riusare quel blocco rende confrontabili i due collaudi.
- **F7 — Fase usata nel rapporto dei tempi.** `resource_policy.formula_after_benchmark` usa «benchmark300M/benchmark30M» senza dire se è
  la media della fase nativa, dello stress o di tutti i 112 passi misurati (il calcolo per passo non dipende dal LR, quindi vanno bene
  tutti e 112): scriverlo, così il preventivo è riproducibile.

### Requisiti del seguito scientifico (prima del training a 5.000)

- **S1 — Identità del codice di training col 30M a 5.000.** Il controllo T100-G a 5.000 è stato addestrato con afb644e. A 05d4bf0 il
  diff su `src/wearusfm`, `scripts/train_jepa.py`, `scripts/slurm/tokenizer_train.sbatch` e `lib_code_dir.sh` è vuoto: oggi la ricetta è
  identica byte per byte. Il wrapper del 300M deve congelare questa identità come fece la campagna tokenizer (`scientific_source_sha256`)
  rispetto allo SHA di afb644e, perché un commit successivo del ramo potrebbe toccare quei file.
- **S2 — Il controllo sta in scratch.** `checkpoint_step0005000.pt` di T100-G (teacher compreso) e le cache di posa a 5.000 sono in
  `$SCRATCH/wearusfm_runs/runs/tokenizer_20261103/T100-G/`, cancellazione automatica dal 13/12/2026. Le estrazioni P1/P2 del controllo a
  5.000 e il confronto vanno fatti prima, o il checkpoint va copiato e ri-pinnato per sha256.
- **S3 — Margine di memoria tra collaudo e produzione.** In produzione il 30M T100-G a 5.000 ha toccato 15,47 GiB per rank (summary del
  trainer), con prefetch, pannelli di posa e valutazione fissa che il collaudo non esercita. Dopo il collaudo, usare il rapporto
  produzione/collaudo del 30M per stimare il picco di produzione del 300M prima di fissare le risorse; il gate al 90 % sul solo passo di
  training non basta da solo. Per ordine di grandezza, scalando le attivazioni con larghezza (2,33×) e profondità, il 300M può avvicinarsi
  ai 57,6 GiB del gate: è proprio ciò che il collaudo deve misurare.
- **S4 — Ripiego dichiarato prima, non dopo.** Se il 300M supera il gate, il ripiego coerente con il contratto delle loss distribuite è
  `accum_steps=2` a batch globale 32 invariato (stesso gradiente del batch globale, stessa dose), non un batch più piccolo né 8 GPU (ripresa
  con `world_size` diverso rifiutata dal trainer). Va scritto nel protocollo prima del collaudo, così non è una scelta a posteriori; se
  serve, un secondo collaudo da 128 batch misura quella variante.
- **S5 — Larghezza del readout.** pool1792 contro pool768 e ridge su 896 contro 384 fanno parte della prova, come dichiarato. Per
  separare «più informazione» da «più capacità della sonda», un controllo secondario a dimensione pareggiata (proiezione casuale fissa di
  pool1792 a 768, o PCA a 768 stimata sul solo train del fold, dichiarata nel protocollo) costa solo fit CPU sulle stesse cache. Facoltativo
  ma consiglio di scriverlo ora, a criteri congelati. Attenzione a UCI (364 finestre): con 1.792 feature il C-grid e l'audit di convergenza
  contano ancora di più.
- **S6 — LR fisso e lettura.** Tenere 1e-4/clip 1 misura «la taglia nella stessa ricetta», non lo scaling: una larghezza 2,33× con lo stesso
  LR è una scelta dichiarata, non neutra. Lo stress rileva solo instabilità grossolane; la via di calibrazione train-only separata, già
  prevista, è quella giusta se le diagnostiche mostrano un problema. La soglia assoluta di rango 38,4 per tutte le taglie è coerente con la
  decisione firmata del 23/10 (`AlarmRule`, commento in `run.py:60-63`): nessun cambiamento.
- **S7 — Lettura appaiata.** Bootstrap per soggetto, 2.000 estrazioni, seme 20261102, margini descrittivi: coerente con la campagna
  tokenizer. Resta valido il pavimento di rumore misurato oggi sulla replica T25-L − B19618 (0,2–0,9 pp per dataset, 0,5 pp su P1): un
  delta 300M − 30M dentro quell'intervallo non è distinguibile dal rumore di run a seme unico.

## 3. Numeri di riferimento usati

| Grandezza | Valore | Fonte |
|---|---|---|
| T100-G 30M a 5.000: tempo del trainer, memoria max per rank | 3.883,03 s (0,777 s/passo), 15,47 GiB | `results/infra/tokenizer_execution_20261103/5000/T100-G/summary.json` |
| Parametri 30M / 300M | 28.362.013 / 292.054.847 | `R.count_parameters` sulla configurazione reale |
| LR nativo ai passi 0 / 63 / 127; stress | 5,10e-8 / 3,26e-6 / 6,52e-6; 1e-4 | `R._lr_at` con la configurazione reale |
| Gate memoria | 0,9 × 64 GiB = 57,6 GiB per rank | piano del collaudo |
| Costo del collaudo | 2 GPU-ore + 16 core-ore allocate; checkout 0,333 core-ore | launcher e `resource_policy.json` |

## 4. Limiti di questa review

Nessuna esecuzione su GPU o in DDP: i test girano su CPU con modelli in miniatura; i pin assoluti degli input su Leonardo non sono stati
ricontrollati da me (lo fa `load_plan` al lancio e lo ha già fatto il preflight del tokenizer del 03/11 con lo stesso piano). Non ho
verificato il comportamento del QoS `boost_qos_dbg` con `--mem=480G`: i nodi sono gli stessi dei job di produzione, ma se SLURM
rifiutasse la richiesta lo si vede alla sottomissione, non dopo.
