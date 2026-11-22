# DRIVER_MAP — mappa del codice congelato per un driver diagnostico esterno (D0 / D1)

Revisore: driver_map. 22/11/2026, 11:23-11:45. Sorgenti letti: `tree_e8c98244` (commit `e8c9824461b5e18494ff462f1be8abda71b76d5e`).
Percorsi `file:riga` relativi a `tree_e8c98244/scripts/pose_temponet/` salvo diversa indicazione.
Legenda: **[D]** dimostrato (codice letto riga per riga, numero ricalcolato dai JSON, o comando eseguito sul Mac);
**[I]** ipotesi / non verificabile sul Mac (niente CUDA, niente dati veri, torch 2.14 invece di 2.2).

Script riproducibili in questa cartella (ambiente del compito, un thread):
`proto_fixture.py` (fixture + calibrazioni vere + selezione vera TCN_post con ricetta di test), `proto_driver.py` (D0 e D1 sulla fixture),
`proto_checks.py` (import, blocchi, contesto numerico), `proto_split.py` (parte affine / profonda). Uscite: `proto_*.out`, `pytest_subset.out`.

```
P=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python ; T=/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244 ; N=.../review/driver_map
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/scripts:$T/src:$N $P proto_fixture.py $N/fixture_out   # 49 s
... $P proto_driver.py $N/fixture_out   # 18 s      ... $P proto_checks.py $N/fixture_out   # 10 s      ... $P proto_split.py $N/fixture_out
```

---

## 0. Cinque fatti che decidono il disegno del driver

1. **[D] L'inizializzazione della TCN non si riproduce sul Mac.** `heads.make_head("TCN_post", seed=0)` sul Mac (torch 2.14.0, arm64) ha
   `state_sha256 = 19ebce0e51b1…e5e9d1`; l'archivio del cluster (torch 2.2.0a0+git6c8c5ad, x86) ha `50fb2294f6be…79aab5`
   (`selection_temporal_TCN_post.json: initial_sha256_per_fold`). Anche MLP differisce (`1427ef3b…` contro `024fde34…`); L0 coincide (`39cee3fe…`, pesi
   senza RNG). Quindi **D1 si fa solo su Leonardo**, e il primo cancello e' `Trajectory.initial_sha256 == 50fb2294f6beb2b54917e99bcb81683ffc24a41bd69ce36b17c08cc53279aab5`.
2. **[D] Bersaglio di training e bersaglio di valutazione sono lo stesso campo del bundle (`FoldBundle.targets`) ma finiscono in due oggetti distinti**:
   il training lo copia in `TensorTrainingSet.target` (`workflow.py:236-237`), la valutazione lo rilegge dal bundle a ogni blocco
   (`workflow.py:204-212`, `253`). Separarli dall'esterno costa una riga: `dataclasses.replace(bundle, targets=bundle.clean_targets)` solo per
   costruire i set di training; l'evaluator resta sul bundle originale. Provato sulla fixture.
3. **[D] `workflow._score_model(..., include_clean=True)` restituisce in un passaggio numeratori e denominatori contro y e contro y\***
   (`workflow.py:242-261`). Usarla cosi' com'e' (blocchi da `BATCH = 32`, `workflow.py:25`) e' l'unico modo di avere la parita' a 1e-12: sulla
   fixture 10/10 punti pooled uguali bit a bit; con blocchi diversi lo scarto relativo e' 3,7e-10 (mse_sum) e 1,1e-9 (dynamic_sse).
4. **[D] `Trajectory.restore` funziona solo nello stesso contesto numerico** (`train.py:341-346`: `numeric == numerical_profile()`, cioe' versione
   di torch, 8 thread, flag deterministici, `device_type`). Su Leonardo/GPU con `configure_numerics()` e' la via piu' rigorosa (verifica anche
   `dataset_sha256` e `initial_sha256`); per la sola inferenza altrove: `heads.make_head` + `load_state_dict(state["model"], strict=True)`.
5. **[D] Il piano congelato non puo' emettere ricevute per il driver**: `provenance.make_receipt` esige un job del piano (`provenance.py:669-670`),
   `runner.run_stage` rifiuta directory e ricevute esistenti (`runner.py:220-221`) e `safe_output_root` accetta solo la radice registrata
   (`provenance.py:105-108`). Servono un ID logico nuovo, una radice di uscita nuova e un manifest proprio del driver; `P.atomic_json` /
   `P.atomic_torch` si possono riusare con un'altra `root` (provato: scrive, e rifiuta la seconda scrittura con `FileExistsError`).

---

## 1. Percorso del job dei controlli (world=temporal, head=TCN_post)

### 1.1 Lancio
- `scripts/slurm/pose_temponet_dispatch.sbatch` -> `lib_code_dir.sh: code_dir_setup` -> `scripts/slurm/pose_temponet.sbatch`:
  `module purge; module load profile/deeplrn; module load cineca-ai/4.3.0; source $WORK/venv_wearusfm/bin/activate`,
  `PYTHONPATH=$CODE_DIR/src:$CODE_DIR/scripts`, `OMP/OPENBLAS/MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK` (8), `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  poi `python3 -m pose_temponet materialize-plan ...` e `python3 -m pose_temponet run --plan $TCN_PLAN --plan-sha256 ... --job-id ...`.
- Valori reali (ricevute e `controls_commands.txt`): `CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1`,
  `EXPECTED_SHA=edebfb6c290cf661989c5cff45916bc6a93e607e` (codice rivisto `a1ae9607681c1a181034d7f72770537054af419d`),
  piano `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117/manifests/b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3.json`
  (= `results/infra/pose_temponet_execution_20261121_A1/plan_controls.json`, sha256 ricalcolato uguale, 76.657 byte).
  Job 65035631: 8 CPU, 1 A100, 32G, limite 05:11:00, durata 4.383,3 s, `boost_usr_prod`, `--qos=normal`.
- `__main__.py:244-251` (comando `run`): `P.file_pin` del piano, `P.load_plan(args.plan, require_review=True)`, `E.validate_graph`, `runner.run_stage(plan, job_id)`.

### 1.2 `runner.run_stage` per lo stage `controls` (`runner.py:208-417`)
| riga | chiamata | nota per il driver |
|---|---|---|
| 213-214 | `P.validate_plan(plan, require_review=True)`, `P.checkout_identity(plan)` | sola lettura (git show / archive); richiedono che `P.ROOT` sia un checkout con sorgenti identici al manifest |
| 215-221 | `_allocation(job)`, radice e directory di uscita, rifiuto se esistono | NON riusare: specifico del job pianificato |
| 222-224 | `train.configure_numerics()`; `numeric = P.numeric_context("gpu")` | da replicare nello stesso ordine |
| 226 | `receipts, pins = _ancestors(plan, job)` | ricevute verificate dei predecessori (sola lettura) |
| 229 | `P.verify_numeric(plan, stage, numeric)` | digest del contesto == `plan["numeric_pins"]["gpu8"]` = `12cb2a0b…b7eb3f` |
| 241-249 | `calibrated = _calibrations(receipts)`; `workflow.combined_closed_form_gate(calibrated)` | `_calibrations` (98-102) = `_torch_artifact(r, "calibration_state")` ordinati per `outer_fold` |
| 303-306 | `cache = P.read_verified_inputs(plan, "train", stage="controls")`; `kwargs = {allowed_train_users: plan["producers"]["train"]["users"], source_identity: _input_identity(plan), expected_folds: previous["train"]["folds"]}` | `previous` = dettagli del preflight (231) |
| 308 | `bundle = workflow.prepare_control_folds(cache, calibrated, job["world"], **kwargs)` | |
| 309 | `result = workflow.run_selection(bundle, job["head"], device="cuda", on_checkpoint=_progress)` | `_progress` (186-194) stampa un evento JSON per punto |
| 317 | `_save_results(root, directory, result, artifacts)` (197-205) | `selection.json` con `P.atomic_json(_plain(value))`, `selection_states.pt` con `P.atomic_torch(value)` = `torch.save` |

`runner._torch_artifact(receipt, name)` (`runner.py:38-44`): `P.verify_pin(pin)`; `torch.load(pin["path"], map_location="cpu", weights_only=False)`;
`P.unchanged(...)`. `runner._json_artifact(receipt, name)` (34-35) = `P.read_json_pin(pin)`.

### 1.3 Lettura degli input verificati
`provenance.read_verified_inputs(plan, split, *, stage, depth=None)` (`provenance.py:467-483`): verifica `pose_stats`, poi
`data.read_cache_pin(pin, producer=..., reference_pin=..., depth=None)` (`data.py:94-119`), poi utenti e dettagli del produttore.
File reali: cache `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/49046/pose/train/train.npz` (474.412.856 byte,
sha `30c0b50d…1eec7e`, 1.907 finestre, 40 utenti); riferimento `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/train_subset.npz`
(sha `c7bc296e…50db23`); `pose_stats.json` su SCRATCH (sha `757f2aee…524a7e`). **Due dei tre input stanno su SCRATCH** (cancellazione a 40 giorni).

Chiavi del dizionario `cache` (`scripts/consolidation_long_pose.py:167-175` + `data.py:112-118`), con `depth=None`:
`pre`, `lat` (N,160,384 FP16), `key_time_valid` (N,160 bool), `pred`, `target` (N,160,20 float32, posa reale normalizzata), `mask` (N,160,20 bool),
`std` (N,20), `pose_var` (N,; NON usato dai controlli: ricalcolato), `meta` (lista di dict con `user`, `n_patches`, ...), `config`, `has_head`, `dim`,
`n_recordings`, `lengths`, `central_mask`, `input_verification`. Niente chiave `features` (compare solo con `depth="post"|"pre"`).
`workflow._features(cache, "post")` (110-114) restituisce `cache["lat"]`.

### 1.4 Costruzione del bundle (`workflow.py`)
`_prepare_real_folds` (132-159):
- utenti = `sorted(set(meta.user))`, devono coincidere con `allowed_train_users` ed essere 40;
- `halves = data.alternating_user_folds(cache, expected_folds)` (`data.py:139-144`): `[users[::2], users[1::2]]`;
- `fitting = (indici del fold 0, indici del fold 1)` come `np.flatnonzero(np.isin(users, half))`; **`held = (fitting[1], fitting[0])`**:
  gli "esclusi" del fold k sono il fitting dell'altro fold (961/946 finestre; con supporto 920/934);
- `support = data.central_support(mask, lengths, key_time_valid)` (`data.py:43-55`; `scripts/pose_temporal_readout.py:195-199`:
  `mask[:, 6:-6] & (t+6 < length)`), forma (N,148,20).

`_prepare_control_folds(cache, artifacts, world, *, allowed_train_users, source_identity, expected_folds=None, expected_user_count=40)` (168-195), per fold:
- `choice = artifact["calibration"]["selected"][world]` -> `a`, `sigma`;
- `ystar, y = controls.synthetic_target(artifact["components"], a, sigma)` (`controls.py:193-198`): `clean = L + a*T`; `observed = clean + sigma*epsilon`;
- **soglia dinamica** `threshold = metrics.training_threshold(ystar[fitting], mask[fitting])` (`metrics.py:60-64`): mediana, sulle finestre di fitting,
  di `dynamic_variance` (42-57: varianza temporale di popolazione per giunto sulla finestra INTERA con `mask`, media sui giunti con >= 2 campioni),
  `max(mediana, 1e-12)`; deve essere `==` a `artifact["outer_reference_statistics"][world+"_threshold"]` (184-185);
- `pose_variances[fold] = metrics.dynamic_variance(ystar, mask)` su tutte le finestre: la selezione "dinamica" viene SEMPRE da y\*, mai da y.

`FoldBundle` (117-129, dataclass frozen): `cache, world, fitting, held, targets (y), clean_targets (y*), pose_variances, thresholds, support, manifest, reference_gate`.
`manifest` (bloccato con `bundle_sha256`): `schema, world, source_identity, population_sha256, users, user_folds, window_ids_sha256, thresholds,
support_sha256, folds[{fit_indices, held_indices, fitting_windows, held_windows, excluded_fitting_windows}], artifacts_sha256, target_sha256[2],
clean_target_sha256[2], reference_gate`.
Valori archiviati (identici nei tre `selection_temporal_*.json`): `bundle_sha256 4e147b1f43f3edf64ac93262d8b0805c762d886cc14d96b7b786c9c867fd60ad`,
`target_sha256 [41519fd2…, e988407c…]`, `clean_target_sha256 [c2452d8b…, f1a6df78…]`, `support_sha256 82bc2849…`, `thresholds [0.5437424309221641, 0.5374366720436082]`,
folds `(961, 946, 41)` e `(946, 961, 12)`, `feature_array_sha256 397527e8…`, `evaluation_identity 8fcf8609…2f5b35`.
**[D sulla fixture]** ricostruendo il bundle dai `calibration.pt` ricaricati con `source_identity = selection["bundle"]["source_identity"]`,
`dict(bundle.manifest) == selection["bundle"]`: un solo confronto copre popolazione, partizioni, supporto, soglie, y, y\* e componenti.

### 1.5 Dove entra il bersaglio
- **Loss**: `_training_set(bundle, indices, depth, fold)` (225-239) -> `T.TensorTrainingSet(features FP16 con zeri oltre la lunghezza,
  torch.from_numpy(bundle.targets[fold][indices]) [float64, N,160,20], support[indices], ids)`; `TensorTrainingSet.batch` (`train.py:195-204`) porta
  x e y a FP32 sul device e ritaglia y a `[6:154]`; `Trajectory.train_to_step` (`train.py:281-287`): `loss = metrics.per_window_mse(prediction[:, 6:-6], y, support)`
  (`metrics.py:24-39`: media sulle finestre del batch di SSE mascherata / numero di elementi supportati).
- **Valutazione/selezione**: `evaluator = lambda model, fold: _score_model(model, bundle, fold, bundle.held[fold], depth, device)` (298);
  `fitting_evaluator` uguale su `bundle.fitting[fold]` (299). `_score_model` (242-261): blocchi `rows = indices[start:start+32]`,
  `x = D.view_inputs(cache, depth, rows)` (FP32, zeri oltre la lunghezza), `prediction = model(x)[:, 6:-6]`, poi
  `metrics.selection_statistics(prediction, _metric_panel(bundle, fold, rows[, clean=True]), bundle.thresholds[fold])`, infine `controls.pool_statistics`.
  `_metric_panel` (204-212): `target = (clean_targets if clean else targets)[fold][indices]`, `pose_var = pose_variances[fold][indices]`, `mask`, `std`, `meta`, `key_time_valid`.
- Stesso campo (`bundle.targets`), due copie indipendenti dopo la costruzione: vedi fatto 2.

### 1.6 `workflow._run_selection` (286-334) e `train._run_search` (`train.py:531-575`)
`run_selection` (337-340) = `T.configure_numerics()` + `_run_selection(bundle, head, control_gate=None, device, on_checkpoint)`.
`_run_selection`: `_assert_bundle` (215-222); `factory = lambda seed, fold: heads.make_head(head, seed=seed)` (296);
`fitting = [_training_set(bundle, ix, depth, fold) ...]` (297); `identity = _identity({"bundle": manifest, "depth": depth, "feature_array_sha256": ...})` (300-301);
`T._run_search(factory, fitting, evaluator, device, recipe, on_checkpoint, evaluation_identity=identity, fitting_evaluator)` (302).
`_run_search`: per `lr` in `selection.INITIAL_LRS = (1e-4, 3e-4, 1e-3, 3e-3)` (+ eventuale `1e-2` o `3e-5` da `next_learning_rate`):
due `Trajectory(factory, data, lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)`, `PairedCandidate(trajectories, evaluator, evaluation_identity)`,
`pair.advance(150, on_checkpoint)`, e `pair.advance(300, ...)` solo se `needs_epoch_extension`; `states[lr] = pair.state()`.
Poi (306-320) il punto scelto viene ripristinato con `Trajectory.restore(pair["best_states"][fold])` e ripunteggiato con `include_clean=True`,
con verifica `np.isclose(..., atol=1e-12, rtol=1e-12)` contro la curva: **modello esatto per D0**.

---

## 2. Classi di `train.py`

### 2.1 `Recipe` (27-37) e `_TestRecipe` (40-46)
`Recipe()`: `initial_epochs=150, maximum_epochs=300, last_epochs=5, batch_size=32, clip_norm=1.0, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, test_only=False`.
`_check_recipe` (49-59): una ricetta non di test deve essere `== Recipe()`; una di test deve essere `_TestRecipe`. Nessun modo lecito di cambiare
orizzonte o batch in produzione: "150 epoche fisse" = chiamare solo `advance(150)`.

### 2.2 `configure_numerics()` (62-82)
Rifiuta `CUBLAS_WORKSPACE_CONFIG` diverso da `:4096:8` e `OMP/MKL/OPENBLAS_NUM_THREADS` diversi da non impostato o `"8"`; imposta 8 thread,
1 interop, `torch.use_deterministic_algorithms(True)`, cudnn deterministico, niente benchmark, niente TF32. Va chiamata prima di inizializzare CUDA.
**[D]** sul Mac con `OMP_NUM_THREADS=1` solleva `ValueError: OMP_NUM_THREADS must be 8 ...`: in locale si usano solo `_run_selection` / le classi con `_TestRecipe`.

### 2.3 `TensorTrainingSet(features, target, support, window_ids, crop=6)` (141-208)
Tensori CPU; `_rows` = indici delle finestre con supporto, **ordinati per ID canonico** (`controls.canonical(meta)`), non per posizione nella cache (169-173);
`size` = finestre con supporto (920 e 934); `identity` (183-193) = sha256 di ids, crop, features, target, support -> **cambia se cambia il bersaglio**;
`batch(ordered_positions, device)` (195-204); `validate_production_shape()` (206-208) esige (N,160,384) e supporto (N,148,20).

### 2.4 `Trajectory(factory, data, *, lr, seed, fold, device="cpu", _recipe=Recipe())` (215-393)
- Costruttore (218-246): `fold in (0,1,2)`, `seed in (0,1,2)`; `self.model = factory(seed, fold).to(device, float32)` dentro `torch.random.fork_rng(devices=[])`;
  `initial_sha256 = heads.state_sha256(model)`; `AdamW(heads.optimizer_groups(model, 0.01), lr, betas, eps)` (decadimento solo su pesi con ndim >= 2);
  semi `heads.trajectory_seeds(seed, fold)` (`heads.py:16-21`): **inizializzazione 100000+seed (indipendente dal fold: `make_head` usa `trajectory_seeds(seed, 0)`),
  dropout 200000+100*fold+seed, shuffle 300000+100*fold+seed**. Seme 0: fold0 dropout 200000 / shuffle 300000; fold1 200100 / 300100.
  - `self.shuffle = torch.Generator("cpu").manual_seed(seme_shuffle)` -> `torch.randperm(data.size, generator=self.shuffle)` a ogni epoca (277);
  - `self.dropout_cpu` = stato di un generatore CPU col seme del dropout; **su CUDA** `self.dropout_cuda = torch.Generator(device=cuda).manual_seed(...).get_state()` (239-240).
- `_rng()` (252-266): salva l'RNG globale CPU (e CUDA del device), installa `dropout_cpu` / `dropout_cuda`, e all'uscita li rilegge e ripristina i globali.
  Su GPU le maschere del dropout escono dall'RNG globale CUDA; lo stream CPU non viene consumato **[I: letto, non eseguibile sul Mac]**.
- `steps_per_epoch = ceil(data.size / batch_size)` (248-250): 29 e 30.
- `train_to_step(target_step)` (268-306): solo in avanti, al massimo `maximum_epochs * steps_per_epoch`; a fine epoca aggiunge
  `{"epoch", "loss_sum", "window_count"}` a `epoch_losses` (`loss_sum` = somma di loss del batch x finestre del batch; media = `loss_sum/window_count`,
  in modalita' train, con dropout, a pesi che cambiano).
- `evaluate(evaluator)` (308-318): `model.eval()`, `no_grad`, dentro `_rng()`; restituisce `selection.FoldScore`.
- `state()` (320-334) / `restore(state)` (336-393): vedi 3.2 e il fatto 4.

### 2.5 `PairedCandidate(trajectories, evaluator, evaluation_identity=None)` (396-501)
Esige fold `[0, 1]`, seme 0, stesso LR e ricetta, stesso `initial_sha256`, e in produzione un'identita' di 64 cifre esadecimali (412-414).
`advance(horizon, on_checkpoint=None)` (421-445): `horizon in (150, 300)`; per ogni `(exposure, fold_steps)` di
`selection.checkpoint_schedule(steps, horizon, after_exposure=...)` (`selection.py:196-211`: esposizioni 0, 0.25, 0.5, 0.75, 1, 2, ...; passi `ceil(exposure*steps)`):
`train_to_step` sui due fold, `point = selection.pool_scores([t.evaluate(evaluator) ...], exposure, fold_steps)`; se `exposure > 0` e
`point.mse < migliore precedente` (stretto) salva `best_states = [t.state() ...]` e `best_exposure`; poi `on_checkpoint(self)`.
`curve()` (447-450); `state()` (452-459); `restore()` (461-501).

### 2.6 Un solo LR a orizzonte fisso con i pezzi esistenti (D1) — provato sulla fixture
```python
factory   = lambda seed, fold: heads.make_head("TCN_post", seed=seed)                                  # workflow.py:296
evaluator = lambda model, fold: W._score_model(model, bundle, fold, bundle.held[fold], "post", device)  # workflow.py:298  (valuta su y)
clean_bundle = dataclasses.replace(bundle, targets=bundle.clean_targets)                                # SOLO per i set di training
sets = {"y":  [W._training_set(bundle,       ix, "post", f) for f, ix in enumerate(bundle.fitting)],
        "y*": [W._training_set(clean_bundle, ix, "post", f) for f, ix in enumerate(bundle.fitting)]}
fresh = lambda s: [T.Trajectory(factory, d, lr=0.003, seed=0, fold=f, device=device, _recipe=T.Recipe()) for f, d in enumerate(s)]

# cancello di parita' (training su y): primi 5 punti della curva archiviata del LR 0.003
gate = fresh(sets["y"]);  assert [t.initial_sha256 for t in gate] == selection["initial_sha256_per_fold"]
for (exposure, fold_steps), archived in zip(S.checkpoint_schedule([t.steps_per_epoch for t in gate], 150)[:5], curve_0003["points"][:5]):
    for t, step in zip(gate, fold_steps): t.train_to_step(step)
    point = S.pool_scores([t.evaluate(evaluator) for t in gate], exposure, fold_steps)   # confronta mse, edyn, fold_steps con archived

# fit sul bersaglio pulito, 150 epoche, selezione su y
saved = {}
def keep(pair):
    if pair.points[-1].exposure in (31.0, 150.0): saved[pair.points[-1].exposure] = [t.state() for t in pair.trajectories]
pair = T.PairedCandidate(fresh(sets["y*"]), evaluator, identity)   # identity = W._identity({...}) ricalcolata == 8fcf8609…
pair.advance(T.Recipe().initial_epochs, keep)                      # e NON chiamare advance(300)
# punto scelto: pair.best_exposure, pair.best_states ; finale: pair.state()["current_states"] ; curva: pair.curve()
```
Esiti sulla fixture (`proto_driver.out`): 5/5 punti del cancello uguali bit a bit; la coppia su y rifatta con `PairedCandidate` riproduce l'archivio
fino a `initial_epochs`; fit su y\* e fit su y hanno **stesso `initial_sha256`, stesso `epoch_order_sha256`, stesso stato dell'RNG del dropout e
dello shuffle**, `dataset_sha256` diverso, pesi diversi; il set di training pulito ha target `==` `clean_targets[fold][fitting]` bit a bit e
features/supporto identici; l'ultimo punto della curva pulita coincide con un ricalcolo indipendente contro y.
Punti archiviati del cancello (LR 0.003): `(0,[0,0])` mse 2.892463349558587, edyn 1.0; `(0.25,[8,8])` 2.1452115131878022 / 0.8222974023158734;
`(0.5,[15,15])` 2.020721989079718 / 0.7844989085655731; `(0.75,[22,23])` 1.9583036504913476 / 0.7692090899847309; `(1,[29,30])` 1.9309589037808752 / 0.7606302236459029.
**[D] Diagnostica a ogni checkpoint senza toccare la traiettoria** (`proto_callback.py`): un `on_checkpoint` che chiama
`W._score_model(t.model, bundle, t.fold, idx, "post", device, include_clean=True)` su esclusi e fitting a ogni punto (curve per fold, contro y e y\*, che
l'archivio non ha) lascia lo stato finale della coppia identico bit a bit (`payload_sha256` uguale con e senza callback) e i suoi valori sugli esclusi
contro y coincidono con i punti della curva. `_score_model` mette da se' il modello in `eval`, non consuma RNG e ripristina la modalita' (`workflow.py:244-259`).
Controlli gratuiti in piu' contro gli stati archiviati: `epoch_order_sha256` dei due fold (prefisso identico), `dropout_cuda` e `shuffle_rng`
agli stessi update (31 -> `best_states`, 150 -> `current_states` del LR 0.003): valgono anche per il fit pulito, perche' non dipendono dal bersaglio **[I su CUDA]**.

---

## 3. Schema dei file

### 3.1 `calibration.pt` (= risultato di `controls.build_positive_control_fold`, salvato intero con `torch.save`; `runner.py:292-294`)
Dimostrato sulla fixture (stessa funzione, stesso salvataggio); il `calibration.json` archiviato ha le stesse chiavi tranne `components`.
```
outer_fold: int                      partition: {fitting_users: [str x20], held_users: [str x20]}   (elementi numpy.str_)
components:
  L, T, epsilon: ndarray float64 (N,160,20)     # L e T GIA' normalizzati; epsilon AR(1) a varianza 1
  normalization: {L: {mean[20], std[20], weight_sum[20], fitting_windows}, T: {...}}
  generator_states: {L: state_dict di nn.Linear(384,20), T: state_dict della Sequential larga 8}   (Tensor)
  generator_hashes: {L, T}   hidden_active_fraction_fitting: [5]   seed: 20261118 + 1000*fold   dtype: "torch.float32"   layernorm_eps: 1e-05   partition
real_reference: {MSE, E_dyn, alpha, mse, boundary_unresolved, status, curve}
calibration: {status, E_real, selected: {linear: {...}, temporal: {a, sigma, alpha, E_cal, E_cleanLS, E_oracle, oracle_gain_fraction, admissible, boundary_unresolved, alpha_curve}}, evaluations: 1936, missing_worlds, curve[1936]}
factor_diagnostics: [...]            timing: {inner_seconds, outer_seconds}        outer_scored_after_inner_selection: bool
outer_reference_statistics: {linear: {constant, oracle, clean_ls, ridge}, linear_threshold, linear_normal_residuals,
                             temporal: {constant, oracle, clean_ls}, temporal_threshold, temporal_normal_residuals,
                             linear_diagnostics: {L0: {...}, L13: {...}}}
```
Contiene array numpy nel pickle: **`weights_only=True` fallisce** (provato); si legge con `weights_only=False` dopo la verifica dell'hash, come `runner._torch_artifact`.
Reali: fold0 228.630.380 byte sha `d08a804e345e9e790a41c51f5649151b6312ddc26648c1ccacf89458d6a34bf2`; fold1 228.396.140 byte sha `441f6486b9ab9446d6e4aa7ddaac6f30ea9b3785d6cf83c7820707e90dde4c91`
(`.../pose_temponet_20261117/wls_calibration/{0,1}/calibration.pt`). Scelte reali: `a=0.4, sigma=1.3, alpha=1.0` nei due fold.

### 3.2 `selection_states.pt` (`workflow.py:331-332`)
```
schema: "pose-head-selection-states-v1"     selection_sha256: str (== selection.json)
candidate_states: {lr (chiave FLOAT: 0.0001, 0.0003, 0.001, 0.003, 0.01): stato della coppia}
  stato della coppia (PairedCandidate.state, train.py:452-459):
    schema "pose-head-paired-v1", lr, recipe (dict; betas tupla), evaluation_identity, approved_horizon (150|300),
    points: [{exposure, mse, edyn, fold_steps (tupla)}],  best_exposure: float,
    best_states: [stato fold0, stato fold1],  current_states: [stato fold0, stato fold1],  payload_sha256
  stato di traiettoria (Trajectory.state, train.py:320-334):
    schema "pose-head-trajectory-v1", recipe, lr, seed, fold, device_type ("cuda" nel vero), dataset_sha256, initial_sha256,
    numeric (dict di numerical_profile), steps_per_epoch,
    model: state_dict CPU (16 tensori FP32: norm.weight/bias, deep.{0,2,4,6,9,12}.weight/bias, affine.weight/bias),
    optimizer: {state: {0..15: {step, exp_avg, exp_avg_sq}}, param_groups: [2 gruppi]},
    updates: int, permutation: None a fine epoca (altrimenti LongTensor), position: int,
    shuffle_rng: Tensor uint8 (5056,), dropout_cpu: Tensor uint8 (5056,), dropout_cuda: Tensor (None su CPU),
    epoch_order_sha256: [str per epoca iniziata], epoch_losses: [{epoch, loss_sum, window_count}], loss_sum, loss_windows, payload_sha256
```
- `best_states` = stato al minimo della MSE pooled sugli esclusi (esposizione > 0); `current_states` = stato all'orizzonte (150 o 300). TCN: 5 LR x 2 x 2 = **20 stati**.
- Checksum: `v = dict(s); d = v.pop("payload_sha256"); T.payload_sha256(v) == d` (provato per coppia e traiettoria); i `points` della coppia `==` curva del JSON (provato).
- Dimensioni reali: TCN_post 18.386.382 byte sha `f9f4d4ad…e1bbf0`; L0 2.197.650 byte sha `7e6cf069…1bea45`; MLP 14.685.978 byte sha `bc45c2aa…9df7f49`
  (`.../controls/temporal/{TCN_post,L0,MLP}/selection_states.pt`). L0 e MLP hanno 4 curve ciascuna -> 16 stati ciascuna.
- Sul Mac (torch 2.14) si carica anche con `weights_only=True`; su torch 2.2 non provato **[I]**: usare `weights_only=False` dopo il pin.

### 3.3 `selection.json`
Chiavi (lette dall'archivio): `bundle` (manifest, 15 chiavi), `closed_form_linear_diagnostics` ({} nei controlli), `closed_form_references`, `curves`
(`[{lr, horizon, points: [{exposure, mse, edyn, fold_steps}]}]`), `evaluation_identity`, `feature_array_sha256`, `head`, `initial_sha256_per_fold`,
`lower_edge_relative_mse_gain`, `minimum_exposures`, `objective_tradeoff`, `optimization_unresolved`, `optimizer_alarms`, `positive_gate_sha256` (None),
`recipe`, `selected {lr, exposure, mse, edyn}`, `selected_checkpoint`, `selected_fitting_diagnostics {"0","1"}` (4 somme, contro y, sul fitting),
`selected_outer_statistics` / `selected_outer_clean_statistics` (`fold0, fold1, pooled`; 4 somme contro y / contro y\*), `selected_outer_values`,
`selected_outer_clean_values`, `selection_locked`, `selection_sha256` (= `T.payload_sha256` del resto, provato), `world`.
**Nelle curve ci sono solo valori pooled**: i valori per fold esistono solo per il punto scelto. Anche il log (920 eventi `selection_checkpoint`) e' pooled.

### 3.4 Ricaricare uno stato per la sola inferenza
```python
v = dict(state); digest = v.pop("payload_sha256"); assert T.payload_sha256(v) == digest      # come proto_driver.load_head
model = heads.make_head("TCN_post", seed=0).to(device=device, dtype=torch.float32)
model.load_state_dict(state["model"], strict=True); model.eval()
observed, clean = W._score_model(model, bundle, fold, indices, "post", device, include_clean=True)
```
Via rigorosa (stesso contesto, come `workflow.py:309-312`): `t = T.Trajectory(factory, W._training_set(bundle, bundle.fitting[fold], "post", fold), lr=lr, seed=0, fold=fold, device="cuda", _recipe=T.Recipe()); t.restore(state)`.
Sulla fixture le due vie danno statistiche identiche bit a bit; `restore` con un numero di thread diverso viene rifiutato.

---

## 4. Funzioni di punteggio da riusare

| funzione | firma | cosa restituisce |
|---|---|---|
| `metrics.selection_statistics` (`metrics.py:161-166`) | `(prediction, data, dynamic_threshold)` | `{mse_sum, window_count, dynamic_sse, dynamic_target_ss}` |
| `metrics.evaluate` (78-158) | `(prediction, data, dynamic_threshold, *, batch_windows=64, per_joint=False)` | gruppi `all/dynamic/static`, per utente, `centered` con `varp, vary, cov, centered_mse_ratio, variance_ratio, pooled_centered_correlation` |
| `workflow._score_model` (242-261) | `(model, bundle, fold, indices, depth, device, *, include_clean=False)` | le 4 somme (o la coppia osservato, pulito) pooled sui blocchi da 32 |
| `workflow._metric_panel` (204-212) | `(bundle, fold, indices, *, clean=False)` | pannello per `selection_statistics` |
| `controls._metric_panel` (`controls.py:540-550`) | `(data, indices, observed, clean)` | pannello con `pose_var` ricalcolata da `clean` |
| `controls.pool_statistics` (415-418), `statistic_values` (421-423) | | somma delle 4 somme; `MSE = mse_sum/window_count`, `E_dyn = dynamic_sse/dynamic_target_ss` |
| `controls.evaluate_outer_references` (553-587) | `(data, components, fit, held, selected, *, block_windows=4, include_l13_diagnostics=True)` | `constant/oracle/clean_ls` (+`ridge` nel lineare) sugli esclusi, soglie, `linear_diagnostics` L0/L13; richiede `data["lat"]`; L13 e' pesante |
| `controls._recovery` (251-260), `temporal_head_gate` (302-318) | | `R = (E_cleanLS - E_TCN)/(E_cleanLS - E_oracle)`, soglia 0,50 |

Definizioni (`scripts/pose_dynamics.py:88-146`): `mse_sum` = somma sulle finestre con supporto di (SSE sul supporto centrale / elementi supportati);
`dynamic_sse` = somma, sulle finestre con `pose_var >= soglia` e sui giunti con >= 2 campioni, di `sum_t ((p - media_p) - (y - media_y))^2` sul supporto centrale;
`dynamic_target_ss` = `sum_t (y - media_y)^2` sugli stessi elementi. Contro y\* cambia solo `target` del pannello: stesse finestre dinamiche.

**y, y\*, L, T, eps dal `calibration.pt`** — `[D sulla fixture]`:
`c = artifact["components"]; ch = artifact["calibration"]["selected"]["temporal"]; ystar, y = controls.synthetic_target(c, ch["a"], ch["sigma"])`.
- y e y\* si ricostruiscono **bit a bit** (aritmetica float64 elemento per elemento su array salvati): uguali a `bundle.targets/clean_targets` e agli hash
  `target_sha256` / `clean_target_sha256` del `selection.json`. `L + a*T` manuale `==` y\*.
- **`y - y*` non e' bit a bit `sigma*eps`** (scarto massimo 1,78e-15): il rumore si prende da `sigma * c["epsilon"]`.
- L e T sono gia' normalizzati (media 0, varianza 1 pesata sul supporto centrale del fitting; `controls.normalize_component`, 89-113); media e scala in `c["normalization"]`.
- I riferimenti chiusi sugli esclusi sono gia' in `artifact["outer_reference_statistics"]["temporal"]`; l'oracolo su qualunque insieme si ricalcola senza fit:
  `M.selection_statistics(bundle.clean_targets[f][idx][:, 6:-6], W._metric_panel(bundle, f, idx), bundle.thresholds[f])` (sulla fixture coincide con l'archivio).
- Parte affine e parte profonda della testa dall'esterno: `h = model.norm(x.float()); lin = model.affine(h); deep = model.deep(h.transpose(1,2)).transpose(1,2)`;
  `lin + deep == model(x)` bit a bit (provato); si punteggiano contro `L` e `a*T` sostituendo `target` nel pannello (`proto_split.py`).

Ricalcolo del cancello dai JSON archiviati **[D]**: R = 0.24173052888786836 / 0.1887545039561508 / 0.21710109906955868 (fold0 / fold1 / pooled), identico a
`positive_controls_gate_20261121.json`. E_cleanLS 0.7104311505676322 / 0.7499203961102027 / 0.7284790286546209; E_oracle 0.6534505840205822 / 0.6911028974346952 / 0.6706589239838878;
E_TCN 0.6966572080798834 / 0.7388183283237657 / 0.7159262203822878.

Bersagli di parita' per D0 (TCN_post, pooled; esposizione migliore `fold_steps` / finale): 1e-4: 181 `[5249,5430]` mse 1.8181615643874114 edyn 0.7256444669656075 / 300 `[8700,9000]` 1.8268597285134838, 0.7281499412886786;
3e-4: 64 `[1856,1920]` 1.8174307513426158, 0.725382121932922 / 150 1.837209816099995, 0.731909906310411; 1e-3: 31 `[899,930]` 1.8139640289997057, 0.722990837297975 / 150 1.8860919742547864, 0.7514958336696763;
3e-3: 31 `[899,930]` 1.8001448213440414, 0.7159262203822878 / 150 `[4350,4500]` 1.8521482905919395, 0.7415560124949256; 1e-2: 93 `[2697,2790]` 1.8492724749293896, 0.7373841024585337 / 150 1.8522916932246416, 0.7376106840187479.
Per fold: solo il punto scelto (`selected_outer_statistics`, `selected_outer_clean_statistics`, `selected_fitting_diagnostics`).

---

## 5. Insidie

1. **[D] Inizializzazione e torch**: vedi fatto 1. Corollario: anche `torch.randperm` e l'ordine dei batch vanno considerati riproducibili solo su torch 2.2/x86 **[I]**.
2. **[D] Blocchi di valutazione**: su CPU i bit FP32 della testa dipendono dalla dimensione del blocco (blocchi da 1-7 uguali fra loro; da 16/19/20 diversi dai
   primi, scarto massimo 1,01e-6). Nel vero 946 % 32 = 18 e 961 % 32 = 1: l'ultimo blocco ha 18 finestre o una sola. Stessi indici, stesso ordine, `BATCH = 32`.
3. **[D] Somme in ordine diverso**: il denominatore dinamico del fold1 vale 2698827.390194636 nei riferimenti (una chiamata) e 2698827.3901946363 nella testa
   (blocchi da 32). I confronti fra denominatori di sorgenti diverse si fanno con tolleranza; `==` solo contro il `selection.json`.
4. **[D] Contesto numerico**: `Trajectory.restore` confronta `numerical_profile()` (versione di torch, thread, flag) e `device_type`; confronta anche i
   `param_groups` dell'ottimizzatore (`train.py:373-378`), le cui chiavi dipendono dalla versione di torch (su 2.14 c'e' `decoupled_weight_decay`;
   su 2.2 non verificato). `P.verify_numeric(plan, "controls", P.numeric_context("gpu"))`
   esige 8 thread ovunque e il digest `gpu8` del piano: e' il cancello giusto per dire "stesso contesto del job originale".
5. **[D] Ordine degli import**: `import pose_temponet.provenance` carica numpy, poi torch e scipy (`data.py:8` -> `consolidation_long_pose.py:14-15`); sklearn entra con
   `numeric_context()`. Il driver importa per primo il pacchetto congelato, poi chiama `configure_numerics()` e `numeric_context("gpu")`, come `runner.py:222-224`.
6. **[D] `weights_only`**: `calibration.pt` richiede `False`. In torch 2.2 il default e' `False`, dal 2.6 `True`: passarlo sempre esplicito.
   `torch.load(mmap=True)` non e' usato dal pacchetto e non serve (gli array sono numpy nel pickle); se lo si usa, il percorso va passato come `str`.
7. **[I] numpy 1.26.4 (cluster) contro 2.3.5 (Mac)**: i pickle del cluster si leggono sul Mac; il verso opposto (file con array numpy scritti sul Mac e letti
   sul cluster) non e' provato: le fixture si generano nell'ambiente che le legge.
8. **[D] Proprieta' CUDA**: `numeric_context("gpu")` usa solo `name, major, minor, total_memory` (`provenance.py:522-523`); niente `uuid`.
9. **[D] `fork_rng`**: `make_head` e `Trajectory.__init__` isolano l'RNG CPU (`fork_rng(devices=[])`); `make_head` lascia intatto l'RNG globale (provato).
   `controls.generate_components` usa invece `torch.manual_seed` dentro il fork (`controls.py:143,147`): vedi difetto N-1. Il driver non ha bisogno di richiamarla.
10. **[D] dtype/dispositivo**: latenti FP16 sull'host, FP32 nel batch; bersaglio float64 sull'host, FP32 nella loss, float64 nel punteggio;
    predizioni FP32 -> numpy -> float64 (`metrics._cropped`). Le feature oltre `n_patches` sono azzerate sia in training (`workflow.py:233-234`) sia in valutazione (`data.py:136`).
11. **[D] Manifest dei sorgenti**: `P.source_manifest` include `src/**/*.py`, `scripts/*.py` (non ricorsivo), `scripts/pose_temponet/**/*.{py,sh,sbatch}`,
    `scripts/slurm/pose_temponet*.sbatch`, `lib_code_dir.sh`, `campaign_checkout.sbatch`. Un driver messo in `scripts/*.py` o in `scripts/pose_temponet/`, o un launcher
    chiamato `scripts/slurm/pose_temponet*.sbatch`, cambia il manifest e fa fallire `validate_plan` ("reader sources changed"). Metterli altrove
    (per esempio `scripts/pose_temponet_diag/` e `scripts/slurm/tcn_diag_*.sbatch`). Provato: `P.source_manifest(tree_e8c98244)` == manifest del piano (247 file, `9aaf1c8b…ec67ea`).
12. **[D] Niente funzioni pubbliche del runner**: `run_stage` scrive e rifiuta l'esistente; `run_search` / `_run_search` percorrono tutti i LR (`selection.INITIAL_LRS`);
    non si patcha lo stato dei moduli congelati (nel prototipo `W.BATCH` e' stato cambiato solo per misurare l'effetto dei blocchi).
13. **[I] Input su SCRATCH**: riferimento dei target e `pose_stats.json` sono su `$SCRATCH`; `read_verified_inputs` fallisce se sono stati cancellati o cambiati.
14. **[D] Finestre senza supporto**: 41 e 12 finestre di fitting non entrano nel training (`TensorTrainingSet._rows`) ma restano negli indici passati a
    `_score_model`, dove non contano nella MSE (`window_count` 920 / 934).

---

## 6. Fixture dei test riusabili

| file | cosa offre | uso per il driver |
|---|---|---|
| `tests/cpu/test_pose_temponet_e2e_chain.py:28-59` `_cache(users, windows, split, rng, weights)` | cache 160x384 FP16 a basso rango, lunghezze 150/160, su cui le calibrazioni VERE si qualificano | **la migliore**: 40 utenti x 1 finestra -> `build_positive_control_fold` (35,7 s per due fold) -> `_prepare_control_folds(expected_user_count=40)` -> `_run_selection(..., "TCN_post", device="cpu", recipe=_TestRecipe(3,6,1,4))` (11,0 s). E' `proto_fixture.py` |
| idem `:62-87` `_cpu_execution(monkeypatch)` | come far girare i punti d'ingresso pubblici su CPU (ricetta, device, `configure_numerics`) | solo se si vuole passare da `run_stage` |
| `tests/cpu/test_pose_temponet_workflow.py:30-100` `cache()` (30), `bundle()` (47), `recipe()` (54), `artifacts(b)` (75), `control_bundle(b, aa, world)` (98) | cache 8 finestre / 4 utenti con forme vere; componenti e riferimenti FABBRICATI (niente calibrazione) | prova rapidissima del cablaggio; `expected_user_count=4`, `_TestRecipe(1,2,1,4)` |
| idem `:312-343` | `W._training_set` + `T.Trajectory(make_head("TCN_post"))` con ricetta di test | esempio minimo di training vero su CPU |
| `tests/cpu/test_pose_temponet_train.py:17-69` `TinyHead`, `factory`, `dataset()`, `evaluator_for()`, `trajectory()` | forme piccole (16x4, 2 giunti) | meccanica di `Trajectory` / `PairedCandidate` / callback / restore |
| `tests/cpu/test_pose_temponet_scientific_integration.py:27-56` `manual_statistics` | oracolo indipendente delle 4 somme | controllo incrociato dei numeratori/denominatori del driver |
| `tests/cpu/test_pose_temponet_controls.py:145-160` `_builder_fixture()` | piccola costruzione dei controlli | regressione dei componenti |

`pytest` sui sei file train/workflow/controls/scientific_integration/heads/selection del commit: **69 passati** in 20,08 s (`pytest_subset.out`).

---

## 7. Scheletro del driver su Leonardo (proposta, da rivedere prima dell'uso)

```python
from pose_temponet import provenance as P, execution as E, runner as R      # PRIMA di numpy/torch
from pose_temponet import controls as C, data as D, heads, metrics as M, selection as S, train as T, workflow as W
import dataclasses, json, numpy as np, torch

T.configure_numerics(); numeric = P.numeric_context("gpu")
plan = P.read_json_pin({"path": PLAN, "sha256": "b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3"})
P.validate_plan(plan, require_review=True); P.checkout_identity(plan); P.verify_numeric(plan, "controls", numeric)
receipts, pins = R._ancestors(plan, E.find_job(plan, "TCNH-selection-L0-20261120"))   # come verify_positive_controls_readonly.py: sola lettura
calibrated = R._calibrations(receipts)
rows = {(R._json_artifact(r, "selection")["world"], R._json_artifact(r, "selection")["head"]): r for r in R._stage_receipts(receipts, "controls")}
selection = R._json_artifact(rows[("temporal", "TCN_post")], "selection"); states = R._torch_artifact(rows[("temporal", "TCN_post")], "states")
previous = R._single(receipts, "preflight")["details"]["preflight"]
cache = P.read_verified_inputs(plan, "train", stage="controls")
bundle = W.prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=plan["producers"]["train"]["users"],
                                 source_identity=R._input_identity(plan), expected_folds=previous["train"]["folds"])
assert dict(bundle.manifest) == selection["bundle"]                      # cancello 1: dati, partizioni, y, y*, soglie
# cancello 2: W._array_hash(W._features(cache, "post")) == selection["feature_array_sha256"] e identity == selection["evaluation_identity"]
# D0: per lr, kind in (best_states, current_states), fold: Trajectory.restore (o make_head+load_state_dict) -> W._score_model(..., include_clean=True)
#     su bundle.held[fold] e bundle.fitting[fold]; parita' pooled con curve e per fold con il punto scelto; epoch_losses dagli stati.
# D1: sezione 2.6.  Uscite con P.atomic_json / P.atomic_torch sotto una radice NUOVA (non OUTPUT_ROOT del piano).
```
Costo **[I, stima dai tempi archiviati]**: il job originale ha impiegato 4.383,3 s per 900 coppie-epoca (300 + 4 x 150) piu' caricamenti, cioe' circa 4,9 s per coppia-epoca:
D1 (150 epoche su y\* + un'epoca di cancello) circa 12-13 minuti di GPU; rifare per intero anche la curva su y altri 12 minuti circa; D0 (20 + 16 + 16 stati, due
insiemi, due bersagli) pochi minuti; caricamento e hash degli input 2-4 minuti. Il preventivo si fa sul limite richiesto (GPU x walltime).

## 8. Che cosa NON ho potuto verificare
- I file veri (`calibration.pt`, `selection_states.pt`, cache) non sono sul Mac: lo schema reale e' dedotto dal codice che li scrive e provato su una fixture
  generata dalle stesse funzioni; `device_type="cuda"`, `dropout_cuda` e il profilo `numeric` di torch 2.2 non li ho visti.
- Nessuna prova su CUDA ne' su torch 2.2: stream del dropout su GPU, riproducibilita' bit a bit fra job diversi, `weights_only=True` su 2.2.
- Non ho verificato che `a1ae9607…` sia antenato del commit in cui vivra' il driver (`checkout_identity` lo richiede): la copia non ha `.git`.
