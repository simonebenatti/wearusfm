# Reviewer DESIGN - notes (review of GPT's temporal-control diagnosis, commit e7670758)

Started 2026-11-21 18:52 CET. Checkout (read-only): /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag
Rules: no cluster, no job, no push; nothing modified in the checkout.

## Task 0 - setup
- Notes file created before any reading.

## Task 1 - Feasibility of D0 from the code (verified by reading code + archived JSON; 18:52-18:58)

States that exist:
- train.py:531-575 `_run_search`: for EVERY learning rate `states[lr] = pair.state()` (line 556); `PairedCandidate.state()` (452-459)
  stores `best_states` (pair of fold states at the pooled-MSE minimum of THAT lr, line 441-443) and `current_states` (pair at the
  end of the horizon). workflow.py:331-334 writes `{"schema","selection_sha256","candidate_states"}` as `selection_states.pt`;
  runner.py:197-205 `_save_results` -> `P.atomic_torch`.
- Receipt `results/infra/pose_temponet_execution_20261121_A1/launcher/TCNH-controls-temporal-TCN_post-20261120.json`
  `.artifacts.states`: path `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117/controls/temporal/TCN_post/selection_states.pt`,
  bytes 18386382, sha256 f9f4d4ad1a59f117b05d8ed411daeefb2bef68709f611f368e379e65a1e1bbf0. Size is consistent with
  5 lr x (2 best + 2 current) trajectory states of a 72,904-parameter model + AdamW moments (~0.9 MB each).
- Counters (selection_temporal_TCN_post.json `.curves[lr=0.003].points`): selected exposure 31 fold_steps [899, 930];
  final exposure 150 fold_steps [4350, 4500]. GPT's 899/930 and 4350/4500 are CORRECT. Steps/epoch 29 and 30
  (920 and 934 supported fitting windows, batch 32). All 5 lr: 1e-4 (horizon 300, best 181, final [8700,9000]),
  3e-4 (best 64), 1e-3 (best 31), 3e-3 (best 31), 1e-2 (best 93); each has best+current.
- Each trajectory state (train.py:320-334) carries model, optimizer, updates, RNG states, dataset_sha256, initial_sha256,
  numeric profile, payload_sha256.

Re-deriving y, y*, masks, thresholds, partition:
- runner.py:292-296: `calibration.pt` = full result of `controls.build_positive_control_fold` INCLUDING `components`
  (L, T, epsilon float64, normalization, generator_states, hashes); the JSON drops only `components`.
  Receipts: wls_calibration/0/calibration.pt 228630380 bytes sha d08a804e...; /1/calibration.pt 228396140 bytes sha 441f6486...
- workflow.py:168-195 `_prepare_control_folds` replays them: `controls.synthetic_target(artifact["components"], a, sigma)`
  (controls.py:193-198), threshold recomputed from y* of the fitting half and REQUIRED equal to the closed-form one (183-185),
  pose_var from y*, support from `D.central_support`, folds from `D.alternating_user_folds`; the manifest stores
  `target_sha256`, `clean_target_sha256`, `thresholds`, `support_sha256`, fold indices and is locked (`bundle_sha256`).
  The archived selection JSON contains that manifest (`.bundle`), so D0 can assert manifest equality = exact identity of
  y, y*, masks, crop, thresholds, partition. Nothing has to be regenerated (no new noise, no new teacher).

Existing scoring functions:
- workflow.py:242-261 `_score_model(model, bundle, fold, indices, depth, device, include_clean=False)`; with include_clean=True it
  returns (observed stats, clean stats); the clean panel = `_metric_panel(..., clean=True)` (204-212): target y*, SAME pose_var
  (from y*), SAME threshold. `selected_outer_clean_statistics` were produced exactly so (workflow.py:308-316) on `bundle.held`.
- Fitting half: `fitting_evaluator` (workflow.py:299) = `_score_model(..., bundle.fitting[fold], ...)` WITHOUT include_clean
  -> the fitting CLEAN error was never computed (GPT's statement is correct). Calling the same function with
  `bundle.fitting[fold]` and include_clean=True gives it; `indices` is a free argument.

New code D0 needs (none of the scoring maths is new):
- a new entry/stage: runner.run_stage only knows the frozen STAGES (provenance.py:49-50) and the frozen job graph
  (execution.py job_graph/validate_graph: max 40 GPU jobs / 7 CPU, `retry: forbidden_without_new_decision`), and refuses an
  existing output directory (runner.py:220-221). D0 cannot be run through the existing plan.
- pinned reads of 2 calibration.pt + selection_states.pt + selection.json + train cache (+ target_reference_train and
  pose_stats, which `read_verified_inputs` opens), loop over lr 0.003 {best,current} x fold x {held,fitting}, both targets,
  reproduction asserts, JSON + receipt.
- `provenance.source_manifest` (236-248) covers scripts/*.py, src/**/*.py and scripts/pose_temponet/**: a new file there
  changes source_sha256 (9aaf1c8b...), so the old receipts cannot be re-verified through `verified_receipt`/`checkout_identity`
  with the new tree -> explicit pins of the old artifacts are needed (GPT says so in section 6, last paragraph).

What can be reproduced first, and limits:
- selected checkpoint: held observed per fold + pooled (`selected_outer_statistics`), held clean (`selected_outer_clean_statistics`),
  fitting observed per fold (`selected_fitting_diagnostics`): full sums archived -> exact check possible.
- FINAL checkpoint (epoch 150): only the POOLED curve point is published (mse 1.8521482905919395, edyn 0.7415560124949256,
  `.curves[3].points[-1]`); the log has 920 `selection_checkpoint` events, all pooled. No per-fold value exists for epoch 150,
  so "reproduce the published score of point 150" can only be a pooled check. GPT's text does not say this.
- Tolerance is unspecified. In-run check was np.isclose(atol=rtol=1e-12) on the same GPU process (workflow.py:318-320).
  `Trajectory.restore` (train.py:341-346) demands the same `device_type` ("cuda") and the same `numerical_profile()`
  (torch 2.2.0a0+git6c8c5ad, 8 threads, CUBLAS :4096:8): a CPU or Mac re-score cannot go through `restore` and cannot be
  bit-identical; loading `state["model"]` into `heads.make_head` works anywhere but then the comparison needs a declared tolerance.

Cluster: YES. Nothing is on the Mac (searched ~/wearusfm_local and the repo: no selection_states.pt / calibration.pt).
  Inputs on Leonardo: selection_states.pt 18.4 MB + 2 calibration.pt 457 MB ($WORK), train cache 474 MB
  (`/leonardo_work/.../consolidation_long_20261105/49046/pose/train/train.npz`), target_reference_train 258 MB and pose_stats on
  $SCRATCH (plan_controls.json `.inputs`). Login node = 10 min CPU, no GPU. => D0 is a Slurm job (1 GPU for bit-level
  reproduction), with logical ID, ledger reservation, walltime and a budget line. GPT's "zero GPU-ore e nessun job" (section 7)
  refers to the analysis already done, not to D0; D0's cost is listed as "missing" but the need for an ID/budget line/decision
  is not stated.

### Task 1 addendum - CPU prototype (~19:02), script wf3_design_work/proto_d0_d1.py (fixtures of tests/cpu/test_pose_temponet_workflow.py, real TCN_post head, _TestRecipe)
Command: PYTHONPATH=<checkout>/src:<checkout>/scripts OMP_NUM_THREADS=1 ... python proto_d0_d1.py <checkout>
Output: "lrs with states: [0.0001, 0.0003, 0.001, 0.003] | keys per lr: [... 'best_states', 'current_states' ...]";
"D0 reproduces: held observed True | held clean True | fitting observed True | final pooled point True";
"D0 new numbers exist (fitting clean, selected): [...]"; "published per-fold values for the FINAL state in selection JSON: []".
=> D0 is composable from heads.make_head + load_state_dict(state['model']) + workflow._score_model(include_clean=True);
   no new scoring code; only pooled reproduction is possible for the final state.

## Task 2 - D1 design audit (part 1: mechanics, verified)
- Training target is bound in workflow.py:225-239 `_training_set`: `torch.from_numpy(np.asarray(bundle.targets[fold][indices]))`.
  Selection target is bound in workflow.py:298 (evaluator -> `_score_model` -> `_metric_panel(bundle, fold, rows)` -> `bundle.targets`).
  Both read the SAME field `bundle.targets`; there is no switch. D1 needs a training set built from `bundle.clean_targets`
  (a new argument or a new builder) while the evaluator keeps `bundle.targets`.
- `train._run_search` (531-575) hard-codes the lr loop (`selection.INITIAL_LRS`, `next_learning_rate`) and the automatic
  150->300 extension; "one lr, 150 fixed" cannot be obtained from it. It can be composed from `Trajectory` +
  `PairedCandidate.advance(recipe.initial_epochs)` (no call to `needs_epoch_extension`).
- Per-fold / clean / fitting curves: `PairedCandidate.points` holds pooled CurvePoint only; extra records must come from the
  evaluator closure (side list). Evaluation runs in eval mode inside `Trajectory.evaluate` -> no RNG draw.
- Prototype (same script): "D1 same init: True | same batch orders: True | same dropout RNG end state: True | same shuffle RNG
  end state: True"; "composed noisy path == frozen curve: True"; dataset identity differs (old states are not restorable into
  the clean-target trajectories, as expected).
  => "same initialisation, same batches, same dropout masks" is achievable EXACTLY for the whole run, not only initially;
  and a parity gate is available: the D1 driver run with target=y must reproduce the published A1 lr-0.003 curve points.
- Cost (TCNH-gpu-benchmark-20261120.json `.details.benchmark.cells.TCN_post`): train median 1.166/1.168 s per fold-epoch,
  held eval INCLUDING the clean score 1.352/1.371 s. An evaluation costs more than a training epoch. D1 with held+fitting
  readings per checkpoint: ~ (1.17 + 1.36 + 1.33) s x 300 fold-epochs ~ 1,160 s ~ 0.32 GPU-h before start-up, reading the
  two 228 MB calibration.pt, quarter-epoch points and margin; GPT's proportional 0.205 GPU-h is ~1.6x too low for the
  design it describes (GPT says it is not a preventive).

## Task 2 - part 2: numbers computed from the archived JSON (~19:04) - launcher/calibration_{0,1}.json, selection_temporal_{TCN_post,L0,MLP}.json
(sv = sums -> ratios; all on the HELD half scored on y unless stated)
- margin cleanLS-oracle: fold0 0.056981, fold1 0.058817 (denominator of the gate; D1 keeps it).
- Price of noise for the LINEAR part alone (gate baseline is fitted on the CLEAN target, the heads on y):
  closed-form WLS L0 fitted on y is behind cleanLS by 0.010884 = 19.1% of the margin (fold0) and 0.009634 = 16.4% (fold1);
  neural L0 by 13.6% / 10.8% (its "recovery" is -13.55% / -10.82%), MLP -12.24% / -10.65%; WLS L13 on y by 85.6% / 69.5%.
  => from the best noisy-trained linear reader (neural L0) to the oracle the TCN recovers 33.2% / 26.8%, and the 50% gate
     corresponds to 56.0% / 54.9% of that distance. D1 (clean fit) removes BOTH the linear-part and the temporal-part noise
     cost: it is like for like with cleanLS, not a one-factor change relative to what the gate measures.
- Clean scale: TCN held clean E_dyn 0.127856 / 0.154431; cleanLS clean E_dyn ~0.1651 / ~0.1916 (= margin SSE / clean target SS,
  exact up to a zero-mean cross term because cleanLS never sees the noise); the 50% gate ~ clean E_dyn <= ~0.083 / ~0.096.
- Noise x error cross term of the TCN at the selected checkpoint (oracle_sse + clean_sse - noisy_sse): +2915.8 = +1.6% of the
  margin SSE (fold0), -817.3 = -0.5% (fold1). => on held the noisy score is the clean error plus a constant to within ~1-2
  points of recovery: selecting the epoch on noisy held MSE while training on y* is sensible (it tracks the clean held error).
- Fitting->held gaps of E_dyn on y: oracle -0.0009 / -0.0021; cleanLS +0.0018 / +0.0003; WLS L0 on y +0.0156 / +0.0137;
  WLS L13 on y +0.0919 / +0.0822; TCN +0.0348 / +0.0282. The noise-free linear fit transfers to the other users with NO gap;
  the TCN gap is ~2.2x that of the 385-feature WLS and ~0.36x that of L13: it has the size expected from noise memorisation and
  is not by itself evidence of poor cross-user generalisation (GPT hedges this; D0's clean fitting error is what decides).
- Fitting references (fold0/fold1): cleanLS 0.708654 / 0.749626, oracle reconstructed 0.654332 / 0.693246, TCN 0.661873 /
  0.710583 -> apparent recovery 86.12% / 69.25% (GPT's numbers reproduced).
- T is mostly NOT linearly readable: T_unread_fraction_fitting L0 0.7117 / 0.8788, L13 0.5163 / 0.7057; L13 fitted on the CLEAN
  target recovers in-sample only 27.09% / 19.93% of the margin. The 50% gate is above what any 13-patch linear reader can
  represent even without noise: it requires learning the non-linear part at a=0.4 vs sigma=1.3.

## Task 3 - protocol compliance (~19:04)
- Protocol docs/pose_temponet_frozen_protocol_20261117.md: line 104 "Ogni condizione richiesta non superata ferma prima dei fit
  reali; nessuna etichetta scientifica negativa o ricerca finché si vince. Eventuale modifica della specifica richiede decisione
  registrata."; line 158 "Un esito negativo dei gate torna alla decisione pertinente, non autorizza mutazioni."; line 186
  "La riserva copre avvii/sforamenti contabilizzati, non retry. [...] Una nuova diagnosi/ripartenza richiede decisione
  registrata."; line 219 "Superamento tetti, cambio scientifico o retry richiedono nuova decisione"; line 106 "gli stessi held
  outer scelgono LR/epoca e giudicano recupero, quindi la qualifica è ottimista".
- docs/decisioni.md (end of file, entries of 21/11): gate recorded ("Arresto scientifico prescritto, non attesa di
  autorizzazione." "Per riprendere serve diagnosi e una consegna distinta se cambia la ricetta."); new entry
  "Diagnosi TCN: seguito proposto, REVIEW_REQUIRED" quotes Simone's request ("perché la TCN non passa il controllo temporale?
  Analizza cosa non funziona e come possiamo risolverlo, poi preparami il prompt per farlo rivedere a claude") and lists D0/D1 as
  proposals. NOT registered: an authorisation to run D0/D1, a budget line, an ID, frozen reading rules.
  docs/claude_GPT_shared: sections 1 and 2 entries of GPT chat 2 (same content; "D0/D1 sono proposte train-only, non invii").
- The diagnosis says "Il residuo del budget A1 non trasforma il gate scientifico negativo in un via a prove diverse" and that
  allocations/budget must be published before release, but never says that a registered decision of Simone is required before
  D0 and D1 (grep for "decisione"/"Simone"/"autorizz" in the document: no such sentence).
- Post-hoc elements: gate, denominators, world, selection metric unchanged. Declared deviations: lr fixed post hoc (0.003),
  no 150->300 extension. The follow-up (one more parsimonious/regularised variant, then "ripetere il controllo originale") IS a
  change of specification made after seeing the failure on this world.
- Circularity: generator and noise seeds are hard-coded (controls.py:141 `base_seed = 20261118 + 1000*outer_fold`, +1, +2),
  folds are the same two alternating halves; the same held halves already chose lr/epoch in A1 (5 lr, 154-304 checkpoints),
  would be read again by D0 and D1, and would then confirm the variant. No fresh world, no fresh noise, no declared number of
  attempts or stop rule if the variant fails. Not protected.

## Task 2 - part 3 (19:08): linear-world decomposition and reading of GPT's table
- Linear world (a=0, sigma=1.35; selection_linear_*.json + calibration_*.json `.outer_reference_statistics.linear`), held E_dyn:
  oracle(=clean_ls) 0.722484 / 0.763721, ridge 0.729602 / 0.770106, TCN_post 0.733731 / 0.774719 (lr 0.003, epoch 28),
  neural L0 0.730603 / 0.770990, MLP 0.735735 / 0.775942.
  => with NO temporal signal the TCN recipe ends 0.011247 / 0.010998 above the oracle = 19.7% / 18.7% of the temporal margin
     (ridge 12.5% / 10.9%; neural L0 14.2% / 12.4%).
  Approximate additive reading of A1 (temporal world TCN is 0.043206 / 0.047715 above the oracle): temporal signal captured
  ~44% / ~38%, noise price ~20% / ~19%, net 24.2% / 18.9%. To pass 50% the TCN must capture ~70% / ~69% at its present noise
  price, or ~62% / ~61% if its noise price fell to the closed-form ridge's. A more parsimonious/regularised variant can gain
  at most ~7-8 points through the noise price; the missing 26-31 points need MORE signal captured under noise, which
  regularisation does not provide by itself. D1 does not measure any of this.
- Rows of GPT's table (section 5):
  row 1 (clean better on held AND recovery >=50% x3): can conclude that architecture+optimiser+20 users suffice without noise
        (excludes optimisation/implementation and pure cross-user shift as sufficient causes). Cannot conclude that any
        regularisation/capacity variant reaches 50% under noise, nor that 50% is attainable under noise at this n. The stated
        consequence ("priorità a ... complessità/regolarizzazione") does not follow from the row.
  row 2 (clean error much lower on fitting than held): "molto inferiore" undefined; confounded with lr 0.003/150 epochs/dropout/wd
        chosen under noise and with selection on noisy held MSE; single seed. cleanLS has no cross-user gap (0.0018/0.0003).
  row 3 (clean fitting poor OR still descending/boundary): a noise-free regression at constant lr is EXPECTED to be still
        descending at 150 epochs; a pass at the boundary matches rows 1 and 3 at once; dropout makes the exact teacher a
        non-stationary point of the training loss (see Task 4 probe) so a clean fitting residual is not an optimisation defect.
  row 4: catches every intermediate outcome (e.g. 35-49% without a large fitting/held gap), with no consequence.
  No precedence between rows, no quantity that decides them, reference for "fitting pulito scarso" undefined
  (cleanLS on y*? zero?). A decidable version exists with the SAME arithmetic: fitting-side recovery
  (E_cleanLS,fit - E_TCN,fit)/(E_cleanLS,fit - E_oracle,fit) and held recovery, both >= or < 0.50.
- A priori most likely: row 1 or an intermediate improvement (my judgement, not verified: the noisy-trained TCN already has
  clean held E_dyn 0.128/0.154 vs ~0.083/~0.096 needed, and the linear part transfers across users without gap).
- Free information GPT's D0 does not list: every `current_states` entry carries `epoch_losses` (train.py:303-305, 330), i.e. the
  per-fold train-mode loss on y for all epochs of all 5 lr; and all 20 saved states (5 lr x best/current x 2 folds) can be
  scored for ~1.4 s each.

## Task 4 - constructive proof (19:05-19:07), script wf3_design_work/embed_check.py and embed_redundant.py
- oracle_embedding.py: builds the student with `heads.make_head("TCN_post", seed=0).eval()` (unmodified head, 72,904
  parameters), teacher and target from `controls.generate_components` + `controls.synthetic_target(c, 0.4, 0.0)`; copies the
  teacher convs into the first 8 channels of the 6 student convs, zeros elsewhere; output layer and affine branch absorb the
  per-joint mean/std normalisation and a; forward through the head's own LayerNorm; compares on all 160 steps and on [6:-6].
- My re-derivation: "fold 0: LN at init (1.0, 1.0, 0.0); eval rmse_central 5.004e-07 max 6.405e-06; clean rms 1.0695;
  a*T rms 0.3931" and fold 1 "5.021e-07 max 4.354e-06" = GPT's JSON. "LN gamma x1.1 -> rmse 1.180e-01": LN is in the path, not
  bypassed; normalisation is inside the head parameters, not outside. Non-zero channels per reader conv [8,8,8,8,8,20] of
  [64,64,64,64,32,20].
- Limits: inputs are 8 windows of Gaussian float16 noise, not real latents (fine for an algebraic identity); teacher weights
  are regenerated on torch 2.14 (Mac), the real controls used torch 2.2.0a0 and the generator hashes are inside `components`,
  which calibration.json drops: identity with the REAL teachers is not checked (irrelevant for representability).
- What it does not show, measured: in TRAIN mode (dropout 0.1 active, the mode in which the loss is minimised) the same weights
  give "train-mode (dropout on) rmse mean 0.3543 = 0.90 x rms(a*T); as fraction of a*T variance 0.813" (fold0) and 0.599 (fold1);
  "loss/grad-norm at embedded point: eval 2.512e-13/1.632e-07; train(dropout) 1.117e-01/6.009e-01". With the most redundant
  exact embedding (8 copies in the 64-wide layers, 4 in the 32-wide; eval rmse 5.9e-07) the train-mode error is still
  0.146 / 0.106 of var(a*T). => the embedded function is NOT a stationary point of the frozen training objective; the proof is
  about inference only, as GPT states ("Dropout è spento a inference"), and says nothing on learnability, on the recipe's
  optimum, on real latents or on generalisation.

## Task 5 - blocking issues
- None found that blocks the scientific convergence on D0. Preconditions before ANY launch (D0 included): registered decision
  of Simone + budget line + logical IDs; decidable reading rule for D1 with precedence, written before the fits; parity gate
  of the new driver. The follow-up "one variant then repeat the original control" is not protected against circularity and its
  premise (regularisation as priority) is not supported by D1 nor by the archived numbers.

## Correction to Task 2 part 3 (19:11)
- The sentence "regularisation does not provide by itself more signal captured" is too strong: my "captured" term also contains
  the estimation variance of the temporal function, which parsimony can reduce. What the archive supports is narrower: at most
  ~7-8 points of margin can come from the component measured in the linear world (TCN noise price 19.7%/18.7% vs ridge
  12.5%/10.9%); the other >= 18-23 points must come from learning the temporal part better UNDER noise, about which A1, D0 and
  D1 give no information (D1 measures it without noise only). So the follow-up priority is "not supported", not "contradicted".
- Size check of selection_states.pt: benchmark `checkpoint_file_bytes` 905,805 per trajectory state x 20 = 18,116,100 vs
  18,386,382 archived: consistent with 5 lr x (best+current) x 2 folds.

## Closing (19:12)
- Work directory wf3_design_work deleted (rm -rf with the full literal path); `git status --short` in the review checkout: 0 lines;
  no reviewer process running. No cluster access, no job, no push.
- Not done: no learnability simulation (real latents are only on Leonardo; a synthetic-feature run would not transfer); existence
  and hashes of the .pt files on Leonardo not re-read; decisioni.md / claude_GPT_shared read only for the 21/11 entries and grep hits.
