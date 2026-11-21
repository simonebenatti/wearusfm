# Reviewer STAT - notes (TCN temporal control diagnosis, commit e7670758)

Started: 2026-11-21 18:52 CET
Checkout (read-only): /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag
Work dir (to delete at end): /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_stat_work

Rules: no cluster, no job, no push, no download; local JSON, code, small CPU simulations only.

## Task 0 - setup
- notes file created before any analysis.

## Task 1 - DENOMINATORS AND SELECTION (done 18:58)

Commands: scratchpad/wf3_stat_work/t1.py, t1b.py, t1c.py, t2.py (read-only on review_diag, run with the prescribed env).

What the gate compares (code): workflow.qualify_controls (workflow.py:343-388) -> controls.temporal_head_gate (controls.py:302-318)
-> _recovery(clean_ls, head, oracle) (controls.py:251-260) per fold0/fold1/pooled.
- reference: controls.evaluate_outer_references (controls.py:553-587): clean_ls = WLS L0 (alpha 0) fitted on y*=L+a*T of the FITTING half,
  predicted on HELD (predict_mixture(held, {"L":1,"T":a}, 0.)); oracle = y* on held; both scored on observed y
  with selection_statistics(pred, panel, threshold), pose_var from y*, threshold = median fitting pose_var of y* (controls.py:566-567).
- head: workflow._score_model (workflow.py:242-261), same target y, same pose_var(y*) and threshold; E_dyn = sum dynamic_sse / sum dynamic_target_ss.
- pooled = sums of the two folds' moments (controls.pool_statistics).
- denominators identical: dynamic_target_ss head vs reference rel. diff 0 (fold0), 1.7e-16 (fold1), 1.6e-16 (pooled).

Recoveries recomputed from raw dynamic_sse/dynamic_target_ss (calibration_{0,1}.json outer_reference_statistics.temporal + selection_temporal_*.json selected_outer_statistics):
- references held: fold0 cleanLS 0.7104311506 oracle 0.6534505840 (margin 0.0569805665, SSE 182,695.6); fold1 0.7499203961 / 0.6911028974 (0.0588174987, SSE 158,738.3);
  pooled 0.7284790287 / 0.6706589240 (0.0578201047, SSE 341,433.9). gap fraction 8.02% / 7.84% / 7.94%.
- TCN_post (LR 0.003, exposure 31): 24.1731% / 18.8755% / 21.7101%  (= gate JSON gate.temporal_head_gate.recovery 0.24173052888786836 / 0.1887545039561508 / 0.21710109906955868). CONFIRMED.
- Heads trained on NOISY y with no temporal ability, same scale:
  * neural L0 (LR 3e-4, exp 183): -13.5531% / -10.8188% / -12.2819%
  * neural MLP (LR 1e-3, exp 24): -12.2400% / -10.6539% / -11.5026% (= gate.temporal_head_gate.mlp_recovery)
  * closed-form WLS L0 fitted on y (linear_diagnostics.L0.worlds.temporal.held): -19.1005% / -16.3797% / -17.8356%
  * closed-form WLS L13 fitted on y: -85.56% / -69.46% / -78.08%
  => the clean-fit baseline costs a noisy-trained head 11-19 points BEFORE any temporal learning (12 for early-stopped neural L0, 18 for unregularised WLS).
- Like-for-like scale (baseline = neural L0 trained on y): (E_L0 - E_TCN)/(E_L0 - E_oracle) = 33.22% / 26.80% / 30.27%. Still < 50%.
  To hit the 50% gate from neural L0 the TCN must close 55.5% (pooled) of the L0(noisy)->oracle distance; it closed 30.3%.
  So: the asymmetry is real (about 12 points) but does NOT by itself explain the failure.
- Protocol: baseline defined as clean-fit (lines 92, 100); line 100 says only qualitatively "banda e margine pulito possono essere piu' severi ... WLS su y* toglie il rumore nel fit";
  line 102: L0/MLP descriptive, "si riportano errori, differenza e quote recuperate". No number, no statement that a noisy-trained head starts below 0.
- GPT diagnosis: gives TCN vs neural L0 = -2.67% E_dyn and clean E_dyn 0.1392 vs 0.1990; does NOT report L0/MLP on the gate scale (-12.3 / -11.5), nor closed-form L0(y) -17.8,
  nor discusses the asymmetry (grep on the md and three READMEs: no hit). control JSON has learned_L0 E_dyn but no recovery.
- 21.88% claim: CONFIRMED for the POOLED panel only. min E_dyn over 915 admitted curve points (5 LRs; 304+4*154 minus 5 step-0) = 0.7158262458538921 at LR 0.003 exposure 27 -> 21.8830%.
  Curves are pooled only (point keys: edyn, exposure, fold_steps, mse): no per-fold curve. Saved states per LR: best_states (pair at min pooled MSE) + current_states (final) only (train.py:452-459);
  the epoch-27 point has no saved state. Per-fold values exist only for the selected pair (epoch 31).
  Pooled recovery is the margin-weighted mean of the fold recoveries (weights 0.535 / 0.465), so no PAIRED checkpoint can pass all three panels: conclusion "no archived checkpoint passes" holds.
- Per-LR best pooled recovery: 1e-4: 5.01% (exp 201), 3e-4: 5.36% (64), 1e-3: 10.81% (28), 3e-3: 21.88% (27), 1e-2: -14.85% (111). End of horizon: 0.57 / -5.93 / -39.81 / -22.62 / -15.79.
  NB monotone rise with LR up to 3e-3 then collapse at 1e-2 (L0 level): grid step between 3e-3 and 1e-2 is a factor 3.3; the winner is interior, no alarm fires.
- Cross term check on held (noisy SSE - clean SSE - oracle SSE): TCN -2,915.8 / +817.3 SSE = -1.6% / +0.5% of the margin: held scoring noise of the recovery is about 1-2 points.

## Task 2 - THE VARIANCE BUDGET (attempt to refute the reviewer's claim) (done 19:03)

Script: wf3_stat_work/t2.py. All on held dynamic windows, centred; "margin" = cleanLS SSE - oracle SSE = 182,695.6 (fold0), 158,738.3 (fold1), 341,433.9 pooled
(= 8.72% / 8.51% of the held noise SS, oracle SSE 2,095,145.5 / 1,865,167.4).

Measured facts (no model):
(a) Decomposition held noisy SSE = clean SSE + oracle SSE + cross: TCN clean SSE 141,448.3 / 127,958.4 = 77.42% / 80.61% of the margin (cross -1.6% / +0.5%).
    So the gate asks: clean dynamic SSE of the head on held <= 50% of the clean dynamic SSE of cleanLS. TCN is at 77/81%, neural L0 at 115/110%, MLP 112/110%.
(b) Closed-form L0 (rank 384 x 20 joints = 7,680 scalar parameters, unregularised) fitted on y vs the same fitted on y* (exact noise-only contrast, linear in the RHS):
    held excess 34,895.9 / 26,000.8 SSE = 0.010884 / 0.009634 E_dyn = 19.10% / 16.38% of the margin; 4.54 / 3.39 SSE per scalar parameter;
    = 1.67% / 1.39% of the noise SS -> empirical n_eff per joint 23,055 / 27,546 against 136,160 / 138,232 time samples (ratio 5.9 / 5.0; AR(1) bound for slow regressors (1+rho)/(1-rho) = 8.04; this ratio also contains cross-user shift, dynamic selection and centring).
    In-sample optimism of the same fit (fitting half): 0.002890 / 0.003756 E -> held excess is 3.77x / 2.57x the in-sample optimism (cross-user amplification).
(c) Closed-form L13 (rank 4,980 x 20 = 99,600 scalar parameters) fitted on y: held SSE above cleanLS 156,309.7 / 110,265.2 = 85.56% / 69.46% of the margin (above the oracle 185.6% / 169.5%).
    Noise-only excess needs the L13 clean-fit held value, NOT archived. ASSUMPTION: its held recovery r is in [0, fitting recovery 0.2709 / 0.1993]
    -> noise-only excess 85.6-112.6% (fold0), 69.5-89.4% (fold1) of the margin = 0.041-0.064 E_dyn; 1.1-2.1 SSE per scalar parameter (context parameters are 2-4x cheaper than L0's).
    If r < 0 (clean L13 generalises worse than clean L0) the noise-only figure is lower than 85.6 / 69.5. In-sample optimism L13 = 0.026700 / 0.029803 E (9.2x / 7.9x L0 with 13x parameters).
(d) Late degradation of the TCN (same LR 0.003): pooled recovery 21.71% (ep 31) -> -22.62% (ep 150) = 44.3 points lost, still falling (ep 125 -> 150: -5.05 points). LR 0.001: 10.81% (ep 28) -> -39.81% (ep 150) = 50.6 points.
    If more training does not increase the bias, the noise-driven variance grows by >= 0.44-0.51 margins after the selected point and has not converged.
(e) Fitting half (oracle reconstructed from L13 clean recovery, as GPT): TCN excess over the fitting oracle 0.007541 / 0.017336 E = 13.88% / 30.75% of the fitting margin (0.054322 / 0.056379).

Counting heuristic (ASSUMPTIONS: unregularised ERM at the optimum; variance = per-parameter cost x number of parameters; per-parameter cost between L13's 1.1-2.1 and L0's 3.4-4.5 SSE; cost of the affine branch as measured for L0):
- whole margin = 40,200-46,900 L0-like scalar parameters, or 88,000-143,000 L13-like.
- 50% bar minus the affine cost (19.1 / 16.4) leaves 30.9 / 33.6 points = 12,400-15,800 L0-like or 27,000-48,000 L13-like parameters for the deep branch.
- TCN_post deep branch = 64,436 parameters (first 1x1 layer 24,640; three k3 64->64 37,056; 2,080; 660). First layer alone at L0 cost = 46-61% (fold0) / 53% (fold1) of the margin. Total heuristic 0.9-1.3 margins.
- width-8 generator-family head: LN 768 + affine 7,700 + deep 3,932 = 12,400 parameters. Variance at the local optimum = affine 19.1 / 16.4 (measured) + deep 2-11 (heuristic) = 19-29% / 19-25% of the margin
  (13-24% if the affine part is early-stopped like the neural L0, whose total cost is 13.6 / 10.8).

[SUPERSEDED - see "Correction to the Task 2 bound" under Task 4: the third assumption below is contradicted by the toy; safe figures are non-noise >= 13.9% / 30.7%, variance <= 63.5% / 49.9%.
 Wherever the verdict below quotes 28-37% / 45-52% or 40-50% / 28-35%, read the corrected figures.]
Bound at the SELECTED checkpoint (ASSUMPTIONS: linear-smoother approximation; b_held >= b_fit as fractions of the margins; noise part of the clean fitting error <= fitted-noise covariance C; held variance = kappa' x C with kappa' between L13's 1.3-2.3 and L0's 2.5-3.6):
  x_fit = b_fit + v_fit - 2c, x_held = b_held + kappa' c  ->  b_held >= (x_held + kappa' x_fit)/(1+kappa').
- fold0: non-noise error on held >= 28-37% of the margin, noise-driven held variance <= 40-50%.
- fold1: non-noise error on held >= 45-52% of the margin, noise-driven held variance <= 28-35%.
- implied prediction for D0: clean recovery of the selected checkpoint on the FITTING half between 52% and 86% (fold0), 39% and 69% (fold1); upper end = no fitted noise.

Verdict on the claim:
- (A) "ERM variance of a 72,904-parameter head ~ the whole margin": NOT refuted, supported in order of magnitude by (c) [0.7-1.1 margins for a 99,600-parameter linear ERM], (d) [>= 0.44 margins added after the selected epoch, not converged] and the counting heuristic [0.9-1.3].
  But "plain ERM" is not the recipe: at the selected (early-stopped on held) checkpoint the held variance is at most about 0.4-0.5 (fold0) / 0.3-0.35 (fold1) of the margin under the stated assumptions. Quoting "the whole margin" for the recipe overstates by about 2x or more.
- (B) "cannot be met without strong effective regularisation": true for converged ERM; the recipe already has early stopping tuned on the held half. The accurate statement: early stopping does not give a good enough trade-off (non-noise error >= 28-37% / 45-52% plus variance).
- (C) "width-8 head loses 15-30% and could plausibly pass": my recomputation 19-29% / 19-25% (13-24% with regularised affine) is a VARIANCE-ONLY figure at a well-optimised optimum. It says the budget does not forbid passing; it says nothing about whether SGD from random initialisation reaches that optimum for an exactly-parameterised student. "Plausibly pass" is not supported by any archived number.
- (D) "compatible with a pure sample-size effect": compatible, not established, and not the only reading: at the selected checkpoint a non-noise error of >= 28-37% (fold0) and >= 45-52% (fold1) of the margin is present (under the assumptions); whether it is early stopping forced by noise, slow learning or a cross-user gap cannot be told from the archive.
CANNOT be concluded from these numbers: (1) the split bias/variance at the selected checkpoint (only bounds); (2) whether ANY head can reach 50% with this recipe, n and noise; (3) what a width-8 head would actually score; (4) the L13 noise-only variance exactly (clean-fit held value missing); (5) effective degrees of freedom of an SGD-trained ReLU network from its parameter count.

## Task 3 - WHAT D0/D1 CAN AND CANNOT SEPARATE (analysis, done 19:04)

Scale used: clean recovery R* = 1 - cleanSSE(head)/cleanSSE(cleanLS) on a half (on held it equals the gate recovery within about 2 points). Known: R*_held(ep31) = 22.6% / 19.4%.
Archived fitting numbers already exclude the PURE form of "optimisation failure with no noise fitting": with no fitted noise the fitting excess 13.9% / 30.7% would be the unlearned function,
i.e. 86% / 69% already learned on the fitting users. Either the function is mostly learned on fitting, or noise is being fitted.

[The D0 bands below (52-63 / 39-47 as lower ends) used the superseded Task 2 bound; after the toy the working figures are: about 51% / 44% if noise damages the fitting function as in the toy, 86% / 69% if it does not.]
Predictions (rough):
H1 optimisation failure: D0 R*_fit(31) at the low end of the admissible band (52-63% fold0, 39-47% fold1) with fitted noise; D1 clean training: fitting clean recovery still < ~80% at 150, held close to fitting (small gap), recovery on y < 50%.
H2 cross-user generalisation without noise: D0 R*_fit(31) near the top (80-86 / 62-69, no fitted noise); D1: fitting > 90% by 150, held stays 20-45%, gap persists, recovery on y < 50%.
H3 noise variance, fixable by a smaller/regularised head: D0 R*_fit(31) intermediate (52-75 / 39-60), at ep 150 noisy fitting E_dyn(y) falls below the fitting oracle (0.6543 / 0.6932) = direct sign of fitted noise; D1: held recovery on y >= 50% in the three panels (70-90% late in training), small fitting/held gap.
H4 noise variance NOT fixable by any head with this recipe at this n (bar unattainable): D0 and D1 IDENTICAL to H3.
H5 race (function learnable but slower than the noise is fitted): D0 as H3; D1 clean run at ISO-EPOCH 31 only 25-45% on held, >= 70% only after ~100 epochs.

Same D0/D1 outcome: (H3, H4) completely. (H3/H4/H5 vs H1) are confused by GPT's row 3 trigger "curve ancora in discesa/bordo": a clean-trained net has nothing that stops it,
so "minimum at 150 / still descending" is expected under EVERY hypothesis where clean training works; as written it would send the most informative outcome to "optimisation to investigate" or "inconclusive".
H5 is separated from H3 only if the iso-epoch readings (clean run at exposure 31 and 150, fitting and held, per fold) are pre-declared.
D0 alone: tells fitted noise present or not (separates pure H2 from the rest); it cannot split held bias from held variance.
Selection on y in D1: harmless, the cross term on held is 0.5-1.6% of the margin (Task 1), so choosing on y or y* differs by 1-2 points.

Cheapest measurement separating H3 from H4 (candidates):
(a) width-8 generator-family head (LN + affine + deep width 8, 12,400 parameters), random init, same noisy y, same recipe: a pass proves the bar attainable by a learnable parsimonious head (H3);
    a failure is ambiguous (variance or narrow-student optimisation). Risk of search: moderate (it is the head most likely to pass; must be declared non-candidate, one width, no scan).
(b) same family initialised at the TRUE weights, trained on noisy y with the recipe, read at a FIXED end (150) and along the curve, never at a held-selected epoch (step 0 is trivially ~100%):
    measures the estimation-variance floor of the minimal well-specified family. < 50% at the end => bar unattainable (H4), the control must be re-specified, not the head.
    >= 50% => attainable, H4 rejected. Risk of search: none (cannot be a candidate). Counting heuristic predicts 71-81% (Task 2).
    NB a wide head with the truth embedded as in GPT's oracle_embedding (other channels all zero) gets exactly zero gradient on the extra channels (ReLU(0): zero activation and zero derivative): it is the width-8 fit in disguise.
    To test the wide head from the truth the extra channels must keep random incoming weights and only zero outgoing weights. (to verify numerically in Task 4)
(c) closed-form WLS on y from [z, true hidden features of T]: exact, CPU only, no optimisation confound; measures only the readout cost (expected ~100 - 19/16 - small = about 80-84%).
    It caps the attainable scale and checks that the scoring chain can produce a pass under noisy fitting; it does NOT separate H3 from H4 (hidden features are given). Risk of search: none.
Recommendation: add (b) and (a) at LR 0.003 / 150 epochs to the same diagnostic package (4 small fits, each a fraction of a D1 fit), (c) optional; pre-declare the three-way reading.

## Task 4 - TOY SIMULATION (optional; done 19:17). LABEL: TOY, synthetic latents, one fold, one seed, one LR.

Scripts (copied next to this file in stat_work_files/): toy.py (gen + run), gradcheck.py. Compute: gen 13 s; 6 fits in two parallel batches, 400 s and 330 s wall each, one thread per process.
Regime actually run:
- latents SYNTHETIC: 40 users x 46 windows x 160 patches x 384 features (float16). 32 AR(1) factors (rho 0.60..0.98, decaying scale), shared mixing + user-specific perturbation (0.35),
  user offset (0.5), per-window level offset, white noise 0.15. 20 users fitting (920 windows), 20 held (920). NOT real latents.
- REAL generator controls.generate_components (outer_fold 0), REAL heads.make_head("TCN_post") (72,904 parameters), REAL loss metrics.per_window_mse, AdamW via heads.optimizer_groups (wd 0.01), clip 1, batch 32, LR 0.003, dropout as in the head.
- small head "S8" = my re-implementation of the same structure with hidden width 8 (12,400 parameters), same init rule and dropout positions.
- a chosen on the fitting half for gap closest to 8% -> a = 0.4 (same as the real calibration), sigma = 1.3, rho = exp(-1/4).
- toy references (held): E_cleanLS 0.8221, E_oracle 0.7655, gap 6.88% (fit 7.46%), margin/noise SS 7.39% (real 8.7 / 8.5), closed-form L0 on y: recovery -8.9% (real -19.1 / -16.4), T unread by L0 on fitting 93% (real 71 / 88).
- my E_dyn re-implementation: centred per window-joint on crop 6, dynamic windows = pose_var(y*) >= median fitting pose_var(y*), ratio of sums.

Results (held recovery on the gate scale; "fit" = fitting half; clean = against y*):
| run | init / target | epochs run | held recovery | notes |
| wide_noisy (TCN_post) | random / y | 28 (time) | selected ep 25: 31.1%; best 31.2% (ep 21); ep 28 30.5% | fitting on y 91.9%, fitting clean 58.6% at ep 25 (real: fitting 86 / 69, held 24 / 19) |
| wide_clean (TCN_post) | random / y* | 28 (time) | ep 10 43.2, ep 15 60.8, ep 20 70.5, ep 25 75.3, ep 28 80.2% still rising | minimum of held MSE at the LAST epoch (edge) |
| wide_truthfree (TCN_post) | truth + free random extra channels / y | 23 (time) | ep 0 100, ep 3 80.7, ep 10 71.8, ep 15 68.7, ep 20 63.5, ep 23 55.0%, still falling | fitting on y 101-105% from ep 15 (below the fitting oracle = fitted noise) |
| s8_noisy | random / y | 94 (time) | ep 30 16.0, ep 60 25.0, ep 94 27.2% (slowly rising) | fitting on y 60%, fitting clean 44% |
| s8_clean | random / y* | 78 (time) | ep 30 33.8, ep 60 41.6, ep 78 44.2% | narrow student learns the clean function slowly |
| s8_truth | truth / y | 94 (time) | ep 3 75.6, ep 20 67.3, ep 94 66.6% (flat 65-67) | variance (+dropout) floor of the minimal family |
Readings:
- The toy reproduces the real signature (wide head on noisy y: about 31% at epoch 21-25, fitting about 92%).
- At iso-epoch 23-25: clean-trained wide head has bias 25 points; truth-initialised wide head on y has lost 45 points (pure variance); random-init wide head on y is at 31% = 100 - 25 - 45 (superposition holds within 1-2 points).
- The ordering "small well-specified head recovers much more" does NOT appear: S8 from random init 27% (noisy), 44% even on the clean target. Its variance floor is good (67% from the truth) but SGD does not get there in 78-94 epochs at this LR.
- A D1-like outcome is a "success" in the toy (80% at epoch 28, minimum at the edge) while NO random-init head tried passes 50% on y. So D1 success does not imply a capacity variant will pass.
- gradcheck.py: with GPT's zero embedding of the truth into TCN_post every extra-channel weight has gradient exactly 0.0 (rows >= 8 and columns >= 8 of all deep convs), true rows non-zero;
  with random extra incoming weights and zero outgoing weights the outgoing columns get gradient (1.25, 1.51, 1.37, 0.90, 0.19) and the function at init is still the truth (RMSE 4.5e-7).
Limits: one fold, one seed, one LR (0.003), no LR search, horizons cut by the time budget (28 epochs wide, 78-94 small), synthetic latents with weaker L0 variance than the real ones, no deterministic-algorithm flags.

### Correction to the Task 2 bound (after the toy)
Assumption "noise part of the clean fitting error <= fitted-noise covariance C" is CONTRADICTED by the toy: at epoch 25 the wide head has noise-driven clean fitting error 30.0 points, C = 16.7 points (ratio 1.8).
So the bounds "non-noise error >= 28-37% / 45-52%, variance <= 40-50% / 28-35%" are NOT safe. With the weaker assumption (ratio <= 2, b_held >= b_fit):
non-noise error on held >= 13.9% / 30.7% of the margin, noise-driven held variance <= 63.5% / 49.9%.
Toy-calibrated central values for the real folds (ratio 1.8, kappa' 2.68, held-fit bias gap 13.3 points): fold0 variance about 47, non-noise 31; fold1 variance about 34, non-noise 47;
predicted clean recovery of the selected checkpoint on the FITTING half (D0): about 51% (fold0), 44% (fold1) if the toy mechanism holds; 86% / 69% if noise does not damage the fitting function (pure cross-user case).
The split at the selected checkpoint is NOT identified by the archive.

## Closing (19:18)
- work dir /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_stat_work deleted (toy_data.npz 297 MB included); no process left; review_diag untouched (git status empty, HEAD e7670758).
- small scripts and logs kept in /Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/stat_work_files/ (100 kB).
- Findings to report, in order: (1) D0+D1 cannot separate "a smaller head would pass" from "no random-init head passes with this recipe at this n"; decision row 1 / par. 6 not licensed (toy shows the case);
  (2) D1 reading rules: edge / still descending is the EXPECTED outcome of clean training, iso-epoch readings must be pre-declared; (3) baseline asymmetry (-12 / -18 points) not reported; like-for-like 30.3%;
  (4) variance claim: order of magnitude right for converged ERM, overstated for the recipe and for the width-8 head; (5) 21.88% is pooled only, epoch 27 has no saved state; (6) zero embedding freezes extra channels.
