# Reviewer REAL-DATA — durable notes

Started: 2026-11-20 14:27 CET. Reviewing tip 0e548bf8218f15ba07fc731efcc2ead512e65034 (sources final at 5d6fb2dc).
Read-only on /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-tcn-users-fix. Own tree: scratchpad/wf2_realdata_tree, work: scratchpad/wf2_realdata_work.

## Task 1a/1b (paper part) — done 14:50

Sources read: scripts/pose_temponet/metrics.py (whole), data.py (whole), provenance.py (whole), runner.py (whole), execution.py,
scripts/pose_dynamics.py, scripts/tokenizer_pose_readout.py:232-302, scripts/consolidation_long_pose.py, scripts/consolidation_long_pose_runtime.py,
scripts/pose_temporal_readout.py:195-199 (central_mask).

1a. ORIGIN OF THE REFERENCE NUMBERS — verified from the archived historical receipt, not guessed:
  `git show 2930d7d6:results/infra/consolidation_long_20261105/execution/receipts/LONG-FIT-G49046-pose-20261105.json`
  (the file is NOT in the working tree of this branch, only 28 of the 103 receipts are; it is reachable as a git object)
  cells.post_ridge.analysis.groups.dynamic.centered.centered_mse_ratio = 0.7859020004638595  (reference - value = 0.0)
  cells.trained_head.analysis.groups.dynamic.centered.centered_mse_ratio = 0.7786289843513977  (reference - value = 0.0)
  counts in the same receipt: n_selected_windows 8329, centered n_windows 8327, n_window_joints 166540, n_valid_elements 20119920, n_users 15 (both readers) = HISTORICAL_REFERENCE.
  (Side fact: for pre_ridge the receipt has 1.0321280032044746 while the comparison "E_dyn point" in proposal.json is ...744: the two
   historical formulas - pooled_summary means ratio vs aggregate sums ratio - can differ by 1 ulp; for the two gated readers they coincide.)
  Historical producer of the numbers: consolidation_long_pose_runtime._fit (lines 272-288) -> P.summarize -> R.summarize
  (tokenizer_pose_readout.py:274-302) -> D.window_joint_moments / D.accumulate / D.pooled_summary (pose_dynamics.py:88-173), BATCH=64.
  New gate: metrics.historical_reference_gate -> metrics.evaluate(batch_windows=64 default) -> the SAME three functions of the SAME module.
  Blob identity of the shared modules, historical fit code a70e29e4 vs tip 0e548bf8 (git rev-parse <commit>:<file>):
    scripts/pose_dynamics.py SAME (a261606a), scripts/tokenizer_pose_readout.py SAME (f0cfe709), scripts/pose_temporal_readout.py SAME (590d8e13),
    scripts/consolidation_long_pose.py SAME (28f1283f). Only consolidation_long_pose_runtime.py differs (amendment A1 lines, not _fit's prediction lines).
  Statement-by-statement differences between R.summarize and metrics.evaluate that touch the E_dyn path: ONE.
    historical: y = data["target"][sl, 6:-6] float32, converted inside window_joint_moments by np.where(mask, y, 0).astype(float64)
    new:        targets = y[:, 6:-6].astype(float64) once in _cropped, then the same np.where(...).astype(float64)
    float32->float64 is exact and commutes with the masking, so the moments are the same numbers.
  Same: users array, selection np.isfinite(pv) & (pv >= 0.2574231028556824) on the cache float32 pose_var, central mask
  (reference.T.central_mask(mask, lengths, offset=6) in both), batch 64, window order of the cache, accumulate(min_n=2, CENTERED_FIELDS), pooled_summary.
  Inputs: plan.json pins == historical fit receipt (sha256 and bytes equal) for cache_val, cache_train, post_ridge_predictions,
  trained_head_predictions, post_ridge_coefficients; historical analysis targets_sha256 == plan.producers.val.details.targets
  (arrays c907b85b..., central_mask 97393047..., metadata 960c8328...), which read_verified_inputs enforces at run time.
  Environment: failed preflight receipt (results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/preflight_receipt.json)
  numpy 1.26.4, python 3.11.6, torch 2.2.0a0, Xeon Platinum 8358; historical fit receipt environment numpy 1.26.4, python 3.11.6, torch 2.2.0. Same numpy.

1b. WRITER OF trained_head.predictions.npz: consolidation_long_pose_runtime.py at a70e29e4, lines 277-285:
    prediction = held["pred"][:, P.OFFSET:-P.OFFSET].astype(np.float64); _save_array(out/"trained_head.predictions.npz", prediction=prediction)
  with P.OFFSET = R.OFFSET = 6 and held = P.read_cache(val cache 36bab32e..., the file pinned as cache_val). The gate compares
  data["pred"][:, 6:-6].astype(float64)[support] with that array[support]: same file, same slice, same cast -> equal on every element;
  NaN cannot break array_equal on support because validate_cache already required isfinite(pred[mask]) and support is a subset of mask[:, 6:-6].

(Correction: the "done 14:50" stamp above was a guess; the paper part of task 1 was finished at about 14:35.)

## Task 1c (measurement) — done, see time stamp of this block in the file's mtime

Script: scratchpad/wf2_realdata_work/r1_measure.py (gen + run). Synthetic arrays at the REAL val size: 18548 windows, the 15 real val user IDs
with 700..1540 windows each, target float32 [N,160,20], masks per time step with 1896 empty windows, 20 windows with one supported centre,
20 with support only in the cropped edges, float32 pose_var with NaN, ridge-like predictions float64 [N,148,20], head float32 [N,160,20].
Historical path = tokenizer_pose_readout.summarize(consolidation_long_pose._target_view(data), prediction) (first statement of P.summarize);
new path = pose_temponet.metrics.evaluate(np.asarray(prediction, float64), data).

Command (from my own detached worktree at 0e548bf8):
  PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 <python> r1_measure.py run <dir>
run twice: ~/.venvs/wearusfm-tok (numpy 2.3.5, py 3.11.14) and ~/anaconda3/envs/appTF (numpy 1.26.4 = cluster numpy, py 3.10.13), arm64.

Decisive output, identical in BOTH numpy versions:
  post_ridge:   historical E_dyn 0.8911610782699846 | evaluate 0.8911610782699846 | abs diff 0.000e+00 | counts equal True (11832, 11817, 32504780, 236340, 15)
                | per-user sums bitwise equal True | all three groups' centered blocks equal True | support hash equal True
  trained_head: historical E_dyn 0.7615891451150004 | evaluate 0.7615891451150004 | abs diff 0.000e+00 | same counts, per-user sums bitwise equal True
  batch_windows 1/7/32/128/1000/18548: abs diff vs historical between 0 and 4.441e-16 (max, post_ridge batch 1)
  shuffled window order (users interleaved): 1.110e-16 / 0.0 ; time axis reversed: 0.0 / 0.0 ; aggregate formula (sum ratio over sorted users): 1.110e-16
  float32 head handed to evaluate (cast inside): 0.0
  historical_reference_gate at real size: passed True, 2.7-2.8 s, peak RSS of the whole script 3.4 GiB; a head prediction shifted by one patch is refused
  ("historical trained-head crop indices/values differ").
Conclusion: the two paths are the same function calls and give bit-identical results; the largest perturbation I could produce by changing what
the code does NOT change (batch size, window order, summation order, formula) is 4.4e-16. 1e-12 absolute leaves a factor ~2250 (3.3 orders of magnitude).
On the cluster the batch (64), the window order, numpy (1.26.4) and the three functions are those of the historical run: expected difference 0.

## Task 2 (val read: peak memory and wall time) — done 14:47

Arrays alive at the peak (real sizes; N=18548, all n_patches=160 so `available` is all True):
  pre  float16 [18548,160,384] 2.123 GiB + lat 2.123 GiB (np.load, consolidation_long_pose.py:169-172; chunked 256 KiB reads, no doubling)
  + a[available]  -> FULL COPY 2.123 GiB (boolean index on the first two axes)        consolidation_long_pose.py:145
  + np.isfinite(copy) -> bool 1.061 GiB                                                 consolidation_long_pose.py:145
  + pred f32 0.221, target f32 0.221, mask 0.055, target_reference (target 0.221, mask 0.055, central 0.051), two meta lists, Python+torch ~0.3
  = about 8.7-9.1 GiB. The statement runs 4 times per val read (pre and lat, in H.read_cache:174 and again in D.validate_cache -> data.py:83).
  Everything in the gate is far below: cached f64 0.409 (metrics.py:175), two 1-D copies cached[support]/trained_head[support] <=0.41 each (176),
  evaluate copies 2 x 0.409 (metrics.py:75), with lat (2.12 GiB, kept by depth="post" although the gate never uses it) -> about 4.8 GiB.
REAL cluster number for the same reader on the same file: historical LONG-FIT-G49046-pose (job 63684845, read train + val with P.read_cache,
  30 GB job) measured_resources.peak_rss_bytes = 9513836544 = 8.86 GiB, elapsed 184.7 s for much more work (two ridge fits, 2 x 22 s predictions,
  three compressed 439 MB writes, three summaries).  -> 24 GB is safe by a factor ~2.7.
My reduced-size measurement (unmocked provenance.preflight, timing wrappers only; build_sim.py train 400 / val 2000; val.npz 497298948 bytes =
  248.6 kB per window, real 247.7 kB per window): scratchpad/wf2_realdata_work/preflight_profile.py, log profile2000.log
  baseline peak RSS 0.261 GiB, peak 1.214 GiB  -> (1.214-0.261) x 18548/2000 + 0.261 = 9.10 GiB (linear in windows; agrees with the real 8.86).
  times on the Mac: total 5.03 s; val part = read_verified_inputs 3.27 s (2 hashes 0.36, read_targets 0.13, inflate 1.93, validate 0.46 + 0.38)
  + read_predictions 0.31 + gate 0.35 = 3.93 s -> x 9.274 = 36 s at the real size (gate at real size measured directly: 2.7-2.8 s).
  hashing on the Mac 2063 MB/s.
Scaling to the cluster (assumption: every phase is linear in the number of windows; cluster/Mac factor taken from the only real number):
  failed job: 23.6 s inside run_stage for check_release + hashing 8.46 GB + the whole train read; the same work on the Mac is ~7.5 s -> factor <= 3.1
  val part on the cluster <= 3.1 x 36 s = 113 s; cross-check from LONG-BENCH-pose (real, train 19618): read_validate 5.94 s for 1907 windows
  -> x 9.73 = 58 s for hash+read_targets+inflate+one validation, + second validation + predictions + gate = about 85 s.
  Whole job: 73 s (what 64931573 used up to the failure, checkout included) + 85..113 s = about 160-190 s. Limit 720 s (+180 s grace): margin about 4x.
Conclusion: 24 GB and 12 minutes are both safe. No finding. Observation: the val cache is hashed twice and validated twice (4 full copies of 2.1 GiB);
harmless here, same pattern in locked_inference (two val reads) and metrics.

## Task 3 (stages right after the preflight, real-format data) — reading and two real-size runs done 14:55 (GPU-logic part: see the addendum below)

Read: benchmark.py (whole), controls.py (whole), linear.py (whole), workflow.py (whole), train.py:1-400, heads.py, report.py, runner.run_stage branches.
Real structural fact used: masks are per time step for all 20 joints (src/wearusfm/data/pose_targets.py:157-179, `mask[has] = True` on whole rows),
consistent with the real counts (166540 = 8327 x 20; first-15 users: supported = max_joint x 20). So linear.mask_groups gives ONE group of 20 joints.
Does anything need every window / every first window to have support?
  - only benchmark._report_measurement (benchmark.py:610-617): the FIRST window of each of the first 15 sorted users must have centred support
    on every joint; real train_support.json: true for all 15 (minimum 41 supported centres), user 76bfc2d385 is 19th and is not used. OK on real data,
    but it is a property of this cache, not guarded by the code (already known from the interrupted reviewer; candidate_fix.py is kept).
  - TensorTrainingSet (train.py:176-180) removes unsupported windows before any minibatch; per_window_mse cannot see an empty minibatch.
  - linear weights use np.divide(..., where=q>0) (linear.py:241, 326; controls.py:108; benchmark.py:461), WeightedQR.add drops zero-weight rows (linear.py:83-86).
  - metrics.dynamic_variance gives NaN for windows with <2 valid rows; training_threshold / controls._threshold use finite values only.
Runs (my worktree at 0e548bf8, single thread, real-size train cache built with the REAL support facts: 53 empty windows, none in the first window of
the first 15 sorted users, first window of 76bfc2d385 empty, two windows with ONE supported centre; train.npz 475589707 bytes vs real 474412856):
  1) kept script partial_realdata/first_statements.py  -> log first_statements.log
     population {'train_windows': 1907, 'fold_windows': [960, 947], 'fit_train_windows': 1854, 'fit_fold_windows': [938, 916], 'excluded_fitting_windows': [22, 31]}
     partition users 20 20 | inner [(480, 480), (480, 480)] | wls fold 0 equals benchmark fold 0: True | window ids unique: True
     training sets supported: [938, 938] | generate_components s 3.0 | production score on 64 held windows: window_count 63
     first windows of the 15 surrogate users have central support: all True | report surrogate OK 2000
     the first delivery's directory list is refused at the benchmark ("benchmark population must contain exactly the declared training users")
  2) my stage3_wls.py = runner.py:281-298 for fold 0 (ONE deviation: L13 outer diagnostic off, too slow for one Mac thread) -> stage3_wls_fold0.log
     unsupported in fit/held 22 31 | one-centre windows 2 | dynamic_variance NaN windows 55
     build_positive_control_fold OK | 29.0 s | status calibrated | evaluations 1936 | outer scored True
     calibration.pt 228543033 bytes, calibration.json 1943605 bytes strict JSON; partition equal after torch and JSON round trips.
  Nothing became empty or divided by zero.
Other checks for the first GPU job (never run on the cluster): runner._allocation reads AllocTRES "gres/gpu"; archived real sacct rows of this project
show the key exactly as parsed: "billing=8,cpu=8,gres/gpu=1,mem=100G,node=1" (138 rows). Heads have no BatchNorm (heads.py:36-58): a short last minibatch is harmless.

### Task 3 addendum (GPU benchmark logic) — done 15:03
  The kept bench_gpu_on_cpu.py is too slow on one Mac thread (after 4 min: L0 complete, 43 cost files, L13 fold0 at epoch 7; estimated 30-40 min)
  and would end in _base_receipt -> _hardware -> torch.cuda.current_device() (not patched by the script). I stopped it (pkill) and removed its files.
  Replacement gpu_cells_smoke.py = the per-cell statements of benchmark.py:288-345 for ALL five heads, fold0 (+ full for L0), ONE epoch, device cpu:
    L0 fold0/full, L13, MLP, TCN_post, TCN_pre(depth pre): train_windows 938 excluded 22 steps/epoch 30 (full: 1854 / 53 / 58), held 947 scored 916,
    state capture/restore, atomic_torch + read back, prediction batches (32,148,20) and (20,148,20), reference proxy multiplier 1.372. All OK.
  gpu_score_cost.py: CPU-side scoring of the whole GPU benchmark (5 heads x 3 parts x 11 passes, include_clean) on the Mac = 75 s; validation surrogate 2.0 s;
    with the <=3.1 cluster factor <= ~4 min, plus GPU epochs and 165 state files: well inside the 1620 s cap. No finding.
  resources.gate_and_pack (read, not run: no CPU receipt here) uses fit_fold_windows / fit_train_windows and checks excluded_fitting_windows:
    consistent with unsupported windows (938+916 = 1854, 22+31 = 53).

## Task 4 (one identity from two sources) — done 15:03
  - plan.producers.{train,val}.users == sorted real cache IDs of tests/cpu/data/pose_temponet_cache_users_20261120.json (40 and 15, all 10-char hex),
    storage_directories == "u"+ID; proposal.json contains no user-ID-like string at all (no second list of users/folds in another format).
  - real embedded config of BOTH caches (headers.json) == plan.producers[split].metadata on all 9 PRODUCER_FIELDS; window recipe 10 / 4.0 / 20261026 / False.
    (val was never validated by the new code on the cluster; this is the comparison data.validate_producer will make.)
  - folds: runner passes previous["train"]["folds"] (preflight receipt, cache-derived) -> same source as the cache; JSON round trip tested in the runs above.
  - resource gate: runner.py:235 requires projection == plan["resource_gate"] EXACTLY; the plan value is computed by `resource-plan` on the operator's
    machine (docs/pose_temponet_execution_20261119.md:169-175, local copies) and recomputed on the cluster. Only numpy call inside gate_and_pack is
    resources.percentile95 = np.quantile(.95, method="linear") on 10 samples: quantile_bits.py, 200000 random sample sets, identical SHA-256
    a9eb09d0... under numpy 1.26.4, 2.0.2 and 2.3.5 (np.median too). Everything else is Python float arithmetic in a fixed order. OK.
  - sacct proof: instructions give `sacct -X -n -P --format=JobIDRaw,State,ExitCode` = the 3 fields execution.verify_completed_predecessors parses
    (the archived preflight_64931573_sacct.tsv has 7 fields and would NOT be accepted as a proof file, but it is not used as one).
  - AllocTRES key for GPU jobs: real rows "billing=8,cpu=8,gres/gpu=1,mem=...G,node=1" match runner._allocation.
  - head/world names: heads.HEADS == workflow.HEADS == resources.HEADS; execution.CONTROL_HEADS/WORLDS == workflow's.
  - NOT verifiable offline: gpu8 numeric pin contains gpu_bytes (card total memory) and cpu_model of the benchmark node; later GPU jobs on another
    Booster node must report the same values (no archived total_memory value in the repo to compare).
  - scratch refresh: scripts/slurm/refresh_scratch.sbatch touches every file under $SCRATCH/{data/processed,wearusfm_runs} (touch -c: mtime changes);
    three pinned inputs live there (two target references, pose_stats). A touch that lands inside a job's read window (seconds; ~2-3 min in the preflight,
    provenance.py:485-495) would raise "pinned input changed while in use". Next refresh is planned for 03/12 09:00.
## Finished 15:04: worktree and work directory removed, no process left (ps checked), delivery worktree clean at 0e548bf8. Scripts and logs kept in /Users/simonebenatti/wearusfm_local/tcn_users_fix_review/realdata_scripts/ (80 kB).
