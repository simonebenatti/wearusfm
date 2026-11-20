# Reviewer TESTS - notes (durable)

Started: 2026-11-20 14:27 CET. Reviewer TESTS started on tip 0e548bf8218f15ba07fc731efcc2ead512e65034 (sources final at 5d6fb2dc), base 31877ad8.
Worktree under review (READ-ONLY): /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-tcn-users-fix
Own tree (mutations): /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf2_tests_tree
Own work dir: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf2_tests_work

## Task 1 - the 8 new tests and the 2 fixtures

Read: `git diff 31877ad8 0e548bf8 -- tests/ scripts/pose_temponet/`, provenance.py, data.py, metrics.py, execution.py, runner.py,
consolidation_long_pose.py (validate_cache/save_cache/read_cache), tokenizer_pose_readout.py (read_targets/check_selection).

One line per test:
- provenance::test_real_producer_users_are_the_ids_stored_in_the_real_caches - real build_plan on the REAL proposal + the two REAL
  receipts; producer["users"] == IDs read from the real caches (fixture), storage_directories == "u"+ID, pin sha/windows tie the fixture
  to the plan. Direct test of the fix. Not mocked.
- provenance::test_storage_directory_maps_back_only_where_the_id_is_unique - unit test of storage_directory/cache_user; the "does not
  round-trip" branch is reachable only by monkeypatching P._STORAGE_UNSAFE (with the real _SAFE_RE r"[^0-9A-Za-z_.@-]" the fullmatch
  r"u[0-9A-Za-z.@-]+" already implies the round trip). Legitimate (guards drift of the imported rule).
- provenance::test_producer_with_ambiguous_storage_directory_is_refused - real receipt; one directory renamed with "_" -> refused.
  Not vacuous: if the rename did not hit any key, pytest.raises would fail (DID NOT RAISE).
- provenance::test_verified_inputs_and_consumers_use_cache_ids_with_the_real_layout - read_verified_inputs comparison logic with
  P.verify_pin, P.unchanged and P.D.read_cache_pin REPLACED; "real layout" in the name is only the real user IDs: the served cache is
  {"meta":[{"user":..}], "input_verification": copy of producer details}. Only ONE consumer is exercised (controls.validate_outer_partition).
- execution::test_second_delivery_never_reuses_the_identity_of_the_failed_first_preflight - 24 unique IDs, all "-20261120", CAMPAIGN, CODE_DIR.
  Two assertions cannot fail: `== "TCNH-preflight-20261120" != "TCNH-preflight-20261119"` (literal inequality) and the
  intersection with the two hard-coded inventory IDs (no stage token can produce them).
- real_layout::test_unpatched_preflight_completes_on_the_real_cache_layout - REAL P.preflight end to end on real-format files.
- real_layout::test_first_delivery_user_list_fails_exactly_like_job_64931573 - anchored message; equals the real log
  (results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/preflight_64931573.log last line:
  "ValueError: cache user population differs from original producer").
- real_layout::test_real_read_path_refuses_drifted_directories_and_each_changed_detail - second check on val + each of the 6 details.

Claim "nothing replaced except metrics.HISTORICAL_REFERENCE" in test_pose_temponet_preflight_real_layout.py: VERIFIED line by line.
The only monkeypatch.setattr in the file is line 122 (M.HISTORICAL_REFERENCE). build_plan, validate_plan, read_verified_inputs,
D.read_cache_pin, historical.read_cache/validate_cache/save_cache, reference.read_targets/check_selection, read_predictions,
verify_pin/file_pin/unchanged, alternating_user_folds, historical_reference_gate are the real functions.
Real-vs-real comparisons that the test does execute: producer metadata from the REAL receipt against the REAL config_json of the
cache (D.validate_producer: 9 producer fields + n_windows/window_s/seed/per_side); real storage directories against real cache IDs.
My probe (wf2_tests_work/probe_layout.py): the file written by the test's _cache through historical.save_cache has exactly the 13
members of the real cache, same dtype descr and rank for all 13 (also the 5 scalars the test does not assert), config_json <U780/<U778
identical to the real header length. Real meta value types (meta_stats.json: str x5, int recording/start/n_patches, float fs) equal the synthetic ones.

Still synthetic in real_layout (what could hide a real-data failure):
 1. R-1 constants are rebound to the output of the same M.evaluate on the same arrays -> `historical_R1.passed` is true by
    construction; trained_head predictions are written from the cache's own pred -> the "crop indices/values" check is true by construction.
    The real 1e-12 parity and the 5 real counts (8329/8327/20119920/166540/15) are not exercised by any test.
 2. The six receipt "details" are recomputed with D.historical.validate_cache on the synthetic file (line 92-93) -> the six
    comparisons are true by construction. Mitigation I verified: `git diff 7d276c98 0e548bf8 -- scripts/consolidation_long_pose.py
    scripts/tokenizer_pose_readout.py scripts/pose_temporal_readout.py scripts/pose_dynamics.py scripts/pose_e3_b.py
    src/wearusfm/harness/pose_eval.py src/wearusfm/ingest/emg2pose.py src/wearusfm/ingest/emg2pose_pose.py` is EMPTY (7d276c98 is
    the producer code_sha): the digests are computed by byte-identical code on sha-pinned files. The real receipts carry exactly the six
    keys with targets={arrays,central_mask,metadata} (read from docs/reviews/pose_temponet_runtime_evidence_20261119/producer_*.json).
 3. Sizes: 80 train / 60 val windows (real 1907 / 18548): memory and the 12-minute limit are not exercised.
 4. Uniform windows per user (2 / 4), n_patches 150 or 160 (real: always 160), array values gaussian, std constant .3.
 5. The target reference is written from the same arrays (real: another model's cache; equality proven on the cluster only for train,
    by the failed run that passed D.read_cache_pin(train)).
 6. Source tree = 3 dummy files, receipts pinned without "commit" (read_json_pin, not git_artifact); checkout_identity,
    validate_plan(require_review=True), run_stage, _allocation, make_receipt/write_receipt are not in this test (P.preflight is called directly).
 7. pose_stats.json is "{}\n" (only hashed by the preflight, never parsed: fine).
Order / external dependence: none found. P.OUTPUT_ROOT is not patched in real_layout (safe_output_root only lstat's /leonardo_work...,
absent on the Mac). No file outside the repository is read. monkeypatch restores M.HISTORICAL_REFERENCE and P.D.read_cache_pin.
pytest.raises blocks: all 12 use a message match; none would pass on an unrelated error.
## Task 2 - new mutations (own tree wf2_tests_tree; runner wf2_tests_work/mutate.py; results wf2_tests_work/mutations_tests_reviewer.json)
Baseline in my tree (same commit 0e548bf8): provenance+execution+preflight_real_layout = "34 passed in 15.28s".
NOTE: the author's run_mutations.py executed only provenance+execution (31 tests); the 3 real_layout tests were never mutation-checked.
Batch 1 (tests: provenance, execution, preflight_real_layout):
- P1 storage_directories reverse order: CAUGHT (2 failed, 3 errors)
- P2 users reverse order: CAUGHT (2 failed, 3 errors)
- P4 first check set==set: SURVIVED (34 passed) - equivalent on a validated plan (list is sorted+unique by construction)
- P5 second check set==set: SURVIVED (34 passed) - same remark
- P6 first check weakened to subset: SURVIVED (34 passed)
- P6b BOTH checks weakened to subset: SURVIVED (34 passed) -> no test serves a cache with one user MISSING
- P7 first check skipped for val: SURVIVED (34 passed) - the second check still covers val
- P7b both checks skipped for val: CAUGHT (real_layout::test_real_read_path_refuses_drifted_directories_and_each_changed_detail)
- P7c both checks skipped for train: CAUGHT (2 tests)
Batch 2 (same 3 test files):
- P8a six details comparisons skipped: CAUGHT only by real_layout::test_real_read_path_refuses_drifted_directories_and_each_changed_detail
- P8b targets dropped / P8c key_time_valid_sha256 dropped / P8d both digests dropped: CAUGHT (same single test)
  -> before file test_pose_temponet_preflight_real_layout.py existed, the details loop had no test at all.
- P8e details comparison skipped for VAL only: SURVIVED (34 passed) -> the six details are tested on train only
- P9 cache_user re.match instead of fullmatch: CAUGHT (2 tests)
- P10 cache_user hex-only regex: CAUGHT; P11 storage_directory other rule (no @): CAUGHT; P12 no ID validation: CAUGHT
- P13 historical_producer does not check the number of user directories (len(users)): SURVIVED (34 passed) (pre-existing line)
Batch 3:
- P14 validate_plan stops comparing the producer entry with the receipt: SURVIVED (34 passed, 3 core files)
- P15 validate_plan ignores users+storage_directories in that comparison: SURVIVED (34 passed)
  (existing test_rewriting_historical_metadata_or_scientific_pins_fails[producer] uses a bare pytest.raises(ValueError): it is
   satisfied by "execution identity differs", producers being part of the execution identity)
- P16 preflight returns before reading val: CAUGHT (real_layout test 1)
- P16b preflight reads val through D.read_cache_pin only (no user/details check on val): SURVIVED (34 passed)
- P16c R-1 gate on train instead of val: CAUGHT (real_layout test 1)
- P17 R-1 tolerance 1e-12 -> 1e-6 (tests: 3 core + metrics): SURVIVED (42 passed)  [pre-existing code, not in the diff]
- P17b R-1 E_dyn comparison removed: CAUGHT (metrics::test_reference_gate_rejects_shift_even_with_identical_shape_and_counts)
- E1 job_id keeps 20261119 for one stage: CAUGHT; E2 CAMPAIGN literal 20261119: CAUGHT; E3 CODE_DIR diverges: CAUGHT (all by the new execution test)
- E4 submit_command --campagna stale (tests: execution, operational, cli, cli_guards): SURVIVED (59 passed)
- E5 preflight export CAMPAIGN=<old name> (same 4 files): SURVIVED (59 passed) -> on the cluster pose_temponet_checkout.sbatch
  line `[ "$CODE_DIR" = "$WORK/wearusfm_campaigns/$CAMPAIGN" ] || exit 2` would stop the job; no test ties the generated export to the wrapper rule.
Batch 4 (20 TCN test files = all but e2e_chain):
- P18 R-1 constants altered in metrics.HISTORICAL_REFERENCE (post_ridge_E_dyn last digits; dynamic_users 15->16): SURVIVED, "270 passed in 55.89s".
  e2e_chain rebinds M.HISTORICAL_REFERENCE too (tests/cpu/test_pose_temponet_e2e_chain.py:157) -> NO test pins the real constants.
- P14all validate_plan producer comparison removed: SURVIVED, "270 passed in 53.79s" (e2e_chain never rewrites producers).
Batch 5:
- R1a runner.py:272 gives benchmark_cpu/benchmark_gpu plan["producers"]["train"]["storage_directories"] as allowed_train_users
  (tests: runner, runtime_fixes, benchmark, resources, e2e_chain = every file that calls run_stage): SURVIVED, "58 passed in 118.85s".
  Reason: runner.py tests and e2e_chain replace benchmark.benchmark_cpu/benchmark_gpu (test_pose_temponet_runner.py:50-51,
  test_pose_temponet_e2e_chain.py:132-133) -> the allow-list handed to the 2nd and 3rd job of the chain is checked by no test.
- R1b runner.py:289 gives calibration the storage directories (tests: runner, e2e_chain): CAUGHT by
  e2e_chain::test_all_24_stages_use_real_selection_refits_inference_and_report ("1 failed, 12 passed in 16.57s").
(preliminary totals removed: see the section 'Task 2 - FINAL counts' below)

## Task 3 - fixtures against independent evidence - all agree
Commands: two python3 scripts over the reviewed worktree (stdlib only). 32 + 20 checks pass; the 6 checks that printed FAIL in the
first script were mis-specified by me (I assumed headers.json holds the CACHE metadata users: it holds them only for the two target
references, read_headers.py: `if "meta_json.npy" in names and key.startswith("target_reference")`); redone correctly in the second script.
- sha256(json.dumps(users)): train 3e05c75e8c63947abd21147f9b3f257d2137569b7e74ad6a818f71d0c534eb7b, val
  905ab1a96d61a800cbd9911243e0b2773269a28b8a31ae72a361e9d1f677f4b9 == fixture users_json_sha256 == operator's actual_users_sha256
  (user_format_diagnosis.json). Extra: sha256(json.dumps(["u"+id])) == operator's producer_users_sha256 (e7a29def..., b6d4a7b2...).
- users == headers.json target_reference_{train,val}.meta.users (40, 15); cache metadata == reference metadata by meta_stats.json
  (same_meta_cache_vs_reference_* true, equal meta_canonical_sha256 and ids_sha256); that digest == receipt details.targets.metadata
  (b270cdd3..., 960c8328...).
- "u"+ID == the 40 / 15 directories after "emg2pose" in the two real receipts (and the same under emg2pose_pose).
- layout fixture == headers.json: config equal, 13 member names, descr and shape equal for all 13, metadata_keys equal, no fortran member.
- metrics.HISTORICAL_REFERENCE (7 values) == proposal.json evaluation historical_reference_gate block (tolerance 1e-12 there too);
  both E_dyn values present in pose_verification.json and scientific_readout.json.

## Task 4 - the missing tests, written in MY tree only and run on the delivered code
File: wf2_tests_work/test_wf2_reviewer_gaps.py (copied to wf2_tests_tree/tests/cpu/, never to the reviewed worktree). Result on the
unmodified delivered sources: "8 passed in 9.49s". So every gap below is a TEST gap; the delivered behaviour is right in each of them:
 1. validate_plan refuses users rewritten to directory names / drifted storage_directories even with execution_sha256 recomputed
    ("historical producer identity was rewritten"), train and val.
 2. read_verified_inputs refuses a cache with one user missing.
 3. P.preflight applies the user check and each of the six details to VAL.
 4. Later stages through the UNPATCHED reader on real-format files: train read for the 6 TRAIN_STAGES (depth None), val refused there;
    benchmark._validate_train, controls.validate_outer_partition, workflow.prepare_real_folds accept plan["producers"]["train"]["users"]
    and refuse storage_directories; folds from the JSON-round-tripped preflight result are accepted; val read for locked_inference
    (post, pre) and metrics (post); the metrics-stage parity gate returns exactly the preflight's result.
 5. metrics.HISTORICAL_REFERENCE == proposal evaluation.historical_reference_gate (7 values), tolerance 1e-12.
 6. tolerance boundary: +5e-13 passes, +3e-12 fails.
 7. preflight submit command: --id TCNH-preflight-20261120, --campagna pose_temponet_20261120, export CAMPAIGN/CODE_DIR satisfy the
    rule of pose_temponet_checkout.sbatch, 00:12:00, 24G, 2 CPUs.
 8. run_stage source uses plan["producers"]["train"]["users"] 3 times and never storage_directories (static check).
Surviving mutations re-run against ONLY my gap tests: P6, P6b, P7, P8e, P14, P15, P16b, P17, P18, E4, E5, R1a are now CAUGHT
(12 of 15). Still surviving: P4, P5 (set vs sorted list: equivalent once validate_plan has run), P13 (count of user directories).

## Task 2 - FINAL counts, from wf2_tests_work/mutations_tests_reviewer.json
38 mutation runs = 37 distinct mutations (P14 run twice: 3 files and 20 files; A0/A1/A2 = author's a/b/f run ONLY against the
real_layout file, which the author's run_mutations.py never executed). CAUGHT 22, SURVIVED 16 runs = 15 distinct mutations.
CAUGHT: P1 P2 P7b P7c P8a P8b P8c P8d P9 P10 P11 P12 P16 P16c P17b E1 E2 E3 R1b A0 A1 A2.
SURVIVED: P4 P5 P6 P6b P7 P8e P13 P14(+P14all) P15 P16b P17 P18 E4 E5 R1a.
A0 (first-delivery behaviour) against real_layout alone: "3 errors in 8.40s", each "ValueError: cache user population differs from
original producer" -> the real-layout fixture would have caught the original defect, with the very message of job 64931573.
A1: real_layout line 162 DID NOT RAISE; A2: real_layout line 152 DID NOT RAISE -> each of the two checks is held by a real-layout test.
Survivals are conclusive for the whole TCN suite: `grep -n "read_verified_inputs\|\.preflight(\|historical_producer" tests/cpu/*.py`
shows that only provenance and real_layout call them unpatched (e2e_chain:127 and runner:40 replace read_verified_inputs);
P18 and P14all were run on the 20 files other than e2e_chain; R1a on every file that calls run_stage.
Probe wf2_tests_work/probe_p13.py on the delivered code: a receipt with one user directory removed or added is refused
("historical cache population differs", 4 of 4 cases) -> P13 is a test gap only.
My gap tests (wf2_tests_work/test_wf2_reviewer_gaps.py, 8 passed on the delivered code) kill 12 of the 15 survivors
(P6 P6b P7 P8e P14 P15 P16b P17 P18 E4 E5 R1a); P4, P5 (equivalent after validate_plan) and P13 remain.

## Task 4 - what no test covers, ranked for the launch
1. R-1 on real data: every test that reaches historical_reference_gate rebinds M.HISTORICAL_REFERENCE (metrics:104, e2e_chain:157,
   real_layout:122); P18 (constants altered) and P17 (tolerance 1e-6) survive. Constants today == proposal block (7 of 7, checked).
   The parity itself cannot be tested without the real arrays: it stays the main unverified point of the next preflight.
2. Benchmark stage (2nd and 3rd job): the allow-list that run_stage hands to benchmark_cpu/gpu is checked by no test (R1a survives);
   delivered code is right (runner.py:272 uses ["users"]; my gap test 4: _validate_train accepts the plan list on real-format data read by
   the unpatched reader and refuses storage_directories).
3. No test runs a stage after the preflight through the unpatched read_verified_inputs on real-format files (e2e_chain and runner
   replace it). My gap test 4 does it for the 6 train stages, locked_inference (post, pre) and metrics: passes.
4. validate_plan's comparison of the producer entry (P14, P14all, P15 survive) - passes when tested (gap test 1).
5. VAL side of the preflight checks: first user check (P7), six details (P8e), whole read_verified_inputs on val (P16b) - passes (gap test 3).
6. Submit command: --campagna and the CAMPAIGN export (E4, E5) - passes (gap test 7). Covered once by the DELIVERY reviewer's dry run.
7. Population with a missing user (P6b) and number of receipt directories (P13) - other layers refuse; passes when tested.
Real ledger check (not a test): the 24 new IDs do not collide with ~/wearusfm_local/jobs/ledger.tsv (885 distinct IDs; TCNH ids there:
TCNH-CPU-inventory-login08-20261120, TCNH-CPU-inventory-login13-20261120, TCNH-preflight-20261119); no state file starts with a new ID.
Committed plan.json: job IDs == job_graph(), producers users == fixture, status REVIEW_REQUIRED, submission_released false.

## Task 5 - reviewed worktree, read-only
cd /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-tcn-users-fix; the three files once, -p no:cacheprovider,
PYTHONDONTWRITEBYTECODE=1: "34 passed in 18.03s". `git status --porcelain | wc -l` 0 before and 0 after, HEAD 0e548bf8.
Extra (my tree): each of the 8 new tests alone by node id: 8 of 8 "1 passed"; the 3 files in fully reversed collection order:
"34 passed in 17.45s" -> no order dependence.
Import claim of provenance.py:23-25: after `import pose_temponet.data`, 'wearusfm.harness.pose_eval' is already in sys.modules and
importing provenance loads no further module outside pose_temponet ("[]"); P._STORAGE_UNSAFE is the ingest object, pattern [^0-9A-Za-z_.@-].
Six details: producer runtime scripts/consolidation_long_pose_runtime.py:136 `details = P.validate_cache(P.read_cache(path, reference,
selection), reference, selection)` on the read-back file; `git diff 7d276c98 0e548bf8` empty for that file and consolidation_long_common.py too.

## Artifacts kept (small): /Users/simonebenatti/wearusfm_local/tcn_users_fix_review/tests_artifacts/
mutate.py, mutations_tests_reviewer.json (53 rows: 38 mutation runs + 15 re-runs against my gap tests), test_wf2_reviewer_gaps.py,
probe_layout.py, probe_p13.py. Own worktree and work directory removed at the end.

## Finished: 2026-11-20 15:01 CET (clock read with `date`; start 14:27)
Cleanup done: `git worktree remove --force .../scratchpad/wf2_tests_tree`, work directory deleted, `git worktree list | grep -c wf2_tests` = 0,
reviewed worktree `git status --porcelain | wc -l` = 0 at HEAD 0e548bf8218f15ba07fc731efcc2ead512e65034, no python/pytest process left (ps count 0).
Verdict: the 8 new tests protect the fix for the right reasons; no blocking or important defect found. 15 distinct mutations survive
(test gaps only: the delivered code behaves correctly in each, shown by 8 extra tests that pass today and kill 12 of the 15).
All tasks 1-5 completed.
