# P1 identity: independent review of mixed provenance, readers and recovery alternatives

16/11/2026. Read-only review of c2ae7ee02b3df5bdc3310b90d6608e211753dd07 in `/private/tmp/wearusfm-p1-identity-review-20261116`. No remote operations, jobs, inference, editing reviewed files or real-cache loading. Independent local reproduction: `/private/tmp/p1-mixed-review-20261116.py` and `.json`, **8 checks PASS**. The synthetic fixture check covers schema/builders, not actual 19k remote payloads.

## Q4: Prefer preserving pose/P2 over rerunning all three

Mixed extraction commits are scientifically acceptable if the difference is exclusively publication/identity IO, and each cache records its actual producer commit. The model checkpoint, numerical extraction scripts and `src/`, preprocessing, import order, runtime, raw selections, teacher/gain handling, recipe, arrays and folds must stay unchanged. Cache SHA bytes will differ when the embedded code SHA changes; that is expected metadata provenance, not permission to change feature values. No reason emerges to repeat successful pose/P2 merely to have the same enclosing Git commit.

The existing readers already support per-cache extraction identities:

- `scaling_probe19618.validate_plan:95–112` checks code SHA per dataset. It enforces one checkpoint per arm, **not** one extraction code SHA per arm. `load_cache:121–145` compares each receipt/provenance/embedded NPZ identity to that cache's pin.
- `scaling_pose_readout19618.validate_plan:77–91` and `validate_artifact:104–113` likewise use per-cache SHA; pose train/val still need the same checkpoint.
- The root blocker is operational: `scaling_plans19618.build_reader_plans:1284–1293` makes one `extraction` object from one `code_sha` argument and uses it for all three jobs. `_extraction_identity:196–204` requires that one SHA for both P1/P2. `candidate_probe_caches:239–259` propagates the receipts' code into per-cache pins once this guard passes.
- Reproduction: a coherent synthetic P1 receipt/provenance set with different code SHA is rejected at the builder (`extraction receipt differs: code_sha`); the resulting reader plan with per-P1 code pins is accepted by the actual reader's `validate_plan`, with unchanged candidate checkpoint.

Minimal bridge: explicit mapping by extraction stage/task from accepted job ID + receipt SHA + code SHA + plan SHA + output directory; use that mapping in candidate builders and enforce the exact permitted old/new pair. Preserve pose/P2 `df2de35` and their original receipts. No arbitrary acceptance of any commit and no rewriting old receipts as if new code produced them. Collect and verify successful pose/P2 before trusting their payloads, since review currently only has scheduler state and claims of receipt existence.

A new reviewed package is needed for the code change. `release_check:1031–1050` allows only new evidence/documents after review. `rereview_check:1057–1068` only allows Mac-side tool/tests/docs/new campaign evidence relative to an earlier deployed checkout. A cluster wrapper modification cannot pass by simply changing the `reviewed` SHA: the old code checkout is already deployed, and `checkout_gate:1217–1226` checks its relation too. Publish a specifically versioned extraction recovery checkout/ID and a narrowly bounded bridge for subsequent readers; do not edit the old checkout or relax the global guard.

## Q3/Q5: A global helper change is neither minimal nor comprehensive

`tokenizer_preflight.file_identity` and `check_unchanged` are shared. **That file is a pinned scientific source in `training_source_contract.json`**, SHA `3bdd90d0a30c1089953345c0b63415f8c40974939af6dd05da63f11d59affe69`; `scaling_train.SCRIPT_PATHS` includes it and `check_source_contract:108–117` validates all exact bytes. Changing it in-place would invalidate the accepted training contract in extraction and readers. Do not silently repin historical scientific sources for an IO policy change.

`scaling_evaluation19618.py` is a strictly derived file; any edit requires a declared derivation/guard amendment or a separate reviewed recovery entry point. Keep the numerical primitive extraction files and historical contract untouched. An explicit new IO helper used by the affected wrapper can avoid changing the old helper, provided no monkeypatch obscures the executed behavior and code/source receipts cover the new helper. The same repair must cover both the initial output hash and the post-validation identity comparison: a fallback only inside the first hash leaves the next full-dictionary comparison vulnerable to a benign metadata transition.

The claim that all readers use the same `check_unchanged` is incomplete:

- Classification uses `ST.resolve_pin` -> `P.file_identity` and `E.P.check_unchanged`, including input caches and some resource plans.
- Pose uses `tokenizer_pose_readout.verify_file:129–135`, which hashes content and checks `pose_temporal_readout.file_state` before/after. That state is `[dev,inode,size,mtime_ns,ctime_ns]` (`pose_temporal_readout:119–122`).
- `scaling_pose_readout19618:143` repeats the stat comparison after validation, and `:349` does so after the whole fit/predict. It does **not** rehash every input at that final comparison.
- A fix only to `P.file_identity/check_unchanged` therefore would not cover pose or every state-based guard. Some referenced primitive scripts are separately pinned in `EXTRACTION_SOURCE_SHA256`, and all current reader plans pin their own source hashes.

Recommendation: diagnose read-only old files using the **actual guard sequences for both families**, including long-gap checks, before spending the long serial chain. Existing validate/benchmark phases already exercise large portions of these sequences; preserve that staged release rather than enqueue all fits before observing them. Do not widen every shared guard speculatively because a just-written producer file had an unexplained failure. If an old immutable cache exhibits metadata-only movement with stable content, apply a reviewed IO policy at the precise guards involved, for both sizes, before fits. If hashes, sizes, object identity, permissions or known pin differ, stop; never reinterpret an arbitrary mutation as a benign timestamp refresh. Even a successful bounded diagnostic is evidence only for its tested conditions, not proof that 43 hours cannot fail.

## Q7: Alternatives and what can honestly be recovered

### A. Operational settling outside unchanged extraction code

A wrapper that merely delays or calls sync **before** `scaling_evaluation19618.execute` cannot insert a barrier after its internal child finishes. The current CLI requires a fresh output directory and always reruns extraction; it has no `validate-existing-output` stage. `stage='validate'` only validates input context and creates a receipt with no extracted outputs. Reusing it would not validate these existing caches.

Creating a second Python process to finish verification from existing outputs, or changing the producer to fsync/atomic-rename, therefore is a new recovery entry point/protocol, not a no-code reuse of this CLI. A completely unchanged identical relaunch is the already failed approach and is not supported as the next experiment.

### B. Post-hoc acceptance of already extracted caches

Potentially cheaper in GPU time, but currently **not** a normal successful extraction. Both jobs remain `FAILED`; first has DB2/DB3 provenance but no DB6 provenance, second only DB2 provenance, neither has a complete receipt. The wrapper writes global receipt only after:

1. child exit and raw source before/after checks;
2. all output hashes and content/target/fold checks;
3. final `P.check_unchanged(states)` for checkpoint, config, plan, sources etc.;
4. scientific-source check and code/STOP verification.

The stack location demonstrates that the raw-source and inventory checks before the output loop were reached successfully; however their concrete snapshots and the final overall state never reached a persisted receipt. Missing checks cannot be invented retrospectively. Embedded P1 NPZ metadata does pin the checkpoint hash, code SHA, recipe, selected inputs, rows and folds, which can be revalidated now. Re-reading all caches and verifying ZIP/arrays/shape/finiteness plus original input/support metadata is useful evidence, but a currently stable hash alone cannot attest everything the failed process did not persist.

A salvage route would need an **explicit recovery receipt** recording failed producer IDs/commits, immutable current cache pins, successful original prefix of checks and its evidence, newly performed validation, and limits on missing start/end observations. No manufacture of the original `complete=true`, no changed scheduler history, no modifying the partial directories. Controller/readers must explicitly recognize this new recovery protocol rather than falsifying the normal complete-extraction receipt. Checkpoint/source and current raw identities should be checked from existing fixed pins; metadata refresh does not license weaker content identity.

This alternative avoids rerunning GPU inference but is a larger provenance amendment than P1-only rerun with a narrow IO repair. It should remain an option after diagnostics, especially if the integrity evidence is sufficient and cost matters, not automatically the chosen route. First attempt DB2/DB3 sidecars versus second DB2 can support comparisons; mixing datasets from different failed attempts creates further lineage complexity and should not be the default.

### C. Copy to node-local staging

Copying existing immutable pinned reader inputs to node-local storage, verifying whole-file SHA against accepted pins before use and after copying, can isolate client metadata trouble while leaving numerical reader code untouched. It changes physical paths/provenance and needs plan support and scratch capacity/timing estimates; extracting to local temp then copying also changes output publication. It does not prove Lustre's cause, cannot repair missing receipts, and is not free under the existing CPU budget. Consider only if diagnostics localize a Lustre metadata defect and the simpler bounded IO repair is insufficient.

## Verdict for my scope

**B con modifiche**: diagnostic evidence first, no automatic third inference. Prefer preserving valid pose/P2 and using an explicit per-task provenance bridge if P1 must rerun. Keep the old shared source helper/contract intact. Harden only demonstrated affected reader guards, before fits, and acknowledge the separate pose stat path. No review finding here authorizes jobs or increases the budget.
