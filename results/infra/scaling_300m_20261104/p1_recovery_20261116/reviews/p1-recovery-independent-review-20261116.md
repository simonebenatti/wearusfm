# Independent review of conditional P1 recovery

Reviewed the current `scripts/scaling_p1_recovery.py` and
`scripts/scaling_p1_recovery_plans.py` in `wearusfm-p1-recovery` on 2026-11-16.

Verdict: **APPROVATO**, conditional on the already specified diagnostic and
operational budget gates. No new release authorization follows from this review.

The execution path reuses `scaling_evaluation19618.load_context`, the original
classification command generator and the unchanged cache validator. Comparison
against the original `execute` confirms preservation of the raw input state and
inventory checks, reference pins, all final `check_unchanged` checks, scientific
source contract verification, code identity and STOP guard. Only newly produced
P1 outputs use the bounded finalization helper. There is one producer invocation,
one new output directory and no loop that retries the producer. An exception at
the producer, finalizer, input guard or STOP guard prevents a complete receipt.

Import order is preserved: the wrapper imports `scaling_evaluation19618` before
the output helper; the evaluation module retains its NumPy-first bootstrap. The
scientific source files and source contract validation are unchanged. A direct
`git diff df2de35 -- src` and the production/extraction/reader files used in this
path is empty.

The recovery manifest requires complete METADATA_ONLY_REPRODUCED evidence and
READERS_CLEAR_IN_TEST, the diagnostic helper source hash equal to the runtime
helper, a passing CPU test receipt covering all declared recovery source hashes,
and the original extraction plan pinned by bytes. Null, incomplete, affected-reader
and integrity-failure diagnostics cannot release recovery. The output barrier is
fixed to none, not selected adaptively from the diagnostic matrix.

Mixed-provenance construction preserves actual P1 recovery identity and original
pose/P2 producers. It hashes collected receipt and sidecar bytes, checks accepted
accounting, rejects both historical FAILED P1 IDs, checks model/checkpoint/source
contract consistency, and returns cache pins acceptable to the existing frozen
pose and probe plan builders. No historical receipt is manufactured or rewritten.

Validation rerun independently:

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_p1_recovery.py tests/cpu/test_scaling_p1_recovery_plans.py`

**47 passed in 6.81 s.** This includes one producer/three finalizations, no repeat
producer or complete receipt on failures, manifest rejection mutations, mixed
lineage rejection mutations and compatibility with both unchanged reader plans.

Scope limits: no remote actions, no inference, no allocation or ledger mutation,
no validation of a real future diagnostic or recovery receipt. Submission-time
accounting, unique logical ID, allocation and cumulative ceilings remain the
operational launch gates; the pure manifest validator is not itself a scheduler
or an accounting collector.
