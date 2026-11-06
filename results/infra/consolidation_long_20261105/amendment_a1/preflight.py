#!/usr/bin/env python3
"""Read-only preflight of the amended fit phase inside its campaign checkout.

It builds the context every job builds (commit, plan, recorded decision, source
identity, closed list of source differences, 28 inputs) and recomputes the two
admissions that precede the fits. It creates no output and submits nothing. It
runs on the login node before the first submission, so that a configuration
error does not consume a logical identifier: under a minute of CPU, about
3.7 GiB hashed (the fifteen classification caches, the three pose train
caches and the artifacts of the pose benchmark), all 28 listed receipts read.
The held pose caches are not read, not even to hash them: a fit opens them
only after both ridge fits. The numeric environment of the fits is checked by
the jobs.

    module purge; module load profile/deeplrn; module load cineca-ai/4.3.0
    source "$WORK/venv_wearusfm/bin/activate"
    cd "$LONG_CODE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=src:scripts python3 results/infra/consolidation_long_20261105/amendment_a1/preflight.py "$LONG_PHASE_SHA"
"""
import json
import sys

import consolidation_long_common as C
import consolidation_long_classification_runtime as CR
import consolidation_long_pose_runtime as PR

PLAN = C.ROOT / 'results/infra/consolidation_long_20261105/plan.json'

if __name__ == '__main__':
    if len(sys.argv) != 2 or len(sys.argv[1]) != 40:
        raise SystemExit('usage: preflight.py <full SHA of the fit phase commit>')
    ctx = C.load_context(PLAN, sys.argv[1])
    gate = CR.amended_gate(CR._benchmarks(ctx), ctx['plan']['amendment']['fit_shards'])
    C.require(gate['all_passed'] is True and gate['original_gate']['all_passed'] is False, 'classification admission differs')
    for step in C.DOSES:
        for dataset in CR.M.DATASETS:
            CR._cache(ctx, step, dataset)
    pose = PR._benchmark_gate(ctx, check_environment=False)
    held = PR._manifest(ctx)['pose']
    for step in C.DOSES:
        PR._entry(ctx, step, 'train')
        entry = held[str(step)]['val']
        report = C.verified_input(ctx, entry['receipt'], 'extract-pose-val')
        C.require(report['checkpoint_step'] == step and entry['cache'] in report['artifacts'].values(),
                  'held pose cache is not covered by its listed receipt')
    print(json.dumps({'preflight': 'passed', 'code_sha': ctx['code_sha'], 'plan_sha256': ctx['plan_pin']['sha256'],
                      'source_sha256': ctx['source_sha256'], 'input_source_sha256': ctx['inputs']['source_sha256'],
                      'listed_inputs': len(ctx['inputs']['receipts']), 'decision': ctx['plan']['amendment']['decision'],
                      'classification_fit_jobs': gate['physical_fit_jobs'],
                      'classification_allocated_core_hours': gate['allocated_core_hours'],
                      'original_rejected_shards': len(gate['original_gate']['rejected_shards']),
                      'pose_benchmark': pose, 'held_pose_caches_read': 0, 'outputs_created': 0},
                     indent=1, ensure_ascii=False))
