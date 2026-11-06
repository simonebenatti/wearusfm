#!/usr/bin/env python3
"""Derive the amended plan.json and launch_plan.json of amendment A1, or check the committed ones.

Inputs: the two plans as published at the end of the benchmark phase (commit
BASE, read with ``git show``), the fifteen collected classification benchmark
receipts, the archived accounting and the frozen amendment module. The walltime
table is the amended gate recomputed on those receipts. No cluster access, no
submission. Run from the repository root with the project environment:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
      python results/infra/consolidation_long_20261105/amendment_a1/apply_to_plans.py --check   (or --write)

``--check`` ignores only what later phase pins and records change: receipt and
manifest pins, the recorded decision, phase SHAs and SLURM identifiers.
"""
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import consolidation_long_common as C
import consolidation_long_classification_runtime as CR

A = C.A
BASE = '2a20316a0bf77c93e354d4ce3f5acee94db459df'
CAMPAIGN = 'results/infra/consolidation_long_20261105'
ROOT = C.ROOT
PROPOSAL = {'branch': 'codex/consolidation-20261104', 'commit': '4bb39d254c26bd3b9666dff0e1b1464f96bfe4db',
            'document': 'docs/consolidation_long_resource_amendment_proposal_20261106.md'}
DESIGN_REVIEW = {'branch': 'worktree-claude-training-engine', 'commit': '7ceedbb5d891168f585229e826baf6535f35c1cc',
                 'document': 'docs/reviews/claude_review_consolidation_long_resource_amendment_20261106.md',
                 'verdict': 'design approved; implementation verifications V-1...V-8; decision of Simone before pin'}
LANE = ('at most two line-2 jobs running or pending in lrd_all_serial, phase checkout and pose fits included, '
        'until the common ledger shows a submitted identifier starting with SCALING-compare- (decision of Simone, '
        '2026-11-06: «Due fit per volta»); afterwards waves within the plan limit of eight outstanding CPU jobs')
PENDING = 'implemented_awaiting_simone_decision'


def need(ok, message):
    if not ok:
        raise SystemExit('apply_to_plans: ' + message)


def base(name):
    return json.loads(subprocess.check_output(['git', 'show', f'{BASE}:{CAMPAIGN}/{name}'], cwd=ROOT))


def core_hours(cpus, seconds):
    return cpus * seconds / 3600


def spent():
    """Consumption of the thirty concluded allocations, from the archived sacct tables."""
    core = gpu = 0.0
    rows = 0
    for phase in ('extract', 'benchmark'):
        text = (ROOT / CAMPAIGN / f'execution/accounting/sacct_{phase}.tsv').read_text()
        for row in csv.DictReader(io.StringIO(text), delimiter='\t'):
            need(row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0', phase + ': an allocation is not COMPLETED 0:0')
            tres = dict(item.split('=', 1) for item in row['AllocTRES'].split(','))
            core += int(row['ElapsedRaw']) * int(row['AllocCPUS']) / 3600
            gpu += int(row['ElapsedRaw']) * int(tres.get('gres/gpu', '0')) / 3600
            rows += 1
    need(rows == 30, 'archived accounting does not hold thirty allocations')
    return core, gpu


def build():
    plan, launch = base('plan.json'), base('launch_plan.json')
    need('amendment' not in plan and len(launch['jobs']) == 82
         and plan['source_sha256'] == A.ORIGINAL_SOURCE == launch['source_sha256']
         and plan['recipe_sha256'] == A.ORIGINAL_RECIPE == launch['recipe_sha256']
         and plan['benchmark_receipts'] == {'classification': None, 'pose': None}, 'unexpected base plans')
    need(all(row['slurm_job_id'] is None for row in launch['jobs'] if row['phase'] in ('fit', 'aggregate')),
         'a fit or aggregate row of the base plan was already submitted')
    reports = {}
    for row in launch['jobs']:
        if row['stage'] == 'benchmark-classification':
            report = json.loads((ROOT / CAMPAIGN / 'execution/receipts' / (row['id'] + '.json')).read_text())
            reports.setdefault(report['dataset'], {})[report['view']] = report
    gate = CR.amended_gate(reports)
    need(gate['all_passed'] is True and gate['original_gate']['all_passed'] is False
         and len(gate['original_gate']['rejected_shards']) == 15, 'gates differ from the reviewed proposal')
    table = {key: {name: shard[name] for name in ('step', 'view', 'dataset', 'folds', 'projected_seconds',
                                                  'walltime_seconds')} for key, shard in gate['shards'].items()}
    source = C.source_identity()
    need(source != A.ORIGINAL_SOURCE, 'the amended source equals the original one')

    old_policy = plan['phase_release']['source_change_policy']
    need(old_policy == launch['phase_source_policy'], 'the base plans disagree on the source policy')
    policy = {
        'choice': 'R-2(i) with amendment A1: exact source identity for every new output; explicit exception '
                  'limited to the 28 original receipts of the input-compatibility manifest',
        'frozen_paths': old_policy['frozen_paths'],
        'freeze_period': 'amended source frozen from the fit checkout through chain completion; GPT development '
                         'isolated and not merged during the chain',
        'amendment_A1': {
            'exception': 'the 12 extraction and 16 benchmark receipts written by the original source are read by '
                         'the amended source through the pinned manifest; fits, comparisons and summary share the '
                         'amended source and record the manifest pin',
            'not_allowed': 'receipt rewrite; generic accepted-source list; any input not listed; extraction or '
                           'benchmark by the amended source; a covered source difference outside the closed list',
            'input_compatibility': dict(A.INPUT_MANIFEST),
            'original_source_sha256': A.ORIGINAL_SOURCE, 'amended_source_sha256': source,
            'changed_paths': list(A.CHANGED_PATHS), 'added_paths': list(A.ADDED_PATHS),
            'amended_definitions': {path: list(names) for path, names in A.AMENDED_DEFINITIONS.items()},
            'added_definitions': {path: list(names) for path, names in A.ADDED_DEFINITIONS.items()},
            'added_after': {path: dict(anchors) for path, anchors in A.ADDED_AFTER.items()},
            'frozen_text': 'in the changed Python files every region outside the amended and added definitions keeps '
                           'its original bytes (decorators, comments and blank lines included) and the top-level '
                           'statements keep their original order; in the launcher only the replaced lines differ',
            'closed_stages': list(A.CLOSED_STAGES),
            'decision': None,
        },
        'invalidation': 'any further covered source change is another source: the exception and the 28 inputs are '
                        'not valid under it without a new review and decision',
        'action': 'stop chain; preserve all artifacts; no receipt rewrite or automatic resubmission',
        'repeat_requires': 'new Claude review, new IDs/outputs, complete new estimate and Simone decision; all '
                           'twelve extractions and sixteen benchmarks repeated',
        'additional_extraction_gpu_hours_cap': old_policy['additional_extraction_gpu_hours_cap'],
        'additional_cpu': old_policy['additional_cpu'],
        'additional_cost_in_current_cap': old_policy['additional_cost_in_current_cap'],
        'repeat_authorized': old_policy['repeat_authorized'],
        'original_r2i': old_policy,
    }

    plan['source_sha256'] = source
    plan['amendment'] = {
        'id': A.AMENDMENT, 'title': 'resource amendment after the complete benchmark phase', 'status': PENDING,
        'post_hoc': 'fixed after the sixteen train-only benchmark timings (15 of 48 original shards over the '
                    '13800 s limit) and before any fit, score or prediction of the long comparison; it concerns '
                    'resources and provenance only and is not a pre-registered rule',
        'unchanged': 'forty scientific cells, four folds each, recipe, C grid, selection, solver, numeric context, '
                     'data, endpoints, reading rule, per-fold projection and safety factors',
        'proposal': PROPOSAL, 'design_review': DESIGN_REVIEW, 'base_commit': BASE,
        'fit_policy': A.FIT_POLICY, 'closed_stages': list(A.CLOSED_STAGES),
        'input_compatibility': dict(A.INPUT_MANIFEST),
        'release': {'order': 'three pose fits first, then the 69 classification fits; the pose gate is independent',
                    'lane': LANE, 'timeout': 'a timeout stops the phase: no requeue, new walltime or new shard'},
        'fit_shards': table,
        'decision': None,
    }
    release = plan['phase_release']
    release['source_change_policy'] = policy
    release['cpu_gate'] = ('amendment A1: all forty classification cells / sixty-nine registered shards pass, with the '
                           'walltime table frozen in the plan, before any classification fit; pose gate separately; '
                           'failed phase has no automatic retry')
    release['source_sha256_scope'] = ('one amended source for every new output; the 28 inputs keep the original '
                                      'source through the input-compatibility manifest')
    plan['authorization']['amendment_A1'] = PENDING

    jobs, superseded, successors = [], [], {}
    fits = []
    for key, shard in gate['shards'].items():
        tag = ''.join(map(str, shard['folds']))
        ident = f"LONG-FIT-G{shard['step']}-{shard['view']}-{shard['dataset']}-F{tag}-20261105-A1"
        out = f"{plan['output_root']}/fits_a1/classification/{shard['step']}/{shard['view']}/{shard['dataset']}/F{tag}"
        fits.append({'id': ident, 'phase': 'fit', 'stage': 'fit-classification', 'out': out,
                     'allocation': {'gpus': 0, 'cpus': A.FIT_POLICY['cpus'], 'memory_gib': A.FIT_POLICY['memory_gib'],
                                    'seconds': shard['walltime_seconds']},
                     'args': ['--stage', 'fit-classification', '--out', out, '--step', str(shard['step']),
                              '--dataset', shard['dataset'], '--view', shard['view'],
                              '--folds', *map(str, shard['folds'])],
                     'prerequisites': ['Amendment A1: decision of Simone recorded in the plan; immutable fit checkout '
                                       'verified; published pins and frozen walltime table; pose fits released first; '
                                       'serial-lane rule respected; fresh quota; no existing logical ID/output'],
                     'slurm_job_id': None, 'amendment': A.AMENDMENT, 'shard': key})
        successors.setdefault((shard['step'], shard['view'], shard['dataset']), []).append((shard['folds'], ident))
    pose = [row for row in launch['jobs'] if row['stage'] == 'fit-pose']
    for row in launch['jobs']:
        if row['stage'] == 'fit-classification':
            args = row['args']
            cell = (int(args[args.index('--step') + 1]), args[args.index('--view') + 1], args[args.index('--dataset') + 1])
            folds = list(map(int, args[args.index('--folds') + 1:]))
            after = [ident for covered, ident in successors[cell] if set(covered) <= set(folds)]
            need(after and sorted(f for covered, _ in successors[cell] if set(covered) <= set(folds) for f in covered)
                 == folds, row['id'] + ': successors do not cover the superseded folds exactly')
            superseded.append({**row, 'disposition': 'superseded by amendment A1; never submitted; not a failed job',
                               'superseded_by': after})
        elif row['stage'] == 'fit-pose':
            continue
        else:
            if row['phase'] == 'aggregate' and pose:
                jobs.extend(pose + fits)
                pose = []
            jobs.append(row)
    need(len(superseded) == 48 and len(jobs) == 103 and len(fits) == 69
         and sorted(i for row in superseded for i in row['superseded_by']) == sorted(row['id'] for row in fits)
         and len({row['id'] for row in jobs + superseded}) == 151 and len({row['out'] for row in jobs + superseded}) == 151,
         'amended launch matrix differs')
    launch['jobs'], launch['superseded_jobs'] = jobs, superseded
    launch['source_sha256'] = source
    launch['phase_source_policy'] = policy

    checkout = launch['caps']['checkout_each']
    each = core_hours(checkout['cpus_accounting_cap'], checkout['seconds'])
    allocation = lambda rows: sum(core_hours(r['allocation']['cpus'], r['allocation']['seconds']) for r in rows)
    concluded = allocation([r for r in jobs if r['phase'] in ('extract', 'benchmark')]) + 2 * each
    fit_cap = allocation(fits)
    other = allocation([r for r in jobs if r['phase'] in ('fit', 'aggregate') and r['stage'] != 'fit-classification']) + 2 * each
    spent_core, spent_gpu = spent()
    cap = launch['caps']['allocated_core_hours']
    need(abs(fit_cap - gate['allocated_core_hours']) < 1e-9 and concluded + fit_cap + other <= cap
         and sum(r['allocation']['gpus'] * r['allocation']['seconds'] / 3600 for r in jobs) == launch['caps']['gpu_hours'],
         'amended allocations exceed the approved caps')
    launch['caps']['amendment_A1'] = {
        'caps_unchanged': True, 'classification_fit_jobs': len(fits),
        'classification_fit_projected_core_hours': gate['projected_core_hours'],
        'classification_fit_allocated_core_hours': fit_cap,
        'other_future_allocated_core_hours': other, 'future_allocated_core_hours': fit_cap + other,
        'new_gpu_hours': 0.0, 'spent_core_hours': spent_core, 'spent_gpu_hours': spent_gpu,
        'spent_plus_future_core_hours': spent_core + fit_cap + other,
        'concluded_phases_allocated_core_hours': concluded,
        'conservative_total_core_hours': concluded + fit_cap + other,
        'margin_to_cap_core_hours': cap - concluded - fit_cap - other,
        'compute_jobs': len(jobs), 'allocations_with_checkouts': len(jobs) + launch['caps']['checkout_count'],
        'allocations_left': len(fits) + 3 + 3 + 2,
    }
    launch['amendment_A1'] = {'status': PENDING, 'proposal': PROPOSAL, 'design_review': DESIGN_REVIEW,
                              'release_order': plan['amendment']['release']['order'], 'lane': LANE,
                              'superseded_jobs': 'the 48 original fit rows are kept under superseded_jobs with their '
                                                 'successors; none was reserved in the ledger or submitted'}
    C.validate_plan(plan, require_review=False)
    need(plan['recipe_sha256'] == C.recipe_identity(plan) == launch['recipe_sha256'], 'recipe identity changed')
    return plan, launch


def neutral(plan, launch):
    """Blank what later pins and records legitimately change."""
    plan, launch = copy.deepcopy(plan), copy.deepcopy(launch)
    for key in ('benchmark_receipts', 'classification_fit_manifest', 'pose_fit_manifest', 'final_reports'):
        plan[key] = None
    plan['amendment']['decision'] = plan['amendment']['status'] = plan['authorization']['amendment_A1'] = None
    plan['phase_release']['source_change_policy']['amendment_A1']['decision'] = None
    launch['phase_source_policy']['amendment_A1']['decision'] = launch['amendment_A1']['status'] = None
    for row in launch['phase_checkouts']:
        row['code_sha'] = None
    for row in launch['jobs']:
        row['slurm_job_id'] = None
    return plan, launch


if __name__ == '__main__':
    mode = sys.argv[1:] or ['--check']
    need(mode in (['--check'], ['--write']), 'usage: apply_to_plans.py [--check|--write]')
    plan, launch = build()
    paths = [ROOT / CAMPAIGN / 'plan.json', ROOT / CAMPAIGN / 'launch_plan.json']
    if mode == ['--write']:
        need('amendment' not in json.loads(paths[0].read_text())
             and 'superseded_jobs' not in json.loads(paths[1].read_text()), 'the plans are already amended')
        for path, value in zip(paths, (plan, launch)):
            path.write_text(json.dumps(value, indent=2) + '\n')
    committed = [json.loads(path.read_text()) for path in paths]
    need(neutral(*committed) == neutral(plan, launch), 'committed plans differ from the derivation')
    print(json.dumps({'source_sha256': plan['source_sha256'], 'jobs': len(launch['jobs']),
                      'superseded': len(launch['superseded_jobs']), **launch['caps']['amendment_A1']}, indent=1))
