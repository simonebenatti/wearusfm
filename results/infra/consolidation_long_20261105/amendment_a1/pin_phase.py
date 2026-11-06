#!/usr/bin/env python3
"""Pin the inputs of the next phase of the long evaluation under amendment A1.

It replaces the pin snippet of section 3 of docs/handoff_line2_claude_20261105.md.
Local files only: collected receipts, audits, plans. No cluster access, no
submission; a manifest that already exists is never overwritten.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
      python results/infra/consolidation_long_20261105/amendment_a1/pin_phase.py fit|aggregate|summary [--dry-run]

fit        the fifteen classification benchmark receipts and the pose one, exactly as listed in the
           input-compatibility manifest; the original gate is recomputed (negative, 15 of 48) and the
           amended gate must equal the walltime table frozen in the plan.
aggregate  the 69 classification fit receipts of the amended source, one per registered shard, and
           the three pose fits; every collected copy must equal its audited remote hash.
summary    the two comparison receipts, in the external summary plan of section 5 of the runbook.

``--dry-run`` writes nothing and does not need the recorded decision; where the
plan already holds the pins of the phase it checks that they equal the derivation.
"""
import hashlib
import json
import sys

import consolidation_long_common as C
import consolidation_long_classification_runtime as CR
import consolidation_long_pose_runtime as PR

A = C.A
ROOT = C.ROOT
CAMPAIGN = ROOT / 'results/infra/consolidation_long_20261105'
COPIES, DEST = CAMPAIGN / 'execution/receipts', CAMPAIGN / 'execution/manifests'
PINNED = ('path', 'sha256', 'bytes')


def need(ok, message):
    if not ok:
        raise SystemExit('pin_phase: ' + message)


def text(value):
    return json.dumps(value, indent=2) + '\n'


def copy_pin(ident, remote):
    raw = (COPIES / (ident + '.json')).read_bytes()
    return json.loads(raw), {'path': remote, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def listed(inputs, row):
    """An original receipt: its collected copy must be the pin listed in the manifest."""
    entry = next(r for r in inputs['receipts'] if r['id'] == row['id'])
    report, pin = copy_pin(row['id'], entry['path'])
    need(pin == {k: entry[k] for k in PINNED} and entry['kind'] == row['stage'] == report['kind']
         and report['source_sha256'] == A.ORIGINAL_SOURCE, row['id'] + ': collected copy is not the listed input')
    return report, pin


def produced(plan, row, code_sha, audit):
    """A receipt written by the amended source: collected copy equal to the audited remote one."""
    report, pin = copy_pin(row['id'], row['out'] + '/report.json')
    checked = audit.get(row['id'], {})
    need(pin == {'path': checked.get('remote_path'), 'sha256': checked.get('sha256'), 'bytes': checked.get('bytes')},
         row['id'] + ': collected copy differs from the audited remote receipt')
    need(report['complete'] is True and report['protocol'] == C.PROTOCOL and report['kind'] == row['stage']
         and report['source_sha256'] == plan['source_sha256'] != A.ORIGINAL_SOURCE
         and report['recipe_sha256'] == plan['recipe_sha256'] and report['amendment'] == A.record()
         and report['code_sha'] == code_sha, row['id'] + ': receipt of the amended source differs')
    return report, pin


def phase_sha(launch, phase):
    sha = next(r for r in launch['phase_checkouts'] if r['name'] == f'consolidation-long-{phase}-20261105')['code_sha']
    need(isinstance(sha, str) and len(sha) == 40, phase + ': phase commit is not recorded in the launch plan')
    return sha


def audited(phase):
    return json.loads((CAMPAIGN / f'execution/audit/receipts_{phase}.json').read_text())['receipts']


def fit(plan, launch, inputs):
    reports, pins = {}, {}
    for row in launch['jobs']:
        if row['stage'] == 'benchmark-classification':
            report, pin = listed(inputs, row)
            reports.setdefault(report['dataset'], {})[report['view']] = report
            pins.setdefault(report['dataset'], {})[report['view']] = pin
    original = CR.resource_gate(reports)
    rejected = sorted(key for key, shard in original['shards'].items() if not shard['passed'])
    need(original['all_passed'] is False and len(rejected) == 15, 'the original gate is not the published negative one')
    gate = CR.amended_gate(reports, plan['amendment']['fit_shards'])
    need(gate['all_passed'] is True and gate['physical_fit_jobs'] == 69
         and gate['original_gate']['rejected_shards'] == rejected, 'amended gate failed')
    rows = {row['shard']: row for row in launch['jobs'] if row['stage'] == 'fit-classification'}
    need(set(rows) == set(gate['shards']) and all(rows[key]['allocation']['seconds'] == shard['walltime_seconds']
                                                  for key, shard in gate['shards'].items()),
         'launch rows differ from the amended gate')
    pose, pose_pin = listed(inputs, next(r for r in launch['jobs'] if r['stage'] == 'benchmark-pose'))
    projection = PR.project_benchmark(pose['timings_seconds'], pose['peak_rss_bytes'])
    need(projection == pose['projection'] and projection['admitted'] is True
         and pose['held_payload_opened'] is False and pose['held_scored'] is False
         and pose['benchmark_policy'] == PR.BENCHMARK_POLICY and pose['fit_resources'] == PR.FIT_RESOURCES,
         'pose benchmark admission differs')
    files = {'measured_resource_gates_a1.json': {'amendment': A.record(), 'classification': gate,
                                                 'original_classification': original, 'pose': projection},
             'classification_benchmark_receipts.json': {'reports': pins}}
    fields = {'benchmark_receipts': {'classification': 'classification_benchmark_receipts.json', 'pose': pose_pin}}
    summary = {'classification_fit_jobs': gate['physical_fit_jobs'], 'original_rejected_shards': len(rejected),
               'allocated_core_hours': gate['allocated_core_hours'], 'pose_seconds_per_dose': projection['seconds_per_dose']}
    return files, fields, summary


def aggregate(plan, launch, inputs):
    code, audit, table = phase_sha(launch, 'fit'), audited('fit'), plan['amendment']['fit_shards']
    reports, pose, keys = [], {}, []
    for row in launch['jobs']:
        if row['stage'] == 'fit-classification':
            report, pin = produced(plan, row, code, audit)
            key = CR.shard_key(report['checkpoint_step'], report['view'], report['dataset'], report['fold_indices'])
            need(key == row['shard'] and report['global_resource_gate_passed'] is True
                 and report['resource_gate']['passed'] is True and report['resource_gate']['folds'] == table[key]['folds']
                 and report['resource_gate']['walltime_seconds'] == table[key]['walltime_seconds']
                 and report['benchmark_manifest'] == plan['benchmark_receipts']['classification'],
                 row['id'] + ': fit receipt differs from its registered shard')
            reports.append(pin)
            keys.append(key)
        elif row['stage'] == 'fit-pose':
            report, pin = produced(plan, row, code, audit)
            need(report['held_payload_opened_after_both_ridge_fits'] is True
                 and report['benchmark_receipt'] == plan['benchmark_receipts']['pose'],
                 row['id'] + ': pose fit receipt differs')
            pose[str(report['checkpoint_step'])] = {'report': pin}
    need(len(reports) == 69 and sorted(keys) == sorted(table) and set(pose) == set(map(str, C.DOSES)),
         'the fit matrix is incomplete: no comparison on a partial matrix')
    files = {'classification_fit_manifest.json': {'reports': reports}, 'pose_fit_manifest.json': {'fits': pose}}
    fields = {'classification_fit_manifest': 'classification_fit_manifest.json', 'pose_fit_manifest': 'pose_fit_manifest.json'}
    return files, fields, {'classification_fit_reports': len(reports), 'pose_fit_reports': len(pose)}


def summary(plan, launch, inputs):
    code, audit, final = phase_sha(launch, 'aggregate'), audited('aggregate'), {}
    for kind in ('classification', 'pose'):
        report, pin = produced(plan, next(r for r in launch['jobs'] if r['stage'] == 'compare-' + kind), code, audit)
        final[kind] = pin
    return {}, {'final_reports': final}, {'final_reports': sorted(final)}


if __name__ == '__main__':
    need(len(sys.argv) in (2, 3) and sys.argv[1] in ('fit', 'aggregate', 'summary')
         and sys.argv[2:] in ([], ['--dry-run']), 'usage: pin_phase.py fit|aggregate|summary [--dry-run]')
    phase, dry = sys.argv[1], sys.argv[2:] == ['--dry-run']
    plan = json.loads((CAMPAIGN / 'plan.json').read_text())
    launch = json.loads((CAMPAIGN / 'launch_plan.json').read_text())
    inputs = json.loads((ROOT / A.INPUT_MANIFEST['path']).read_text())
    need(C.file_pin(ROOT / A.INPUT_MANIFEST['path'])['sha256'] == A.INPUT_MANIFEST['sha256']
         and plan['amendment']['input_compatibility'] == A.INPUT_MANIFEST, 'input-compatibility manifest differs')
    files, fields, report = {'fit': fit, 'aggregate': aggregate, 'summary': summary}[phase](plan, launch, inputs)
    pins = {}
    for name, value in files.items():
        raw = text(value).encode()
        pins[name] = {'path': str((DEST / name).relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw)}
    resolve = lambda value: (pins[value] if isinstance(value, str) else
                             {k: resolve(v) for k, v in value.items()} if isinstance(value, dict)
                             and set(value) != set(PINNED) else value)
    derived = {key: resolve(value) for key, value in fields.items()}
    held = {key: plan[key] for key in derived}
    empty = lambda value: value is None or isinstance(value, dict) and all(empty(v) for v in value.values())
    if dry:
        need(all(empty(held[key]) or held[key] == derived[key] for key in derived),
             'the pins already in the plan differ from the derivation')
    else:
        need(all(empty(value) for value in held.values()), 'this phase is already pinned in the plan')
    plan.update(derived)
    need(plan['source_sha256'] == C.source_identity() == launch['source_sha256']
         and plan['recipe_sha256'] == C.recipe_identity(plan) == launch['recipe_sha256'], 'plan identities differ')
    C.validate_plan(plan, require_review=not dry)
    if not dry:
        DEST.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            with (DEST / name).open('x') as stream:
                stream.write(text(value))
        target = CAMPAIGN / ('execution/summary_plan.json' if phase == 'summary' else 'plan.json')
        with target.open('x' if phase == 'summary' else 'w') as stream:
            stream.write(text(plan))
    print(json.dumps({'phase': phase, 'dry_run': dry, **report, 'pins': derived}, indent=1))
