"""Mutation check of the corrected benchmark-plan test: every broken constraint must make it fail.

usage: mutate_r1.py <temporary checkout of GPT's tip>
Each mutation edits committed plan files inside the temporary checkout, runs the single test, and restores the bytes.
"""
import hashlib
import json
import os
import subprocess
import sys

root = sys.argv[1]
folder = os.path.join(root, 'results/infra/logsquare_evaluation_20261104')
inference = os.path.join(folder, 'inference_plan.json')
extraction = os.path.join(folder, 'extraction_plan.json')
TEST = 'tests/cpu/test_logsquare_infer_benchmark.py::test_review_plan_freezes_selection_aggregation_and_null_candidate_pins'
saved = {path: open(path, 'rb').read() for path in (inference, extraction)}


def run():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    code = "import sys; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=line', sys.argv[1]]))"
    done = subprocess.run([sys.executable, '-c', code, TEST], cwd=root, env=env, capture_output=True, text=True)
    line = [l for l in done.stdout.splitlines() if 'Error' in l or 'assert' in l]
    return done.returncode, (line[0][:150] if line else '')


def write(path, obj):
    raw = (json.dumps(obj, indent=2) + '\n').encode()
    open(path, 'wb').write(raw)
    return raw


def repin(plan, raw):
    plan['extraction_plan']['sha256'] = hashlib.sha256(raw).hexdigest()
    if 'bytes' in plan['extraction_plan']:
        plan['extraction_plan']['bytes'] = len(raw)


def restore():
    for path, raw in saved.items():
        open(path, 'wb').write(raw)


def inference_only(change):
    def apply():
        plan = json.loads(saved[inference])
        change(plan)
        write(inference, plan)
    return apply


def extraction_change(change, with_pin):
    def apply():
        plan = json.loads(saved[inference])
        e = json.loads(saved[extraction])
        change(e, plan)
        raw = write(extraction, e)
        if with_pin:
            repin(plan, raw)
        write(inference, plan)
    return apply


def reformat_only():
    """Control: both files rewritten with the same content and a consistent pin. The test must still pass."""
    plan = json.loads(saved[inference])
    e = json.loads(saved[extraction])
    raw = write(extraction, e)
    repin(plan, raw)
    write(inference, plan)


def status_review_with_pin(e, plan):
    e['status'] = plan['status'] = 'REVIEW_REQUIRED'


def review_state(e, plan):
    e['status'] = plan['status'] = 'REVIEW_REQUIRED'
    e['checkpoints']['5000'] = None


mutations = [
    ('control: unchanged files', lambda: None, 0),
    ('control: same content rewritten, pin updated', reformat_only, 0),
    ('control: both plans back to the review state, 5000 pin null', extraction_change(review_state, True), 0),
    ('status of the inference plan set to APPROVED_FOR_EXTRACTION', inference_only(lambda p: p.update(status='APPROVED_FOR_EXTRACTION')), 1),
    ('selection changed', inference_only(lambda p: p.update(selection={'changed': True})), 1),
    ('aggregation changed', inference_only(lambda p: p.update(aggregation={'changed': True})), 1),
    ('cap raised to 1 GPU-hour', inference_only(lambda p: p['allocation'].update(gpu_hours_cap=1.0)), 1),
    ('16 CPUs instead of 8', inference_only(lambda p: p['allocation'].update(cpus=16)), 1),
    ('walltime raised to 3600 s', inference_only(lambda p: p['allocation'].update(walltime_seconds=3600)), 1),
    ('extraction plan edited without updating its pin', extraction_change(lambda e, p: e.update(inference_receipt={'x': 1}), False), 1),
    ('approved plan with the 5000 pin removed, pin updated', extraction_change(lambda e, p: e['checkpoints'].update({'5000': None}), True), 1),
    ('19618 pin filled, pin updated', extraction_change(lambda e, p: e['checkpoints'].update({'19618': e['checkpoints']['5000']}), True), 1),
    ('inference receipt filled, pin updated', extraction_change(lambda e, p: e.update(inference_receipt={'x': 1}), True), 1),
    ('extraction allocations filled, pin updated', extraction_change(lambda e, p: e.update(allocations={'pose': {'gpus': 1}}), True), 1),
    ('review status with the 5000 pin still present, pin updated', extraction_change(status_review_with_pin, True), 1),
    ('5000 checkpoint path moved, pin updated', extraction_change(lambda e, p: e['checkpoints']['5000'].update(path='/tmp/other.pt'), True), 1),
    ('5000 release hash altered, pin updated', extraction_change(lambda e, p: e['checkpoints']['5000'].update(release_sha256='0' * 64), True), 1),
    ('statuses of the two plans differ, pin updated', extraction_change(lambda e, p: e.update(status='REVIEW_REQUIRED'), True), 1),
]

results = []
try:
    for name, apply, expected in mutations:
        restore()
        apply()
        code, line = run()
        results.append({'mutation': name, 'expected': 'pass' if expected == 0 else 'fail',
                        'observed': 'pass' if code == 0 else 'fail', 'as_expected': (code == 0) == (expected == 0), 'first_line': line})
finally:
    restore()
for path, raw in saved.items():
    assert open(path, 'rb').read() == raw
json.dump({'test': TEST, 'results': results, 'all_as_expected': all(r['as_expected'] for r in results)}, sys.stdout, indent=1)
