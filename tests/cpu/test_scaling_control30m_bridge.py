"""Independent-control provenance joins the original 49k numerical reader plans."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_plans49046 as S
import scaling_control30m_bridge as B

CONTROL_CODE = 'a' * 40
CANDIDATE_SHA = 'b' * 64


def test_control_plan_contains_only_the_actual_preserved_30m_endpoint():
    plan = B.build_control_plan(S, ROOT)
    old = S._load(B.OLD + '/evaluation/extraction_plan.json', ROOT)
    receipt = S._load(B.CONTROL_RECEIPT, ROOT)
    assert plan['arms'] == [S.ARMS[0]] and plan['control_only'] is True
    assert set(plan['checkpoints']) == {S.ARMS[0]}
    pin = plan['checkpoints'][S.ARMS[0]]
    assert {k: pin[k] for k in ('path', 'sha256', 'bytes')} == {
        k: receipt['persistent_checkpoint'][k] for k in ('path', 'sha256', 'bytes')}
    assert pin['receipt_path'] == S.checkout('code') + '/' + B.CONTROL_RECEIPT
    assert pin['receipt_sha256'] == S.digest((ROOT / B.CONTROL_RECEIPT).read_bytes())
    assert plan['step'] == 49046 and plan['protocol'] == S.E.PROTOCOL
    for key in ('comparison', 'inputs', 'paths', 'pose_references', 'classification_references'):
        assert plan[key] == old[key]


@pytest.mark.parametrize('change', [
    lambda r: r.update(complete=False), lambda r: r.update(step=19618),
    lambda r: r['summary'].update(steps=49045), lambda r: r['summary'].update(stopped='time_limit'),
    lambda r: r['persistent_checkpoint'].update(sha256='f' * 64),
    lambda r: r['persistent_checkpoint'].update(bytes=1),
    lambda r: r['persistent_checkpoint'].update(path='checkpoint_postdecay.pt'),
])
def test_control_plan_rejects_an_unaccepted_endpoint(monkeypatch, change):
    original = S._load
    def load(name, root):
        result = original(name, root)
        if name == B.CONTROL_RECEIPT:
            change(result)
        return result
    monkeypatch.setattr(S, '_load', load)
    with pytest.raises(ValueError, match='preserved 30M endpoint'):
        B.build_control_plan(S, ROOT)


@pytest.fixture
def joined(monkeypatch):
    """Plausible future receipts, with dataset identities from accepted real plans.

    Cache bytes are not read by this pure builder. Actual cache validation still
    belongs to the unchanged reader; this fixture tests the provenance wiring.
    """
    control = B.build_control_plan(S, ROOT)
    control_sha = S.digest(S.dumps(control))
    final = deepcopy(control)
    del final['control_only']
    final['arms'] = list(S.ARMS)
    final['checkpoints'][S.ARMS[1]] = {'path': '/candidate/checkpoint_postdecay.pt',
                                      'sha256': CANDIDATE_SHA, 'bytes': 4673783740}
    final_sha = S.digest(S.dumps(final))
    original_load = S._load
    probe = original_load(B.OLD + '/readers/probe_plan.json', ROOT)
    files = {}
    counter = 0
    for arm in S.ARMS:
        cp = control['checkpoints'][arm] if arm == S.ARMS[0] else final['checkpoints'][arm]
        code, plan_sha = (CONTROL_CODE, control_sha) if arm == S.ARMS[0] else (B.READER_CODE, final_sha)
        for task, names in (('pose', ('train', 'val')), *S.TASKS.items()):
            common = {'protocol': S.E.PROTOCOL, 'arm': arm, 'which': 'teacher', 'checkpoint_step': 49046,
                      'checkpoint_sha256': cp['sha256'], 'code_sha': code, 'plan_sha256': plan_sha,
                      'complete': True, 'checkpoint_unchanged': True, 'postload_recipe_passed': True,
                      'stage': 'extract-pose' if task == 'pose' else 'extract-classification'}
            receipt = {**common, 'outputs': {}}
            prefix = arm + '/' + task + '/'
            for name in names:
                counter += 1
                stem = 'train_subset' if task == 'pose' and name == 'train' else name
                path = '/early-join-fixture/' + prefix + stem + '.npz'
                provenance = {**common, 'cache': path, 'cache_sha256': f'{counter:064x}', 'cache_bytes': 1234}
                if task == 'pose':
                    provenance['split'] = name
                else:
                    provenance.update(dataset=name, task=task,
                        **{k: probe['datasets'][name][k] for k in ('input_sha256', 'splits_sha256', 'extraction_recipe')})
                data = S.dumps(provenance)
                files[prefix + stem + '.npz.provenance.json'] = (data, S.pin(path + '.provenance.json', data))
                receipt['outputs'][name] = {'path': path, 'sha256': provenance['cache_sha256'], 'bytes': 1234,
                                            'provenance_path': path + '.provenance.json', 'provenance_sha256': S.digest(data)}
            raw = S.dumps(receipt)
            files[prefix + 'receipt.json'] = (raw, S.pin('/early-join-fixture/' + prefix + 'receipt.json', raw))
    manifest = {'stage': 'extractions', 'jobs': {job['id']: {'job_slurm': str(i + 1)}
                                               for i, job in enumerate(S.jobs('extractions'))}}
    relative = S.PACKAGE + '/' + S.DOCUMENTS['extraction_plan'][1]
    monkeypatch.setattr(S, '_load', lambda name, root: deepcopy(final) if name == relative else original_load(name, root))
    monkeypatch.setattr(S, 'document', lambda name, root: S.pin('/fixture/final-plan.json', S.dumps(final)))
    monkeypatch.setattr(S, 'evidence', lambda stage, root: (files, manifest))
    return {'control': control, 'control_sha': control_sha, 'final': final, 'files': files,
            'manifest': manifest, 'code': CONTROL_CODE, 'reader_code': B.READER_CODE}


def run_join(fixture):
    return B.mixed_reader_plans(S, fixture['reader_code'], ROOT, fixture['control'],
                                fixture['control_sha'], fixture['code'])


def test_join_preserves_per_arm_producer_pins_and_original_numerical_sources(joined):
    before = deepcopy(joined)
    pose, probe, lineage = run_join(joined)
    assert joined == before
    assert pose['sources'] == S.P.source_hashes() and probe['sources'] == S.C.source_hashes()
    for arm in S.ARMS:
        wanted_code = CONTROL_CODE if arm == S.ARMS[0] else B.READER_CODE
        wanted_plan = joined['control_sha'] if arm == S.ARMS[0] else S.digest(S.dumps(joined['final']))
        assert {pin['extraction_code_sha'] for pin in pose['caches'][arm].values()} == {wanted_code}
        assert {pin['extraction_code_sha'] for pin in probe['caches'][arm].values()} == {wanted_code}
        assert {pin['extraction_plan_sha256'] for pin in probe['caches'][arm].values()} == {wanted_plan}
        assert lineage['producers'][arm]['plan_sha256'] == wanted_plan
        assert set(lineage['receipts'][arm]) == {'pose', 'p1', 'p2'}
    assert lineage['reader_code_sha'] == B.READER_CODE
    assert lineage['common_extraction_inputs_identical'] is True
    assert lineage['caches_rewritten'] is lineage['control_extracted_again'] is False
    assert pose['recipe'] == S.P.recipe() and probe['recipe'] == S.C.recipe()


@pytest.mark.parametrize('change,match', [
    (lambda f: f.update(reader_code='f' * 40), 'original frozen'),
    (lambda f: f.update(code=B.READER_CODE), 'separate released'),
    (lambda f: f.update(code='not-a-sha'), 'separate released'),
    (lambda f: f.update(control_sha='f' * 64), 'released hash'),
    (lambda f: f['control'].update(step=19618), 'released hash'),
    (lambda f: f['final'].update(arms=[S.ARMS[1]]), 'both registered'),
    (lambda f: f['final'].update(control_only=False), 'both registered'),
    (lambda f: f['final']['checkpoints'][S.ARMS[0]].update(bytes=1), 'checkpoint/receipt pins'),
    (lambda f: f['final']['inputs'].update(extra='different'), 'common inputs'),
    (lambda f: f['final']['paths'].update(extra='different'), 'common inputs'),
    (lambda f: f['final']['pose_references']['val'].update(sha256='f' * 64), 'common inputs'),
    (lambda f: f['final']['training_source_contract'].update(sha256='f' * 64), 'common inputs'),
    (lambda f: f['manifest']['jobs'].pop(next(iter(f['manifest']['jobs']))), 'six original'),
])
def test_mixed_join_stops_on_wrong_commit_plan_common_inputs_or_incomplete_collection(joined, change, match):
    change(joined)
    with pytest.raises(ValueError, match=match):
        run_join(joined)


@pytest.mark.parametrize('arm', S.ARMS)
@pytest.mark.parametrize('task', ('pose', 'p1', 'p2'))
@pytest.mark.parametrize('field,value', [('code_sha', 'f' * 40), ('plan_sha256', 'f' * 64),
                                         ('complete', False), ('checkpoint_step', 19618)])
def test_each_of_six_receipts_must_match_its_own_producer(joined, arm, task, field, value):
    key = arm + '/' + task + '/receipt.json'
    raw, pin = joined['files'][key]
    receipt = json.loads(raw)
    receipt[field] = value
    raw = S.dumps(receipt)
    joined['files'][key] = (raw, S.pin(pin['path'], raw))
    with pytest.raises(ValueError, match='extraction receipt differs: ' + field):
        run_join(joined)


def test_changed_numerical_source_is_not_accepted(joined, monkeypatch):
    real = S._blob_sha256
    monkeypatch.setattr(S, '_blob_sha256', lambda commit, name, root: 'f' * 64
                        if name == 'scripts/scaling_probe49046.py' else real(commit, name, root))
    with pytest.raises(ValueError, match='numerical reader sources'):
        run_join(joined)
