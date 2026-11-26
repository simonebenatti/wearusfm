"""Resubmission of the cancelled second training segment: only the logical ID changes, frozen operator files stay as they are."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_plans49046 as S
import scaling_operations49046 as O
import scaling_terminal_builders as T


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load('resubmit_train_final', ROOT / S.INFRA / 'operations/resubmit_train_final.py')
R = load('release49046_helpers', ROOT / 'tests/cpu/test_scaling_release49046.py')
CODE = '87da975bc0fd4824684e166659bb996e118ff107'


@pytest.fixture
def installed(monkeypatch):
    # monkeypatch restores both names after each test
    monkeypatch.setattr(T, 'training_job', T.training_job)
    monkeypatch.setattr(S, 'ledger_job', S.ledger_job)
    return lambda: A.install()


def entry(ledger, job_id, slurm, state='sottomesso'):
    folder = ledger / job_id
    folder.mkdir(parents=True)
    (folder / 'state.json').write_text(json.dumps(
        {'agente': 'gpt', 'campagna': S.CAMPAIGN, 'stato': state, 'job_slurm': slurm}))


def test_only_the_id_of_the_second_segment_changes(installed):
    before = {stage: S.stage_table(stage) for stage in ('train-predecay', 'train-final', 'archive', 'extractions')}
    installed()
    installed()
    after = {stage: S.stage_table(stage) for stage in before}
    assert before['train-final'][0]['id'] == A.OLD_ID and after['train-final'][0]['id'] == A.NEW_ID
    assert {**after['train-final'][0], 'id': A.OLD_ID} == before['train-final'][0]
    for stage in ('train-predecay', 'archive', 'extractions'):
        assert after[stage] == before[stage]


def test_submission_line_is_the_published_one_with_the_new_id(installed, monkeypatch):
    monkeypatch.setattr(S, 'ledger_job', lambda *a, **k: '64837401')
    old = S.stage_lines('train-final', CODE)
    installed()
    new = S.stage_lines('train-final', CODE)
    assert len(old) == len(new) == 1 and new[0] == old[0].replace(A.OLD_ID, A.NEW_ID) and A.NEW_ID in new[0]
    assert 'SCALING_PREDECESSOR_JOB=64837401' in new[0] and '-- --time=10:00:00' in new[0] and '--gpu-ore 40' in new[0]


def test_ledger_lookup_follows_the_new_job_and_ignores_the_cancelled_one(installed, tmp_path):
    entry(tmp_path, A.OLD_ID, A.OLD_JOB)
    installed()
    for name in (A.OLD_ID, A.NEW_ID):
        with pytest.raises((OSError, ValueError)):
            S.ledger_job(name, tmp_path)
    entry(tmp_path, A.NEW_ID, '70000001')
    assert S.ledger_job(A.NEW_ID, tmp_path) == S.ledger_job(A.OLD_ID, tmp_path) == '70000001'


def test_gate_request_treats_the_new_job_as_not_yet_submitted(installed, monkeypatch, tmp_path):
    installed()
    monkeypatch.setattr(S, 'checkout_commit', lambda *a, **k: CODE)
    monkeypatch.setattr(O, 'dependencies', lambda *a, **k: {})
    entry(tmp_path, A.OLD_ID, A.OLD_JOB)
    _, _, _, table, submitted = S.gate_request('train-final', ledger=tmp_path)
    assert [job['id'] for job in table] == [A.NEW_ID] and submitted == {}
    assert not (tmp_path / A.NEW_ID).exists()


def test_later_stages_bind_to_the_new_job_end_to_end(installed, monkeypatch, tmp_path):
    installed()
    R.copy_training_documents(tmp_path)
    plan = json.loads((tmp_path / R.B.BASE / 'training_49046.json').read_text())
    receipt = {'complete': True, 'slurm_job_id': '12345678', 'checkpoint_step': 49046, 'start_step': 39237,
               'plan_sha256': S.digest((tmp_path / R.B.BASE / 'training_49046.json').read_bytes()),
               'checkpoint': plan['run_dir'] + '/checkpoint_postdecay.pt', 'checkpoint_sha256': 'c' * 64,
               'protocol': R.L.PROTOCOL, 'scientific_endpoint': True, 'previous_run_unchanged': True,
               'additional_steps': 9809, 'code_sha': 'c' * 40,
               'run_inventory': {'checkpoint_postdecay.pt': {'sha256': 'c' * 64, 'bytes': 4673783756}}}
    R.collect_synthetic_stage(tmp_path, 'train-final', {'receipt.json': receipt,
                                                        'summary.json': {'stopped': 'max_steps', 'steps': 49046}})
    entry(tmp_path / 'ledger', A.NEW_ID, '12345678')
    assert O.dependencies(S, 'archive', tmp_path, tmp_path / 'ledger') == {A.NEW_ID: '12345678'}
    asked = []
    monkeypatch.setattr(S, 'ledger_job', lambda name: asked.append(name) or '12345678')
    installed()  # the stub is wrapped again: the literal old ID of the builder must arrive as the new one
    monkeypatch.setattr(S, 'checkout_commit', lambda *a: 'c' * 40)
    T.extraction_plan_from_receipts(S, tmp_path)
    assert asked == [A.NEW_ID]


def row(state='CANCELLED', start='None', elapsed='0', job=A.OLD_JOB, code=0):
    return {'exit': code, 'stdout': f'JobIDRaw|State|ExitCode|ElapsedRaw|Start\n{job}|{state}|0:0|{elapsed}|{start}\n'}


def test_replaced_job_must_be_cancelled_before_starting():
    for answer in (row(), row(state='CANCELLED by 127686', start='Unknown')):
        assert A.replaced_job_check(answer)['JobIDRaw'] == A.OLD_JOB
    bad = [row(state='PENDING', start='Unknown'), row(state='RUNNING', start='2026-11-25T08:00:00'),
           row(state='COMPLETED'), row(start='2026-11-25T08:00:00'), row(elapsed='5'), row(job='1'), row(code=1), None]
    for answer in bad:
        with pytest.raises(ValueError):
            A.replaced_job_check(answer)


def test_frozen_operator_files_are_untouched():
    changed = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', CODE, '--', *O.OPERATOR_FILES])
    assert changed.returncode == 0
