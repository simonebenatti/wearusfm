"""Closed early-control stages, resource amendment and emitted commands."""
import copy
import json
from pathlib import Path
import shlex
import shutil
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_control30m_ops as A
S = A.S


def test_same_three_ids_and_outputs_no_training_or_new_gpu_work():
    original = [j for j in S.jobs('extractions') if j['flags'][0] == S.ARMS[0]]
    assert A.table('extractions') == original
    assert len(original) == 3
    assert sum(j['seconds'] / 3600 for j in original) == 7
    assert all(j['gpus'] == 1 and j['cpus'] == 8 and j['memory'] == '100G' for j in original)
    assert A.CHECKOUT_ID not in {j['id'] for j in S.jobs('extractions')}
    with pytest.raises(ValueError): A.table('train-final')


def test_only_one_checkout_added_to_cpu_budget():
    p = A.policy()
    assert p['cpu_only_worst_case_with_margin'] == pytest.approx(348.06666666666666)
    assert p['checkout_stage_cap_with_margin'] == 1.6
    assert p['additional_checkout_core_hours_with_margin'] == pytest.approx(4 / 15)
    assert p['all_extractions_gpu_hours_requested'] == 14
    assert p['cpu_only_worst_case_with_margin'] < p['cpu_only_cap'] == 400


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    for relative in (A.B.OLD + '/evaluation/extraction_plan.json', A.B.CONTROL_RECEIPT):
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dst)
    original = S._blob_sha256
    monkeypatch.setattr(S, '_blob_sha256', lambda commit, relative, root: original(commit, relative, ROOT))
    A.prepare(tmp_path)
    return tmp_path


def test_real_plan_contains_only_existing_control_and_is_idempotent(prepared):
    raw = (prepared / A.PLAN).read_bytes()
    A.prepare(prepared)
    assert raw == (prepared / A.PLAN).read_bytes()
    p = json.loads(raw)
    assert p['control_only'] is True and p['arms'] == [S.ARMS[0]]
    assert set(p['checkpoints']) == {S.ARMS[0]}
    assert p['checkpoints'][S.ARMS[0]]['sha256'] == S.E.CONTROL_CHECKPOINT_SHA256


def reading(stage, root):
    commands = {'squeue': {'exit': 0, 'stdout': ''},
                'quota': {'exit': 0, 'stdout': f'Disk quotas for prj 20147512\n{S.WORK}\n1000 1073741824 0\n'},
                'head:base': {'exit': 0, 'stdout': A.BASE_CODE},
                'status:base': {'exit': 0, 'stdout': ''}}
    files = []
    if stage == 'extractions':
        commands.update({'head:control': {'exit': 0, 'stdout': 'c' * 40},
                         'status:control': {'exit': 0, 'stdout': ''}})
        files.append({**A.plan_pin(root), 'canonical': True})
    return {'commands': commands, 'files': files, 'exists': {j['out']: False for j in A.table(stage)},
            'sacct': {'123': {'exit': 0, 'stdout': 'JobIDRaw|State|ExitCode\n123|COMPLETED|0:0\n'}},
            'time_utc': '2026-11-21T18:00:00Z'}


def snap():
    return {'dependencies': {'base': '123'}, 'submitted': {}, 'evidence_sha256': {}}


@pytest.mark.parametrize('stage', A.STAGES)
def test_gate_accepts_only_declared_control_jobs(prepared, stage):
    checks, facts = A.checks(stage, 'c' * 40, reading(stage, prepared), snap(), prepared)
    assert all(ok for _, ok in checks)
    assert facts['tranche'] == [j['id'] for j in A.table(stage)]


@pytest.mark.parametrize('key', ['squeue', 'quota'])
@pytest.mark.parametrize('exit_code', [1, False, None])
def test_failed_or_boolean_read_does_not_become_empty_queue(prepared, key, exit_code):
    r = reading('checkout', prepared)
    r['commands'][key]['exit'] = exit_code
    with pytest.raises(ValueError): A.checks('checkout', 'c' * 40, r, snap(), prepared)


@pytest.mark.parametrize('kind', ['full_queue', 'line_limit', 'quota', 'dirty', 'wrong_head', 'sacct_failed', 'output_exists'])
def test_checkout_gate_stops_before_any_submission(prepared, kind):
    r = reading('checkout', prepared)
    if kind == 'full_queue': r['commands']['squeue']['stdout'] = ''.join(f'{i}|OTHER|PD|2|Priority\n' for i in range(10))
    if kind == 'line_limit': r['commands']['squeue']['stdout'] = ''.join(f'{i}|SCALING-{i}|PD|2|Priority\n' for i in range(4))
    if kind == 'quota': r['commands']['quota']['stdout'] = f'prj 20147512\n{S.WORK}\n1052770304 1073741824\n'
    if kind == 'dirty': r['commands']['status:base']['stdout'] = ' M script.py'
    if kind == 'wrong_head': r['commands']['head:base']['stdout'] = 'f' * 40
    if kind == 'sacct_failed': r['sacct']['123']['exit'] = 1
    if kind == 'output_exists': r['exists'][A.CODE_DIR] = True
    checks, _ = A.checks('checkout', 'c' * 40, r, snap(), prepared)
    assert not all(ok for _, ok in checks)


@pytest.mark.parametrize('field,value', [('sha256', 'f' * 64), ('bytes', 0), ('canonical', False)])
def test_extraction_gate_checks_real_plan_bytes(prepared, field, value):
    r = reading('extractions', prepared)
    r['files'][0][field] = value
    checks, _ = A.checks('extractions', 'c' * 40, r, snap(), prepared)
    assert not all(ok for _, ok in checks)


def test_emitted_commands_keep_ledger_ids_allocations_and_exact_plan(prepared):
    jobs = A.table('extractions')
    lines = A.raw_lines('extractions', 'c' * 40, [j['id'] for j in jobs], prepared)
    assert len(lines) == 3
    for line, job in zip(lines, jobs):
        argv = shlex.split(line)
        assert argv[argv.index('--id') + 1] == job['id']
        assert float(argv[argv.index('--gpu-ore') + 1]) == job['seconds'] / 3600
        assert '--time=' + S.clock(job['seconds']) in argv
        assert A.CODE_DIR + '/scripts/slurm/scaling_control30m_evaluation.sbatch' in argv
        assert A.plan_pin(prepared)['path'] in argv
        assert job['out'] == argv[-1]
        assert not any('300M' in a or 'training_49046.json' in a for a in argv)
    checkout = shlex.split(A.raw_lines('checkout', 'c' * 40, [A.CHECKOUT_ID], prepared)[1])
    assert '/dev/stdin' == checkout[-1]
    assert '--script-stdin' in checkout and '--cpus-per-task=2' in checkout


def test_lines_require_fresh_unchanged_gate_and_ledger(prepared, monkeypatch):
    ledger = prepared / 'ledger'; ledger.mkdir()
    a = {'commit': 'c' * 40}
    monkeypatch.setattr(A, 'attestation', lambda *args: a)
    monkeypatch.setattr(A, 'state_snapshot', lambda *args: snap())
    record = {'protocol': A.PROTOCOL, 'stage': 'extractions', 'commit': 'c' * 40,
              'passed': True, 'read_at_utc': '2026-11-21T18:00:00Z', 'attestation': a,
              'snapshot': snap(), 'facts': {'tranche': [j['id'] for j in A.table('extractions')]}}
    now = A.calendar.timegm(A.time.strptime(record['read_at_utc'], '%Y-%m-%dT%H:%M:%SZ'))
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now)
    path = A.gate_path('extractions', prepared) / '20261121T180000Z.json'
    A.write_new(path, record)
    assert len(A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now)) == 3
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now + 1801)
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'd' * 40, prepared, ledger, now)
    monkeypatch.setattr(A, 'state_snapshot', lambda *args: {**snap(), 'submitted': {'new': '456'}})
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now)
    monkeypatch.setattr(A, 'state_snapshot', lambda *args: snap())
    (ledger / record['facts']['tranche'][0]).mkdir()
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now)
    refused = A.gate_path('extractions', prepared) / '20261121T180001Z_refused.json'
    A.write_new(refused, {'protocol': A.PROTOCOL, 'passed': False})
    with pytest.raises(ValueError): A.admitted_lines('extractions', 'c' * 40, prepared, ledger, now)


def test_checkout_accounting_without_rss_matches_collection(tmp_path):
    ledger = tmp_path / 'ledger'
    job = A.table('checkout')[0]
    folder = A.evidence_dir('checkout', tmp_path) / job['local']
    folder.mkdir(parents=True)
    state = ledger / job['id'] / 'state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'stato': 'sottomesso', 'job_slurm': '123',
                                 'agente': 'gpt', 'campagna': S.CAMPAIGN}))
    raw = 'JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|Timelimit|AllocTRES\n' \
          '123|COMPLETED|0:0|18|2||00:05:00|cpu=2,node=1\n'
    (folder / 'sacct.txt').write_text(raw)
    measured = S.accounting(raw, '123', 2, 300, 0, need_rss=False)
    assert measured['MaxRSS_bytes'] is None
    (folder / 'accounting.json').write_text(json.dumps(measured))
    (folder.parent / 'collection.json').write_text(json.dumps({'jobs': {job['id']: {'job_slurm': '123'}}}))
    assert A.accounting('checkout', tmp_path, ledger) == {job['id']: '123'}
    measured['ElapsedRaw'] = 19
    (folder / 'accounting.json').write_text(json.dumps(measured))
    with pytest.raises(ValueError, match='accounting differs'):
        A.accounting('checkout', tmp_path, ledger)
