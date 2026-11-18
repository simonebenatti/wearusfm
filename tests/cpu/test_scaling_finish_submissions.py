"""Offline operational-driver checks; never execute SSH or submit jobs."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'results/infra/scaling_300m_20261104/cpu_fit_release_20261105/finish_submissions.py'
spec = importlib.util.spec_from_file_location('finish_submissions', FILE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture_probe():
    plan = M.frozen()
    job = copy.deepcopy(plan['jobs'][1])
    p = M.read(ROOT / 'results/infra/scaling_300m_20261104/cpu_readers_20261105/probe_plan.json')
    b = M.read(ROOT / 'results/infra/scaling_300m_20261104/cpu_recovery_20261105/epn612_completed/benchmark.json')
    arm, view = p['recipe']['cells'][job['cell']].values()
    rows = []
    for i, c in enumerate([.001, .01, .1, 1., 10., 100., 1.]):
        r = copy.deepcopy(b['fits'][0])
        r.update(cell=job['cell'], arm=arm, readout=view, task=job['task'], dataset=job['datasets'][0], fold=0,
                 C=c, fit_phase='candidate' if i < 6 else 'refit', code_sha=M.CODE,
                 plan_sha256=job['argv'][job['argv'].index('/dev/stdin') + 3],
                 projection_sha256=p['projection']['sha256'], cache_sha256=p['caches'][arm][job['datasets'][0]]['sha256'],
                 checkpoint_sha256=p['caches'][arm][job['datasets'][0]]['checkpoint_sha256'])
        rows.append(r)
    auditpath, oofpath = job['out'] + '/audit.jsonl', job['out'] + '/shard.oof.npz'
    audit = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()
    resources = M.read(ROOT / plan['probe_resources']['path'])
    allocation = next(a for a in resources['shards'] if all(a[k] == job[k] for k in ('cell', 'task', 'datasets', 'folds')))
    report = dict(protocol='scaling-probe-20261104-v1', kind='shard', complete=True, step=5000, recipe=p['recipe'],
        plan_sha256=job['argv'][job['argv'].index('/dev/stdin') + 3], code_sha=M.CODE, cell=job['cell'], arm=arm, readout=view,
        task=job['task'], plan_content_sha256=M.jhash(p), projection_sha256=p['projection']['sha256'],
        numeric_context=b['numeric_context'], numeric_sha256=b['numeric_sha256'],
        resource_verification={'allocation': allocation, 'resource_plan': {'sha256': plan['probe_resources']['sha256']}, 'estimated_seconds_with_margin': 123.},
        audit=auditpath, audit_sha256=M.sha(audit), oof=oofpath, oof_sha256='a'*64,
        datasets={job['datasets'][0]: {'folds': [{'fold': 0}]}}, audit_summary=M.summary(rows))
    return plan, job, report, {auditpath: audit}, {oofpath: {'sha256': 'a'*64}}


def test_real_published_pins_and_probe_recipe_shape():
    values = fixture_probe()
    verdict = M.validate_report(values[1], values[2], values[3], values[4], values[0], 123)
    assert verdict['solver_unresolved'] is False


@pytest.mark.parametrize('mutation', ['code', 'fold', 'output', 'numeric', 'summary', 'allocation', 'coverage'])
def test_rejects_corrupted_probe_metadata(mutation):
    plan, job, report, files, outputs = fixture_probe()
    if mutation == 'code': report['code_sha'] = 'bad'
    if mutation == 'fold': report['datasets'][job['datasets'][0]]['folds'][0]['fold'] = 1
    if mutation == 'output': report['oof_sha256'] = 'b'*64
    if mutation == 'numeric': report['numeric_context']['new'] = 'bad'
    if mutation == 'summary': report['audit_summary']['fits'] = 8
    if mutation == 'allocation': report['resource_verification']['allocation'] = {}
    if mutation == 'coverage': report['datasets']['extra'] = {'folds': []}
    with pytest.raises(RuntimeError): M.validate_report(job, report, files, outputs, plan, 123)


def test_unresolved_solver_is_accepted_and_flagged():
    plan, job, report, files, outputs = fixture_probe()
    records = [json.loads(line) for line in files[report['audit']].decode().splitlines()]
    records[0].update(n_iter=12000, reached_limit=True, converged_by_criterion=False)
    files[report['audit']] = ('\n'.join(json.dumps(r) for r in records) + '\n').encode()
    report['audit_sha256'] = M.sha(files[report['audit']])
    report['audit_summary'] = M.summary(records)
    assert M.validate_report(job, report, files, outputs, plan, 123)['solver_unresolved'] is True


def test_inconsistent_convergence_is_rejected():
    _, _, report, files, _ = fixture_probe()
    rows = [json.loads(line) for line in files[report['audit']].decode().splitlines()]
    rows[0]['converged_by_criterion'] = False
    with pytest.raises(RuntimeError, match='inconsistent'): M.summary(rows)


def snapshot_one(state='PENDING', reason='Priority'):
    return {'existing_outputs': [], 'sacct': 'JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocCPUS|Partition|\n123|one|' + state + '|0:0|1|8|lrd_all_serial|\n',
            'squeue': '' if state == 'COMPLETED' else '123|one|' + state + '|lrd_all_serial|8|' + reason + '|normal\n'}


@pytest.mark.parametrize('bad', ['FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'UNKNOWN'])
def test_bad_scheduler_state_stops(bad):
    with pytest.raises(RuntimeError): M.status({'jobs': [{'id': 'one'}]}, {'one': {'job_slurm': 123}}, snapshot_one(bad))


def test_completed_and_pending_states():
    for state in ['PENDING', 'RUNNING', 'COMPLETED']:
        completed, _, _ = M.status({'jobs': [{'id': 'one'}]}, {'one': {'job_slurm': 123}}, snapshot_one(state))
        assert len(completed) == (state == 'COMPLETED')


@pytest.mark.parametrize('bad', ['duplicate', 'unknown', 'missing', 'dependency', 'output', 'eleven', 'cpu16'])
def test_scheduler_ambiguity_and_caps(bad):
    snap = snapshot_one()
    found = {'one': {'job_slurm': 123}}
    if bad == 'duplicate': snap['squeue'] += '124|one|PENDING|lrd_all_serial|8|Priority|normal\n'
    if bad == 'unknown': found = {}
    if bad == 'missing': snap['squeue'] = ''; snap['sacct'] = snap['sacct'].splitlines()[0]+'\n'
    if bad == 'dependency': snap = snapshot_one(reason='DependencyNeverSatisfied')
    if bad == 'output': snap['existing_outputs'] = ['out']
    if bad == 'eleven': snap['squeue'] += ''.join(f'{j}|other{j}|PENDING|lrd_all_serial|8|Priority|normal\n' for j in range(200, 210))
    if bad == 'cpu16': snap['squeue'] = '123|one|RUNNING|lrd_all_serial|8|None|normal\n124|other|RUNNING|lrd_all_serial|8|None|normal\n'
    with pytest.raises(RuntimeError): M.status({'jobs': [{'id': 'one'}]}, found, snap)


def mock_loop(monkeypatch, tmp_path, fail=False):
    plan = M.frozen()
    found = {j['id']: {'job_slurm': i+1000} for i, j in enumerate(plan['jobs'][:10])}
    calls = []
    monkeypatch.setattr(M, 'SUB', tmp_path)
    monkeypatch.setattr(M, 'frozen', lambda: plan)
    monkeypatch.setattr(M, 'receipts', lambda p: copy.deepcopy(found))
    monkeypatch.setattr(M, 'remote', lambda p: {})
    monkeypatch.setattr(M, 'status', lambda p, f, s: ([], [[str(v['job_slurm']), k, 'PENDING', 'lrd_all_serial', '8', 'Dependency', 'normal'] for k, v in list(f.items())[-9:]], []))
    def submit(argv, **kw):
        assert kw['shell'] is False
        calls.append(argv)
        j = next(j for j in plan['jobs'] if j['argv'] == argv)
        if not fail: found[j['id']] = {'job_slurm': 2000 + len(calls)}
        return SimpleNamespace(returncode=1 if fail else 0, stdout='', stderr='rejected' if fail else '')
    monkeypatch.setattr(M.subprocess, 'run', submit)
    return plan, found, calls


def test_only_five_immutable_argv_submitted(monkeypatch, tmp_path):
    plan, _, calls = mock_loop(monkeypatch, tmp_path)
    assert M.finish()['status'] == 'all_15_submitted'
    assert calls == [j['argv'] for j in plan['jobs'][10:]]


def test_no_retry_after_submitter_failure_or_restart(monkeypatch, tmp_path):
    _, _, calls = mock_loop(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='submitter refused'): M.finish()
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='previous attempt unresolved'): M.finish()
    assert len(calls) == 1


def test_deadline_prevents_submission(monkeypatch, tmp_path):
    _, _, calls = mock_loop(monkeypatch, tmp_path)
    times = iter([0, 0, M.MAX_WAIT + 1])
    monkeypatch.setattr(M.time, 'monotonic', lambda: next(times))
    with pytest.raises(RuntimeError, match='deadline'): M.finish()
    assert calls == []


def test_completed_collection_before_any_new_submission(monkeypatch, tmp_path):
    plan, found, calls = mock_loop(monkeypatch, tmp_path)
    order = []
    def status(p, f, s):
        return ([p['jobs'][0]], [[str(v['job_slurm']), k, 'PENDING', 'lrd_all_serial', '8', 'Dependency', 'normal'] for k, v in list(f.items())[-9:]], [])
    monkeypatch.setattr(M, 'status', status)
    def collect(*args):
        order.append(len(calls))
        raise RuntimeError('collected integrity failure')
    monkeypatch.setattr(M, 'collect', collect)
    with pytest.raises(RuntimeError, match='integrity'): M.finish()
    assert order == [0] and calls == []


def test_one_independent_start_no_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(M, 'SUB', tmp_path)
    monkeypatch.setattr(M.sys, 'argv', [str(FILE), '--start'])
    calls = []
    def popen(argv, **kw):
        calls.append((argv, kw)); return SimpleNamespace(pid=12345)
    monkeypatch.setattr(M.subprocess, 'Popen', popen)
    M.main()
    assert len(calls) == 1 and calls[0][0][-1] == '--run'
    assert calls[0][1]['start_new_session'] is True and calls[0][1]['shell'] is False
    assert M.read(tmp_path/'finish_startup.json')['max_wait_seconds'] == 28800
    with pytest.raises(RuntimeError, match='already started'): M.main()
    assert len(calls) == 1


def accounting_rows():
    parent = {'JobIDRaw': '123', 'State': 'COMPLETED', 'ExitCode': '0:0', 'AllocCPUS': '8', 'Partition': 'lrd_all_serial',
              'NodeList': 'login08', 'ElapsedRaw': '3784', 'Timelimit': '01:00:00', 'AllocTRES': 'cpu=8,mem=30G,node=1', 'MaxRSS': ''}
    batch = dict(parent, JobIDRaw='123.batch', MaxRSS='1048576K')
    return [parent, batch]


def test_accounting_uses_parent_actual_even_overtime_184():
    a = M.account({'walltime_seconds': 3600}, 123, accounting_rows(), 'raw')
    assert a['core_hours'] == 3784 * 8 / 3600
    assert a['peak_rss_bytes'] == 2**30
    assert M.slurm_seconds('2-01:02:03') == 176523
    assert M.slurm_seconds('59:00') == 3540


@pytest.mark.parametrize('field,value', [('NodeList','other'),('Timelimit','02:00:00'),('AllocTRES','cpu=8,mem=30G,gres/gpu=1'),('AllocTRES','cpu=8,mem=31G'),('AllocCPUS','16')])
def test_accounting_rejects_allocation_mismatch(field, value):
    rows = accounting_rows(); rows[0][field] = value
    with pytest.raises(RuntimeError): M.account({'walltime_seconds': 3600}, 123, rows, 'raw')


@pytest.mark.parametrize('field,value', [('MaxRSS','31G'),('MaxRSS',''),('State','FAILED'),('ExitCode','1:0')])
def test_accounting_rejects_failed_step_or_memory(field, value):
    rows = accounting_rows(); rows[1][field] = value
    with pytest.raises(RuntimeError): M.account({'walltime_seconds': 3600}, 123, rows, 'raw')


def test_envelope_uses_completed_actual_and_outstanding_caps(monkeypatch, tmp_path):
    monkeypatch.setattr(M, 'SUB', tmp_path)
    plan = M.frozen()
    M.envelope(plan)
    original = M.read(tmp_path / 'finish_envelope.json')['total_core_hours']
    assert original < 310
    folder = tmp_path / 'collected_jobs' / plan['jobs'][0]['id']
    M.write(folder/'operational_acceptance.json', {})
    M.write(folder/'accounting.json', {'core_hours': 1})
    M.envelope(plan)
    assert M.read(tmp_path/'finish_envelope.json')['total_core_hours'] == pytest.approx(original - 8.4 + 1)
    M.write(folder/'accounting.json', {'core_hours': 400})
    with pytest.raises(RuntimeError, match='310'): M.envelope(plan)


def test_failed_process_spawn_leaves_reservation_no_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(M, 'SUB', tmp_path)
    monkeypatch.setattr(M.sys, 'argv', [str(FILE), '--start'])
    def fail(*args, **kwargs): raise OSError('spawn failed')
    monkeypatch.setattr(M.subprocess, 'Popen', fail)
    with pytest.raises(OSError): M.main()
    assert M.read(tmp_path/'finish_startup.json')['pid'] is None
    with pytest.raises(RuntimeError, match='already started'): M.main()


def test_second_run_lock_does_not_overwrite_active_result(monkeypatch, tmp_path):
    import fcntl, os
    monkeypatch.setattr(M, 'SUB', tmp_path)
    monkeypatch.setattr(M.sys, 'argv', [str(FILE), '--run'])
    M.write(tmp_path/'finish_startup.json', {'pid': os.getpid()})
    M.write(tmp_path/'finish_result.json', {'sentinel': 'preserve'})
    with (tmp_path/'finish_submissions.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): M.main()
    assert M.read(tmp_path/'finish_result.json') == {'sentinel': 'preserve'}


def test_rejected_report_preserves_raw_evidence_without_acceptance(monkeypatch, tmp_path):
    plan, job, report, files, outputs = fixture_probe()
    report['code_sha'] = 'bad'
    reportpath = job['out'] + '/shard.json'
    logpath = M.LOGROOT + '/scaling_readers_recovery_123.out'
    files[reportpath] = json.dumps(report).encode()
    files[logpath] = b'completion log\n'
    bundle = {'utc':'test', 'files':[{'path':p, 'sha256':M.sha(data), 'bytes':len(data), 'base64':base64.b64encode(data).decode()} for p,data in files.items()],
              'outputs':[dict(pin, path=p, bytes=1) for p,pin in outputs.items()]}
    monkeypatch.setattr(M, 'SUB', tmp_path)
    monkeypatch.setattr(M, 'remote', lambda _: bundle)
    with pytest.raises(RuntimeError, match='probe runtime'):
        M.collect(job, 123, {'sacct':'raw accounting'}, [], plan)
    folder = tmp_path/'collected_jobs'/job['id']
    assert (folder/'shard.json').read_bytes() == files[reportpath]
    assert (folder/'audit.jsonl').exists() and (folder/'sacct.txt').exists()
    assert not (folder/'operational_acceptance.json').exists()
