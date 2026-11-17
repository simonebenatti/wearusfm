"""Actual accepted P1-r2 receipts through the explicit, CPU-only controller profile."""
import copy
import calendar
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import scaling_plans19618 as S
import scaling_p1_reader_bridge as B


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    root, ledger = tmp_path/'root', tmp_path/'ledger'
    names = [B.BASE, S.RECOVERY['plan'], S.PACKAGE+'/control19618_identity',
             S.INFRA+'/cpu_readers_20261105/pose_plan.json', S.INFRA+'/cpu_readers_20261105/probe_plan.json',
             'results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json',
             'results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json', *B.MAC_SOURCES]
    for name in names:
        source, destination = ROOT/name, root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copyfile(source, destination)
    for name, (job, agent) in B.EXTRACTIONS.items():
        write(ledger/name/'state.json', {'id': name, 'stato': 'sottomesso', 'job_slurm': job,
              'agente': agent, 'campagna': S.CAMPAIGN})
    return root, ledger


def test_real_mixed_receipts_build_full_grid_without_adopting_failed_jobs(case):
    root, _ = case
    historical = copy.deepcopy(S.RELAUNCH)
    pose, probe, lineage = B.plans(S, root)
    assert sum(len(arm) for arm in pose['caches'].values()) == 4
    assert sum(len(arm) for arm in probe['caches'].values()) == 10
    assert lineage['tasks']['p1']['job_id'] == '64620434'
    assert lineage['tasks']['p1']['code_sha'] == B.P1_CODE
    assert lineage['tasks']['pose']['code_sha'] == lineage['tasks']['p2']['code_sha'] == B.CODE
    S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)
    assert S._load(S.PACKAGE+'/'+S.DOCUMENTS['probe_plan'][1], root) == probe
    assert S._load(S.PACKAGE+'/'+S.DOCUMENTS['pose_plan'][1], root) == pose
    assert S.RELAUNCH == historical
    with pytest.raises(FileExistsError):
        S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)


def test_accepted_evidence_mutation_refused(case):
    root, _ = case
    path = root/B.BASE/'evidence/p1/receipt.json'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='accepted evidence changed'):
        B.plans(S, root)


def test_prerequisites_require_only_the_new_p1_accepted_job(case):
    root, ledger = case
    assert set(B.dependencies(S, 'pose-validate', root, ledger).values()) == {'64571999', '64572001', '64620434'}
    name = next(n for n in B.EXTRACTIONS if 'p1-r2' in n)
    path = ledger/name/'state.json'
    value = json.loads(path.read_text())
    value['job_slurm'] = '64596915'
    write(path, value)
    with pytest.raises(ValueError, match='only the accepted'):
        B.dependencies(S, 'pose-validate', root, ledger)


@pytest.mark.parametrize('stage', ['pose-benchmark', 'probe-benchmark', 'pose-ridge', 'probe-fit', 'probe-aggregate', 'compare'])
def test_missing_previous_stage_never_releases_a_later_stage(case, stage):
    root, ledger = case
    with pytest.raises((ValueError, OSError)):
        B.dependencies(S, stage, root, ledger)


@pytest.mark.parametrize('stage', ['training', 'extractions', 'identity-precheck', 'checkout-training', 'checkout-code'])
def test_profile_cannot_regenerate_any_training_or_extraction(stage):
    with pytest.raises(ValueError, match='only the remaining CPU'):
        B.allowed_stage(stage)
    with pytest.raises(ValueError):
        S.stage_lines(stage, B.CODE, p1_reader_recovery=True)


def test_live_failed_dependency_and_changed_documents_stop_the_gate(case):
    root, ledger = case
    S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)
    reading = {'sacct': {job: {'exit': 0, 'stdout': 'JobIDRaw|State|ExitCode\n'+job+'|COMPLETED|0:0\n'}
                         for job, _ in B.EXTRACTIONS.values()}}
    checks, context = B.checks(S, 'pose-validate', reading, root, ledger)
    assert all(ok for _, ok in checks)
    assert context['accepted_p1_job'] == '64620434'
    reading['sacct']['64620434']['stdout'] = 'JobIDRaw|State|ExitCode\n64620434|FAILED|1:0\n'
    checks, _ = B.checks(S, 'pose-validate', reading, root, ledger)
    assert not all(ok for _, ok in checks)
    path = root/S.PACKAGE/S.DOCUMENTS['probe_plan'][1]
    value = json.loads(path.read_text())
    value['caches'][S.ARM]['ninapro_db3']['extraction_code_sha'] = B.CODE
    write(path, value)
    with pytest.raises(ValueError, match='published reader plan differs'):
        B.context(S, 'pose-validate', root, ledger)


def test_profile_budget_separates_prior_cpu_from_gpu_allocations(case):
    root, ledger = case
    budget = B.budget(S, root, ledger)
    assert budget['within_total_cap'] is True and budget['within_stage_caps'] is True
    assert 349.95 < budget['total']['core_hours'] < 350


def test_gpt_ledger_allowed_only_for_registered_remaining_cpu_ids(tmp_path):
    state = {'id': 'J', 'stato': 'sottomesso', 'job_slurm': 123, 'agente': 'gpt', 'campagna': S.CAMPAIGN}
    for name in ('J', 'SCALING-19618-code-checkout-20261106', 'SCALING-300M-eval19618-p1-r1-20261116'):
        write(tmp_path/name/'state.json', {**state, 'id': name})
        with pytest.raises(ValueError):
            S.ledger_job(name, tmp_path)
    for name in [S.checkout_job('plans')['id'], *(j['id'] for j in S.jobs('pose-benchmark')),
                 *(j['id'] for j in S.jobs('probe-benchmark'))]:
        write(tmp_path/name/'state.json', {**state, 'id': name})
        assert S.ledger_job(name, tmp_path) == '123'


def test_profile_submission_uses_gpt_common_submitter_and_frozen_code(case):
    root, _ = case
    S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)
    lines = S.stage_lines('pose-benchmark', B.CODE, root, p1_reader_recovery=True)
    assert len(lines) == 3
    for line in lines:
        argv = shlex.split(line)
        assert argv[:2] == ['python3', str(root/'scripts/submit_job.py')]
        assert argv[argv.index('--agente')+1] == 'gpt'
        assert argv[argv.index('--ssh')+1] == shlex.join(S.SSH)
        assert '--export=ALL,CODE_DIR='+S.checkout('code')+',EXPECTED_SHA='+B.CODE in argv
    assert '--dopo' not in shlex.split(lines[0])
    assert '--dopo' in shlex.split(lines[1])
    checkout = S.checkout_lines('plans', 'a'*40, True, root)
    assert '/dev/stdin' == shlex.split(checkout[1])[-1]
    assert shlex.split(checkout[0])[:3] == ['git', '-C', str(root)]


def test_cli_profile_is_explicit_and_rejects_recovery_training_before_other_reads():
    for command in (['recovery-lines', '--sha', 'a'*40], ['gate', 'extractions'], ['collect', 'training']):
        with pytest.raises(ValueError):
            S.main(['--p1-reader-recovery', *command])
    result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/scaling_plans19618.py'),
                             '--p1-reader-recovery', 'recovery-lines', '--sha', 'a'*40], capture_output=True, text=True)
    assert result.returncode != 0 and 'cannot release training/extractions' in result.stderr


def ledger_state(ledger, job, slurm, agent='gpt', export=None):
    argv = ['sbatch', '--cpus-per-task='+str(job['cpus']), '--mem='+job['memory'], '--time='+S.clock(job['seconds'])]
    if export:
        argv.append(export)
    write(ledger/job['id']/'state.json', {'id': job['id'], 'stato': 'sottomesso', 'job_slurm': slurm,
          'agente': agent, 'campagna': S.CAMPAIGN, 'gpu_ore_stimate': 0, 'sbatch': argv})


@pytest.mark.parametrize('mutation', ['profile', 'document_bytes', 'checkout', 'mac_source'])
def test_lines_bind_actual_mixed_profile_documents_and_all_checkouts(case, mutation):
    root, ledger = case
    S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)
    commits = {'code': B.CODE, 'plans': 'a'*40}
    for n, (name, sha) in enumerate(commits.items()):
        export = '--export=ALL,CAMPAIGN='+S.CHECKOUTS[name]+',CAMPAIGN_SHA='+sha+',EXPECTED_SHA='+sha
        ledger_state(ledger, S.checkout_job(name), str(700+n), 'claude' if name == 'code' else 'gpt', export)
    documents = {'pose_plan': S.document('pose_plan', root)}
    record = {'protocol': S.GATE_PROTOCOL, 'stage': 'pose-validate', 'passed': True,
              'read_at_utc': '2026-11-17T08:00:00Z', 'commits': commits, 'documents': documents,
              'submitted': {}, 'facts': {'tranche': [S.jobs('pose-validate')[0]['id']]},
              'p1_reader_recovery': B.context(S, 'pose-validate', root, ledger)}
    path = root/S.PACKAGE/'evidence/gates/pose-validate_20261117T080000Z.json'
    write(path, record)
    now = calendar.timegm((2026, 11, 17, 8, 1, 0))
    lines = S.admitted_lines('pose-validate', root=root, ledger=ledger, now=now, p1_reader_recovery=True)
    assert len(lines) == 1 and shlex.split(lines[0])[shlex.split(lines[0]).index('--agente')+1] == 'gpt'
    with pytest.raises(ValueError, match='explicit P1 reader profile'):
        S.admitted_lines('pose-validate', root=root, ledger=ledger, now=now)
    if mutation == 'profile':
        record.pop('p1_reader_recovery')
        write(path, record)
    elif mutation == 'document_bytes':
        p = root/S.PACKAGE/S.DOCUMENTS['pose_plan'][1]
        p.write_bytes(p.read_bytes()+b'\n')
    elif mutation == 'checkout':
        job = S.checkout_job('plans')
        statepath = ledger/job['id']/'state.json'
        state = json.loads(statepath.read_text())
        state['sbatch'] = [v.replace('a'*40, 'b'*40) for v in state['sbatch']]
        write(statepath, state)
    else:
        p = root/'scripts/scaling_p1_reader_bridge.py'
        p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError):
        S.admitted_lines('pose-validate', root=root, ledger=ledger, now=now, p1_reader_recovery=True)


def test_completed_validated_inputs_release_benchmarks_and_partial_report_does_not(case):
    root, ledger = case
    S.build_reader_plans(B.CODE, root, p1_reader_recovery=True)
    job, slurm = S.jobs('pose-validate')[0], '900'
    ledger_state(ledger, job, slurm)
    plan = S._load(S.PACKAGE+'/'+S.DOCUMENTS['pose_plan'][1], root)
    report = {'protocol': S.P.PROTOCOL, 'complete': True, 'step': S.STEP, 'recipe': plan['recipe'],
              'plan_sha256': S.document('pose_plan', root)['sha256'],
              'runtime': {'code_sha': B.CODE, 'sources': plan['sources']},
              'status': 'validated_input_metadata', 'scientific_metrics_computed': False, 'slurm_job_id': slurm,
              'inspections': {a+'/'+s: {} for a in S.ARMS for s in ('train', 'val')}}
    for key, field in (('input_pins', 'caches'), ('reference_pins', 'references'),
                       ('projection_pin', 'projection'), ('training_source_contract', 'training_source_contract')):
        report[key] = plan[field]
    directory = S.evidence_dir('pose-validate', root)
    raw = ('JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|Timelimit|AllocTRES\n'
           '900|COMPLETED|0:0|10|2|1G|00:30:00|cpu=2,mem=30G,node=1\n')
    write(directory/'accounting.json', S.accounting(raw, slurm, 2, 1800, need_rss=True))
    (directory/'sacct.txt').write_text(raw)
    def put_report():
        data = S.dumps(report)
        (directory/'report.json').write_bytes(data)
        write(directory/'collection.json', {'jobs': {job['id']: {'job_slurm': slurm}},
              'files': {'report.json': {**S.pin(job['files']['report.json'], data), 'stored': 'plain'}}})
    put_report()
    assert B.dependencies(S, 'pose-benchmark', root, ledger)[job['id']] == slurm
    assert B.dependencies(S, 'probe-benchmark', root, ledger)[job['id']] == slurm
    report['complete'] = False
    put_report()
    with pytest.raises(ValueError, match='pose report provenance differs'):
        B.dependencies(S, 'probe-benchmark', root, ledger)


def test_missing_fit_plan_cannot_hide_previously_submitted_fit(case):
    root, ledger = case
    write(ledger/'SCALING-19618-probe-fit-01-20261106'/'state.json', {'stato': 'sottomesso'})
    with pytest.raises(ValueError, match='outside the complete published stage tables'):
        B.budget(S, root, ledger)
