"""Recovery changes only dispatch/ceilings; exercise the real shell guards.

Reuse the original launcher's scheduler/git/controller fixture, with the exact
historical code pin. No cluster calls, scientific fits or synthetic readiness.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = 'aef665ba7ea66fc80291846487907bb3ef9107df'
LAUNCHER = ROOT/'scripts/slurm/scaling_readers_recovery.sbatch'
spec = importlib.util.spec_from_file_location('original_reader_fixture',
                                             ROOT/'tests/cpu/test_scaling_readers_launcher.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
CELLS, TASKS = original.CELLS, original.TASKS


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(original, 'LAUNCHER', LAUNCHER)
    monkeypatch.setattr(original, 'CODE', PIN)
    fixture = original.harness.__wrapped__(tmp_path)
    h = next(fixture)
    run = h['run']

    def pinned_run(*args, environment=None, **kwargs):
        return run(*args, environment={'TEST_GIT_HEAD': PIN, **(environment or {})}, **kwargs)

    h['run'] = pinned_run
    try:
        yield h
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


def test_diff_is_only_the_declared_operational_allowlist():
    frozen = subprocess.check_output(['git', 'show', PIN+':scripts/slurm/scaling_readers.sbatch'], cwd=ROOT)
    assert hashlib.sha256(frozen).hexdigest() == '2da66c150fbc238dcf9b211a2eb672bb41adab8de96fc6f3f0c0ea5e716f6355'
    assert (ROOT/'scripts/slurm/scaling_readers.sbatch').read_bytes() == frozen
    edits = [
        ('#SBATCH --time=02:00:00', '#SBATCH --time=03:00:00'),
        ('#SBATCH -J scaling_readers', '#SBATCH -J scaling_readers_recovery'),
        ('logs/scaling_readers_%j.out', 'logs/scaling_readers_recovery_%j.out'),
        ('# pose ridge 8CPU/60min; probe benchmark 8CPU/45min (300M-natural90min),\n'
         '# all five datasets in one job; probe fit 8CPU/published shard <=120min;',
         '# pose ridge 8CPU/60min; recovery benchmark only300M-natural, one dataset:\n'
         '# EPN612 8CPU/120min, UCI 8CPU/10min; fit <=15shards, <=180min, <=240coreh;'),
        ("p.add_argument('--cell', choices=CELLS)",
         "p.add_argument('--cell', choices=CELLS)\np.add_argument('--dataset', choices=TASKS['p2'])"),
        ("'probe-benchmark': ('cell',)", "'probe-benchmark': ('cell', 'dataset')"),
        ("expected_code = os.environ['EXPECTED_SHA']",
         "expected_code = os.environ['EXPECTED_SHA']\n"
         "require(expected_code == 'aef665ba7ea66fc80291846487907bb3ef9107df', 'recovery requires the original frozen reader code')"),
        ("cpus = 2 if a.stage in ('pose-validate', 'probe-aggregate') else 8",
         "if a.stage == 'probe-benchmark':\n"
         "    require(a.cell == 'T100-G-300M-natural', 'recovery benchmark only for300M-natural')\n"
         "cpus = 2 if a.stage in ('pose-validate', 'probe-aggregate') else 8"),
        ("'probe-benchmark': 5400 if a.cell == 'T100-G-300M-natural' else 2700,",
         "'probe-benchmark': 7200 if a.dataset == 'epn612' else 600,"),
        ("0 < s['walltime_seconds'] <= 7200", "0 < s['walltime_seconds'] <= 10800"),
        ("<= 136*3600, 'probe fits exceed136 core-hours'", "<= 240*3600, 'probe fits exceed240 core-hours'"),
        ('# The five calls are sequential; any failure stops the chain, no retry.',
         '# Exactly one missing dataset; any failure stops without retry.'),
        ('for ds in DATASETS]', 'for ds in (a.dataset,)]'),
    ]
    expected = frozen.decode()
    for before, after in edits:
        assert expected.count(before) == (2 if before == 'logs/scaling_readers_%j.out' else 1)
        expected = expected.replace(before, after)
    assert LAUNCHER.read_text() == expected
    subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], check=True)


def benchmark(h, dataset='epn612', **kwargs):
    fields = {'TimeLimit': '02:00:00' if dataset == 'epn612' else '00:10:00', **kwargs.pop('fields', {})}
    flags = kwargs.pop('flags', ['--cell', CELLS[1], '--dataset', dataset])
    return h['run']('probe-benchmark', cell=CELLS[1], flags=flags, fields=fields, **kwargs)


@pytest.mark.parametrize('dataset', ['epn612', 'uci_emg'])
def test_one_missing_dataset_only_and_original_numeric_dispatch(harness, dataset):
    h = harness
    got = benchmark(h, dataset)
    assert got.returncode == 0, got.stdout+got.stderr
    rows = [json.loads(s) for s in h['record'].read_text().splitlines()]
    assert len(rows) == 1
    row, = rows
    argv = row['argv']
    assert argv[:2] == ['scripts/scaling_probe.py', 'benchmark']
    assert argv[argv.index('--dataset')+1] == dataset
    assert argv[argv.index('--cell')+1] == CELLS[1]
    assert argv[argv.index('--expected-code-sha')+1] == PIN
    assert argv[argv.index('--expected-plan-sha256')+1] == h['plan_pin']['sha256']
    assert Path(argv[argv.index('--out')+1]).name == dataset
    assert row['cwd'] == str(h['code'])
    assert row['env'] == {'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                          'OPENBLAS_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': ''}
    assert 'import scaling_compare;sys.argv=sys.argv[1:];runpy.run_path' in row['bootstrap']


@pytest.mark.parametrize('flags', [
    ['--cell', CELLS[0], '--dataset', 'epn612'],
    ['--cell', CELLS[2], '--dataset', 'epn612'],
    ['--cell', CELLS[1]],
    ['--cell', CELLS[1], '--dataset', 'ninapro_db2'],
    ['--cell', CELLS[1], '--datasets', 'epn612', 'uci_emg'],
    ['--cell', CELLS[1], '--dataset', 'epn612', '--dataset', 'uci_emg', '--max-iter', '100'],
])
def test_recovery_scope_is_not_an_arbitrary_rerun(harness, flags):
    assert benchmark(harness, flags=flags).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('dataset,clock', [('epn612', '01:30:00'), ('epn612', '02:00:01'),
                                         ('uci_emg', '00:09:59'), ('uci_emg', '00:10:01')])
def test_benchmark_walltime_is_exact(harness, dataset, clock):
    got = benchmark(harness, dataset, fields={'TimeLimit': clock})
    assert got.returncode != 0 and 'actual walltime differs' in got.stderr
    assert not harness['record'].exists()


def test_controller_failure_stops_without_retry_or_second_dataset(harness):
    got = benchmark(harness, environment={'TEST_FAIL_CONTROLLER': '1'})
    assert got.returncode != 0
    assert len(harness['record'].read_text().splitlines()) == 1


@pytest.mark.parametrize('fields', [
    {'NodeList': 'login13'}, {'NumCPUs': '4'}, {'NumNodes': '2'}, {'JobState': 'PENDING'},
    {'AllocTRES': 'cpu=8,mem=30G,node=1,gres/gpu=1'}, {'AllocTRES': 'cpu=8,mem=31G,node=1'},
])
def test_scheduler_and_zero_gpu_guards_remain_strict(harness, fields):
    assert benchmark(harness, fields=fields).returncode != 0
    assert not harness['record'].exists()


def test_even_matching_other_checkout_and_expected_sha_are_rejected(harness):
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40, 'TEST_GIT_HEAD': 'b'*40})
    assert got.returncode != 0 and 'original frozen reader code' in got.stderr
    assert not harness['record'].exists()


@pytest.mark.parametrize('kind', ['bad_plan_hash', 'input_changed', 'existing_output', 'dirty_code'])
def test_pins_and_fresh_outputs_still_guard_the_recovery(harness, tmp_path, kind):
    h = harness
    kwargs = {}
    if kind == 'bad_plan_hash': kwargs['plan_override'] = {**h['plan_pin'], 'sha256': 'a'*64}
    if kind == 'input_changed': kwargs['environment'] = {'TEST_MUTATE_INPUT': h['plan_pin']['path']}
    if kind == 'existing_output': kwargs['out'] = tmp_path
    if kind == 'dirty_code': kwargs['environment'] = {'TEST_DIRTY': '1'}
    got = benchmark(h, **kwargs)
    assert got.returncode != 0
    assert h['record'].exists() == (kind == 'input_changed')


def publish_resource(h, shards):
    h['resource']['caps']['walltime_seconds'] = 10800
    h['resource']['shards'] = shards
    h['repin'](h['resource_pin'], h['resource'])


def fit(h, shard=None, **kwargs):
    shard = shard or h['resource']['shards'][0]
    flags = ['--cell', shard['cell'], '--task', shard['task'], '--datasets', *shard['datasets'],
             '--folds', *map(str, shard['folds']), '--resource-plan', h['resource_pin']['path'],
             '--resource-sha256', h['resource_pin']['sha256']]
    seconds = shard['walltime_seconds']
    clock = f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}'
    fields = {'TimeLimit': clock, **kwargs.pop('fields', {})}
    return h['run']('probe-fit', flags=flags, fields=fields, **kwargs)


def fifteen_shards(h):
    return [{**s, 'folds': folds} for s in h['resource']['shards']
            for folds in ([[0], [1], [2], [3]] if s['task'] == 'p1' else [[0, 1, 2, 3]])]


def test_exact_10800_seconds_is_allowed_and_dispatched_to_same_controller(harness):
    h = harness
    shards = copy.deepcopy(h['resource']['shards'])
    for s in shards: s['walltime_seconds'] = 10800
    publish_resource(h, shards)
    got = fit(h)
    assert got.returncode == 0, got.stderr
    row = json.loads(h['record'].read_text())
    assert row['argv'][:2] == ['scripts/scaling_probe.py', 'fit-shard']
    assert row['argv'][row['argv'].index('--expected-code-sha')+1] == PIN


def test_fifteen_shards_and_exact240corehours_are_accepted(harness):
    h = harness
    shards = fifteen_shards(h)
    assert len(shards) == 15 and sum(s['walltime_seconds']*s['cpus'] for s in shards) == 240*3600
    publish_resource(h, shards)
    got = fit(h)
    assert got.returncode == 0, got.stderr


@pytest.mark.parametrize('kind,expected', [
    ('10801_seconds', 'published envelope'), ('over240corehours', '240 core-hours'),
    ('16shards', '1..15'), ('missing_coverage', 'exactly once'), ('duplicate_coverage', 'exactly once'),
    ('wrong_cpu', 'published envelope'), ('wrong_memory', 'published envelope'),
    ('wrong_code', 'resource plan identity'), ('wrong_plan', 'resource plan identity'),
])
def test_fit_limits_and_entire_grid_stay_enforced(harness, kind, expected):
    h = harness
    shards = fifteen_shards(h)
    if kind == '10801_seconds': shards[0]['walltime_seconds'] = 10801
    if kind == 'over240corehours': shards[0]['walltime_seconds'] = 7201
    if kind == '16shards':
        i = next(i for i, s in enumerate(shards) if s['task'] == 'p2')
        s = shards.pop(i)
        shards += [{**s, 'folds': [0, 1]}, {**s, 'folds': [2, 3]}]
        for s in shards: s['walltime_seconds'] = 3600
    if kind == 'missing_coverage': shards.pop()
    if kind == 'duplicate_coverage': shards[1] = copy.deepcopy(shards[0])
    if kind == 'wrong_cpu': shards[0]['cpus'] = 4
    if kind == 'wrong_memory': shards[0]['memory_gib'] = 31
    if kind == 'wrong_code': h['resource']['code_sha'] = 'b'*40
    if kind == 'wrong_plan': h['resource']['plan_content_sha256'] = 'a'*64
    publish_resource(h, shards)
    got = fit(h)
    assert got.returncode != 0 and expected in got.stderr
    assert not h['record'].exists()


def test_fit_allocation_must_equal_actual_scheduler_walltime(harness):
    got = fit(harness, fields={'TimeLimit': '03:00:00'})
    assert got.returncode != 0 and 'actual walltime differs' in got.stderr
    assert not harness['record'].exists()


@pytest.mark.parametrize('stage', ['pose-validate', 'pose-benchmark', 'pose-ridge', 'probe-aggregate', 'compare'])
def test_other_stages_retain_original_commands_and_limits(harness, stage):
    got = harness['run'](stage)
    assert got.returncode == 0, got.stderr
    row = json.loads(harness['record'].read_text())
    assert row['argv'][0] == ('scripts/scaling_compare.py' if stage == 'compare' else
                              'scripts/scaling_probe.py' if stage == 'probe-aggregate' else 'scripts/scaling_pose_readout.py')
    assert '--dataset' not in row['argv']
    assert row['argv'][row['argv'].index('--expected-code-sha')+1] == PIN
