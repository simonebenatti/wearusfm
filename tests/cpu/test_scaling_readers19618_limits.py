"""Per-dataset benchmarks and fit ceilings of the 19618 CPU launcher; real shell guards.

Derived from test_scaling_readers_recovery_launcher.py by the replacements declared in scaling_derive19618.py.
Reuses the scheduler/git/controller fixture of test_scaling_readers19618_launcher.py. The
launcher has no fixed code pin: it requires the frozen checkout given as EXPECTED_SHA.
No cluster calls, scientific fits or synthetic readiness.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT/'scripts/slurm/scaling_readers19618.sbatch'
spec = importlib.util.spec_from_file_location('reader19618_fixture',
                                             ROOT/'tests/cpu/test_scaling_readers19618_launcher.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
CELLS, TASKS = original.CELLS, original.TASKS
PIN = original.CODE  # the fixture's frozen checkout; the launcher itself pins no commit
FREEZE = json.loads((ROOT/'results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json').read_text())
SECONDS = FREEZE['readers']['stages']['probe_benchmark']['seconds_by_cell_and_dataset']
GRID = [(cell, dataset) for cell in CELLS for dataset in original.DATASETS]


def hms(seconds):
    return f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}'


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


def test_launcher_is_the_declared_derivation_of_the_reviewed_recovery_launcher_with_the_frozen_table():
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    import scaling_derive19618 as D
    target = 'scripts/slurm/scaling_readers19618.sbatch'
    source, pinned, _ = D.DERIVATIONS[target]
    assert source == 'scripts/slurm/scaling_readers_recovery.sbatch'
    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == pinned
    text = LAUNCHER.read_text()
    assert text == D.derive(target)
    table = ast.literal_eval(text.split('BENCHMARK_SECONDS = ', 1)[1].split('\ncpus = ', 1)[0])
    assert table == SECONDS and FREEZE['readers']['stages']['probe_benchmark']['jobs'] == len(GRID) == 15
    assert FREEZE['readers']['stages']['probe_benchmark']['datasets_per_job'] == 1
    assert 'aef665ba7ea66fc80291846487907bb3ef9107df' not in text
    subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], check=True)


def benchmark(h, dataset='epn612', cell=CELLS[1], **kwargs):
    fields = {'TimeLimit': hms(SECONDS[cell][dataset]), **kwargs.pop('fields', {})}
    flags = kwargs.pop('flags', ['--cell', cell, '--dataset', dataset])
    return h['run']('probe-benchmark', cell=cell, flags=flags, fields=fields, **kwargs)


@pytest.mark.parametrize('cell,dataset', GRID)
def test_each_of_the_fifteen_benchmarks_runs_one_dataset_with_its_frozen_limit(harness, cell, dataset):
    h = harness
    got = benchmark(h, dataset, cell)
    assert got.returncode == 0, got.stdout+got.stderr
    rows = [json.loads(s) for s in h['record'].read_text().splitlines()]
    assert len(rows) == 1
    row, = rows
    argv = row['argv']
    assert argv[:2] == ['scripts/scaling_probe19618.py', 'benchmark']
    assert argv[argv.index('--dataset')+1] == dataset
    assert argv[argv.index('--cell')+1] == cell
    assert argv[argv.index('--expected-code-sha')+1] == PIN
    assert argv[argv.index('--expected-plan-sha256')+1] == h['plan_pin']['sha256']
    assert Path(argv[argv.index('--out')+1]).name == dataset
    assert row['cwd'] == str(h['code'])
    assert row['env'] == {'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                          'OPENBLAS_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': ''}
    assert 'import scaling_compare19618;sys.argv=sys.argv[1:];runpy.run_path' in row['bootstrap']


@pytest.mark.parametrize('flags', [
    ['--cell', CELLS[1]],
    ['--dataset', 'epn612'],
    ['--cell', 'T100-G-30M-matched', '--dataset', 'epn612'],
    ['--cell', CELLS[1], '--dataset', 'other'],
    ['--cell', CELLS[1], '--datasets', 'epn612', 'uci_emg'],
    ['--cell', CELLS[1], '--dataset', 'epn612', '--dataset', 'uci_emg', '--max-iter', '100'],
])
def test_benchmark_needs_exactly_one_registered_cell_and_dataset(harness, flags):
    assert benchmark(harness, flags=flags).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('cell,dataset', GRID)
def test_benchmark_limit_of_another_cell_or_dataset_is_refused(harness, cell, dataset):
    other = next(s for s in sorted({v for row in SECONDS.values() for v in row.values()}) if s != SECONDS[cell][dataset])
    got = benchmark(harness, dataset, cell, fields={'TimeLimit': hms(other)})
    assert got.returncode != 0 and 'actual walltime differs' in got.stderr
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


def test_no_fixed_code_pin_but_expected_sha_must_be_the_frozen_checkout_head(harness):
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40})
    assert got.returncode != 0 and not harness['record'].exists()
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40, 'TEST_GIT_HEAD': 'b'*40})
    assert got.returncode == 0, got.stderr
    row = json.loads(harness['record'].read_text())
    assert row['argv'][row['argv'].index('--expected-code-sha')+1] == 'b'*40


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
    assert row['argv'][:2] == ['scripts/scaling_probe19618.py', 'fit-shard']
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
    assert row['argv'][0] == ('scripts/scaling_compare19618.py' if stage == 'compare' else
                              'scripts/scaling_probe19618.py' if stage == 'probe-aggregate' else 'scripts/scaling_pose_readout19618.py')
    assert '--dataset' not in row['argv']
    assert row['argv'][row['argv'].index('--expected-code-sha')+1] == PIN
