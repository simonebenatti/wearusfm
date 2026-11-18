"""Execute the CPU launcher guards; intercept only scientific subprocesses.

No cluster, data, GPU or fitting is used. Scheduler/git/library identities are
fixtures. The real shell, frozen-checkout helper, pins and budget math execute.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT/'scripts/slurm/scaling_readers.sbatch'
CODE = 'c'*40
CELLS = ('T100-G-30M-natural', 'T100-G-300M-natural', 'T100-G-300M-matched')
TASKS = {'p1': ['ninapro_db2', 'ninapro_db3', 'ninapro_db6'], 'p2': ['epn612', 'uci_emg']}
DATASETS = TASKS['p1']+TASKS['p2']


def jhash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def harness(tmp_path):
    root = tmp_path.resolve()
    work, bins = root/'work', root/'bin'
    code = work/'wearusfm_campaigns/code_C'
    plans = work/'wearusfm_campaigns/later_plan_D'
    (code/'scripts/slurm').mkdir(parents=True)
    plans.mkdir()
    bins.mkdir()
    shutil.copyfile(ROOT/'scripts/slurm/lib_code_dir.sh', code/'scripts/slurm/lib_code_dir.sh')
    (work/'venv_wearusfm/bin').mkdir(parents=True)
    (work/'venv_wearusfm/bin/activate').write_text('# fixture activation\n')
    def save(name, obj):
        path = plans/name
        path.write_text(json.dumps(obj))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    sources = {'pose.py': 'd'*64}
    numeric = {'code_sha': CODE, 'sources': sources, 'versions': {'python': '3.test', 'numpy': '2.test', 'torch': '2.test'},
               'threads': {k: '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
               'torch_threads': 8, 'threadpools': [{'internal_api': 'openblas', 'prefix': 'libopenblas', 'version': 'test',
               'num_threads': 8, 'architecture': 'test-cpu', 'threading_layer': 'pthreads'}]}
    plan = {'caches': {'content': 'cachepins'}, 'references': {'content': 'referencepins'},
            'projection': {'sha256': 'e'*64}, 'training_source_contract': {'sha256': 'f'*64}}
    plan_pin = save('plan.json', plan)
    common = {'protocol': 'scaling-pose-readout-20261104-v1', 'step': 5000, 'recipe': {'test': 'recipe'},
              'complete': True, 'runtime': numeric, 'plan_sha256': plan_pin['sha256'], 'input_pins': plan['caches'],
              'reference_pins': plan['references'], 'projection_pin': plan['projection'], 'training_source_contract': plan['training_source_contract']}
    validation = {**common, 'status': 'validated_input_metadata', 'scientific_metrics_computed': False, 'slurm_job_id': '100',
        'inspections': {a+'/'+s: {} for a in ('T100-G-30M', 'T100-G-300M') for s in ('train', 'val')}}
    validation['runtime'] = copy.deepcopy(numeric)
    validation['runtime']['threads'] = {k: '2' for k in numeric['threads']}
    validation['runtime']['torch_threads'] = 2
    validation['runtime']['threadpools'][0]['num_threads'] = 2
    accounting = {'JobIDRaw': '100', 'State': 'COMPLETED', 'ExitCode': '0:0', 'ElapsedRaw': 10, 'AllocCPUS': 2, 'MaxRSS_bytes': 2**30}
    benches, benchmark_accounting = {}, {}
    for index, cell in enumerate(CELLS):
        arm = 'T100-G-30M' if cell == CELLS[0] else 'T100-G-300M'
        b = {**common, 'status': 'train_only_timing', 'cell': cell, 'slurm_job_id': str(200+index), 'inspections': {arm+'/train': {}},
             'timings': {'statistics_seconds': 2., 'three_eigensystems_seconds': 1., 'n_features': 896 if cell == CELLS[1] else 384,
                         'n_users': 40, 'valid_fit_elements': 4813620}}
        benches[cell] = save(cell+'.json', b)
        benchmark_accounting[cell] = save(cell+'_accounting.json', {'JobIDRaw': str(200+index), 'State': 'COMPLETED',
             'ExitCode': '0:0', 'AllocCPUS': 8, 'ElapsedRaw': 60, 'MaxRSS_bytes': 1024**3})
    # ceil60(1.5 * (2*10 + 3*(11*2+13*1)) + 120) =360.
    gate = {'protocol': 'scaling-pose-resources-20261104-v1', 'complete': True, 'code_sha': CODE,
            'plan_sha256': plan_pin['sha256'], 'allocation': {'cpus': 8, 'memory_gib': 30, 'walltime_seconds': 3600},
            'validation_report': save('validation.json', validation), 'validation_accounting': save('accounting.json', accounting),
            'benchmarks': benches, 'benchmark_accounting': benchmark_accounting, 'derived_seconds': 360}
    gate_pin = save('gate.json', gate)
    shards = [{'cell': c, 'task': t, 'datasets': ds, 'folds': [0, 1, 2, 3], 'cpus': 8, 'memory_gib': 30, 'walltime_seconds': 7200}
              for c in CELLS for t, ds in TASKS.items()]
    resource = {'protocol': 'scaling-probe-resources-20261104-v1', 'complete': True, 'code_sha': CODE,
                'plan_content_sha256': jhash(plan), 'caps': {'cpus': 8, 'memory_gib': 30, 'walltime_seconds': 7200}, 'shards': shards}
    resource_pin = save('resource.json', resource)
    pose_report = {**common, 'status': 'completed'}
    inputs = {'protocol': 'scaling-comparison-inputs-20261104-v1', 'checkpoint_step': 5000,
              'pose_report': save('pose_completed.json', pose_report)}
    inputs_pin = save('inputs.json', inputs)
    shard = plans/'completed_shard.json'
    shard.write_text('{}')
    record = root/'commands.jsonl'
    def executable(name, body):
        path = bins/name
        path.write_text('#!'+sys.executable+'\n'+body)
        path.chmod(0o755)
    executable('module', 'pass\n')
    executable('scontrol', "import os\nprint(os.environ['TEST_JOB_INFO'])\n")
    executable('git', """import os,sys
if sys.argv[3:]==['rev-parse','--is-inside-work-tree']: print('true')
elif sys.argv[3:]==['rev-parse','HEAD']: print(os.environ.get('TEST_GIT_HEAD','c'*40))
elif sys.argv[3:]==['status','--porcelain','--untracked-files=all']:
    if os.environ.get('TEST_DIRTY'): print(' M file')
else: raise AssertionError(sys.argv)
""")
    executable('python3', """import copy,json,os,socket,subprocess,sys,types
assert sys.argv[1]=='-'
socket.gethostname=lambda:os.environ.get('TEST_HOST','login08')
runtime=json.loads(os.environ['TEST_RUNTIME'])
if os.environ.get('TEST_RUNTIME_NUMPY'): runtime['versions']['numpy']=os.environ['TEST_RUNTIME_NUMPY']
if os.environ.get('TEST_RUNTIME_THREADS'):
    runtime['threadpools'][0]['num_threads']=int(os.environ['TEST_RUNTIME_THREADS'])
module=types.SimpleNamespace(PROTOCOL='scaling-pose-readout-20261104-v1',recipe=lambda:{'test':'recipe'},source_hashes=lambda:runtime['sources'],
    Q=types.SimpleNamespace(runtime_identity=lambda sha:copy.deepcopy(runtime)))
sys.modules['scaling_pose_readout']=module
sys.modules['scaling_compare']=types.ModuleType('scaling_compare')
def dispatch(cmd,check):
    assert check and cmd[1]=='-c' and 'import scaling_compare;sys.argv=sys.argv[1:];runpy.run_path' in cmd[2]
    assert cmd[3] in ['scripts/scaling_pose_readout.py','scripts/scaling_probe.py','scripts/scaling_compare.py']
    with open(os.environ['TEST_RECORD'],'a') as f:
        f.write(json.dumps({'argv':cmd[3:],'bootstrap':cmd[2],'cwd':os.getcwd(),'env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES']}})+'\\n')
    if os.environ.get('TEST_MUTATE_INPUT'):
        with open(os.environ['TEST_MUTATE_INPUT'],'a') as f: f.write(' ')
    if os.environ.get('TEST_FAIL_CONTROLLER'): raise subprocess.CalledProcessError(9,cmd)
subprocess.run=dispatch
sys.argv=sys.argv[1:]
exec(compile(sys.stdin.read(),'<real-launcher-guards>','exec'))
""")
    for path in [*code.rglob('*'), code]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    env = {**os.environ, 'PATH': str(bins)+os.pathsep+os.environ['PATH'], 'WORK': str(work), 'CODE_DIR': str(code),
           'EXPECTED_SHA': CODE, 'SLURM_JOB_ID': '12345', 'SLURM_JOB_PARTITION': 'lrd_all_serial', 'SLURM_JOB_NUM_NODES': '1',
           'SLURM_NTASKS': '1', 'TEST_RECORD': str(record), 'TEST_RUNTIME': json.dumps(numeric)}
    def run(stage='pose-validate', cell=CELLS[0], flags=None, fields=None, environment=None, plan_override=None, out=None):
        cpus = 2 if stage in ('pose-validate', 'probe-aggregate') else 8
        times = {'pose-validate': '00:30:00', 'pose-benchmark': '00:10:00', 'pose-ridge': '01:00:00',
                 'probe-benchmark': '01:30:00' if cell == CELLS[1] else '00:45:00', 'probe-fit': '02:00:00',
                 'probe-aggregate': '00:10:00', 'compare': '00:30:00'}
        default = {'pose-validate': [], 'pose-benchmark': ['--cell', cell],
                   'pose-ridge': ['--pose-budget', gate_pin['path'], '--pose-budget-sha256', gate_pin['sha256']],
                   'probe-benchmark': ['--cell', cell],
                   'probe-fit': ['--cell', cell, '--task', 'p1', '--datasets', *TASKS['p1'], '--folds', '0', '1', '2', '3',
                                 '--resource-plan', resource_pin['path'], '--resource-sha256', resource_pin['sha256']],
                   'probe-aggregate': ['--cell', cell, '--task', 'p1', '--shards', str(shard)], 'compare': []}
        info = {'JobId': '12345', 'JobState': 'RUNNING', 'Partition': 'lrd_all_serial', 'NodeList': 'login08', 'NumNodes': '1',
                'NumCPUs': str(cpus), 'NumTasks': '1', 'TimeLimit': times[stage], 'AllocTRES': f'cpu={cpus},mem=30G,node=1'}
        info.update(fields or {})
        chosen = plan_override or (inputs_pin if stage == 'compare' else plan_pin)
        args = [str(LAUNCHER), stage, chosen['path'], chosen['sha256'], str(out or root/'output'), *(flags if flags is not None else default[stage])]
        return subprocess.run(['/bin/bash', *args], cwd=root, env={**env, 'SLURM_CPUS_PER_TASK': str(cpus),
            'TEST_JOB_INFO': ' '.join(k+'='+v for k,v in info.items()), **(environment or {})}, capture_output=True, text=True, timeout=20)
    def repin(pin_value, value):
        updated = save(Path(pin_value['path']).name, value)
        pin_value.clear()
        pin_value.update(updated)
    yield dict(run=run, record=record, code=code, plans=plans, plan=plan, plan_pin=plan_pin, gate=gate, gate_pin=gate_pin,
               resource=resource, resource_pin=resource_pin, inputs=inputs, inputs_pin=inputs_pin, numeric=numeric,
               save=save, repin=repin, work=work)
    for path in [code, *code.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize('stage', ['pose-validate', 'pose-benchmark', 'pose-ridge', 'probe-fit', 'probe-aggregate', 'compare'])
def test_stages_dispatch_exact_controller_without_science_overrides(harness, stage):
    h = harness
    got = h['run'](stage)
    assert got.returncode == 0, got.stdout+got.stderr
    rows = [json.loads(line) for line in h['record'].read_text().splitlines()]
    assert len(rows) == 1
    cmd = rows[0]['argv']
    assert cmd[0] == ('scripts/scaling_compare.py' if stage == 'compare' else
                      'scripts/scaling_pose_readout.py' if stage.startswith('pose') else 'scripts/scaling_probe.py')
    assert cmd[cmd.index('--expected-code-sha')+1] == CODE
    assert '--step' not in cmd and '--seed' not in cmd
    assert str(h['plans']) in ' '.join(cmd) and rows[0]['cwd'] == str(h['code'])
    threads = '2' if stage in ('pose-validate', 'probe-aggregate') else '8'
    assert rows[0]['env'] == {k: threads for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')} | {'CUDA_VISIBLE_DEVICES': ''}


@pytest.mark.parametrize('cell', CELLS)
def test_probe_benchmark_all_five_datasets_sequentially(harness, cell):
    h = harness
    got = h['run']('probe-benchmark', cell=cell)
    assert got.returncode == 0, got.stderr
    cmds = [json.loads(line)['argv'] for line in h['record'].read_text().splitlines()]
    assert [cmd[cmd.index('--dataset')+1] for cmd in cmds] == DATASETS
    assert all(cmd[1] == 'benchmark' and cmd[cmd.index('--cell')+1] == cell for cmd in cmds)
    assert len({cmd[cmd.index('--out')+1] for cmd in cmds}) == 5


def test_benchmark_failure_stops_without_retries(harness):
    h = harness
    got = h['run']('probe-benchmark', environment={'TEST_FAIL_CONTROLLER': '1'})
    assert got.returncode != 0
    assert len(h['record'].read_text().splitlines()) == 1


@pytest.mark.parametrize('fields', [{'JobId': '999'}, {'JobState': 'PENDING'}, {'Partition': 'other'}, {'NodeList': 'login07'},
    {'NumNodes': '2'}, {'NumCPUs': '8'}, {'NumTasks': '2'}, {'TimeLimit': '00:29:59'}, {'TimeLimit': 'UNLIMITED'},
    {'TimeLimit': '00:60:00'}, {'AllocTRES': 'cpu=2,mem=30G,node=1,gres/gpu=1'},
    {'AllocTRES': 'cpu=2,mem=31G,node=1'}, {'AllocTRES': 'cpu=4,mem=30G,node=1'}])
def test_scheduler_guards_before_controller(harness, fields):
    assert harness['run'](fields=fields).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('environment', [{'TEST_HOST': 'login07'}, {'SLURM_CPUS_PER_TASK': '8'}, {'SLURM_JOB_NUM_NODES': '2'},
    {'TEST_DIRTY': '1'}, {'TEST_GIT_HEAD': 'a'*40}])
def test_environment_and_frozen_code_guards(harness, environment):
    assert harness['run'](environment=environment).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('kind', ['bad_hash', 'relative', 'existing', 'code_output', 'unregistered_override'])
def test_paths_pins_and_no_arbitrary_cli(harness, kind, tmp_path):
    h = harness
    kwargs = {}
    if kind == 'bad_hash': kwargs['plan_override'] = {**h['plan_pin'], 'sha256': 'a'*64}
    if kind == 'relative': kwargs['plan_override'] = {**h['plan_pin'], 'path': 'relative.json'}
    if kind == 'existing': kwargs['out'] = tmp_path
    if kind == 'code_output': kwargs['out'] = h['code']/'output'
    if kind == 'unregistered_override': kwargs['flags'] = ['--max-iter', '10']
    assert h['run'](**kwargs).returncode != 0
    assert not h['record'].exists()


@pytest.mark.parametrize('stage,flags', [('pose-validate', ['--cell', CELLS[0]]), ('pose-ridge', []), ('probe-fit', ['--cell', CELLS[0]]),
    ('compare', ['--task', 'p1']), ('probe-benchmark', ['--cell', CELLS[0], '--datasets', 'epn612'])])
def test_whitelisted_flags_exact_per_stage(harness, stage, flags):
    assert harness['run'](stage, flags=flags).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('kind', ['missing_cell', 'stale_hash', 'wrong_support', 'wrong_code', 'failed_validate', 'different_job',
    'wrong_dimension', 'budget_too_small', 'excess_timing', 'not_train_only', 'changed_projection', 'runtime_mismatch',
    'missing_benchmark_accounting', 'benchmark_failed', 'benchmark_wrong_job', 'benchmark_over_memory', 'benchmark_missing_rss', 'validation_missing_rss'])
def test_pose_ridge_gate_requires_three_real_pinned_benchmarks(harness, kind):
    h = harness
    gate = h['gate']
    if kind == 'missing_cell': del gate['benchmarks'][CELLS[2]]
    elif kind == 'missing_benchmark_accounting': del gate['benchmark_accounting'][CELLS[2]]
    elif kind.startswith('benchmark_'):
        pin = gate['benchmark_accounting'][CELLS[2]]
        obj = json.loads(Path(pin['path']).read_text())
        if kind == 'benchmark_failed': obj['ExitCode'] = '1:0'
        if kind == 'benchmark_wrong_job': obj['JobIDRaw'] = '999'
        if kind == 'benchmark_over_memory': obj['MaxRSS_bytes'] = 31*2**30
        if kind == 'benchmark_missing_rss': obj.pop('MaxRSS_bytes')
        h['repin'](pin, obj)
    elif kind == 'budget_too_small': gate['derived_seconds'] = 300
    elif kind in ('failed_validate', 'different_job', 'validation_missing_rss'):
        pin = gate['validation_accounting']
        obj = json.loads(Path(pin['path']).read_text())
        if kind == 'validation_missing_rss': obj.pop('MaxRSS_bytes')
        else: obj['State' if kind == 'failed_validate' else 'JobIDRaw'] = 'FAILED' if kind == 'failed_validate' else '999'
        h['repin'](pin, obj)
    else:
        pin = gate['benchmarks'][CELLS[2]]
        obj = json.loads(Path(pin['path']).read_text())
        if kind == 'wrong_support': obj['timings']['valid_fit_elements'] -= 1
        if kind == 'wrong_dimension': obj['timings']['n_features'] = 896
        if kind == 'wrong_code': obj['runtime']['code_sha'] = 'a'*40
        if kind == 'excess_timing': obj['timings']['statistics_seconds'] = 10000
        if kind == 'not_train_only': obj['status'] = 'completed'
        if kind == 'changed_projection': obj['projection_pin'] = {'sha256': 'a'*64}
        if kind == 'runtime_mismatch': obj['runtime']['versions']['numpy'] = 'different'
        h['repin'](pin, obj)
        if kind == 'stale_hash': Path(pin['path']).write_text('{}')
    h['repin'](h['gate_pin'], gate)
    got = h['run']('pose-ridge')
    assert got.returncode != 0 and not h['record'].exists()


@pytest.mark.parametrize('kind', ['missing_coverage', 'duplicated_coverage', 'too_many_shards', 'too_long', 'wrong_cpu', 'wrong_mem', 'wrong_code', 'wrong_plan'])
def test_probe_allocation_requires_published_complete_nonoverlapping_grid(harness, kind):
    h = harness
    b = h['resource']
    if kind == 'missing_coverage': b['shards'].pop()
    if kind == 'duplicated_coverage': b['shards'].append(copy.deepcopy(b['shards'][0]))
    if kind == 'too_many_shards': b['shards'] *= 3
    if kind == 'too_long': b['shards'][0]['walltime_seconds'] = 7201
    if kind == 'wrong_cpu': b['shards'][0]['cpus'] = 4
    if kind == 'wrong_mem': b['shards'][0]['memory_gib'] = 31
    if kind == 'wrong_code': b['code_sha'] = 'a'*40
    if kind == 'wrong_plan': b['plan_content_sha256'] = 'a'*64
    h['repin'](h['resource_pin'], b)
    assert h['run']('probe-fit').returncode != 0 and not h['record'].exists()


def test_probe_actual_walltime_must_equal_selected_shard(harness):
    h = harness
    h['resource']['shards'][0]['walltime_seconds'] = 3600
    h['repin'](h['resource_pin'], h['resource'])
    assert h['run']('probe-fit').returncode != 0 and not h['record'].exists()


def test_probe_total136corehour_cap_is_enforced(harness):
    h = harness
    # Twelve nonoverlapping half-fold shards need192coreh at their7200s caps.
    h['resource']['shards'] = [{**s, 'folds': f} for s in h['resource']['shards'] for f in ([0, 1], [2, 3])]
    h['repin'](h['resource_pin'], h['resource'])
    got = h['run']('probe-fit')
    assert got.returncode != 0 and '136 core-hours' in got.stderr and not h['record'].exists()


@pytest.mark.parametrize('environment', [{'TEST_RUNTIME_NUMPY': 'other'}, {'TEST_RUNTIME_THREADS': '2'}])
def test_compare_current_runtime_must_match_pose_exactly(harness, environment):
    h = harness
    got = h['run']('compare', environment=environment)
    assert got.returncode != 0 and 'runtime differs' in got.stderr and not h['record'].exists()


def test_operational_pin_mutation_detected_after_controller(harness):
    h = harness
    got = h['run'](environment={'TEST_MUTATE_INPUT': h['plan_pin']['path']})
    assert got.returncode != 0 and 'inputs changed' in got.stderr
    assert h['record'].exists()


def test_equivalent_slurm_memory_clock_and_no_gpu_zero(harness):
    got = harness['run'](fields={'TimeLimit': '0-00:30:00', 'AllocTRES': 'cpu=2,mem=30720M,node=1,gres/gpu=0'})
    assert got.returncode == 0, got.stderr


def test_shell_syntax_and_explicit_cpu_only_node():
    assert subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], capture_output=True).returncode == 0
    text = LAUNCHER.read_text()
    assert '#SBATCH --nodelist=login08' in text and '#SBATCH -p lrd_all_serial' in text
    assert '#SBATCH --gres=gpu' not in text
    # Stage ceilings179coreh plus checkout1/3coreh leave2/3coreh reserve.
    assert 24+4+8+136+2+4+1 == 179


def test_real_common_preload_keeps_numeric_libraries_identical_for_all_controllers():
    """Exercise actual imports/runpy/CLI, including sklearn's extra runtime.

    One thread is intentional for this local CPU test. Production requires8;
    --help stops before any fitting, cache access or cluster operations.
    """
    code = '''import ast,os,runpy,subprocess,sys
from pathlib import Path
sys.path.insert(0,'scripts')
# Read the exact operational bootstrap, not an imitation in the test.
shell=Path('scripts/slurm/scaling_readers.sbatch').read_text()
embedded=shell.split("<<'PY'\\n",1)[1].rsplit('\\nPY\\n',1)[0]
tree=ast.parse(embedded)
bootstrap=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in n.targets))
import scaling_compare
import scaling_pose_readout as P
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
before=P.Q.runtime_identity(head)
assert before['threadpools']
assert before['torch_threads']==1
for script in ('scaling_pose_readout.py','scaling_probe.py','scaling_compare.py'):
    sys.argv=['-c','scripts/'+script,'--help']
    try: exec(bootstrap)
    except SystemExit as exc: assert exc.code==0
    assert P.Q.runtime_identity(head)==before,script
# Execute each exact bootstrap again, allowing source loading but intercepting
# only the final main entry. --help alone cannot expose relative __file__ bugs.
original=runpy.run_path
def inspect_source(path,run_name):
    assert run_name=='__main__' and Path(path).is_absolute()
    namespace=original(path,run_name='__bootstrap_source_test__')
    assert Path(namespace['__file__']).is_absolute()
    assert namespace['source_hashes']()
    return namespace
runpy.run_path=inspect_source
for script in ('scaling_pose_readout.py','scaling_probe.py','scaling_compare.py'):
    sys.argv=['-c','scripts/'+script,'--help']
    exec(bootstrap)
    assert P.Q.runtime_identity(head)==before,script
'''
    env = {**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    got = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
    assert got.returncode == 0, got.stdout+got.stderr
