#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=login08
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH -J scaling_readers19618
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_readers19618_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_readers19618_%j.out

# Thin operational dispatcher: no fitting or scientific overrides here.
# Args: <stage> <absolute plan/inputs> <SHA256> <new absolute output> [flags]
# Frozen reader code C may consume pinned plans from a later published checkout D.
# Exact resources: validate 2CPU/30min; pose benchmark 8CPU/10min;
# pose ridge 8CPU/60min; probe benchmark 8CPU, one cell and one dataset per job,
# limits of the budget freeze v2; fit <=15shards, <=180min, <=240coreh;
# aggregate 2CPU/10min; compare 8CPU/30min. All use30G and zero GPUs.
# Every CPU process preloads scaling_compare19618, including its numeric libraries,
# before executing the unchanged controller. F2 observes this same module set.
# Submit stage-specific CPU/time overrides through the common submitter/ledger.
set -euo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
export SCALING_READER_JOB_INFO="$JOB_INFO"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""

python3 - "$@" <<'PY'
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

CELLS = ('T100-G-30M-natural', 'T100-G-300M-natural', 'T100-G-300M-matched')
TASKS = {'p1': ('ninapro_db2', 'ninapro_db3', 'ninapro_db6'), 'p2': ('epn612', 'uci_emg')}
DATASETS = sum(TASKS.values(), ())
STAGES = ('pose-validate', 'pose-benchmark', 'pose-ridge', 'probe-benchmark', 'probe-fit', 'probe-aggregate', 'compare')
p = argparse.ArgumentParser(description='Scaling19618 CPU dispatch; no scientific overrides')
p.add_argument('stage', choices=STAGES)
p.add_argument('plan', type=Path)
p.add_argument('plan_sha256')
p.add_argument('out', type=Path)
p.add_argument('--cell', choices=CELLS)
p.add_argument('--dataset', choices=DATASETS)
p.add_argument('--task', choices=TASKS)
p.add_argument('--datasets', nargs='+', choices=DATASETS)
p.add_argument('--folds', nargs='+', type=int)
p.add_argument('--shards', nargs='+', type=Path)
p.add_argument('--resource-plan', type=Path)
p.add_argument('--resource-sha256')
p.add_argument('--pose-budget', type=Path)
p.add_argument('--pose-budget-sha256')
a = p.parse_args()

def require(ok, message):
    if not ok:
        raise SystemExit(message)

allowed = {'pose-validate': (), 'pose-benchmark': ('cell',),
           'pose-ridge': ('pose_budget', 'pose_budget_sha256'),
           'probe-benchmark': ('cell', 'dataset'),
           'probe-fit': ('cell', 'task', 'datasets', 'folds', 'resource_plan', 'resource_sha256'),
           'probe-aggregate': ('cell', 'task', 'shards'), 'compare': ()}
optional = set(sum(allowed.values(), ()))
require(all((getattr(a, k) is not None) == (k in allowed[a.stage]) for k in optional),
        'stage requires exactly its whitelisted flags')
code, work = Path(os.environ['CODE_DIR']).resolve(strict=True), Path(os.environ['WORK']).resolve(strict=True)
expected_code = os.environ['EXPECTED_SHA']
require(code.is_relative_to(work/'wearusfm_campaigns'), 'frozen campaign code required')
states = {}

def state(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def pin(raw):
    require(isinstance(raw, dict) and re.fullmatch('[0-9a-f]{64}', str(raw.get('sha256', ''))), 'invalid input SHA256 pin')
    path = Path(raw['path'])
    require(path.is_absolute() and path == path.resolve(strict=True) and path.is_file(), 'canonical absolute input required')
    before = state(path)
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == raw['sha256'] and
            ('bytes' not in raw or type(raw['bytes']) is int and raw['bytes'] == len(content)), 'input hash/size differs')
    require(before == state(path), 'input changed while hashing')
    require(states.setdefault(str(path), before) == before, 'input changed between uses')
    return json.loads(content)

def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

plan = pin({'path': str(a.plan), 'sha256': a.plan_sha256})
out = a.out.resolve()
require(a.out.is_absolute() and a.out == out and not a.out.exists() and not a.out.is_symlink(), 'canonical new absolute output required')
require(not any(out.is_relative_to(root) for root in (code, work/'wearusfm', work/'wearusfm_campaigns')), 'output inside code')
if a.stage == 'probe-aggregate':
    require(all(not q.exists() and not q.is_symlink() for q in (out.with_suffix('.oof.npz'), out.with_suffix('.audit.jsonl'))), 'aggregate output sidecar exists')
    require(len(set(a.shards)) == len(a.shards), 'duplicate shard paths')
    for q in a.shards:
        require(q.is_absolute() and q == q.resolve(strict=True) and q.is_file(), 'canonical completed shard paths required')

# Budget freeze v2, readers.stages.probe_benchmark: fifteen jobs, one dataset each.
BENCHMARK_SECONDS = {
    'T100-G-30M-natural': {'ninapro_db2': 2700, 'ninapro_db3': 1200, 'ninapro_db6': 1200, 'epn612': 1800, 'uci_emg': 600},
    'T100-G-300M-natural': {'ninapro_db2': 3600, 'ninapro_db3': 1800, 'ninapro_db6': 3600, 'epn612': 7200, 'uci_emg': 600},
    'T100-G-300M-matched': {'ninapro_db2': 2700, 'ninapro_db3': 1200, 'ninapro_db6': 1200, 'epn612': 1800, 'uci_emg': 600}}
cpus = 2 if a.stage in ('pose-validate', 'probe-aggregate') else 8
seconds = {'pose-validate': 1800, 'pose-benchmark': 600, 'pose-ridge': 3600,
           'probe-benchmark': BENCHMARK_SECONDS[a.cell][a.dataset] if a.stage == 'probe-benchmark' else None,
           'probe-aggregate': 600, 'compare': 1800}.get(a.stage)
if a.stage == 'probe-fit':
    require(a.datasets and len(set(a.datasets)) == len(a.datasets) and set(a.datasets) <= set(TASKS[a.task]) and
            a.folds == sorted(set(a.folds)) and all(f in range(4) for f in a.folds), 'invalid shard coverage')
    budget = pin({'path': str(a.resource_plan), 'sha256': a.resource_sha256})
    require(budget.get('protocol') == 'scaling-probe-resources-19618-20261106-v1' and budget.get('complete') is True and
            budget.get('code_sha') == expected_code and budget.get('plan_content_sha256') == jhash(plan), 'probe resource plan identity differs')
    shards = budget.get('shards', [])
    require(isinstance(shards, list) and 1 <= len(shards) <= 15, 'probe allocation requires1..15 published shards')
    coverage = []
    for s in shards:
        require(s.get('cell') in CELLS and s.get('task') in TASKS and s.get('cpus') == 8 and s.get('memory_gib') == 30 and
                type(s.get('walltime_seconds')) is int and 0 < s['walltime_seconds'] <= 10800, 'probe shard exceeds published envelope')
        ds, folds = s.get('datasets'), s.get('folds')
        require(isinstance(ds, list) and ds and len(set(ds)) == len(ds) and set(ds) <= set(TASKS[s['task']]) and
                isinstance(folds, list) and folds and folds == sorted(set(folds)) and
                all(type(f) is int and f in range(4) for f in folds), 'invalid published shard coverage')
        coverage.extend((s['cell'], d, f) for d in ds for f in folds)
    expected = {(c, d, f) for c in CELLS for d in DATASETS for f in range(4)}
    require(len(coverage) == len(set(coverage)) and set(coverage) == expected, 'resource plan must cover every cell/dataset/fold exactly once')
    require(sum(s['cpus']*s['walltime_seconds'] for s in shards) <= 240*3600, 'probe fits exceed240 core-hours')
    wanted = dict(cell=a.cell, task=a.task, datasets=a.datasets, folds=a.folds)
    selected = [s for s in shards if all(s.get(k) == v for k, v in wanted.items())]
    require(len(selected) == 1, 'requested shard absent/duplicate')
    seconds = selected[0]['walltime_seconds']

fields = dict(x.split('=', 1) for x in os.environ['SCALING_READER_JOB_INFO'].split() if '=' in x)
job = os.environ['SLURM_JOB_ID']
require(job.isdigit() and int(job) > 0 and fields.get('JobId') == job and fields.get('JobState') == 'RUNNING', 'running scheduler job identity required')
require(all(fields.get(k) == v for k, v in {'Partition': 'lrd_all_serial', 'NodeList': 'login08', 'NumNodes': '1',
        'NumTasks': '1', 'NumCPUs': str(cpus)}.items()) and socket.gethostname().split('.')[0] == 'login08', 'wrong CPU partition/node/allocation')
require(os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_serial' and os.environ.get('SLURM_JOB_NUM_NODES') == '1' and
        os.environ.get('SLURM_NTASKS', '1') == '1' and os.environ.get('SLURM_CPUS_PER_TASK') == str(cpus), 'SLURM environment allocation differs')
tres = dict(x.split('=', 1) for x in fields.get('AllocTRES', '').split(',') if '=' in x)
require(tres.get('cpu') == str(cpus) and tres.get('node') == '1' and tres.get('mem') in ('30G', '30720M') and
        not any('gpu' in k and v != '0' for k, v in tres.items()), 'actual CPU/RAM/GPU allocation differs')
clock = fields.get('TimeLimit', '')
match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?', clock)
require(match is not None, 'invalid SLURM TimeLimit')
days, h, m, sec = match.groups()
if sec is None:
    require(days is None, 'invalid day TimeLimit')
    h, m, sec = '0', h, m
require(int(m) < 60 and int(sec) < 60, 'invalid TimeLimit fields')
actual_seconds = int(days or 0)*86400 + int(h)*3600 + int(m)*60 + int(sec)
require(actual_seconds == seconds, 'actual walltime differs from published stage/shard')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = str(cpus)
sys.path.insert(0, str(code/'scripts'))
# A common preload prevents an extra sklearn/OpenMP library appearing only in
# the comparison child after the parent's runtime check.
import scaling_compare19618

runtime_before = None
def current_pose_runtime():
    import scaling_pose_readout19618 as P
    r = P.Q.runtime_identity(expected_code)
    r['sources'] = P.source_hashes()
    return r

def require_runtime(recorded, current):
    require(recorded.get('code_sha') == expected_code and recorded.get('sources') == current.get('sources'), 'pose runtime code/source differs')
    for key in ('versions', 'threads', 'torch_threads', 'threadpools'):
        require(recorded.get(key) == current.get(key), 'pose runtime differs: '+key)
    require(current.get('torch_threads') == 8 and all(v == '8' for v in current.get('threads', {}).values()) and
            current.get('threadpools') and all(t.get('num_threads') == 8 for t in current['threadpools']), 'eight-thread numeric runtime required')

if a.stage == 'pose-ridge':
    import scaling_pose_readout19618 as P
    gate = pin({'path': str(a.pose_budget), 'sha256': a.pose_budget_sha256})
    require(gate.get('protocol') == 'scaling-pose-resources-19618-20261106-v1' and gate.get('complete') is True and
            gate.get('code_sha') == expected_code and gate.get('plan_sha256') == a.plan_sha256 and
            gate.get('allocation') == {'cpus': 8, 'memory_gib': 30, 'walltime_seconds': 3600}, 'pose budget identity/allocation differs')
    validation = pin(gate['validation_report'])
    accounting = pin(gate['validation_accounting'])
    require(validation.get('status') == 'validated_input_metadata' and validation.get('scientific_metrics_computed') is False and
            set(validation.get('inspections', {})) == {arm+'/'+split for arm in ('T100-G-30M', 'T100-G-300M') for split in ('train', 'val')}, 'complete four-cache validation required')
    require(accounting.get('JobIDRaw') == validation.get('slurm_job_id') and str(accounting.get('JobIDRaw', '')).isdigit() and
            accounting.get('State') == 'COMPLETED' and accounting.get('ExitCode') == '0:0' and accounting.get('AllocCPUS') == 2 and
            type(accounting.get('ElapsedRaw')) is int and 0 < accounting['ElapsedRaw'] <= 1800 and
            type(accounting.get('MaxRSS_bytes')) is int and 0 < accounting['MaxRSS_bytes'] <= 30*2**30,
            'verified successful validation accounting and RSS within30GiB required')
    require(set(gate.get('benchmarks', {})) == set(CELLS) and set(gate.get('benchmark_accounting', {})) == set(CELLS),
            'all three actual pose benchmarks and accounting required')
    runtime_before = current_pose_runtime()
    reports = [validation]
    estimate = 2*accounting['ElapsedRaw']
    for cell in CELLS:
        report = pin(gate['benchmarks'][cell])
        measured = pin(gate['benchmark_accounting'][cell])
        require(measured.get('JobIDRaw') == report.get('slurm_job_id') and str(measured.get('JobIDRaw', '')).isdigit() and
                measured.get('State') == 'COMPLETED' and measured.get('ExitCode') == '0:0' and measured.get('AllocCPUS') == 8 and
                type(measured.get('ElapsedRaw')) is int and 0 < measured['ElapsedRaw'] <= 600 and
                type(measured.get('MaxRSS_bytes')) is int and 0 < measured['MaxRSS_bytes'] <= 30*2**30,
                'successful benchmark accounting with measured RSS within30GiB required')
        arm = 'T100-G-30M' if cell == CELLS[0] else 'T100-G-300M'
        require(report.get('status') == 'train_only_timing' and report.get('cell') == cell and
                set(report.get('inspections', {})) == {arm+'/train'}, 'train-only pose benchmark identity differs')
        require_runtime(report.get('runtime', {}), runtime_before)
        t = report['timings']
        require(t.get('n_features') == (896 if cell == CELLS[1] else 384) and t.get('n_users') == 40 and
                t.get('valid_fit_elements') == 4813620, 'pose benchmark train support/dimension differs')
        S, E = t.get('statistics_seconds'), t.get('three_eigensystems_seconds')
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (S, E)), 'invalid measured pose timings')
        estimate += 11*S + 13*E
        reports.append(report)
    for report in reports:
        require(report.get('protocol') == P.PROTOCOL and report.get('complete') is True and type(report.get('step')) is int and
                report['step'] == 19618 and report.get('recipe') == P.recipe() and report.get('plan_sha256') == a.plan_sha256 and
                report.get('runtime', {}).get('code_sha') == expected_code and report['runtime'].get('sources') == P.source_hashes(), 'pose budget report provenance differs')
        for key, field in (('input_pins', 'caches'), ('reference_pins', 'references'), ('projection_pin', 'projection'), ('training_source_contract', 'training_source_contract')):
            require(report.get(key) == plan.get(field), 'pose budget report inputs differ: '+key)
    needed = math.ceil((1.5*estimate + 120)/60)*60
    require(type(gate.get('derived_seconds')) is int and gate['derived_seconds'] == needed and needed <= 3600, 'measured pose budget exceeds gate or derivation differs')
elif a.stage == 'compare':
    require(plan.get('protocol') == 'scaling-comparison-inputs-19618-20261106-v1' and type(plan.get('checkpoint_step')) is int and plan['checkpoint_step'] == 19618, 'wrong comparison manifest/dose')
    report = pin(plan['pose_report'])
    require(report.get('protocol') == 'scaling-pose-readout-19618-20261106-v1' and report.get('complete') is True and report.get('status') == 'completed' and report.get('step') == 19618, 'completed pose report required')
    runtime_before = current_pose_runtime()
    require_runtime(report.get('runtime', {}), runtime_before)

base = ['--plan', str(a.plan), '--expected-code-sha', expected_code]
commands = []
if a.stage.startswith('pose-'):
    cmd = [sys.executable, 'scripts/scaling_pose_readout19618.py', a.stage[5:], *base, '--out', str(out)]
    if a.stage == 'pose-benchmark': cmd += ['--cell', a.cell]
    commands.append(cmd)
elif a.stage.startswith('probe-'):
    base += ['--expected-plan-sha256', a.plan_sha256, '--cell', a.cell]
    if a.stage == 'probe-benchmark':
        # Exactly one dataset per job; any failure stops without retry.
        out.mkdir(parents=True, exist_ok=False)
        commands = [[sys.executable, 'scripts/scaling_probe19618.py', 'benchmark', *base, '--dataset', ds, '--out', str(out/ds)] for ds in (a.dataset,)]
    elif a.stage == 'probe-fit':
        commands = [[sys.executable, 'scripts/scaling_probe19618.py', 'fit-shard', *base, '--task', a.task,
            '--datasets', *a.datasets, '--folds', *map(str, a.folds), '--resource-plan', str(a.resource_plan),
            '--expected-resource-plan-sha256', a.resource_sha256, '--out', str(out)]]
    else:
        commands = [[sys.executable, 'scripts/scaling_probe19618.py', 'aggregate', *base, '--task', a.task,
            '--shards', *map(str, a.shards), '--out', str(out)]]
else:
    commands = [[sys.executable, 'scripts/scaling_compare19618.py', '--inputs', str(a.plan), '--expected-inputs-sha256', a.plan_sha256,
                 '--expected-code-sha', expected_code, '--out', str(out)]]

bootstrap = "import sys,runpy;from pathlib import Path;sys.path.insert(0,'scripts');import scaling_compare19618;sys.argv=sys.argv[1:];runpy.run_path(str(Path(sys.argv[0]).resolve()),run_name='__main__')"
for command in commands:
    require(states == {path: state(Path(path)) for path in states}, 'pinned operational inputs changed')
    subprocess.run([command[0], '-c', bootstrap, *command[1:]], check=True)
require(states == {path: state(Path(path)) for path in states}, 'pinned operational inputs changed')
if runtime_before is not None:
    require(current_pose_runtime() == runtime_before, 'numeric runtime changed during reader execution')
PY
code_dir_setup
