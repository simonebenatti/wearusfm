import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('p1_recovery_launch', ROOT / 'scripts/scaling_p1_recovery_launch.py')
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


@pytest.fixture
def case(tmp_path):
    work = tmp_path / 'work'
    code = work / 'wearusfm_campaigns/recovery'
    code.mkdir(parents=True)
    output = tmp_path / 'scratch/T100-G-300M/p1-r2'
    manifest = copy.deepcopy(L.EXPECTED)
    manifest['out'] = str(output)
    args = {'code': code, 'manifest': 'recovery.json', 'job': '123', 'work': work,
            'expected_output': output,
            'job_info': 'JobId=123 JobName=' + L.LOGICAL_ID + ' JobState=RUNNING NumNodes=1 NumCPUs=8 '
                        'NumTasks=1 Partition=boost_usr_prod QOS=normal TimeLimit=02:00:00 '
                        'AllocTRES=cpu=8,mem=100G,node=1,gres/gpu=1'}

    def write(value):
        data = (json.dumps(value, sort_keys=True) + '\n').encode()
        (code / 'recovery.json').write_bytes(data)
        args['manifest_sha256'] = hashlib.sha256(data).hexdigest()

    write(manifest)
    return args, manifest, write


def test_exact_allocation_and_fixed_manifest(case):
    args, _, _ = case
    assert L.validate(**args)
    args['job_info'] = args['job_info'].replace('mem=100G', 'mem=102400M')
    assert L.validate(**args)


@pytest.mark.parametrize('old,new', [
    ('JobId=123', 'JobId=124'), (L.LOGICAL_ID, 'another-p1'), ('RUNNING', 'PENDING'),
    ('NumNodes=1', 'NumNodes=2'), ('NumCPUs=8', 'NumCPUs=16'), ('NumTasks=1', 'NumTasks=2'),
    ('boost_usr_prod', 'lrd_all_serial'), ('QOS=normal', 'QOS=boost_qos_dbg'),
    ('02:00:00', '02:30:00'), ('cpu=8,', 'cpu=16,'), ('mem=100G', 'mem=99G'),
    ('node=1,', 'node=2,'), ('gres/gpu=1', 'gres/gpu=2'), ('gres/gpu=1', 'gres/gpu=0'),
])
def test_actual_allocation_or_job_identity_mismatch(case, old, new):
    args, _, _ = case
    args['job_info'] = args['job_info'].replace(old, new)
    with pytest.raises(ValueError):
        L.validate(**args)


@pytest.mark.parametrize('key,value', [
    ('protocol', 'other'), ('arm', 'T100-G-30M'), ('step', 5000), ('step', 19618.0),
    ('task', 'p2'), ('logical_job_id', 'another-p1'), ('output_barrier', 'sync'),
    ('diagnostic_protocol', 'other'), ('failed_attempts', []),
    ('allocation', {'nodes': True, 'gpus': 1, 'cpus': 8, 'memory_gib': 100, 'walltime_seconds': 7200}),
    ('budget', {'p1_cumulative_gpu_hours_cap': 3.10, 'extractions_gpu_hours_cap': 7.30,
                'p1_max_attempts': 3, 'retry_authorized': True}),
])
def test_manifest_does_not_widen_recovery(case, key, value):
    args, manifest, write = case
    manifest[key] = value
    write(manifest)
    with pytest.raises(ValueError):
        L.validate(**args)


def test_manifest_pin_and_out_must_be_exact(case):
    args, manifest, write = case
    args['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='pin'):
        L.validate(**args)
    manifest['out'] = str(Path(manifest['out']).parent / 'p1-r3')
    write(manifest)
    with pytest.raises(ValueError, match='out'):
        L.validate(**args)


def test_existing_and_symlink_output_are_refused(case):
    args, _, _ = case
    out = args['expected_output']
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match='new canonical'):
        L.validate(**args)
    out.rmdir()
    out.symlink_to(out.parent / 'does-not-exist')
    with pytest.raises(ValueError, match='new canonical'):
        L.validate(**args)


def test_code_and_manifest_paths_do_not_escape(case):
    args, _, _ = case
    absolute = args['code'] / 'recovery.json'
    args['manifest'] = str(absolute)
    with pytest.raises(ValueError, match='relative contained'):
        L.validate(**args)
    args['manifest'] = '../recovery/recovery.json'
    with pytest.raises(ValueError, match='relative contained'):
        L.validate(**args)
    external = args['work'] / 'external.json'
    external.write_bytes(absolute.read_bytes())
    (args['code'] / 'escape.json').symlink_to(external)
    args['manifest'] = 'escape.json'
    with pytest.raises(ValueError, match='contained'):
        L.validate(**args)


@pytest.mark.parametrize('count', [0, 2, 4])
def test_visible_cuda_device_count_is_exact(count):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: count))
    with pytest.raises(ValueError, match='one visible CUDA'):
        L.validate_cuda(torch)


def test_one_visible_cuda_device():
    L.validate_cuda(types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1)))


def test_launcher_syntax_resources_and_separate_process():
    path = ROOT / 'scripts/slurm/scaling_p1_recovery.sbatch'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for directive in ('--qos=normal', '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8',
                      '--mem=100G', '--time=02:00:00', '-J ' + L.LOGICAL_ID):
        assert '#SBATCH ' + directive in text
    assert '"$#" -ne 2' in text
    assert text.index('source "$WORK/venv_wearusfm/bin/activate"') < text.index('python3 scripts/scaling_p1_recovery_launch.py')
    assert text.index('python3 scripts/scaling_p1_recovery_launch.py') < text.index('exec python3 scripts/scaling_p1_recovery.py')
    assert 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1' in text
    assert '--expected-code-sha "$EXPECTED_SHA"' in text
