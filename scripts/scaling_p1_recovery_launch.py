"""Allocation and manifest gate for the single conditional P1-r2 extraction.

This process ends before the numerical wrapper starts. Its CUDA visibility check
therefore cannot change the wrapper's frozen numpy-before-torch import order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

WORK = Path('/leonardo_work/IscrB_WearUsFM')
PROTOCOL = 'scaling-p1-recovery-20261116-v1'
LOGICAL_ID = 'SCALING-300M-eval19618-p1-r2-20261116'
OUTPUT = Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M/p1-r2')
EXPECTED = {
    'protocol': PROTOCOL, 'arm': 'T100-G-300M', 'step': 19618, 'task': 'p1',
    'logical_job_id': LOGICAL_ID, 'out': str(OUTPUT), 'output_barrier': 'none',
    'diagnostic_protocol': 'SCALING-P1-IDENTITY-DIAG-v1',
    'allocation': {'nodes': 1, 'gpus': 1, 'cpus': 8, 'memory_gib': 100, 'walltime_seconds': 7200},
    'budget': {'p1_cumulative_gpu_hours_cap': 3.10, 'extractions_gpu_hours_cap': 7.30,
               'p1_max_attempts': 3, 'retry_authorized': False},
    'failed_attempts': [{'job_id': '64572000', 'state': 'FAILED', 'exit_code': '1:0'},
                        {'job_id': '64596915', 'state': 'FAILED', 'exit_code': '1:0'}],
}


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_exact(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate(code, manifest, manifest_sha256, job, job_info, *, work=WORK, expected_output=OUTPUT):
    code = Path(code)
    if (not code.is_absolute() or code.resolve(strict=True) != code or
            not code.is_relative_to(work / 'wearusfm_campaigns') or code == work / 'wearusfm_campaigns'):
        raise ValueError('canonical frozen campaign path required')
    relative = Path(manifest)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative contained recovery manifest required')
    path = (code / relative).resolve(strict=True)
    if not path.is_relative_to(code) or not path.is_file():
        raise ValueError('recovery manifest must be contained in frozen code')
    data = path.read_bytes()
    if (not isinstance(manifest_sha256, str) or re.fullmatch('[0-9a-f]{64}', manifest_sha256) is None or
            hashlib.sha256(data).hexdigest() != manifest_sha256):
        raise ValueError('recovery manifest pin mismatch')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('recovery manifest must be an object')
    expected = {**EXPECTED, 'out': str(expected_output)}
    for key, wanted in expected.items():
        if not _exact(value.get(key), wanted):
            raise ValueError('fixed P1 recovery manifest differs: ' + key)
    out = Path(value['out'])
    if not out.is_absolute() or out.resolve() != out or out.exists() or out.is_symlink():
        raise ValueError('new canonical P1-r2 output required')
    if any(out.is_relative_to(parent) for parent in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
        raise ValueError('recovery output inside code')
    fields = dict(part.split('=', 1) for part in job_info.split() if '=' in part)
    required = {'JobId': str(job), 'JobName': LOGICAL_ID, 'JobState': 'RUNNING',
                'NumNodes': '1', 'NumCPUs': '8', 'NumTasks': '1',
                'Partition': 'boost_usr_prod', 'QOS': 'normal', 'TimeLimit': '02:00:00'}
    if not str(job).isdigit() or int(job) <= 0 or any(fields.get(k) != v for k, v in required.items()):
        raise ValueError('actual P1-r2 job identity/allocation mismatch')
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    if (any(tres.get(k) != v for k, v in {'cpu': '8', 'node': '1', 'gres/gpu': '1'}.items()) or
            tres.get('mem') not in ('100G', '102400M')):
        raise ValueError('actual P1-r2 resources differ from 1GPU/8CPU/100GiB')
    return True


def validate_cuda(torch):
    if torch.cuda.device_count() != 1:
        raise ValueError('exactly one visible CUDA device required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'manifest', 'manifest-sha256', 'job', 'job-info'):
        parser.add_argument('--' + name, required=True)
    validate(**vars(parser.parse_args(argv)))
    import torch
    validate_cuda(torch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
