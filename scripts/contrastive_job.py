"""Execute one reviewed preflight cell; fixed commands, no submission or retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch  # trainer import order
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrastive_prepare import ROOT, _read, source_digest, validate_input_bindings

GPU_MODES = {'cuda', 'calibration', 'full', 'split'}
MODES = GPU_MODES | {'audit', 'readout'}
# The 100/125 ms recipe needs a two-hour audit; the readout keeps its one-hour allocation.
ALLOCATED_SECONDS = {**{mode: 1800 for mode in GPU_MODES}, 'audit': 7200, 'readout': 3600}
DATA_ROOTS = {Path('/leonardo_work/IscrB_WearUsFM/data/processed'),
              Path('/leonardo_scratch/large/userexternal/sbenatti/data/processed')}


def checked_file(spec):
    if (not isinstance(spec, dict) or not {'path', 'sha256'} <= set(spec) or set(spec) - {'path', 'sha256', 'bytes'} or
            not isinstance(spec['path'], str) or not spec['path'] or not isinstance(spec['sha256'], str) or
            len(spec['sha256']) != 64 or any(c not in '0123456789abcdef' for c in spec['sha256'])):
        raise ValueError('complete path/SHA256 pin required')
    path = Path(spec['path'])
    if not path.is_absolute(): path = ROOT / path
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec['sha256'] or ('bytes' in spec and (type(spec['bytes']) is not int or spec['bytes'] != len(content))):
        raise ValueError('input pin mismatch: '+str(path))
    return path.resolve()


def script(name):
    return str(ROOT / 'scripts' / name)


def calibration_directory(spec):
    path = checked_file(spec)
    if path.name != 'summary.json':
        raise ValueError('calibration summary.json input required')
    return path.parent


def rank_exit_report(out):
    out = Path(out).resolve()
    return out.parent / (out.name + '.rank_exit_probe') / 'report.json'


def prepare_cuda_probe(job):
    """Run the intentional rank failure first; a failed probe prevents the harness."""
    from contrastive_split_benchmark import checked_command
    from contrastive_rank_exit import validate_receipt
    report = rank_exit_report(job['out'])
    command = [sys.executable, script('contrastive_rank_exit.py'), '--out-dir', str(report.parent), '--timeout-s', '60']
    checked_command(command, time.monotonic() + 90)
    validate_receipt(report, expected_source_sha256=source_digest())
    return report


def build_command(job, kind, allocated_seconds):
    mode = job.get('mode')
    if (job.get('status') != 'APPROVED_FOR_PREFLIGHT' or mode not in MODES or
            job.get('source_sha256') != source_digest() or
            kind != ('gpu' if mode in GPU_MODES else 'cpu') or
            allocated_seconds != ALLOCATED_SECONDS.get(mode)):
        raise ValueError('wrong mode, release, source or allocation')
    checked_file(job['review'])
    out = Path(job['out']).resolve()
    results = Path(os.environ['WORK']).resolve() / 'wearusfm_runs'
    if not out.is_relative_to(results) or out.exists():
        raise ValueError('fresh output below WORK/wearusfm_runs required')
    python = [sys.executable]
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    if mode == 'readout':
        command = python + [script('contrastive_benchmark_readout.py'), '--cohort', str(checked_file(job['cohort'])),
                            '--cuda-report', str(checked_file(job['cuda_report'])), '--out', str(out),
                            '--cpu-audit', str(checked_file(job['audit'])),
                            '--calibration-report', str(checked_file(job['calibration_report'])),
                            '--calibration-run-dir', str(calibration_directory(job['calibration_summary']))]
        for key, arms in (('full', ('C0','CN','CR')), ('resumed', ('CN','CR'))):
            if set(job[key]) != set(arms): raise ValueError('readout arm set differs')
            for arm in arms:
                summary = checked_file(job[key][arm])
                if summary.name != 'summary.json': raise ValueError('summary input required')
                command += ['--'+key, arm+'='+str(summary.parent)]
        return command
    config = checked_file(job['config'])
    if mode in {'audit', 'cuda'}:
        from contrastive_run import validate_recipe
        validate_recipe(_read(config), 'calibration')
    if set(job.get('inputs', {})) != {'manifest','scales','window_seconds','pose_stats'}:
        raise ValueError('complete exact comparison inputs required')
    inputs = {k: checked_file(job['inputs'][k]) for k in ('manifest','scales','window_seconds','pose_stats')}
    validate_input_bindings({k: job['inputs'][k]['sha256'] for k in inputs})
    roots = [Path(p).resolve(strict=True) for p in job['roots']]
    if len(roots) != len(DATA_ROOTS) or set(roots) != DATA_ROOTS or any(not p.is_dir() for p in roots):
        raise ValueError('data roots differ')
    common = ['--manifest', str(inputs['manifest']), '--scales', str(inputs['scales']),
              '--window-seconds', str(inputs['window_seconds'])]
    for root in roots: common += ['--root', str(root)]
    if mode == 'audit':
        return python + [script('contrastive_audit.py'), '--config', str(config), '--steps', '1024', '--out', str(out)] + common
    if mode == 'cuda':
        command = torchrun + [script('contrastive_cuda_check.py'), '--config', str(config),
                             '--config-sha256', job['config']['sha256'], '--expected-source-sha256', source_digest(),
                             '--calibration-checkpoint', str(checked_file(job['calibration_checkpoint'])),
                             '--calibration-checkpoint-sha256', job['calibration_checkpoint']['sha256'],
                             '--rank-exit-report', str(rank_exit_report(out)),
                             '--pose-stats-sha256', job['inputs']['pose_stats']['sha256'], '--out', str(out)] + common
        for key in ('manifest','scales','window_seconds'):
            command += ['--'+key.replace('_','-')+'-sha256', job['inputs'][key]['sha256']]
        return command
    release = checked_file(job['release'])
    stage = 'calibration' if mode == 'calibration' else 'benchmark'
    payload = _read(release)
    expected_time = 1800 if mode == 'split' else 1770
    if (payload.get('status') != 'APPROVED_FOR_PREPARATION' or payload.get('stage') != stage or
            payload.get('config_sha256') != job['config']['sha256'] or
            payload.get('source_sha256') != job['source_sha256'] or payload.get('time_limit_s') != expected_time or
            payload.get('inputs') != {k: job['inputs'][k]['sha256'] for k in inputs} or
            payload.get('audit_sha256') != job['audit']['sha256'] or
            (mode != 'calibration' and payload.get('phase') != mode)):
        raise ValueError('job stage/config/time/input pins differ from the execution release')
    common += ['--audit-report', str(checked_file(job['audit']))]
    if mode != 'calibration':
        common += ['--cohort', str(checked_file(job['cohort'])),
                   '--calibration-report', str(checked_file(job['calibration_report'])),
                   '--calibration-run-dir', str(calibration_directory(job['calibration_summary']))]
        if payload.get('cohort_sha256') != job['cohort']['sha256'] or payload.get('calibration_sha256') != job['calibration_report']['sha256']:
            raise ValueError('job cohort/calibration pins differ from the execution release')
    if mode == 'split':
        return python + [script('contrastive_split_benchmark.py'), '--split-config', str(config),
                         '--resume-config', str(checked_file(job['resume_config'])), '--release', str(release),
                         '--out-dir', str(out), '--total-time-limit-s', '1800', '--'] + common
    return torchrun + [script('contrastive_run.py'), '--config', str(config), '--release', str(release),
                       '--stage', stage, '--out-dir', str(out), '--time-limit-s', '1770'] + common


def validate_allocation(kind):
    expected_cpu = 32 if kind == 'gpu' else 8
    expected_memory = 240 * 1024 if kind == 'gpu' else 30 * 1024
    if (int(os.environ.get('SLURM_JOB_NUM_NODES','0')) != 1 or int(os.environ.get('SLURM_NTASKS','0')) != 1 or
            int(os.environ.get('SLURM_CPUS_PER_TASK','0')) != expected_cpu or
            int(os.environ.get('SLURM_MEM_PER_NODE','0')) != expected_memory):
        raise ValueError('unexpected node/task/CPU/memory allocation')
    if kind == 'gpu' and torch.cuda.device_count() != 4:
        raise ValueError('exactly four GPUs required')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--kind', choices=('gpu','cpu'), required=True)
    ap.add_argument('--allocated-seconds', type=int, required=True)
    a = ap.parse_args()
    jobpath = a.job.resolve(strict=True)
    if not jobpath.is_relative_to(ROOT): raise ValueError('job specification must be in frozen checkout')
    job = _read(jobpath)
    command = build_command(job, a.kind, a.allocated_seconds)
    validate_allocation(a.kind)
    if job['mode'] == 'cuda':
        prepare_cuda_probe(job)
    print(json.dumps({'command': command, 'job_sha256': hashlib.sha256(jobpath.read_bytes()).hexdigest(),
                      'source_sha256': source_digest()}, allow_nan=False), flush=True)
    os.execv(command[0], command)


if __name__ == '__main__': main()
