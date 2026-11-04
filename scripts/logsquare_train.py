"""Execute one reviewed T100-LQ phase through the unchanged production trainer.

Fresh seed-zero initialization, then own resumable 5000 -> 19618 continuation.
A review-only plan cannot execute: the release must pin the R1 reading, Claude
review, measured R2 budget and the exact controller/scientific source files.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

import torch

REPO = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


L = module('logsquare_train_preflight', 'scripts/logsquare_preflight.py')
T = module('logsquare_train_reference', 'scripts/tokenizer_train.py')
CLI = module('logsquare_train_cli', 'scripts/train_jepa.py')
P, R, require = L.P, L.R, L.require
PROTOCOL = 'tokenizer-logsquare-training-20261104-v1'
ARM = 'T100-LQ'
CONTROLLER_FILES = ('scripts/logsquare_train.py', 'scripts/slurm/logsquare_train.sbatch')


def read_pinned_json(pin):
    identity = P.pinned_file(pin)
    path = Path(identity['path'])
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data), identity


def validate_release(plan):
    """A persisted, pinned review decision; operational consent is already given."""
    require(plan.get('status') == 'APPROVED_FOR_TRAINING' and isinstance(plan.get('training_release'), dict),
            'training review/release pending; no launch')
    release, release_identity = read_pinned_json(plan['training_release'])
    require(release.get('training_approved') is True and release.get('epsilon') == 0.001,
            'training or fixed epsilon not approved')
    for key in ('preflight_report', 'r1_analysis', 'resource_proposal'):
        require(release.get(key + '_sha256') == plan[key]['sha256'], 'release pin mismatch: ' + key)
    require(set(release.get('controller_sha256', {})) == set(CONTROLLER_FILES), 'controller review incomplete')
    identities = {'training_release': release_identity}
    for relative, digest in release['controller_sha256'].items():
        identities[relative] = P.pinned_file({'path': relative, 'sha256': digest})
    review = P.pinned_file(release['claude_review'])
    require(review['bytes'] > 0, 'empty Claude review')
    identities['claude_review'] = review
    return identities


def validate_preflight(report, plan):
    require(report.get('protocol') == L.PROTOCOL and report.get('complete') is True, 'incomplete preflight')
    require(report.get('code_sha') == plan['preflight_code_sha'], 'wrong preflight code')
    require(report.get('epsilon') == .001 and report.get('batches') == 64 and
            report.get('windows_each_arm') == 2048 and report.get('runtime', {}).get('world_size') == 4,
            'preflight epsilon/dose/world mismatch')
    require(report.get('paired_support') is True and
            report.get('initial_shared_weights_rng_and_function_exact') is True, 'preflight pairing/initialization failed')
    require(set(report.get('initial_function_per_rank', {})) == {'0', '1', '2', '3'} and
            all(v.get('status') == 'passed' for v in report['initial_function_per_rank'].values()),
            'initial-function rank checks incomplete')
    require(report['inputs']['plan']['sha256'] == plan['preflight_plan']['sha256'], 'preflight plan mismatch')
    for name in L.ARMS:
        initial = report['arms'][name]['initial']
        require(initial.get('new_weight_zero') is True and initial['shared_sha256'] == plan['shared_initial_sha256'],
                'preflight shared initialization mismatch')
    return report['update_mean_after8_ratio_LQ_over_L']


def validate_allocations(plan, rho):
    require(type(rho) in (float, int) and math.isfinite(rho) and rho > 0, 'invalid rho_update')
    for step, historical_cap, extra in ((5000, 5400, 5000), (19618, 14400, 14618)):
        cap = math.ceil(historical_cap * max(1., rho) / 900) * 900
        allocation = plan['allocations'][str(step)]
        require(cap <= 86400, 'allocation exceeds registered partition limit; review required')
        expected = {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': '480G', 'partition': 'boost_usr_prod',
                    'walltime_seconds': cap, 'additional_steps': extra,
                    'gpu_hours_cap': 4 * cap / 3600, 'allocated_core_hours_cap': 32 * cap / 3600}
        require(all(allocation.get(k) == v for k, v in expected.items()), 'allocation does not follow R2')


def command_for(comparison, cfg, run_dir, step, seconds):
    require(step in (5000, 19618) and seconds > 300, 'unregistered phase or insufficient walltime')
    argv = list(comparison['arm_configurations']['T100-L']['validated_trainer_argv'])
    require(argv[0] == 'scripts/train_jepa.py' and argv.count('--max-steps') == 1 and
            argv.count('--out-dir') == 1 and not any(x in argv for x in
            ('--time-limit-s', '--init-from', '--branch-from', '--branch-state-from', '--tokenizer-logsquare',
             '--tokenizer-logsquare-eps')), 'unregistered trainer arguments')
    argv[argv.index('--max-steps') + 1] = str(step)
    argv[argv.index('--out-dir') + 1] = str(run_dir)
    argv += ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '0.001', '--time-limit-s', str(seconds - 300)]
    args = CLI.build_parser().parse_args(argv[1:])
    effective = CLI.build_config(args)
    require(R.config_to_dict(effective) == R.config_to_dict(replace(cfg, max_steps=step)), 'candidate CLI/config mismatch')
    require(args.num_workers == 6 and args.distributed and args.init_from is None and args.branch_from is None,
            'wrong stream or warm start')
    return ['torchrun', '--standalone', '--nproc_per_node=4', *argv]


def verify_initialization(cfg, expected):
    """Audit construction on CPU; trainer still constructs its own fresh model."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        student = P.WearUsFM(cfg.model)
        values = student.state_dict()
        new = values[L.NEW_WEIGHT]
        require(new.shape == (cfg.model.dim, 64) and not bool(new.count_nonzero()), 'new branch not zero initialized')
        shared = P.state_sha256({k: v for k, v in values.items() if k != L.NEW_WEIGHT})
        require(shared == expected, 'fresh shared initialization differs from historical T100-L')
    return {'shared_sha256': shared, 'logsquare_weight_zero': True, 'new_parameters': new.numel()}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan.get('protocol') == PROTOCOL and plan.get('arm') == ARM and plan.get('endpoints') == [5000, 19618],
            'wrong training protocol/arm/dose')
    identities = validate_release(plan)
    reference = P.pinned_file(plan['preflight_plan'])
    _, configs, inputs = L.load_plan(reference['path'])
    identities.update(inputs)
    for key in ('preflight_report', 'r1_analysis', 'resource_proposal'):
        value, identity = read_pinned_json(plan[key])
        identities[key] = identity
        if key == 'preflight_report':
            report = value
    rho = validate_preflight(report, plan)
    validate_allocations(plan, rho)
    for relative, digest in plan['scientific_source_sha256'].items():
        identities[relative] = P.pinned_file({'path': relative, 'sha256': digest})
    require(set(plan['scientific_source_sha256']) == set(plan['required_scientific_sources']), 'incomplete source inventory')
    identities['training_plan'] = P.file_identity(path)
    return plan, configs[ARM], identities


def checkpoint_identity(path, cfg, step, signature):
    # Reuse the tested four-rank RNG/AdamW/data-signature completeness checks.
    identity = T.checkpoint_identity(path, cfg, step, signature)
    state = torch.load(path, map_location='cpu', weights_only=False)
    with torch.random.fork_rng(devices=[]):
        expected_model = P.WearUsFM(cfg.model)
    expected_state = expected_model.state_dict()
    for kind in ('student', 'teacher'):
        actual = state[kind]
        require(actual.keys() == expected_state.keys(), 'incomplete or extra model state: ' + kind)
        require(all(isinstance(actual[k], torch.Tensor) and actual[k].shape == v.shape and
                    actual[k].dtype == v.dtype for k, v in expected_state.items()),
                'model state shape/dtype mismatch: ' + kind)
    parameters = [(name, p) for name, p in expected_model.named_parameters() if p.requires_grad]
    optimizer = state['optimizer']
    # AdamW state_dict assigns integer IDs in group parameter order. The trainer
    # uses exactly this ordered list; names/shapes prevent foreign-sized moments.
    ids = optimizer['param_groups'][0]['params']
    require(ids == list(range(len(parameters))), 'optimizer parameter inventory/order mismatch')
    for param_id, moments in optimizer['state'].items():
        name, parameter = parameters[param_id]
        require(all(moments[key].shape == parameter.shape and moments[key].dtype == parameter.dtype
                    for key in ('exp_avg', 'exp_avg_sq')), 'optimizer moment shape/dtype mismatch: ' + name)
    log_id = next(i for i, (name, _) in enumerate(parameters) if name == L.NEW_WEIGHT)
    require(log_id in optimizer['state'], 'missing log-square optimizer moments')
    # Other parameters may legitimately have no moments when no gradient ever
    # reached them; do not invent zero moments or require inactive objectives.
    require(P.file_identity(path)['sha256'] == identity['sha256'], 'checkpoint changed during branch check')
    return identity


def execute(plan_path, step, expected_code, out, allocated_seconds):
    code = P.code_identity(expected_code)
    plan, base_cfg, inputs = load_plan(plan_path)
    require(step in (5000, 19618), 'unregistered endpoint')
    cfg = replace(base_cfg, max_steps=step)
    require(allocated_seconds == plan['allocations'][str(step)]['walltime_seconds'], 'allocation differs from plan')
    comparison = json.loads(Path(inputs['comparison']['path']).read_text())
    run_dir = Path(plan['run_dir'])
    command = command_for(comparison, base_cfg, run_dir, step, allocated_seconds)
    roots = (REPO, Path('/leonardo_work/IscrB_WearUsFM/wearusfm'), Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns'))
    require(run_dir.is_absolute() and out.is_absolute(), 'absolute output paths required')
    require(not any(run_dir.resolve().is_relative_to(r) or out.resolve().is_relative_to(r) for r in roots), 'output inside code')
    require(not out.exists() and not out.is_symlink(), 'receipt directory already exists')
    require(str((out / 'receipt.json').resolve()) == plan['phase_receipts'][str(step)], 'receipt differs from plan')
    scales = json.loads(Path(inputs['scales']['path']).read_text())['scales']
    windows = json.loads(Path(inputs['windows']['path']).read_text())['window_s']
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs['manifest']['path'], scales, windows)))
    plan_hash = inputs['training_plan']['sha256']
    old_checkpoint = initial = None
    if step == 5000:
        require(not run_dir.exists() and not run_dir.is_symlink(), 'fresh training requires new run directory')
        initial = verify_initialization(base_cfg, plan['shared_initial_sha256'])
    else:
        rec = json.loads(Path(plan['phase_receipts']['5000']).read_text())
        named = run_dir / 'checkpoint_step0005000.pt'
        require(rec.get('protocol') == PROTOCOL and rec.get('complete') is True and rec.get('arm') == ARM and
                rec.get('checkpoint_step') == 5000 and rec.get('code_sha') == code and rec.get('plan_sha256') == plan_hash,
                'missing completed same-arm/code/plan screening receipt')
        require(not (run_dir / 'STOP').exists(), 'STOP present')
        old_checkpoint = checkpoint_identity(named, base_cfg, 5000, signature)
        resume = checkpoint_identity(run_dir / 'checkpoint.pt', base_cfg, 5000, signature)
        require(rec.get('checkpoint') == str(named) and old_checkpoint['sha256'] == rec.get('checkpoint_sha256') and
                rec.get('resume_checkpoint') == str(run_dir / 'checkpoint.pt') and resume['sha256'] == rec.get('resume_checkpoint_sha256') and
                old_checkpoint['trained_state_sha256'] == resume['trained_state_sha256'], 'not own verified 5000 state')
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / 'launch.json', {'arm': ARM, 'step': step, 'plan_sha256': plan_hash, 'code_sha': code,
                                     'command': command, 'previous_checkpoint': old_checkpoint, 'initialization': initial})
    subprocess.run(command, cwd=REPO, check=True)
    summary = json.loads((run_dir / 'summary.json').read_text())
    require(summary.get('stopped') == 'max_steps' and summary.get('steps') == step, 'training phase incomplete')
    require(not (run_dir / 'STOP').exists(), 'STOP present after phase')
    require(json.loads((run_dir / 'config.json').read_text()) == R.config_to_dict(cfg), 'final config mismatch')
    named = run_dir / f'checkpoint_step{step:07d}.pt'
    identity = checkpoint_identity(named, cfg, step, signature)
    last = checkpoint_identity(run_dir / 'checkpoint.pt', cfg, step, signature)
    require(identity['trained_state_sha256'] == last['trained_state_sha256'], 'named/final trained states differ')
    if old_checkpoint is not None:
        old_now = P.file_identity(old_checkpoint['path'])
        require(all(old_now[k] == v for k, v in old_checkpoint.items() if k != 'trained_state_sha256'), '5000 checkpoint was modified')
    P.check_unchanged(inputs)
    require(P.code_identity(code) == code, 'code changed')
    config_file = out / 'config.json'
    P.write_json(config_file, R.config_to_dict(cfg))
    P.write_json(out / 'summary.json', summary)
    receipt = T.receipt_for(ARM, step, named, run_dir / 'checkpoint.pt', config_file, code, plan_hash, command)
    receipt.update(protocol=PROTOCOL, preflight_report_sha256=plan['preflight_report']['sha256'],
                   release_sha256=inputs['training_release']['sha256'])
    P.write_json(out / 'receipt.json', receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=(5000, 19618), required=True)
    parser.add_argument('--expected-code-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allocated-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    execute(args.plan, args.step, args.expected_code_sha, args.out, args.allocated_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
