"""Execute one reviewed T100-LQ phase through the unchanged production trainer.

Fresh seed-zero initialization, then own resumable 5000 -> 19618 continuation.
A review-only plan cannot execute: the release must pin the R1 reading, Claude
review, measured R2 budget and the exact controller/scientific source files.

Initial state (05/11/2026, after job 63532335): the reviewed guard is verified
on the model every training rank actually builds (scripts/logsquare_train_rank.py),
not only on the temporary model of the pre-check here. See
scripts/logsquare_initial_state.py for the rule.
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
G = module('logsquare_train_initial_state', 'scripts/logsquare_initial_state.py')
P, R, require = L.P, L.R, L.require
PROTOCOL = 'tokenizer-logsquare-training-20261104-v1'
ARM = 'T100-LQ'
RANK_ENTRY = 'scripts/logsquare_train_rank.py'
CONTROLLER_FILES = ('scripts/logsquare_train.py', 'scripts/slurm/logsquare_train.sbatch', RANK_ENTRY,
                    'scripts/logsquare_initial_state.py')


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


def command_for(comparison, cfg, run_dir, step, seconds, guard=None, records=None):
    """The historical T100-L trainer arguments plus the branch; the fresh phase starts every rank behind the guard."""
    require(step in (5000, 19618) and seconds > 300, 'unregistered phase or insufficient walltime')
    require((guard is not None and records is not None) if step == 5000 else (guard is None and records is None),
            'the guard belongs to the fresh phase, and only to it')
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
    entry = [] if step != 5000 else [RANK_ENTRY, '--initial-state-guard', guard['path'],
                                     '--initial-state-guard-sha256', guard['sha256'],
                                     '--initial-state-records', str(records)]
    return ['torchrun', '--standalone', '--nproc_per_node=4', *entry, *argv]


def verify_initialization(cfg, guard):
    """Pre-check on a temporary CPU model, before any output exists.

    It fails early and says why. It is not the guarantee: the trainer builds its
    own models, and every training rank verifies those (RANK_ENTRY).
    """
    require(G.NEW_WEIGHT == L.NEW_WEIGHT, 'guard written for another branch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        student = P.WearUsFM(cfg.model)
        record = G.verify(student.state_dict(), guard, 'controller pre-check, temporary CPU model')
    return {**record, 'scope': 'pre-check on a temporary model; the trained model is verified in every training rank'}


def rank_attestation(directory, guard_sha256, verify_only=False):
    """After the fresh phase: all four ranks verified every model they built, and it is one state.

    `verify_only` reads the records of the diagnostic dry run (the entry's --verify-only): the same
    requirements, on ranks that stopped before the trainer received the student. Such records are
    never a training attestation, and training records are never a dry run.
    """
    rows = []
    for rank in range(4):
        path = Path(directory) / f'rank{rank}.json'
        require(path.is_file(), f'missing initial-state record of rank {rank}')
        row = json.loads(path.read_text())
        require(row.get('rank') == rank and row.get('world_size') == 4 and row.get('guard_sha256') == guard_sha256 and
                row.get('passed') is True and row.get('verify_only') is verify_only and
                row.get('trainer_returned') is (not verify_only), f'unverified initial state: rank {rank}')
        # Rank zero first counts parameters on an unseeded model that is never trained (recorded, not judged);
        # then every rank builds exactly one student, the model it trains.
        roles = [c.get('role') for c in row.get('constructions', ())]
        require(roles == (['parameter_count', 'student'] if rank == 0 else ['student']) and
                row['constructions'][-1].get('passed') is True and
                row['constructions'][-1].get('classification') == G.PASSED, f'unexpected model constructions: rank {rank}')
        rows.append(row)
    students = [row['constructions'][-1] for row in rows]
    require(len({(c['shared_sha256'], c['fe_gain_hex']) for c in students}) == 1, 'ranks did not start from one state')
    first = students[0]
    return {'verified_on': ('the student each of the four ranks built in verification-only mode; nothing was trained' if verify_only
                            else 'the student each of the four training ranks built and trained'),
            'classification': first['classification'], 'shared_sha256': first['shared_sha256'],
            'reference_gain_shared_sha256': first['reference_gain_shared_sha256'],
            'fe_gain': first['fe_gain'], 'fe_gain_hex': first['fe_gain_hex'],
            'models_built_per_rank': [len(row['constructions']) for row in rows],
            'unjudged_parameter_count_gain_hex': rows[0]['constructions'][0]['fe_gain_hex'],
            'records': {f'rank{rank}.json': P.file_identity(Path(directory) / f'rank{rank}.json')['sha256']
                        for rank in range(4)}}


def plan_guard(plan):
    """The pinned guard, bound to this plan's reference and re-read from its historical records."""
    require(isinstance(plan.get('initial_state_guard'), dict),
            'plan without the reviewed initial-state guard: new release required')
    identity = P.pinned_file(plan['initial_state_guard'])
    guard = G.load_guard(identity['path'], identity['sha256'])
    require(guard['reference']['shared_sha256'] == plan['shared_initial_sha256'] and
            G.same_pin(guard['reference']['source'], plan['preflight_report']), 'guard reference is not the plan reference')
    G.validate_provenance(guard)
    return identity


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan.get('protocol') == PROTOCOL and plan.get('arm') == ARM and plan.get('endpoints') == [5000, 19618],
            'wrong training protocol/arm/dose')
    identities = validate_release(plan)
    identities['initial_state_guard'] = plan_guard(plan)
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


def checkpoint_identity(path, cfg, step, signature, guard=None):
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
        # The gain is a buffer no update touches: the saved one is the one training started with.
        require(guard is None or G.bits(actual[G.GAIN].item()) == G.bits(guard['training_gain']['fe_gain']),
                'checkpoint gain is not the registered historical T100-L training gain: ' + kind)
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
    guard_file = inputs['initial_state_guard']
    guard = G.load_guard(guard_file['path'], guard_file['sha256'])
    records = out / 'initial_state'
    command = (command_for(comparison, base_cfg, run_dir, step, allocated_seconds, guard_file, records) if step == 5000
               else command_for(comparison, base_cfg, run_dir, step, allocated_seconds))
    roots = (REPO, Path('/leonardo_work/IscrB_WearUsFM/wearusfm'), Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns'))
    require(run_dir.is_absolute() and out.is_absolute(), 'absolute output paths required')
    require(not any(run_dir.resolve().is_relative_to(r) or out.resolve().is_relative_to(r) for r in roots), 'output inside code')
    require(not out.exists() and not out.is_symlink(), 'receipt directory already exists')
    require(str((out / 'receipt.json').resolve()) == plan['phase_receipts'][str(step)], 'receipt differs from plan')
    scales = json.loads(Path(inputs['scales']['path']).read_text())['scales']
    windows = json.loads(Path(inputs['windows']['path']).read_text())['window_s']
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs['manifest']['path'], scales, windows)))
    plan_hash = inputs['training_plan']['sha256']
    old_checkpoint = precheck = initial = None
    if step == 5000:
        require(not run_dir.exists() and not run_dir.is_symlink(), 'fresh training requires new run directory')
        require(guard['new_parameter']['shape'] == [base_cfg.model.dim, 64], 'guard written for another model width')
        precheck = verify_initialization(base_cfg, guard)
    else:
        rec = json.loads(Path(plan['phase_receipts']['5000']).read_text())
        named = run_dir / 'checkpoint_step0005000.pt'
        require(rec.get('protocol') == PROTOCOL and rec.get('complete') is True and rec.get('arm') == ARM and
                rec.get('checkpoint_step') == 5000 and rec.get('code_sha') == code and rec.get('plan_sha256') == plan_hash,
                'missing completed same-arm/code/plan screening receipt')
        require(not (run_dir / 'STOP').exists(), 'STOP present')
        initial = rec.get('trained_initial_state')
        require(isinstance(initial, dict) and initial.get('classification') == G.PASSED and
                rec.get('initial_state_guard_sha256') == guard_file['sha256'],
                'screening phase without the verified initial state of this guard')
        old_checkpoint = checkpoint_identity(named, base_cfg, 5000, signature, guard)
        resume = checkpoint_identity(run_dir / 'checkpoint.pt', base_cfg, 5000, signature, guard)
        require(rec.get('checkpoint') == str(named) and old_checkpoint['sha256'] == rec.get('checkpoint_sha256') and
                rec.get('resume_checkpoint') == str(run_dir / 'checkpoint.pt') and resume['sha256'] == rec.get('resume_checkpoint_sha256') and
                old_checkpoint['trained_state_sha256'] == resume['trained_state_sha256'], 'not own verified 5000 state')
    out.mkdir(parents=True, exist_ok=False)
    if step == 5000:
        records.mkdir()
    P.write_json(out / 'launch.json', {'arm': ARM, 'step': step, 'plan_sha256': plan_hash, 'code_sha': code,
                                     'command': command, 'previous_checkpoint': old_checkpoint, 'initialization': precheck,
                                     'initial_state_guard_sha256': guard_file['sha256']})
    subprocess.run(command, cwd=REPO, check=True)
    if step == 5000:
        initial = rank_attestation(records, guard_file['sha256'])
    summary = json.loads((run_dir / 'summary.json').read_text())
    require(summary.get('stopped') == 'max_steps' and summary.get('steps') == step, 'training phase incomplete')
    require(not (run_dir / 'STOP').exists(), 'STOP present after phase')
    require(json.loads((run_dir / 'config.json').read_text()) == R.config_to_dict(cfg), 'final config mismatch')
    named = run_dir / f'checkpoint_step{step:07d}.pt'
    identity = checkpoint_identity(named, cfg, step, signature, guard)
    last = checkpoint_identity(run_dir / 'checkpoint.pt', cfg, step, signature, guard)
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
                   release_sha256=inputs['training_release']['sha256'],
                   initial_state_guard_sha256=guard_file['sha256'], trained_initial_state=initial,
                   checkpoint_gain_is_historical_training_gain=True)
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
