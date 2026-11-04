"""Pinned T100-LQ teacher extraction, with unchanged production numerical paths.

Missing checkpoint pins, review state or measured inference allocations fail closed.
The only implicit historical defaults are the two fields introduced with LQ.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

import tokenizer_evaluation as E
import logsquare_preflight as L
from wearusfm.model.fm import FMConfig

P, R, require, REPO = E.P, E.R, E.require, E.REPO
PROTOCOL = 'tokenizer-logsquare-evaluation-20261104-v1'
ARM = 'T100-LQ'
TRAINING_PROTOCOL = 'tokenizer-logsquare-training-20261104-v1'
NEW_WEIGHT = 'tokenizer.logsquare_weight'
DEFAULTS = {'tokenizer_logsquare': False, 'tokenizer_logsquare_eps': .001}
EVALUATOR_SCRIPTS = ('scripts/logsquare_evaluation.py', 'scripts/logsquare_infer_benchmark.py',
    'scripts/tokenizer_evaluation.py', 'scripts/logsquare_preflight.py', 'scripts/tokenizer_preflight.py',
    'scripts/train_jepa.py', 'scripts/pose_eval.py', 'scripts/channel_readout.py',
    'scripts/calibration_probe.py', 'scripts/tokenizer_pose_readout.py',
    'scripts/slurm/logsquare_evaluation.sbatch', 'scripts/slurm/logsquare_infer_benchmark.sbatch')


def source_files():
    return tuple(sorted(set(EVALUATOR_SCRIPTS) | {str(p.relative_to(REPO)) for p in (REPO / 'src/wearusfm').rglob('*.py')}))


def validate_sources(plan):
    hashes = plan.get('source_sha256', {})
    require(set(hashes) == set(source_files()), 'incomplete evaluation source inventory')
    return {'source_' + relative: E.pin_file({'path': relative, 'sha256': digest}) for relative, digest in hashes.items()}



def benchmark_context(plan):
    """Stable across the later endpoint pin and budget publication, not data changes.

    Called before repo-relative split paths are resolved inside a checkout.
    The original pin bytes/relative identifiers, not checkout location, are bound.
    """
    keys = ('protocol', 'arm', 'inputs', 'paths', 'pose_references', 'classification_references',
            'training_plan', 'candidate_config', 'source_sha256')
    value = {key: copy.deepcopy(plan[key]) for key in keys}
    value['checkpoint_5000'] = copy.deepcopy(plan['checkpoints']['5000'])
    return value


def context_sha256(plan):
    return hashlib.sha256(json.dumps(benchmark_context(plan), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_config(value):
    """Canonical comparison only; never rewrite the pinned source bytes."""
    value = copy.deepcopy(value)
    model = value['model']
    for key, default in DEFAULTS.items():
        model.setdefault(key, default)
    require(set(model) == {f.name for f in fields(FMConfig)}, 'missing or extra model fields beyond two legacy LQ defaults')
    return E.canonical(value)


def log_weight_diagnostics(state):
    """All 64 column norms, student and teacher; descriptive, no selection gate."""
    result = {}
    for kind in ('student', 'teacher'):
        require(kind in state and NEW_WEIGHT in state[kind], 'missing log weights: ' + kind)
        weight = state[kind][NEW_WEIGHT]
        require(weight.ndim == 2 and weight.shape[1] == 64 and torch.isfinite(weight).all(), 'invalid log weight matrix')
        norms = torch.linalg.vector_norm(weight.double(), dim=0).cpu().numpy()
        groups = {'fourier': norms[:24], 'spline': norms[24:48], 'mlp': norms[48:], 'non_mlp': norms[:48]}
        result[kind] = {'column_l2_norms': norms.tolist(), 'families': {
            name: {'n': len(v), 'mean': float(v.mean()), 'median': float(np.median(v)),
                   'min': float(v.min()), 'max': float(v.max()), 'sum_squared_norms': float(v @ v)}
            for name, v in groups.items()}, 'descriptive_only': True}
    return result


def validate_checkpoint(pin, expected_config, step, stats_sha256, data_signature, *, arm=ARM):
    require(arm in (ARM, 'T100-L') and type(step) is int and step in (5000, 19618), 'unregistered arm/dose')
    require(isinstance(pin, dict), 'checkpoint pin absent; training must finish first')
    path = Path(pin['path'])
    require(path.is_absolute() and path.name == f'checkpoint_step{step:07d}.pt', 'immutable named endpoint required')
    require(not (path.parent / 'STOP').exists(), 'STOP present')
    states = {'checkpoint': E.pin_file(pin), 'config': E.pin_file({'path': pin['config_path'], 'sha256': pin['config_sha256']}),
              'receipt': E.pin_file({'path': pin['receipt_path'], 'sha256': pin['receipt_sha256']})}
    config = json.loads(Path(states['config']['path']).read_text())
    receipt = json.loads(Path(states['receipt']['path']).read_text())
    expected = canonical_config(expected_config)
    require(canonical_config(config) == expected, 'full frozen recipe mismatch')
    required = {'protocol': TRAINING_PROTOCOL if arm == ARM else 'tokenizer-training-20261103-v1',
                'complete': True, 'arm': arm, 'checkpoint_step': step, 'checkpoint': str(path),
                'checkpoint_sha256': pin['sha256'], 'config_path': states['config']['path'],
                'config_file_sha256': pin['config_sha256']}
    for key, value in required.items():
        require(receipt.get(key) == value, 'training receipt mismatch: ' + key)
    require(len(pin.get('training_code_sha', '')) == 40 and receipt.get('code_sha') == pin['training_code_sha'], 'training code pin mismatch')
    if arm == ARM:
        for key, pk in (('plan_sha256', 'training_plan_sha256'), ('release_sha256', 'release_sha256')):
            require(len(pin.get(pk, '')) == 64 and receipt.get(key) == pin[pk], 'training release/plan pin mismatch: ' + key)
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(type(state.get('step')) is int and state['step'] == step, 'checkpoint step mismatch')
    require(state.get('config') == config, 'checkpoint/source configuration mismatch')
    require(state.get('pose_stats_sha256') == stats_sha256 and state.get('data_signature') == data_signature, 'checkpoint data/statistics mismatch')
    teacher = state.get('teacher')
    require(isinstance(teacher, dict) and teacher and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in teacher.values()), 'invalid teacher state')
    model = E.FF.load_model(str(path), 'cpu', 'teacher')  # strict state_dict load, no evaluation calibration
    require(E.canonical(asdict(model.cfg)) == expected['model'], 'loaded model config differs')
    require(model.cfg.window_ms == 100. and model.tokenizer.window_s == .1 and
            model.cfg.tokenizer_activation == model.tokenizer.tokenizer_activation == 'identity', 'tokenizer window/activation mismatch')
    require(model.cfg.tokenizer_logsquare == model.tokenizer.tokenizer_logsquare == (arm == ARM) and
            model.cfg.tokenizer_logsquare_eps == model.tokenizer.tokenizer_logsquare_eps == .001, 'LQ flag/epsilon mismatch')
    actual = model.state_dict()
    require(actual.keys() == teacher.keys() and all(v.dtype == teacher[k].dtype and torch.equal(v, teacher[k]) for k, v in actual.items()), 'teacher not restored exactly')
    gain = float(model.tokenizer.fe_gain)
    require(math.isfinite(gain) and gain > 0 and torch.equal(model.tokenizer.fe_gain, teacher['tokenizer.fe_gain']), 'saved gain not restored')
    if arm == ARM:
        require(teacher[NEW_WEIGHT].shape == (model.cfg.dim, 64) and torch.equal(model.tokenizer.logsquare_weight, teacher[NEW_WEIGHT]), 'log weight shape/value mismatch')
        diagnostics = log_weight_diagnostics(state)
    else:
        require(NEW_WEIGHT not in teacher and model.tokenizer.logsquare_weight is None, 'linear control unexpectedly has LQ weights')
        diagnostics = None
    model_config = E.canonical(asdict(model.cfg))
    del model, state, teacher, actual
    gc.collect()
    P.check_unchanged(states)
    require(not (path.parent / 'STOP').exists(), 'STOP appeared during validation')
    return {'states': states, 'model_config': model_config, 'fe_gain': gain, 'log_weights': diagnostics,
            'checkpoint_sha256': pin['sha256'], 'checkpoint_step': step, 'training_code_sha': receipt['code_sha'],
            'postload_recipe_passed': True, 'gain_policy': 'exact saved buffer; no data calibration'}


def load_context(plan_path, step, expected_code_sha, expected_plan_sha256, *, benchmark=False):
    code = P.code_identity(expected_code_sha)
    states = {'plan': E.pin_file({'path': str(plan_path), 'sha256': expected_plan_sha256})}
    plan = json.loads(Path(states['plan']['path']).read_text())
    require(plan.get('protocol') == PROTOCOL and plan.get('arm') == ARM and step in (5000, 19618), 'wrong extraction protocol/arm/dose')
    allowed = ('APPROVED_FOR_BENCHMARK', 'APPROVED_FOR_EXTRACTION') if benchmark else ('APPROVED_FOR_EXTRACTION',)
    require(plan.get('status') in allowed, 'evaluation plan review/pins pending')
    states.update(validate_sources(plan))
    plan['_benchmark_context_sha256'] = context_sha256(plan)
    require(set(plan['inputs']) == E.INPUTS, 'input pins incomplete')
    require(set(plan['pose_references']) == {'train', 'val'} and set(plan['classification_references']) == set(sum((list(v) for v in E.DATASETS.values()), [])), 'reference coverage incomplete')
    for split, selection in E.Q.SELECTION.items():
        require(all(plan['pose_references'][split].get(k) == v for k, v in selection.items()), 'historical pose selection differs')
    for key in ('pose_root', 'processed_root', 'uci_root'):
        require(Path(plan['paths'][key]).is_absolute(), 'relative data root')
    require(all(Path(p).is_absolute() for key in ('roots', 'epn_roots') for p in plan['paths'][key]), 'relative data root')
    states.update({key: E.pin_file(pin) for key, pin in plan['inputs'].items()})
    for key in E.INPUTS:
        plan['inputs'][key]['path'] = states[key]['path']
    states['training_plan'] = E.pin_file(plan['training_plan'])
    training = json.loads(Path(states['training_plan']['path']).read_text())
    require(training.get('protocol') == TRAINING_PROTOCOL and training.get('status') == 'APPROVED_FOR_TRAINING', 'training release missing')
    states['training_release'] = E.pin_file(training['training_release'])
    states['candidate_config'] = E.pin_file(plan['candidate_config'])
    _, configs, preflight_states = L.load_plan(training['preflight_plan']['path'])
    # The preflight plan is pinned independently; none of its inputs are replaced.
    require(E.pin_file(training['preflight_plan'])['sha256'] == preflight_states['plan']['sha256'], 'preflight plan identity mismatch')
    states.update({'preflight_' + k: v for k, v in preflight_states.items()})
    cfg = replace(configs[ARM], max_steps=step)
    require(R.config_to_dict(configs[ARM]) == json.loads(Path(states['candidate_config']['path']).read_text()), 'candidate config pin differs')
    require(cfg.model.dim == 384 and cfg.model.k_latents == 64 and cfg.model.patch_ms == 25 and cfg.model.pose_head and cfg.jepa.jepa_weight == 0, 'unregistered model recipe')
    actual, historical = E.canonical(asdict(cfg.loader)), E.canonical(asdict(R.sanity_config().loader))
    require(all(actual[k] == historical[k] for k in E.PREPROCESSING), 'preprocessing differs from production pose path')
    require(str(Path(plan['paths']['pose_root']) / 'pose_stats.json') == states['pose_stats']['path'] == str(Path(cfg.loader.pose_root) / 'pose_stats.json'), 'pose root/statistics mismatch')
    scales = json.loads(Path(states['scales']['path']).read_text())['scales']
    windows = json.loads(Path(states['windows']['path']).read_text())['window_s']
    signature = E.canonical(R.calibration_signature(cfg, states['manifest']['path'], scales, windows))
    pin = plan['checkpoints'][str(step)]
    require(isinstance(pin, dict) and pin.get('training_plan_sha256') == states['training_plan']['sha256'] and
            pin.get('release_sha256') == states['training_release']['sha256'], 'checkpoint not from pinned training release')
    guard = validate_checkpoint(pin, R.config_to_dict(cfg), step, states['pose_stats']['sha256'], signature)
    states.update(guard['states'])
    return plan, states, guard, code


def allocation_for(plan, stage, step, task, allocated_seconds):
    key = f'pose_{step}' if stage == 'extract-pose' else task + '_19618'
    require(key in ('pose_5000', 'pose_19618', 'p1_19618', 'p2_19618'), 'unregistered extraction cell')
    identity = E.pin_file(plan['inference_receipt'])
    report = json.loads(Path(identity['path']).read_text())
    require(report.get('protocol') == 'tokenizer-logsquare-inference-20261104-v1' and report.get('complete') is True, 'missing complete inference benchmark')
    # A later extraction plan pins the earlier benchmark plan; the benchmark
    # itself pins its pre-budget extraction context. No circular hash equality.
    import logsquare_infer_benchmark as B
    infer_identity = E.pin_file(plan['inference_plan'])
    infer_plan = json.loads(Path(infer_identity['path']).read_text())
    require(infer_plan.get('protocol') == B.PROTOCOL and infer_plan.get('status') == 'APPROVED_FOR_BENCHMARK', 'inference plan not released')
    require(report.get('plan_sha256') == infer_identity['sha256'] and len(plan.get('inference_code_sha', '')) == 40 and
            report.get('code_sha') == plan['inference_code_sha'], 'inference code/plan identity mismatch')
    require(infer_plan.get('aggregation') == report.get('aggregation') == B.POLICY and infer_plan.get('selection') == B.SELECTION, 'inference rule mismatch')
    require(report.get('benchmark_context_sha256') == plan['_benchmark_context_sha256'], 'inference input/config/root context differs')
    require(report.get('source_sha256') == infer_plan.get('source_sha256') == plan.get('source_sha256'), 'inference/extraction scientific sources differ')
    validate_sources(infer_plan)
    groups = B.group_ratios(report['records'])
    history, historical_states = B.historical_jobs(infer_plan['historical_jobs'])
    require(report.get('groups') == groups and report.get('historical_jobs') == history, 'inference timing/count reconstruction differs')
    computed = B.resource_proposal(groups, history)
    require(report.get('resource_proposal') == computed, 'rho_infer budget reconstruction differs')
    require(report.get('control_checkpoint_sha256') == infer_plan['control_checkpoint']['sha256'] and report.get('checkpoint_step') == 5000, 'inference control/endpoint differs')
    expected = computed['jobs'][key]
    require(plan.get('allocations', {}).get(key) == expected, 'allocation differs from measured rho_infer proposal')
    require(expected['walltime_seconds'] == allocated_seconds and allocated_seconds <= 86400, 'walltime differs from measured allocation')
    require(expected['gpus'] == 1 and expected['cpus'] == 8 and expected['memory'] == '100G', 'unexpected extraction resources')
    require(report['candidate_checkpoint_sha256'] == plan['checkpoints']['5000']['sha256'], 'benchmark checkpoint differs')
    require(P.file_identity(identity['path']) == identity and P.file_identity(infer_identity['path']) == infer_identity, 'inference evidence changed')
    return key, {'inference_receipt': identity, 'inference_plan': infer_identity, **historical_states}


def new_output(out):
    out = Path(out)
    require(out.is_absolute() and not out.exists() and not out.is_symlink() and out.resolve() == out, 'new canonical absolute output required')
    require(not any(out.is_relative_to(p) for p in (REPO, Path('/leonardo_work/IscrB_WearUsFM/wearusfm'), Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns'))), 'output inside code')
    return out


def execute(stage, plan_path, step, expected_code_sha, expected_plan_sha256, out, task=None, device='cuda', allocated_seconds=None):
    started = time.perf_counter()
    require(stage in ('validate', 'extract-pose', 'extract-classification'), 'unknown stage')
    require(stage != 'extract-classification' or (step == 19618 and task in E.DATASETS), 'classification only at endpoint')
    plan, states, guard, code = load_context(plan_path, step, expected_code_sha, expected_plan_sha256, benchmark=stage == 'validate')
    checkpoint = plan['checkpoints'][str(step)]['path']
    out = new_output(out)
    if stage != 'validate':
        _, budget_states = allocation_for(plan, stage, step, task, allocated_seconds)
        states.update(budget_states)
    common = {'protocol': PROTOCOL, 'arm': ARM, 'which': 'teacher', 'checkpoint_step': step,
              'checkpoint_sha256': guard['checkpoint_sha256'], 'code_sha': code, 'plan_sha256': states['plan']['sha256'],
              'postload_recipe_passed': True, 'model_config': guard['model_config'], 'stats_sha256': states['pose_stats']['sha256']}
    out.mkdir(parents=True, exist_ok=False)
    outputs, commands, raw = {}, {}, {}
    if stage != 'validate':
        commands = E.extraction_commands(stage, plan, checkpoint, out, task, device)
        for key, command in commands.items():
            if stage == 'extract-pose':
                records, raw_before = E.pose_sources(plan, key)
                reference = plan['pose_references'][key]
                E.Q.validate_reference(reference)
                require(len(records) == reference['n_recordings'], 'recording coverage differs')
            else:
                raw_before = E.classification_sources(plan, task)
                for dataset in E.DATASETS[task]:
                    states['reference_' + dataset] = E.pin_file(plan['classification_references'][dataset])
            with (out / (key + '.log')).open('x') as log:
                subprocess.run(command, cwd=REPO, env={**os.environ, 'WEARUSFM_CODE_SHA': code}, stdout=log, stderr=subprocess.STDOUT, check=True)
            require(E.source_states([Path(p) for p in raw_before]) == raw_before, 'raw inputs changed')
            if stage == 'extract-pose':
                require(E.pose_sources(plan, key)[0] == records, 'recording inventory changed')
                paths = {key: out / ('train_subset.npz' if key == 'train' else 'val.npz')}
            else:
                require(E.classification_sources(plan, task) == raw_before, 'classification inventory changed')
                paths = {dataset: out / (dataset + '.npz') for dataset in E.DATASETS[task]}
            raw[key] = {'file_states': raw_before, 'unchanged': True}
            for name, path in paths.items():
                before = P.file_identity(path)
                details = E.verify_pose_cache(path, plan['pose_references'][name], name, checkpoint, code, states['pose_stats']['sha256'], guard['model_config']) if stage == 'extract-pose' else E.verify_classification_cache(path, plan['classification_references'][name], name, checkpoint, guard['checkpoint_sha256'], code, states['splits'])
                require(P.file_identity(path) == before, 'cache changed during verification')
                pp = Path(str(path) + '.provenance.json')
                P.write_json(pp, {**common, **details, 'cache': str(path), 'cache_sha256': before['sha256'], 'cache_bytes': before['bytes']})
                outputs[name] = {'path': str(path), 'sha256': before['sha256'], 'bytes': before['bytes'], 'provenance_path': str(pp), 'provenance_sha256': R.sha256_file(pp)}
    P.check_unchanged(states)
    require(P.code_identity(code) == code and not (Path(checkpoint).parent / 'STOP').exists(), 'code/STOP changed')
    P.write_json(out / 'receipt.json', {**common, 'complete': True, 'stage': stage, 'checkpoint_unchanged': True,
        'outputs': outputs, 'checkpoint_guard': guard, 'source_states': states, 'raw_inputs': raw, 'commands': commands,
        'log_weight_diagnostics': guard['log_weights'], 'elapsed_seconds': time.perf_counter() - started,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': device}})
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('validate', 'extract-pose', 'extract-classification'), required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--step', type=int, choices=(5000, 19618), required=True)
    p.add_argument('--task', choices=tuple(E.DATASETS))
    p.add_argument('--expected-code-sha', required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--allocated-seconds', type=int)
    a = p.parse_args(argv)
    execute(a.stage, a.plan, a.step, a.expected_code_sha, a.expected_plan_sha256, a.out, a.task, a.device, a.allocated_seconds)


if __name__ == '__main__':
    main()
