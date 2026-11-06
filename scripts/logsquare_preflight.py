"""Paired train-only calibration of T100-L and its zero-init log-square branch.

Uses the existing trainer update, input stream and support audits. No validation,
checkpoint or training launch. Production invocation requires an approved frozen
checkout and exactly four CUDA ranks; output paths must be new and outside code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('logsquare_reference_preflight', REPO / 'scripts/tokenizer_preflight.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
R, J, require = P.R, P.J, P.require
ARMS = ('T100-L', 'T100-LQ')
PROTOCOL = 'tokenizer-logsquare-preflight-20261104-v1'
NEW_WEIGHT = 'tokenizer.logsquare_weight'


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan['protocol'] == PROTOCOL and plan['epsilon'] == 1e-3, 'unregistered protocol/epsilon')
    reference = P.pinned_file(plan['reference_plan'])
    original, configs, states = P.load_plan(reference['path'])
    require(plan['batches'] == original['batches'] == 64, 'exactly64paired batches required')
    historical = P.pinned_file(plan['historical_report'])
    require(configs['T100-L'].model.window_ms == 100. and configs['T100-L'].model.tokenizer_activation == 'identity',
            'not the frozen100ms linear control')
    baseline = configs['T100-L']
    candidate = replace(baseline, model=replace(baseline.model, tokenizer_logsquare=True,
                                              tokenizer_logsquare_eps=plan['epsilon']))
    states['candidate_config'] = P.pinned_file(plan['candidate_config'])
    require(R.config_to_dict(candidate) == json.loads(Path(states['candidate_config']['path']).read_text()),
            'candidate config differs from registered branch')
    states['reference_plan'] = states['plan']
    states['plan'] = P.file_identity(path)
    states['historical_report'] = historical
    return {**original, **plan}, {'T100-L': baseline, 'T100-LQ': candidate}, states


class LogMoments(P.FeatureMoments):
    """FP64 aggregate diagnostics, no raw signals and no data-dependent selection."""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def add(self, key, values, activation):
        super().add(key, values, activation)
        if not values.numel():
            return
        # Exactly the FP32 arithmetic used by the tokenizer, then FP64 statistics.
        x = values.detach().float().reshape(-1, 64)
        y = torch.log(x.square() + self.epsilon)
        derivative = 2 * x / (self.epsilon + x.square())
        require(bool(torch.isfinite(y).all() and torch.isfinite(derivative).all()), 'nonfinite log branch')
        additions = {'log_sum': y.double().sum(0), 'log_square': y.double().square().sum(0),
                     'below_floor': (x.square() < self.epsilon).sum(0).double(),
                     'derivative_square': derivative.double().square().sum(0)}
        group = self.groups[key]
        for name, value in additions.items():
            a = value.cpu().numpy()
            group[name] = group.get(name, np.zeros_like(a)) + a

    def summary(self):
        result = super().summary()
        for key, group in self.groups.items():
            n = group['n']
            mean = group['log_sum'] / n
            result[key]['logsquare'] = {
                'epsilon': self.epsilon, 'mean_per_kernel': mean.tolist(),
                'std_per_kernel': np.sqrt(np.maximum(group['log_square'] / n - mean ** 2, 0)).tolist(),
                'below_floor_fraction_per_kernel': (group['below_floor'] / n).tolist(),
                'derivative_rms_per_kernel': np.sqrt(group['derivative_square'] / n).tolist(),
                'analytic_max_abs_derivative': 1 / np.sqrt(self.epsilon)}
        return result


def create_arms(configs, ctx, device, historical_hash, epsilon):
    require(configs['T100-LQ'].model.tokenizer_logsquare_eps == epsilon, 'diagnostic/model epsilon mismatch')
    arms = {}
    for name in ARMS:
        cfg = configs[name]
        torch.manual_seed(cfg.seed)
        student = P.WearUsFM(cfg.model).to(device)
        teacher = J.make_teacher(student).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if ctx.active:
            torch.manual_seed(cfg.seed * 1_000_003 + ctx.rank + 1)
        wrapped = R._ddp_wrap(student, cfg, ctx, device) if ctx.active else None
        if ctx.active:
            for value in teacher.state_dict().values():
                torch.distributed.broadcast(value, src=0)
        state = student.state_dict()
        initial = {'shared_sha256': P.state_sha256({k: v for k, v in state.items() if k != NEW_WEIGHT}),
                   'parameters': sum(p.numel() for p in student.parameters()),
                   'fe_gain': float(student.tokenizer.fe_gain),
                   'new_weight_zero': student.tokenizer.logsquare_weight is None or
                                      not bool(student.tokenizer.logsquare_weight.count_nonzero())}
        groups = P.selected_parameters(student)
        if student.tokenizer.logsquare_weight is not None:
            groups['logsquare'] = {NEW_WEIGHT: student.tokenizer.logsquare_weight}
        arm = {'cfg': cfg, 'student': student, 'teacher': teacher, 'optimizer': optimizer, 'wrapped': wrapped,
               'rng': P.rng_state(device), 'initial': initial, 'features': LogMoments(epsilon), 'observed': [],
               'groups': groups, 'records': []}
        original_decode = student.decode

        def capture(enc, ch, qt, offsets, arm=arm, decode=original_decode):
            arm['observed'].append({'ch': ch.detach(), 'time': qt.detach(), 'offsets': offsets.detach(),
                                   'visible': enc.visible.detach(), 'patch_valid': enc.patch_valid.detach()})
            return decode(enc, ch, qt, offsets)

        student.decode = capture
        arms[name] = arm
    require(all(a['initial']['shared_sha256'] == historical_hash for a in arms.values()),
            'shared initialization does not replicate historical T100-L')
    require(all(a['initial']['new_weight_zero'] for a in arms.values()), 'log branch not zero initialized')
    require(len({P.state_sha256(a['rng']) for a in arms.values()}) == 1, 'initial RNG mismatch')
    require(arms['T100-LQ']['initial']['parameters'] - arms['T100-L']['initial']['parameters'] == 64 * configs['T100-L'].model.dim,
            'unexpected capacity difference')
    return arms


def compare_outputs(first, second):
    result = {}
    for name, a, b in zip(('local', 'z'), first, second):
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        result[name] = {'exact': torch.equal(a, b) and finite, 'finite': finite,
                        'max_abs_difference': float((a.double() - b.double()).abs().max()) if finite else None}
    return result


@torch.no_grad()
def check_initial_function(arms, micro, device, diagnostic_path=None):
    def encode(name):
        arm = arms[name]
        arm['student'].eval()
        P.restore_rng(arm['rng'], device)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=str(device).startswith('cuda') and arm['cfg'].amp_bf16):
            out = arm['student'].encode(micro.inp, micro.visible)
        return out.local.clone(), out.z.clone()

    control = encode('T100-L')
    self_check = compare_outputs(control, encode('T100-L'))
    diagnostic = {'control_self_comparison': self_check, 'between_arms': None}
    if not all(v['finite'] for v in self_check.values()):
        diagnostic['status'] = 'nonfinite_control_forward'
    elif not all(v['exact'] for v in self_check.values()):
        diagnostic['status'] = 'nondeterministic_control_forward'
    else:
        diagnostic['between_arms'] = compare_outputs(control, encode('T100-LQ'))
        diagnostic['status'] = ('passed' if all(v['exact'] for v in diagnostic['between_arms'].values())
                                else 'initial_function_mismatch')
    if diagnostic_path is not None:
        P.write_json(diagnostic_path, diagnostic)  # retain diagnostics also on failure
    require(diagnostic['status'] == 'passed', 'initial function gate: ' + json.dumps(diagnostic, allow_nan=False))
    return diagnostic


def clipping_summary(parts, configs):
    """DDP ranks are replicas, not independent observations; retain every rank."""
    result = {name: {'per_rank': {}, 'factor_formula': 'min(1, grad_clip / (g + 1e-6))',
                     'scope': 'Factors multiply gradients before AdamW, not parameter updates.'} for name in ARMS}
    for part in parts:
        records = {name: part['arms'][name]['records'] for name in ARMS}
        steps = [r['step'] for r in records['T100-L']]
        require(steps and steps == [r['step'] for r in records['T100-LQ']], 'unpaired clipping records')
        norms, factors = {}, {}
        for name in ARMS:
            norms[name] = [float(r['grad_norm_before_clip']) for r in records[name]]
            require(all(math.isfinite(g) and g >= 0 for g in norms[name]), 'invalid gradient norm')
            # PyTorch clip_grad_norm_ uses 1e-6 in the denominator.
            factors[name] = [min(1., configs[name].grad_clip / (g + 1e-6)) for g in norms[name]]
        ratios = [q / l for q, l in zip(factors['T100-LQ'], factors['T100-L'])]
        new_norms = [float(r['gradient_norms_before_clip']['logsquare']) for r in records['T100-LQ']]
        require(all(math.isfinite(g) and g >= 0 for g in new_norms), 'invalid log gradient norm')
        shares = [(w / g) ** 2 if g else None for w, g in zip(new_norms, norms['T100-LQ'])]
        for name in ARMS:
            result[name]['per_rank'][str(part['rank'])] = {
                'steps': steps, 'gradient_norm_before_clip': norms[name],
                'clip_factor_per_step': factors[name], 'clip_factor_LQ_over_L_per_step': ratios,
                'logsquare_gradient_squared_share_per_step': shares if name == 'T100-LQ' else None,
                'fraction_steps_clipped': sum(f < 1. for f in factors[name]) / len(steps)}
        q0, l0, w0 = norms['T100-LQ'][0], norms['T100-L'][0], new_norms[0]
        result['T100-LQ']['per_rank'][str(part['rank'])]['first_step_norm_decomposition_relative_residual'] = (
            (q0*q0 - l0*l0 - w0*w0) / (q0*q0) if q0 else None)
    return result


def journal_step(path, step, batch_hash, records):
    # A failed run leaves the preceding completed updates available for diagnosis.
    with Path(path).open('x' if step == 0 else 'a') as stream:
        json.dump({'step': step + 1, 'batch_sha256': batch_hash, 'arms': records}, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()


def verify_pair(records):
    a, b = (records[name] for name in ARMS)
    require(a['support'] == b['support'], 'different input/target/query support')
    require(a['rng_after_sha256'] == b['rng_after_sha256'], 'different RNG consumption')


def run(plan, configs, states, out, ctx, device, code):
    began = time.perf_counter()
    cfg = configs['T100-L']
    manifest = Path(states['manifest']['path'])
    scales = json.loads(Path(states['scales']['path']).read_text())['scales']
    windows = json.loads(Path(states['windows']['path']).read_text())['window_s']
    R.check_input_files(manifest, Path(states['windows']['path']), cfg.nested, states['manifest']['sha256'])
    index = R.load_index(manifest, [Path(p) for p in plan['roots']], cfg.datasets, cfg.nested)
    if cfg.nested is not None:
        R.check_derived_binding(index.params, cfg, windows)
    require(all(row['split'] == 'pretraining' for row in index.rows), 'not a train-only index')
    guard = P.SourceGuard(index, cfg.loader.pose_root)
    historical = json.loads(Path(states['historical_report']['path']).read_text())['arms']['T100-L']['initial']['all_sha256']
    arms = create_arms(configs, ctx, device, historical, plan['epsilon'])
    initial = {name: a['initial'] for name, a in arms.items()}
    peers = [None] * ctx.world_size
    torch.distributed.all_gather_object(peers, initial)
    require(all(p == initial for p in peers), 'initialization differs across ranks')
    loader = torch.utils.data.DataLoader(
        R.BatchStream(index, cfg.loader, cfg.batch_size // ctx.world_size, cfg.seed, scales, windows,
                      rank=ctx.rank, start=0, segment=cfg.stream_segment), batch_size=None,
        num_workers=plan['num_workers'], persistent_workers=True,
        generator=torch.Generator().manual_seed(cfg.seed), collate_fn=R._to_shared)
    iterator = iter(loader)
    exposure, fingerprints = Counter(), []
    for step in range(plan['batches']):
        bad, = R._any_flag(ctx, device, time.perf_counter() - began > 1500)
        require(not bad, 'preflight exceeded internal25min deadline')
        batch = R._from_shared(next(iterator))
        require(batch.index == step and len(batch.signals) == 8, 'batch identity mismatch')
        guard.observe(batch.rows)
        before = P.batch_sha256(batch)
        micro, = R._stage(batch, cfg.accum_steps, device)
        digest = R._micro_digest([micro])
        if step == 0:
            initial_function = check_initial_function(arms, micro, device, out / f'initial_function_rank{ctx.rank}.json')
        records = {name: P.update_arm(arms[name], micro, batch.rows, step, ctx, device) for name in ARMS}
        journal_step(out / f'rank{ctx.rank}.jsonl', step, before, records)
        verify_pair(records)
        require(P.batch_sha256(batch) == before and R._micro_digest([micro]) == digest, 'mutated batch')
        fingerprints.append(before)
        for row, fs in zip(batch.rows, batch.fs):
            exposure[f"{row['dataset']}/{float(fs):g}"] += 1
        if ctx.is_main:
            print(json.dumps({'step': step+1, 'paired': True, 'normal_update_s':
                              {name: r['normal_update_s'] for name, r in records.items()}}), flush=True)
    guard.finish()
    P.check_unchanged(states)
    P.code_identity(code)
    for name, arm in arms.items():
        for group in arm['groups']:
            require(any(r['gradient_norms_before_clip'][group] > 0 for r in arm['records']), f'zero gradients:{name}/{group}')
            require(any(r['update_norms'][group] > 0 for r in arm['records']), f'zero updates:{name}/{group}')
    raw = {'protocol': PROTOCOL, 'complete': True, 'rank': ctx.rank, 'initial': initial,
           'initial_function': initial_function,
           'exposure_windows': dict(exposure), 'batch_sha256': fingerprints,
           'arms': {name: {'records': a['records'], 'feature_moments': a['features'].serializable()}
                    for name, a in arms.items()}, 'wall_seconds': time.perf_counter() - began}
    P.write_json(out / f'rank{ctx.rank}.json', raw)
    torch.distributed.barrier()
    if ctx.is_main:
        parts = [json.loads((out / f'rank{r}.json').read_text()) for r in range(ctx.world_size)]
        results, total_exposure = {}, Counter()
        for part in parts:
            require(part['complete'] and len(part['batch_sha256']) == 64, 'incomplete rank')
            total_exposure.update(part['exposure_windows'])
        require(any(k.endswith('/200') for k in total_exposure) and any(k.endswith('/2000') for k in total_exposure),
                'missing native200Hz or2000Hz training coverage; do not launch training')
        clipping = clipping_summary(parts, configs)
        for name in ARMS:
            moments = LogMoments(plan['epsilon'])
            for part in parts:
                moments.merge(part['arms'][name]['feature_moments'])
            timing = [sum(max(part['arms'][name]['records'][i][component] for part in parts)
                          for component in ('forward_backward_s', 'clip_optimizer_ema_s')) for i in range(64)]
            results[name] = {'initial': initial[name], 'features': moments.summary(), 'clipping': clipping[name],
                             'update_mean_after8_s': float(np.mean(timing[8:])),
                             'update_p90_after8_s': float(np.quantile(timing[8:], .9)),
                             'max_allocated_bytes_two_arms_resident': max(r['memory_peak_allocated_bytes']
                                 for part in parts for r in part['arms'][name]['records'])}
        report = {'protocol': PROTOCOL, 'complete': True, 'code_sha': code, 'inputs': states,
                  'epsilon': plan['epsilon'], 'batches': 64, 'windows_each_arm': 2048,
                  'exposure_windows': dict(total_exposure), 'arms': results,
                  'paired_support': True, 'initial_shared_weights_rng_and_function_exact': True,
                  'initial_function_per_rank': {str(p['rank']): p['initial_function'] for p in parts},
                  'update_mean_after8_ratio_LQ_over_L': results['T100-LQ']['update_mean_after8_s'] / results['T100-L']['update_mean_after8_s'],
                  'scope': 'Train-only diagnostic, no scientific checkpoint. Epsilon floor fractions require review before training.',
                  'timing_scope': 'Existing DDP update; excludes loader/diagnostics. Two complete models resident, not single-run memory.',
                  'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(device), 'world_size': ctx.world_size},
                  'rank_journals': [P.file_identity(out / f'rank{r}.jsonl') for r in range(ctx.world_size)],
                  'initial_function_reports': [P.file_identity(out / f'initial_function_rank{r}.json')
                                               for r in range(ctx.world_size)],
                  'rank_reports': [P.file_identity(out / f'rank{r}.json') for r in range(ctx.world_size)]}
        P.write_json(out / 'report.json', report)
    torch.distributed.barrier()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-code-sha', required=True)
    args = parser.parse_args(argv)
    code = P.code_identity(args.expected_code_sha)
    plan, configs, states = load_plan(args.plan)
    ctx = R.dist_from_env()
    require(ctx.world_size == 4 and torch.cuda.is_available(), 'exactly four CUDA ranks required')
    forbidden = (REPO, Path('/leonardo_work/IscrB_WearUsFM/wearusfm'), Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns'))
    require(args.out.is_absolute() and not any(args.out.resolve().is_relative_to(p) for p in forbidden), 'unsafe output path')
    torch.cuda.set_device(ctx.local_rank)
    device = f'cuda:{ctx.local_rank}'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.distributed.init_process_group('nccl')
    try:
        bad, = R._any_flag(ctx, device, args.out.exists() or args.out.is_symlink())
        require(not bad, 'output already exists; no retry or overwrite')
        if ctx.is_main:
            args.out.mkdir(parents=True, exist_ok=False)
        torch.distributed.barrier()
        run(plan, configs, states, args.out, ctx, device, code)
    finally:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
