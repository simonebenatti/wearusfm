"""Independent, offline audit of the frozen paired LQ preflight and R1/R2.

No trainer/collector imports; requires the original report and all rank artifacts.
Thresholds are the prospectively published R1 rules, not fitted to these data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('T100-L', 'T100-LQ')
CODE = '43cad1f760fbac2f10d8c16bc6f5d4eedc957b61'
REPORT_SHA = '311ac329731c84f91e9199c78c92e4890cc43d2de598361f11f4e3f7465e5b08'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b, message):
    require(np.allclose(a, b, rtol=1e-12, atol=1e-12), message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def accounting(path, job):
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='|') if r['JobIDRaw'] == str(job)]
    require(len(rows) == 1 and rows[0]['State'] == 'COMPLETED' and rows[0]['ExitCode'] == '0:0', 'accounting incomplete')
    r = rows[0]
    return int(r['ElapsedRaw']), int(r['AllocCPUS'])


def r1_rows(features, epsilon):
    rows, frequencies = [], []
    for key in sorted(k for k in features if k.endswith('/unmasked_same_visible')):
        prefix = key.rsplit('/', 1)[0]
        masked = features[prefix + '/masked_visible']
        unmasked = features[key]
        require(masked['n_tokens'] == unmasked['n_tokens'], 'unpaired feature support counts')
        phase, dataset, fs = prefix.split('/')
        m, u = masked['logsquare'], unmasked['logsquare']
        for i in range(64):
            pm, pu = (x['below_floor_fraction_per_kernel'][i] for x in (m, u))
            dm, du = (x['derivative_rms_per_kernel'][i] for x in (m, u))
            delta = m['mean_per_kernel'][i] - u['mean_per_kernel'][i]
            std = u['std_per_kernel'][i]
            rows.append({'phase': phase, 'dataset': dataset, 'fs': int(fs), 'kernel': i,
                         'family': 'fourier' if i < 24 else 'spline' if i < 48 else 'mlp',
                         'n_tokens_each_mode': unmasked['n_tokens'],
                         'p_unmasked': pu, 'p_masked': pm, 'delta_p': pm-pu,
                         'd_unmasked': du, 'd_masked': dm, 'delta_d': dm-du,
                         'd_normalized_unmasked': du*math.sqrt(epsilon),
                         'd_normalized_masked': dm*math.sqrt(epsilon),
                         'delta_log_mean': delta, 'unmasked_log_std': std,
                         'mask_mean_std_ratio': abs(delta)/std if std else None,
                         'floor_trigger': pu >= .5,
                         'derivative_trigger_unmasked': du*math.sqrt(epsilon) >= .5,
                         'derivative_trigger_masked': dm*math.sqrt(epsilon) >= .5,
                         'mask_floor_crossing_trigger': (pm >= .5) != (pu >= .5),
                         'mask_log_mean_trigger': abs(delta) >= std if std else delta != 0})
    # All available 200/2000 dataset pairs, separately by phase and mode.
    for key, a in sorted(features.items()):
        phase, dataset, fs, mode = key.split('/')
        if fs != '200':
            continue
        for other_key, b in sorted(features.items()):
            ph, ds, freq, md = other_key.split('/')
            if (ph, freq, md) != (phase, '2000', mode):
                continue
            p, q = (v['logsquare']['below_floor_fraction_per_kernel'] for v in (a, b))
            frequencies.append({'group_200': key, 'group_2000': other_key,
                                'n_tokens_200': a['n_tokens'], 'n_tokens_2000': b['n_tokens'],
                                'compatible_fs_attenuation_kernels': [i for i in range(64) if p[i] >= .5 and q[i] < .5],
                                'p_200': p, 'p_2000': q, 'exempted_kernels': []})
    return rows, frequencies


def audit(folder, repo):
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_text())
    require(hashlib.sha256(report_path.read_bytes()).hexdigest() == REPORT_SHA, 'report differs from remote receipt')
    require(report['complete'] and report['code_sha'] == CODE and report['batches'] == 64, 'wrong/incomplete run')
    require(report['epsilon'] == .001 and report['runtime']['world_size'] == 4, 'wrong epsilon/ranks')
    artifacts = []
    for category in ('rank_reports', 'rank_journals', 'initial_function_reports'):
        for identity in report[category]:
            path = folder / Path(identity['path']).name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == identity['sha256'] and path.stat().st_size == identity['bytes'], 'artifact changed: '+path.name)
            artifacts.append({'name': path.name, 'remote_path': identity['path'], 'sha256': digest, 'bytes': path.stat().st_size})
    pinned_local = {}
    for name, identity in report['inputs'].items():
        marker = '/logsquare-preflight-20261104/'
        if marker in identity['path']:
            path = repo / identity['path'].split(marker, 1)[1]
            require(hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256'], 'local pinned input changed: '+name)
            pinned_local[name] = identity['sha256']
    parts = [json.loads((folder / f'rank{r}.json').read_text()) for r in range(4)]
    historical = json.loads((repo / 'results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json').read_text())
    expected_initial = historical['arms']['T100-L']['initial']['all_sha256']
    for arm in ARMS:
        init = report['arms'][arm]['initial']
        require(init['new_weight_zero'] and init['shared_sha256'] == expected_initial, 'historical initialization mismatch')
    require(report['arms']['T100-LQ']['initial']['parameters'] - report['arms']['T100-L']['initial']['parameters'] == 24576,
            'unexpected new capacity')
    exposure = Counter()
    for rank, part in enumerate(parts):
        require(part['rank'] == rank and part['complete'], 'wrong rank')
        require(len(part['batch_sha256']) == 64, 'incomplete batches')
        initial = part['initial_function']
        require(initial == report['initial_function_per_rank'][str(rank)], 'initial diagnostics differ')
        require(initial == json.loads((folder / f'initial_function_rank{rank}.json').read_text()), 'initial artifact differs')
        require(initial['status'] == 'passed', 'initial function failed')
        for group in ('control_self_comparison', 'between_arms'):
            require(all(x == {'exact': True, 'finite': True, 'max_abs_difference': 0.0} for x in initial[group].values()), 'nonexact initial function')
        require(part['initial'] == {a: report['arms'][a]['initial'] for a in ARMS}, 'initial rank identity mismatch')
        exposure.update(part['exposure_windows'])
        journals = [json.loads(line) for line in (folder / f'rank{rank}.jsonl').read_text().splitlines()]
        require(len(journals) == 64, 'incomplete journal')
        for arm in ARMS:
            records = part['arms'][arm]['records']
            require([r['step'] for r in records] == list(range(1, 65)), 'wrong steps')
            require(finite(records), 'nonfinite record')
            for name in records[0]['gradient_norms_before_clip']:
                require(any(r['gradient_norms_before_clip'][name] > 0 for r in records), 'zero gradient group')
                require(any(r['update_norms'][name] > 0 for r in records), 'zero update group')
        for i, journal in enumerate(journals):
            a, b = (part['arms'][arm]['records'][i] for arm in ARMS)
            require(journal['step'] == i+1 and journal['batch_sha256'] == part['batch_sha256'][i], 'journal batch differs')
            require(journal['arms'] == {arm: part['arms'][arm]['records'][i] for arm in ARMS}, 'journal record differs')
            require(a['support'] == b['support'] and a['rng_after_sha256'] == b['rng_after_sha256'], 'unpaired support/RNG')
        for arm in ARMS:
            records = part['arms'][arm]['records']
            clip = report['arms'][arm]['clipping']['per_rank'][str(rank)]
            cfg = json.loads((repo / 'results/infra/logsquare_review_20261104/config_T100-LQ.json').read_text())
            norms = np.array([r['grad_norm_before_clip'] for r in records])
            factors = np.minimum(1., cfg['grad_clip']/(norms+1e-6))
            close(norms, clip['gradient_norm_before_clip'], 'gradient norm aggregation differs')
            close(factors, clip['clip_factor_per_step'], 'clipping factors differ')
            close(np.mean(factors < 1), clip['fraction_steps_clipped'], 'clipping fraction differs')
            other = part['arms']['T100-L' if arm == 'T100-LQ' else 'T100-LQ']['records']
            other_factors = np.minimum(1., cfg['grad_clip']/(np.array([r['grad_norm_before_clip'] for r in other])+1e-6))
            ratio = factors/other_factors if arm == 'T100-LQ' else other_factors/factors
            close(ratio, clip['clip_factor_LQ_over_L_per_step'], 'clip ratios differ')
            if arm == 'T100-LQ':
                new = np.array([r['gradient_norms_before_clip']['logsquare'] for r in records])
                close((new/norms)**2, clip['logsquare_gradient_squared_share_per_step'], 'new gradient share differs')
                old_norm = other[0]['grad_norm_before_clip']
                residual = (norms[0]**2-old_norm**2-new[0]**2)/norms[0]**2
                close(residual, clip['first_step_norm_decomposition_relative_residual'], 'initial gradient decomposition differs')
    require(dict(exposure) == report['exposure_windows'] and sum(exposure.values()) == report['windows_each_arm'] == 2048, 'exposure differs')
    require(any(k.endswith('/200') for k in exposure) and any(k.endswith('/2000') for k in exposure), 'frequency missing')
    for arm in ARMS:
        features = report['arms'][arm]['features']
        require(set(features) == set().union(*(set(p['arms'][arm]['feature_moments']) for p in parts)), 'feature groups differ')
        for key, feature in features.items():
            raw = [p['arms'][arm]['feature_moments'][key] for p in parts if key in p['arms'][arm]['feature_moments']]
            n = sum(r['n'] for r in raw)
            require(n == feature['n_tokens'], 'token aggregation differs')
            sums = {name: sum((np.array(r[name], dtype=float) for r in raw)) for name in ('log_sum', 'log_square', 'below_floor', 'derivative_square')}
            mean = sums['log_sum']/n
            stats = {'mean_per_kernel': mean,
                     'std_per_kernel': np.sqrt(np.maximum(sums['log_square']/n-mean**2, 0)),
                     'below_floor_fraction_per_kernel': sums['below_floor']/n,
                     'derivative_rms_per_kernel': np.sqrt(sums['derivative_square']/n)}
            require(finite(feature['logsquare']), 'nonfinite feature statistics')
            for name, values in stats.items():
                close(values, feature['logsquare'][name], 'moment aggregation differs: '+name)
            require(np.max(stats['derivative_rms_per_kernel']) <= 1/math.sqrt(.001), 'analytic bound violated')
        times = [sum(max(p['arms'][arm]['records'][i][k] for p in parts) for k in ('forward_backward_s', 'clip_optimizer_ema_s')) for i in range(64)]
        close(np.mean(times[8:]), report['arms'][arm]['update_mean_after8_s'], 'timing mean differs')
        close(np.quantile(times[8:], .9), report['arms'][arm]['update_p90_after8_s'], 'timing quantile differs')
    rho = report['arms']['T100-LQ']['update_mean_after8_s']/report['arms']['T100-L']['update_mean_after8_s']
    close(rho, report['update_mean_after8_ratio_LQ_over_L'], 'rho differs')
    rows, frequency_pairs = r1_rows(report['arms']['T100-LQ']['features'], .001)
    summary = []
    for phase, dataset, fs in sorted({(r['phase'], r['dataset'], r['fs']) for r in rows}):
        group = [r for r in rows if (r['phase'], r['dataset'], r['fs']) == (phase, dataset, fs)]
        summary.append({'phase': phase, 'dataset': dataset, 'fs': fs, 'n_tokens_each_mode': group[0]['n_tokens_each_mode'],
                        **{k: sum(r[k] for r in group) for k in group[0] if k.endswith('trigger') or k.startswith('derivative_trigger_')},
                        'max_p_unmasked': max(r['p_unmasked'] for r in group),
                        'max_normalized_derivative': max(max(r['d_normalized_masked'], r['d_normalized_unmasked']) for r in group),
                        'zero_std_nonzero_delta_count': sum(r['unmasked_log_std'] == 0 and r['delta_log_mean'] != 0 for r in group),
                        'max_mask_mean_std_ratio': max(r['mask_mean_std_ratio'] or 0 for r in group)})
    triggers = {k: sum(r[k] for r in rows) for k in rows[0] if k.endswith('trigger') or k.startswith('derivative_trigger_')}
    phases = {}
    historical_accounting = repo / 'results/infra/tokenizer_execution_20261103/19618/sacct_completed.txt'
    for step, job, cap in ((5000, 63386190, 5400), (19618, 63386337, 14400)):
        seconds, cpus = accounting(historical_accounting, job)
        require(cpus == 32, 'unexpected historical CPU allocation')
        ceiling = math.ceil(cap*max(1., rho)/900)*900
        phases[str(step)] = {'historical_wall_seconds': seconds, 'estimated_wall_seconds': seconds*rho,
                            'walltime_cap_seconds': ceiling, 'estimated_gpu_hours': seconds*rho*4/3600,
                            'cap_gpu_hours': ceiling*4/3600, 'cap_allocated_core_hours': ceiling*32/3600}
    clip = report['arms']['T100-LQ']['clipping']['per_rank']['0']
    consumed_accounting = repo / 'results/infra/logsquare_execution_20261104/sacct_completed.txt'
    elapsed, cpus = accounting(consumed_accounting, 63525730)
    setup_elapsed, setup_cpus = accounting(consumed_accounting, 63525639)
    return {'protocol': 'logsquare-preflight-offline-analysis-v1', 'report_sha256': REPORT_SHA,
            'scientific_code_sha': CODE, 'verification': {'passed': True, 'rank_files': 4, 'paired_updates': 256,
            'global_batches': 64, 'windows_each_arm': 2048, 'artifact_identities': artifacts, 'local_input_pins': pinned_local,
            'limitations': 'Source immutability and complete parameter finiteness attested by frozen driver; no raw source rehash on collector.'},
            'R1': {'epsilon': .001, 'kernel_group_pairs': len(rows), 'mode_kernel_groups': len(rows)*2,
                   'triggers': triggers, 'groups': summary, 'all_kernel_rows': rows, 'frequency_pairs': frequency_pairs,
                   'exempted_kernels': [], 'decision_proposal': 'KEEP_0.001_REQUIRES_CLAUDE_REVIEW',
                   'reason': 'Sensitivity trigger present; no floor/masking trigger or technical failure. Keep registered epsilon rather than optimize it on short calibration.',
                   'limitations': '64 early warmup steps only; final LR 3.26198e-6 versus peak 1e-4. Initial and trajectory samples overlap; tokens/ranks are not independent. W_log gradient share does not measure all sensitivity propagated to the bank.',
                   'training_release': False},
            'clipping_rank0_descriptive': {'steps': 64, 'ranks_are_replicas': True,
                'fraction_clipped_each_arm': {a: report['arms'][a]['clipping']['per_rank']['0']['fraction_steps_clipped'] for a in ARMS},
                'LQ_over_L_min_median_max': [float(f(clip['clip_factor_LQ_over_L_per_step'])) for f in (np.min, np.median, np.max)],
                'logsquare_gradient_squared_share_min_median_max': [float(f(clip['logsquare_gradient_squared_share_per_step'])) for f in (np.min, np.median, np.max)]},
            'R2': {'rho_update': rho, 'training_phases': phases, 'rho_infer': None,
                   'training_authorized_by_simone': True, 'training_approved_by_review': False,
                   'scope': 'Update excludes loader and diagnostics; historical time includes I/O. Ratio below one is not demonstrated speedup; caps use max(1,rho).'},
            'consumed': {'checkout_job': 63525639, 'preflight_job': 63525730, 'gpu_hours': elapsed*4/3600,
                         'allocated_core_hours': (elapsed*cpus+setup_elapsed*setup_cpus)/3600,
                         'cpu_only_core_hours': setup_elapsed*setup_cpus/3600}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, Path(__file__).resolve().parents[1])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'verified': result['verification']['passed'], 'R1': result['R1']['triggers'], 'rho_update': result['R2']['rho_update']}))


if __name__ == '__main__':
    main()
