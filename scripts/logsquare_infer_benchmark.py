"""Train-only timings of actual pose FP32 and classification BF16+decoder paths.

Production preprocessing remains in PE.extract / CR.extract_channel_readouts.
Scoped instrumentation repeats their actual prepared inputs; nothing is fit and
sample outputs are discarded. Future extraction budgets use this receipt only.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
import platform
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import logsquare_evaluation as V
from wearusfm.harness import benchmarks as B
from wearusfm.model.pose_head import latent_mean

E, P, R, require = V.E, V.P, V.R, V.require
C, CR, PE = E.C, E.C.CR, E.PE
PROTOCOL = 'tokenizer-logsquare-inference-20261104-v1'
ARMS = ('T100-L', 'T100-LQ')
GROUPS = {'pose': 2000., 'ninapro_db2': 2000., 'ninapro_db3': 2000., 'ninapro_db6': 2000., 'epn612': 200., 'uci_emg': 1000.}
POLICY = {'cap_ratio': 'maximum_representative_group_ratio_per_job',
          'point_ratio': 'historical_window_count_weighted_mean_group_ratio',
          'weight_source': 'pinned_historical_receipt_output_provenances',
          'missing_group_or_count': 'stop', 'cap_round_seconds': 300,
          'cap_never_below_historical': True, 'continuation_waits_for_benchmark': False}
SELECTION = {'pose': 'first_two_recordings_of_frozen_40user_200record_train_subset; require10windows_each',
             'p1': 'first_fold0_train_subject; first_session_with_at_least64windows; first64',
             'p2': 'fold0_train_subjects_in_original_loader_order; first64rows_plus_scale_completion',
             'scale_completion': 'retain_at_least_first10rows_of_every_subject_present; extra_tail_not_timed',
             'classification_timed_calls': 'first_full64batch_of_each_crop',
             'fold': 0, 'classification_batch': 64, 'pose_batch': 32,
             'pose_windows': 10, 'warmup': 2, 'repeats': 6, 'order': 'L_then_LQ_on_even_rounds_reverse_on_odd'}


def pose_forward(model, inp):
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        enc = model.encode(inp, None)
        z = latent_mean(enc.z).float()
        pred = model.pose_head(z).float()
    return enc, z, pred


def class_forward(model, inp, original_readouts=CR.readouts):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=inp.signals[0].device.type == 'cuda'):
        return original_readouts(model, inp)


def input_identity(inp):
    return {'sha256': P.state_sha256({'signals': inp.signals, 'fs': inp.fs, 'counts': inp.counts,
        'codes': inp.codes, 'sets': inp.sets, 'qc_valid': inp.qc_valid,
        'band_limit': inp.band_limit, 'edge_guard': inp.edge_guard}),
        'fs': list(map(float, inp.fs)), 'counts': list(inp.counts),
        'shapes': [list(x.shape) for x in inp.signals], 'n_windows': len(inp.signals)}


def paired_measure(models, inp, forward, *, warmup, repeats, sync, clock=time.perf_counter):
    """Same objects/support, alternating fixed order; sync outside each measured call."""
    require(set(models) == set(ARMS) and warmup == 2 and repeats == 6, 'unregistered measurement recipe')
    before = input_identity(inp)
    times = {arm: [] for arm in ARMS}
    rows, saved = [], {}
    for round_index in range(warmup + repeats):
        order = ARMS if round_index % 2 == 0 else ARMS[::-1]
        for arm in order:
            sync()
            started = clock()
            value = forward(models[arm], inp)
            sync()
            elapsed = clock() - started
            tensors = list(value.values()) if isinstance(value, dict) else [value[0].local, value[0].z, value[1], value[2]]
            require(all(bool(torch.isfinite(v).all()) for v in tensors), 'nonfinite inference output')
            require(math.isfinite(elapsed) and elapsed > 0, 'invalid measured duration')
            rows.append({'round': round_index, 'arm': arm, 'warmup': round_index < warmup, 'seconds': elapsed})
            if round_index >= warmup:
                times[arm].append(elapsed)
            if arm == 'T100-L':
                saved[arm] = value
    require(input_identity(inp) == before, 'prepared input/support mutated by forward')
    means = {arm: float(np.mean(v)) for arm, v in times.items()}
    return {'input': before, 'calls': rows, 'mean_seconds': means, 'rho_infer': means[ARMS[1]] / means[ARMS[0]]}, saved['T100-L']


class Meter:
    def __init__(self, models, device):
        self.models, self.device, self.records = models, device, []
        self.sync = (lambda: torch.cuda.synchronize(device)) if str(device).startswith('cuda') else lambda: None

    def measure(self, group, inp, forward):
        n = len(inp.signals)
        expected_n = 10 if group == 'pose' else 64
        require(n == expected_n and set(map(float, inp.fs)) == {GROUPS[group]}, 'representative group coverage/shape mismatch')
        result, value = paired_measure(self.models, inp, forward, warmup=2, repeats=6, sync=self.sync)
        result['group'] = group
        result['precision'] = 'fp32_no_autocast' if group == 'pose' else 'bf16_cuda_with_decoder'
        self.records.append(result)
        return value


class PoseProxy:
    """PE.extract remains the sole owner of record preprocessing/window batching."""
    def __init__(self, meter):
        self.meter, self.cfg, self.cached_prediction = meter, meter.models['T100-L'].cfg, None

    def eval(self):
        return self

    def encode(self, inp, visible):
        require(visible is None, 'unexpected mask in pose inference')
        enc, _, self.cached_prediction = self.meter.measure('pose', inp, pose_forward)
        return enc

    def pose_head(self, z):
        require(self.cached_prediction is not None and z.shape[:2] == self.cached_prediction.shape[:2], 'pose instrumentation order mismatch')
        result, self.cached_prediction = self.cached_prediction, None
        return result


def first64_with_scale_completion(windows, subjects):
    """Keep the first ten original scale windows for every included subject."""
    require(len(windows) >= 64, 'fewer than64 training windows; cannot use registered batch')
    subjects = np.asarray(subjects)
    selected = np.arange(64)
    for subject in np.unique(subjects[:64]):
        first = np.flatnonzero(subjects == subject)[:10]
        selected = np.union1d(selected, first)
    return selected


def load_train_p2(plan, dataset):
    """Reuse production low-level readers; never open held/val subject signals."""
    if dataset == 'epn612':
        files = [(f'{Path(root).name}/{p.name}', p / (p.name + '.json'))
                 for root in plan['paths']['epn_roots'] for p in sorted(Path(root).glob('user*'))
                 if (p / (p.name + '.json')).exists()]
    else:
        files = [(p.name, p) for p in sorted(Path(plan['paths']['uci_root']).rglob('*')) if p.is_dir() and p.name.isdigit()]
    all_subjects = sorted({subject for subject, _ in files})
    internal = C.CP.p2_internal_subjects(all_subjects)
    role = C.CP.fold_roles(internal)[0]
    allowed = set(role['train'])
    require(allowed and not (allowed & set(role['val'] + role['held'])), 'invalid fold train set')
    ws, ss, source_paths = [], [], []
    # All train subjects until at least64 rows; complete each opened subject so
    # its first10 calibration windows are exactly those of the production reader.
    for subject, path in files:
        if subject not in allowed:
            continue
        if dataset == 'epn612':
            before = E.source_states([path])
            w, _, fs = B.load_epn_user(path)
            require(fs == B.EPN_FS, 'unexpected EPN sampling rate')
            w = [B.fix_length(x, int(round(5. * fs))) for x in w]
            require(E.source_states([path]) == before, 'EPN source changed during load')
            source_paths.append(path)
        else:
            w, fs = [], B.UCI_FS
            for file in sorted(path.glob('*.txt')):
                before = E.source_states([file])
                samples, labels = B.load_uci_file(file)
                samples = B.to_myo_codes(samples, str(file))
                chunks, _ = B.windowize(samples, labels, window_samples=1000, stride_samples=1000, classes=(1, 2, 3, 4, 5, 6))
                w += chunks
                require(E.source_states([file]) == before, 'UCI source changed during load')
                source_paths.append(file)
        ws += w
        ss += [subject] * len(w)
        if len(ws) >= 64:
            break
    require(len(ws) >= 64, 'train-only P2 selection too small')
    windows, subjects = np.stack(ws), np.asarray(ss)
    ix = first64_with_scale_completion(windows, subjects)
    require(set(subjects[ix]) <= allowed, 'nontrain subject in benchmark')
    return windows[ix], subjects[ix], fs, E.FF.myo8_montage(dataset, fs), {
        'subjects': sorted(set(subjects[ix].tolist())), 'fold_train': role['train'], 'fold_roles_sha256': E.fingerprint(role),
        'selected_rows': ix.tolist(), 'source_states': E.source_states(source_paths), 'all_subject_inventory': all_subjects}


def load_train_p1(plan, dataset):
    splits = json.loads(Path(plan['inputs']['splits']['path']).read_text())
    pre = C.CP.pretraining_subjects(splits, dataset)
    role = C.CP.fold_roles(pre)[0]
    subject = role['train'][0]
    root = C._root_of([Path(p) for p in plan['paths']['roots']], dataset, subject)
    for session in C.P1.sessions_of(root, dataset, subject):
        files = [session / n for n in ('metadata.json', 'data_int16.npy', 'labels.npz')]
        before = E.source_states(files)
        sw = C.P1.session_windows(session, dataset, window_s=C.CP.WINDOW_S)
        require(E.source_states(files) == before, 'P1 source changed during load')
        if len(sw.windows) >= 64:
            key = np.full(64, f'{subject}/{sw.session}')
            return sw.windows[:64], key, sw.fs, sw.montage, {
                'subjects': [subject], 'fold_train': role['train'], 'fold_roles_sha256': E.fingerprint(role),
                'session': sw.session, 'selected_rows': list(range(64)), 'source_states': before}
    raise ValueError('first fold0 train subject has no session with64 windows; stop without reselection')


def measure_classification(meter, group, windows, subjects, fs, montage):
    original = CR.readouts
    seen = []
    def instrumented(model, inp):
        # Scale-completion rows can create a final short batch. It is executed
        # once through the baseline, excluded from the registered full64 group.
        if len(inp.signals) == 64:
            seen.append(input_identity(inp))
            return meter.measure(group, inp, lambda m, x: class_forward(m, x, original))
        return original(model, inp)
    with patch.object(CR, 'readouts', instrumented):
        result = CR.extract_channel_readouts(meter.models['T100-L'], windows, fs, montage, subjects, batch=64, device=meter.device)
    require(seen, 'no representative classification batch')
    for layer in CR.LAYERS:
        require(np.isfinite(result[layer]).all(), 'nonfinite descriptive sample output')
    return {'n_sample_windows': len(windows), 'timed_calls': len(seen), 'output_discarded': True}


def group_ratios(records):
    require({r['group'] for r in records} == set(GROUPS), 'representative group coverage missing')
    result = {}
    for r in records:
        require(r.get('precision') == ('fp32_no_autocast' if r['group'] == 'pose' else 'bf16_cuda_with_decoder'), 'timing precision mismatch')
        require(r['input']['n_windows'] == (10 if r['group'] == 'pose' else 64) and set(r['input']['fs']) == {GROUPS[r['group']]}, 'representative timing geometry differs')
        calls = r.get('calls', [])
        require(len(calls) == 16, 'missing measured/warmup calls')
        measured = {arm: [] for arm in ARMS}
        for i, call in enumerate(calls):
            round_index, position = divmod(i, 2)
            order = ARMS if round_index % 2 == 0 else ARMS[::-1]
            require(call.get('arm') == order[position] and call.get('round') == round_index and call.get('warmup') is (round_index < 2), 'timing order/warmup changed')
            seconds = call.get('seconds')
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, 'invalid measured call')
            if round_index >= 2: measured[call['arm']].append(seconds)
        means = {arm: float(np.mean(v)) for arm, v in measured.items()}
        require(r['mean_seconds'] == means and r['rho_infer'] == means[ARMS[1]] / means[ARMS[0]], 'timing mean/ratio reconstruction differs')
    for group in GROUPS:
        rows = [r for r in records if r['group'] == group]
        shapes = {(tuple(r['input']['counts']), tuple(map(tuple, r['input']['shapes'])), tuple(r['input']['fs'])) for r in rows}
        require(len(shapes) == 1, 'multiple shapes without historical group weights')
        base = sum(r['mean_seconds']['T100-L'] for r in rows)
        candidate = sum(r['mean_seconds']['T100-LQ'] for r in rows)
        require(math.isfinite(base + candidate) and base > 0 and candidate > 0, 'invalid inference durations')
        result[group] = {'rho_infer': candidate / base, 'n_representative_batches': len(rows),
                         'control_mean_seconds': base / len(rows), 'candidate_mean_seconds': candidate / len(rows)}
    return result


def historical_jobs(specifications):
    """Weights from original pinned provenance only; reads no historical arrays/scores."""
    require(set(specifications) == {'pose_5000', 'pose_19618', 'p1_19618', 'p2_19618'}, 'historical cell coverage missing')
    jobs, states = {}, {}
    for key, specification in specifications.items():
        identity = E.pin_file(specification['receipt'])
        states['history_' + key] = identity
        receipt = json.loads(Path(identity['path']).read_text())
        step = 5000 if key.endswith('5000') else 19618
        require(receipt.get('protocol') == E.PROTOCOL and receipt.get('complete') is True and receipt.get('arm') == 'T100-L' and receipt.get('which') == 'teacher' and receipt.get('checkpoint_step') == step, 'historical receipt mismatch')
        outputs = receipt['outputs']
        expected = {'train', 'val'} if key.startswith('pose') else set(E.DATASETS[key.split('_')[0]])
        require(set(outputs) == expected, 'historical output coverage mismatch')
        weights = {}
        for name, output in outputs.items():
            ident = E.pin_file({'path': output['provenance_path'], 'sha256': output['provenance_sha256']})
            states['history_' + key + '_' + name] = ident
            prov = json.loads(Path(ident['path']).read_text())
            require(prov.get('arm') == 'T100-L' and prov.get('checkpoint_step') == step and prov.get('which') == 'teacher' and
                    prov.get('cache_sha256') == output['sha256'] and prov.get('code_sha') == receipt['code_sha'] and
                    prov.get('checkpoint_sha256') == receipt['checkpoint_sha256'], 'historical count provenance mismatch')
            n = prov.get('n_windows')
            require(type(n) is int and n > 0, 'missing positive historical window count')
            group = 'pose' if key.startswith('pose') else name
            weights[group] = weights.get(group, 0) + n
        elapsed = receipt.get('elapsed_seconds')
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0, 'missing historical elapsed time')
        require(specification['walltime_seconds'] == (1800 if key == 'p1_19618' else 2700), 'wrong historical walltime')
        jobs[key] = {'historical_window_counts': weights, 'historical_elapsed_seconds': elapsed,
                     'historical_walltime_seconds': specification['walltime_seconds'], 'receipt_sha256': identity['sha256']}
    return jobs, states


def resource_proposal(groups, history):
    jobs = {}
    for key, old in history.items():
        counts = old['historical_window_counts']
        require(counts and all(g in groups and type(n) is int and n > 0 for g, n in counts.items()), 'missing group/count weights')
        ratios = [groups[g]['rho_infer'] for g in counts]
        require(all(math.isfinite(r) and r > 0 for r in ratios), 'invalid group rho')
        cap_rho = max(ratios)
        point_rho = sum(groups[g]['rho_infer'] * n for g, n in counts.items()) / sum(counts.values())
        cap = math.ceil(old['historical_walltime_seconds'] * max(1., cap_rho) / 300) * 300
        require(cap <= 86400, 'proposed extraction exceeds partition limit; review required')
        jobs[key] = {'nodes': 1, 'gpus': 1, 'cpus': 8, 'memory': '100G', 'partition': 'boost_usr_prod',
            'walltime_seconds': cap, 'gpu_hours_cap': cap / 3600, 'allocated_core_hours_cap': 8 * cap / 3600,
            'rho_infer_cap': cap_rho, 'rho_infer_point': point_rho, 'historical_window_counts': counts,
            'expected_seconds_proxy': old['historical_elapsed_seconds'] * point_rho}
    return {'jobs': jobs, 'gpu_hours_cap': sum(j['gpu_hours_cap'] for j in jobs.values()),
        'allocated_core_hours_cap': sum(j['allocated_core_hours_cap'] for j in jobs.values()),
        'gpu_hours_point_proxy': sum(j['expected_seconds_proxy'] for j in jobs.values()) / 3600,
        'interpretation': 'forward ratio scales historical walltime including I/O/loader; proxy, not measured speedup; representative shapes, not exhaustive tails'}


def execute(plan_path, expected_code, expected_plan, out, device='cuda'):
    started = time.perf_counter()
    code = P.code_identity(expected_code)
    states = {'benchmark_plan': E.pin_file({'path': str(plan_path), 'sha256': expected_plan})}
    plan = json.loads(Path(states['benchmark_plan']['path']).read_text())
    require(plan.get('protocol') == PROTOCOL and plan.get('status') == 'APPROVED_FOR_BENCHMARK', 'benchmark review/pins pending')
    states.update(V.validate_sources(plan))
    require(plan.get('selection') == SELECTION and plan.get('aggregation') == POLICY, 'benchmark selection/aggregation changed')
    require(plan.get('allocation') == {'nodes': 1, 'gpus': 1, 'cpus': 8, 'memory': '100G', 'walltime_seconds': 1800, 'gpu_hours_cap': .5, 'allocated_core_hours_cap': 4.}, 'benchmark allocation mismatch')
    extraction, context_states, guard, _ = V.load_context(plan['extraction_plan']['path'], 5000, code, plan['extraction_plan']['sha256'], benchmark=True)
    states.update(context_states)
    states['benchmark_plan'] = E.pin_file({'path': str(plan_path), 'sha256': expected_plan})
    history, history_states = historical_jobs(plan['historical_jobs'])
    states.update(history_states)
    cp = extraction['checkpoints']['5000']
    control_cfg = copy.deepcopy(json.loads(Path(guard['states']['config']['path']).read_text()))
    control_cfg['model']['tokenizer_logsquare'] = False
    checkpoint_state = torch.load(cp['path'], map_location='cpu', weights_only=False)
    signature = checkpoint_state['data_signature']
    del checkpoint_state
    control = V.validate_checkpoint(plan['control_checkpoint'], control_cfg, 5000, states['pose_stats']['sha256'], signature, arm='T100-L')
    states.update({'control_' + k: v for k, v in control['states'].items()})
    out = V.new_output(out)
    out.mkdir(parents=True, exist_ok=False)
    require(device == 'cuda' and torch.cuda.device_count() == 1, 'one CUDA device required for production timing')
    models = {'T100-L': E.FF.load_model(plan['control_checkpoint']['path'], device, 'teacher'),
              'T100-LQ': E.FF.load_model(cp['path'], device, 'teacher')}
    for model in models.values():
        model.eval()
    torch.cuda.reset_peak_memory_stats()
    meter = Meter(models, device)
    sources, selections, totals = {}, {}, {}
    records, _ = E.pose_sources(extraction, 'train')
    records = records[:2]
    require(len(records) == 2, 'pose training sample incomplete')
    pose_files = []
    for rec in records:
        path = Path(rec['path'])
        pose = E.PT.pose_dir(Path(extraction['paths']['pose_root']), path)
        pose_files += [path / 'metadata.json', path / 'data_int16.npy', pose / E.PT.POSE_FILE, pose / E.PT.VALID_FILE, pose / E.PT.META_FILE]
    sources['pose'] = E.source_states(pose_files)
    store = E.PT.PoseStore(extraction['paths']['pose_root'])
    t = time.perf_counter()
    result = PE.extract(PoseProxy(meter), records, store, PE.EvalConfig(n_windows=10, window_s=4., seed=20261026, batch=32, device=device), log=lambda _: None)
    totals['pose_pipeline_seconds'] = time.perf_counter() - t
    require(len(result['meta']) == 20, 'pose sample must have10 windows from eachrecording')
    require(np.isfinite(result['lat']).all() and np.isfinite(result['pred']).all(), 'nonfinite pose sample output')
    selections['pose'] = {'records': records, 'windows': result['meta'], 'outputs_discarded': True}
    del result
    for dataset in tuple(E.DATASETS['p1']) + tuple(E.DATASETS['p2']):
        t = time.perf_counter()
        loader = load_train_p1 if dataset in E.DATASETS['p1'] else load_train_p2
        windows, subjects, fs, montage, selection = loader(extraction, dataset)
        totals[dataset + '_loader_seconds'] = time.perf_counter() - t
        sources[dataset] = selection['source_states']
        selection['raw_windows_sha256'] = E.fingerprint(windows, subjects, fs, montage)
        t = time.perf_counter()
        selection.update(measure_classification(meter, dataset, windows, subjects, fs, montage))
        totals[dataset + '_pipeline_seconds'] = time.perf_counter() - t
        selections[dataset] = selection
        del windows
    groups = group_ratios(meter.records)
    proposal = resource_proposal(groups, history)
    for old in sources.values():
        require(E.source_states([Path(p) for p in old]) == old, 'selected raw source changed')
    P.check_unchanged(states)
    require(P.code_identity(code) == code, 'code changed during benchmark')
    require(all(not (Path(p).parent / 'STOP').exists() for p in (cp['path'], plan['control_checkpoint']['path'])), 'STOP appeared')
    # Payloads never enter scientific caches. Timed calls exclude source loading,
    # preprocessing/transfers, CPU copies and discarded-output packaging.
    measured_all = sum(c['seconds'] for r in meter.records for c in r['calls'])
    totals['paired_forward_seconds_including_warmup'] = measured_all
    totals['pipeline_nonforward_seconds_combined'] = sum(v for k, v in totals.items() if k.endswith('_pipeline_seconds')) - measured_all
    totals['nonforward_breakdown'] = 'production path aggregate: preprocessing, transfers, CPU copies and packing; not individually instrumented'
    receipt = {'protocol': PROTOCOL, 'complete': True, 'code_sha': code, 'plan_sha256': states['benchmark_plan']['sha256'],
        'candidate_checkpoint_sha256': cp['sha256'], 'control_checkpoint_sha256': plan['control_checkpoint']['sha256'],
        'checkpoint_step': 5000, 'benchmark_context_sha256': extraction['_benchmark_context_sha256'], 'source_sha256': plan['source_sha256'], 'groups': groups, 'records': meter.records, 'selection': selections, 'sources': sources,
        'historical_jobs': history, 'resource_proposal': proposal, 'aggregation': POLICY, 'input_states': states,
        'timings': totals, 'elapsed_seconds': time.perf_counter() - started,
        'memory': {'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'two_teachers_resident': True},
        'log_weight_diagnostics': guard['log_weights'], 'no_scientific_payload_saved': True,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                    'torch_threads': torch.get_num_threads(),
                    'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}}
    P.write_json(out / 'receipt.json', receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-code-sha', required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    execute(a.plan, a.expected_code_sha, a.expected_plan_sha256, a.out)


if __name__ == '__main__':
    main()
