"""Read-only compute-node audit of B5000 and the R3c stable checkpoint.

No training, model construction, checkpoint copying or sampler iteration. Only
JSON metadata is written to a new result path; tensors are summarized or hashed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

CODE_SHA = '24139ab901c052357a46a65bc20466f2eb1e8358'
B_SHA = '815956b98d27cd2ff71f1255d0e220e2b0e08e467e7bedf93a54d0984b10783e'
B_CONFIG_SHA = 'f55fa6a5f117ad8898c20bbaf8b20f288c169a9a9a12e7bf3bfdad26fef1d157'
MANIFEST_SHA = '6d6d8167b882eb4079fcc566748028dc6de81692859f220a0246291c198574ec'
STATS_SHA = '757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e'
CSV_SHA = '0e7ad04a79a0f162385d22ec8d6f43af9b8360978ca11ef9d601bbf9594b09e6'
WORK = Path('/leonardo_work/IscrB_WearUsFM')
SCRATCH = Path('/leonardo_scratch/large/userexternal/sbenatti')
RUNS = SCRATCH / 'wearusfm_runs/runs/pilot_posa'
B = RUNS / 'PILOT-POSA-R3-B-30M-s0-20261028/checkpoint.pt'
R3C = RUNS / 'PILOT-POSA-R3C-ancore-30M-s0/checkpoint_step0019618.pt'
OUT = WORK / 'wearusfm_runs/results/pilot_posa/B19618_preflight_20261028.json'
MANIFEST = WORK / 'wearusfm_runs/results/passo6/derived_v1_1/manifest_v1_1_nested_1.json.gz'
SCALES = WORK / 'wearusfm_runs/results/passo6/loader_59623700/session_scales.json'
WINDOWS = WORK / 'wearusfm_runs/results/passo6/window_seconds_59425316.json'
STATS = SCRATCH / 'data/processed/emg2pose_pose/pose_stats.json'
CSV = WORK / 'data/raw/emg2pose_mini/emg2pose_metadata.csv'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def data_signature(config, manifest_sha, scales, windows):
    loader = dict(config['loader'])
    if loader.get('pose_root') is None:
        loader.pop('pose_root', None)
        loader.pop('pose_unprompted', None)
    return {'target_version': config['target_version'], 'readout_version': config['readout_version'],
            'manifest_sha256': manifest_sha, 'loader': loader, 'datasets': config['datasets'],
            'scales_sha256': json_sha(scales), 'window_seconds_sha256': json_sha(windows)}


def rng_metadata(value):
    import torch
    require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
            'RNG state must be a nonempty uint8 vector')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()}


def summarize_state(state, expected_step, config, signature):
    import torch
    require(int(state['step']) == expected_step, 'wrong checkpoint step')
    require(state['config'] == config, 'checkpoint/config mismatch')
    require(state.get('world_size') == 4, 'checkpoint is not four-rank DDP')
    require(state.get('stream') == {'num_workers': 6, 'segment': 1200, 'world_size': 4}, 'unexpected sampler stream')
    require(state.get('data_signature') == signature, 'checkpoint/input data signature mismatch')
    ranks = state.get('rank_rng')
    require(isinstance(ranks, list) and len(ranks) == 4, 'missing per-rank RNG states')
    rank_summary = [{'rank': i, 'keys': sorted(r), 'torch_rng': rng_metadata(r['torch_rng']),
                     'cuda_rng': rng_metadata(r['cuda_rng'])} for i, r in enumerate(ranks)]
    models = {}
    for name in ('student', 'teacher'):
        values = state[name]
        require(isinstance(values, dict) and values, 'missing model state: ' + name)
        shapes = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in values.items()}
        models[name] = {'tensor_count': len(values), 'numel': sum(v.numel() for v in values.values()),
                        'keys_and_shapes_sha256': json_sha(shapes),
                        'pose_head_keys': sorted(k for k in values if k.startswith('pose_head.'))}
    optimizer = state['optimizer']
    require(optimizer.get('state') and optimizer.get('param_groups'), 'missing optimizer state')
    counts = Counter()
    for item in optimizer['state'].values():
        require(all(k in item for k in ('step', 'exp_avg', 'exp_avg_sq')), 'incomplete AdamW moments')
        step = item['step']
        require(not isinstance(step, torch.Tensor) or step.numel() == 1, 'optimizer step is not scalar')
        counts[str(float(step))] += 1
    groups = [{**{k: v for k, v in g.items() if k != 'params'}, 'n_parameters': len(g['params'])}
              for g in optimizer['param_groups']]
    return {'keys': sorted(state), 'step': expected_step, 'config': config, 'config_canonical_sha256': json_sha(config),
            'stream': state['stream'], 'stream_sha256': json_sha(state['stream']),
            'sampler_resume': {'start_global_step': expected_step, 'seed': config['seed'],
                               'batch_per_rank': config['batch_size'] // 4,
                               'method': 'deterministic segmented replay; no mutable loader cursor saved or inspected'},
            'data_signature': signature, 'data_signature_sha256': json_sha(signature),
            'rank_rng': rank_summary, 'torch_rng': rng_metadata(state['torch_rng']),
            'top_level_cuda_rng_count': len(state.get('cuda_rng') or []),
            'world_size': state['world_size'], 'alarm_counts': list(state['alarm_counts']),
            'pose_stats_sha256': state.get('pose_stats_sha256'), 'model_states': models,
            'optimizer': {'type': 'AdamW as frozen trainer', 'state_entries': len(optimizer['state']),
                          'step_counts': dict(sorted(counts.items())), 'param_groups': groups}}


def inspect_checkpoint(path, expected_step, scales, windows, expected_sha=None):
    import torch
    before = sha(path)
    require(expected_sha is None or before == expected_sha, 'checkpoint SHA mismatch: ' + str(path))
    config_path = path.parent / 'config.json'
    config = json.loads(config_path.read_text())
    # Both paths are trusted products of this project; B additionally has its frozen digest.
    state = torch.load(path, map_location='cpu', weights_only=False)
    result = summarize_state(state, expected_step, config, data_signature(config, MANIFEST_SHA, scales, windows))
    del state
    require(sha(path) == before, 'checkpoint changed during inspection')
    result.update(path=str(path), bytes=path.stat().st_size, sha256=before,
                  config_path=str(config_path), config_file_sha256=sha(config_path))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--wrapper-sha256', required=True)
    args = parser.parse_args(argv)
    require(os.environ.get('WEARUSFM_CODE_SHA') == CODE_SHA, 'wrong frozen environment')
    require(sha(__file__) == args.source_sha256, 'source SHA mismatch')
    require(not OUT.exists() and not OUT.is_symlink(), 'output must be new')
    require(OUT.resolve().parent == (WORK / 'wearusfm_runs/results/pilot_posa').resolve(), 'unexpected output directory')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')) <= 2, 'preflight allocation exceeds two CPUs')
    for path, digest in ((MANIFEST, MANIFEST_SHA), (STATS, STATS_SHA), (CSV, CSV_SHA), (B.parent / 'config.json', B_CONFIG_SHA)):
        require(sha(path) == digest, 'input SHA mismatch: ' + str(path))
    scales = json.loads(SCALES.read_text())['scales']
    windows = json.loads(WINDOWS.read_text())['window_s']
    b = inspect_checkpoint(B, 5000, scales, windows, B_SHA)
    r3c = inspect_checkpoint(R3C, 19618, scales, windows)
    require(b['pose_stats_sha256'] == STATS_SHA and r3c['pose_stats_sha256'] is None, 'unexpected pose provenance')
    require(b['config']['model']['pose_head'] and not r3c['config']['model']['pose_head'], 'wrong pose heads')
    require(b['config']['schedule'] == r3c['config']['schedule'], 'different original schedules')
    require(b['stream'] == r3c['stream'], 'different original streams')
    with gzip.open(MANIFEST, 'rt') as f:
        manifest = json.load(f)
    columns = manifest['columns']
    rows = [dict(zip(columns, row)) for row in manifest['rows']]
    training = {str(r['subject']).removeprefix('u') for r in rows if r['dataset'] == 'emg2pose' and r['split'] == 'pretraining'}
    with CSV.open(newline='') as f:
        validation = {r['user'].removeprefix('u') for r in csv.DictReader(f) if r['split'] == 'val'}
    require(len(training) == 158 and len(validation) == 15 and not training & validation, 'split identity mismatch')
    fixed_hashes = {name: sha(path.parent / 'fixed_eval_set.json') for name, path in [('B5000', B), ('R3c19618', R3C)]}
    fixed_b, fixed_r3c = [json.loads((path.parent / 'fixed_eval_set.json').read_text()) for path in (B, R3C)]
    require(fixed_b['windows'] == fixed_r3c['windows'] and fixed_b['seed'] == fixed_r3c['seed'],
            'fixed-evaluation window selection differs')
    result = {'schema_version': 1, 'completed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Read-only metadata audit; no proof of numerical resume equality and no training.',
              'B5000': b, 'R3c19618': r3c, 'fixed_eval_set_hashes': fixed_hashes,
              'fixed_eval_sets_identical': len(set(fixed_hashes.values())) == 1,
              'fixed_eval_windows_identical': True, 'fixed_eval_seeds_identical': True,
              'fixed_eval_note': 'Whole batch hashes include pose targets when present, so equality between B and R3c is not required.',
              'input_files': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (MANIFEST, SCALES, WINDOWS, STATS, CSV)},
              'manifest': {'n_rows': len(rows), 'dataset_split_counts': dict(Counter(r['dataset'] + '/' + r['split'] for r in rows)),
                           'params': manifest.get('params')},
              'pose_split': {'pretraining_users': len(training), 'validation_users': len(validation), 'overlap': len(training & validation)},
              'provenance': {'source_sha256': args.source_sha256, 'wrapper_sha256': args.wrapper_sha256,
                             'environment_code_sha': CODE_SHA, 'torch_version': torch.__version__,
                             'torch_threads': torch.get_num_threads(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'completed': True, 'output': str(OUT), 'B_step': b['step'], 'R3c_step': r3c['step'],
                      'B_sha256': b['sha256'], 'R3c_sha256': r3c['sha256'],
                      'fixed_eval_sets_identical': result['fixed_eval_sets_identical']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
