"""Immutable evidence and execution guards for the frozen slot-contrast diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import pose_b19618_eval_guard as G
import pose_dynamics as D
import pose_slot_features as F
import pose_temporal_readout as T

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = T.ARCHIVE
PROTOCOL = 'pose-slot-readout-20261102-v1'
OUTPUT_ROOT = '/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pose_slots_20261102'
PROJECTIONS = 'results/infra/pose_slots_20261102/projections.npz'
MODELS = ('B', 'R3c')
SPLITS = ('train', 'val')
MEAN_GATE = {'normalized_patch_rms_max': 1e-3, 'normalized_max_abs': 1e-2,
             'denominator_relative_max': 1e-3, 'eps': 1e-5,
             'support': 'every real patch; never filtered by target mask'}
require = D.require
file_state = T.file_state


def file_sha256(path):
    return D.file_sha256(path)


def source_hashes():
    names = ['scripts/pose_slot_plan.py', 'scripts/pose_slot_features.py',
             'scripts/pose_slot_extract.py', 'scripts/pose_slot_readout.py',
             'scripts/pose_temporal_readout.py', 'scripts/pose_e3_b.py',
             'scripts/pose_dynamics.py', 'scripts/pose_b19618_eval_guard.py',
             'scripts/slurm/pose_slots.sbatch', 'scripts/slurm/lib_code_dir.sh',
             PROJECTIONS]
    # The new extractor imports model/loader code directly. Freeze that code too.
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src/wearusfm').rglob('*.py'))]
    for model in MODELS:
        base = ARCHIVE/'extraction_metadata'/model/'pose'
        names += [str((base/name).relative_to(ROOT)) for name in
                  ('extraction_receipt.json', 'train_subset.npz.provenance.json', 'val.npz.provenance.json')]
    return {name: file_sha256(ROOT/name) for name in sorted(set(names))}


def make_plan():
    old = T.make_plan()
    models = {}
    for model in MODELS:
        receipt = json.loads((ARCHIVE/'extraction_metadata'/model/'pose/extraction_receipt.json').read_text())
        models[model] = {
            'checkpoint': receipt['checkpoint'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
            'config_path': str(Path(receipt['checkpoint']).parent/'config.json'),
            'config_sha256': receipt['config_file_sha256'],
            'receipt': receipt['receipt'], 'receipt_sha256': receipt['receipt_sha256'],
        }
    return {
        'protocol': PROTOCOL, 'recipe': F.recipe(), 'mean_parity_gate': MEAN_GATE,
        'output_root': OUTPUT_ROOT, 'models': models, 'caches': old['caches'],
        'fold_users': old['fold_users'], 'dynamic_threshold': T.THRESHOLD,
        'stats_sha256': G.STATS_SHA, 'sources': source_hashes(),
        'projections': {'path': PROJECTIONS, 'sha256': file_sha256(ROOT/PROJECTIONS)},
        'paths': {
            'csv': '/leonardo_work/IscrB_WearUsFM/data/raw/emg2pose_mini/emg2pose_metadata.csv',
            'manifest': '/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/derived_v1_1/manifest_v1_1_nested_1.json.gz',
            'processed_root': '/leonardo_scratch/large/userexternal/sbenatti/data/processed/emg2pose',
            'pose_root': '/leonardo_scratch/large/userexternal/sbenatti/data/processed/emg2pose_pose',
        },
        'input_hashes': {'csv': G.CSV_SHA, 'manifest': G.MANIFEST_SHA, 'stats': G.STATS_SHA},
        'preflight_recording_indices': [0, 1, 2, 3],
        'bootstrap': {'replicates': 2000, 'seed': 20261102, 'unit': 'user'},
        'support': 'same central support as temporal diagnostic: 4 <= t and t+4 < n_patches',
        'scope': 'Exploratory 7 fixed slot contrasts, not full slots; frozen encoders; no online or endpoint verdict.',
    }


def validate_plan(plan):
    require(plan == make_plan(), 'plan differs from frozen slot protocol/evidence/sources')


def runtime_identity(expected_code_sha):
    result = T.runtime_identity(expected_code_sha)
    result['sources'] = source_hashes()
    result['projection_recipe'] = F.recipe()
    result['accelerator'] = {
        'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    return result


def validate_reference(plan, model, split):
    require(model in MODELS and split in SPLITS, 'invalid model/split')
    return T.validate_artifact(plan['caches'][model][split], model, split)


def validate_input_files(plan):
    paths = plan['paths']
    targets = {'csv': Path(paths['csv']), 'manifest': Path(paths['manifest']),
               'stats': Path(paths['pose_root'])/'pose_stats.json'}
    states = {}
    for key, path in targets.items():
        require(file_sha256(path) == plan['input_hashes'][key], 'input hash mismatch: '+key)
        states[str(path)] = file_state(path)
    return states


def validate_checkpoint(plan, model):
    pin = plan['models'][model]
    paths = [Path(pin[k]) for k in ('checkpoint', 'config_path', 'receipt')]
    states = {str(p): file_state(p) for p in paths}
    for name, digest_key in (('checkpoint', 'checkpoint_sha256'), ('config_path', 'config_sha256'),
                             ('receipt', 'receipt_sha256')):
        require(file_sha256(pin[name]) == pin[digest_key], 'checkpoint evidence hash mismatch: '+name)
    # Original guarded checker verifies trusted pickle only after its receipt's SHA matches.
    receipt = json.loads(Path(pin['receipt']).read_text())
    ckpt, config = G.check_checkpoint(Path(pin['checkpoint']), model, receipt, Path(pin['receipt']))
    require(ckpt == pin['checkpoint_sha256'] and config == pin['config_sha256'], 'checkpoint receipt differs')
    require(states == {str(p): file_state(p) for p in paths}, 'checkpoint evidence changed during validation')
    return {'model': model, **pin, 'states': states,
            'config': json.loads(Path(pin['config_path']).read_text())}


def fresh_output(path, plan):
    path, root = Path(path), Path(plan['output_root'])
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), 'output must be new and absolute')
    require(path == root or root in path.parents, 'output outside approved slot campaign')
    require(path.resolve() == path, 'noncanonical/symlinked output')


def validate_mean_parity(receipt, n):
    require(receipt.get('mean_parity_gate') == MEAN_GATE, 'residual parity gate mismatch')
    parity = receipt.get('mean_parity', {})
    require(parity.get('passed') is True, 'residual parity not passed')
    for key in ('normalized_patch_rms_max', 'normalized_max_abs', 'denominator_relative_max'):
        value = parity.get(key)
        require(isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= MEAN_GATE[key],
                'residual parity outside gate: '+key)
    require(parity.get('n_elements') == n * 160 * 384, 'residual parity support incomplete')


def validate_preflight(plan, expected_code_sha, expected_plan_sha256):
    """A failed or partial preflight cannot authorize full extraction."""
    root = Path(plan['output_root'])/'preflight'
    parent_path = root/'receipt.json'
    paths = [parent_path] + [root/m/'train/receipt.json' for m in MODELS]
    states = {str(p): file_state(p) for p in paths}
    parent = json.loads(parent_path.read_text())
    require(parent.get('complete') is True and parent.get('stage') == 'preflight', 'incomplete preflight')
    require(set(parent.get('outputs', {})) == {f'{m}/train' for m in MODELS}, 'preflight models differ')
    for receipt in [parent] + [json.loads(p.read_text()) for p in paths[1:]]:
        require(receipt.get('protocol') == PROTOCOL and receipt.get('code_sha') == expected_code_sha
                and receipt.get('plan_sha256') == expected_plan_sha256 and receipt.get('complete') is True,
                'preflight identity mismatch')
    for model, rp in zip(MODELS, paths[1:]):
        receipt = json.loads(rp.read_text())
        entry = parent['outputs'][f'{model}/train']
        require(entry['receipt'] == str(rp) and entry['receipt_sha256'] == file_sha256(rp),
                'preflight split receipt mismatch')
        require(receipt.get('model') == model and receipt.get('split') == 'train'
                and receipt.get('which') == 'student' and receipt.get('checkpoint_step') == 19618
                and receipt.get('checkpoint_sha256') == plan['models'][model]['checkpoint_sha256']
                and receipt.get('reference_cache_sha256') == plan['caches'][model]['train']['sha256'],
                'preflight source identity mismatch')
        require(receipt.get('projection_record') == F.recipe()
                and receipt.get('selected_original_recording_indices') == plan['preflight_recording_indices'],
                'preflight selection/projection mismatch')
        n = receipt.get('n_windows')
        require(type(n) is int and n > 0 and entry['n_windows'] == n, 'invalid preflight support')
        validate_mean_parity(receipt, n)
    require(states == {str(p): file_state(p) for p in paths}, 'preflight changed during validation')
    return {'path': str(parent_path), 'sha256': file_sha256(parent_path), 'states': states}


def validate_residual(plan, model, split, *, expected_code_sha=None, expected_plan_sha256=None):
    """Only complete full-split extractions can enter readout; preflight is in another tree."""
    require(model in MODELS and split in SPLITS, 'invalid residual model/split')
    directory = Path(plan['output_root'])/model/split
    path, rp = directory/'residual.npy', directory/'receipt.json'
    parent_path = directory.parent/'receipt.json'
    states = {str(p): file_state(p) for p in (path, rp, parent_path)}
    receipt = json.loads(rp.read_text())
    parent = json.loads(parent_path.read_text())
    require(parent.get('complete') is True and parent.get('stage') == 'extract'
            and parent.get('protocol') == PROTOCOL, 'incomplete model extraction')
    require(all(parent.get(k) == receipt.get(k) and parent.get(k) is not None
                for k in ('code_sha', 'plan_sha256')), 'model/split extraction identity mismatch')
    require(set(parent.get('outputs', {})) == {f'{model}/{s}' for s in SPLITS},
            'model extraction missing full splits')
    parent_output = parent['outputs'][f'{model}/{split}']
    require(parent_output.get('receipt') == str(rp)
            and parent_output.get('receipt_sha256') == file_sha256(rp)
            and parent_output.get('n_windows') == receipt.get('n_windows'),
            'model/split extraction receipt mismatch')
    require(receipt.get('complete') is True, 'incomplete residual extraction')
    for key, value in {'protocol': PROTOCOL, 'model': model, 'split': split, 'which': 'student',
                       'checkpoint_sha256': plan['models'][model]['checkpoint_sha256'],
                       'reference_cache_sha256': plan['caches'][model][split]['sha256']}.items():
        require(receipt.get(key) == value, 'residual identity mismatch: '+key)
    if expected_code_sha is not None:
        require(receipt.get('code_sha') == expected_code_sha, 'residual code mismatch')
    if expected_plan_sha256 is not None:
        require(receipt.get('plan_sha256') == expected_plan_sha256, 'residual plan mismatch')
    require(receipt.get('projection_record') == F.recipe(), 'residual projection mismatch')
    require(receipt.get('checkpoint_step') == 19618, 'residual checkpoint step mismatch')
    n = plan['caches'][model][split]['n_windows']
    require(receipt.get('reference_row_indices') == list(range(n)), 'residual rows incomplete or reordered')
    validate_mean_parity(receipt, n)
    info = receipt['residual']
    require(info['path'] == str(path) and info['shape'] == [n, 160, 336] and info['dtype'] == '<f4',
            'residual shape/path/dtype mismatch')
    require(path.stat().st_size == info['bytes'] and file_sha256(path) == info['sha256'], 'residual SHA/size mismatch')
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    require(arr.shape == (n, 160, 336) and arr.dtype == np.float32, 'invalid residual NPY header')
    require(receipt.get('n_windows') == n, 'incomplete residual windows')
    with np.load(plan['caches'][model][split]['path'], allow_pickle=False) as reference:
        metadata = json.loads(str(reference['meta_json']))
    require(receipt.get('metadata_sha256') == T.json_hash(metadata), 'residual window ordering mismatch')
    require(states == {str(p): file_state(p) for p in (path, rp, parent_path)}, 'residual changed during validation')
    return {'path': str(path), 'receipt': receipt, 'states': states, 'receipt_sha256': file_sha256(rp)}
