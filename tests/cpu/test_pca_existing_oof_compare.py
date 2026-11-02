"""Prediction-only checks: pins, exact pairing and shared subject resampling."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'pca_existing_oof_compare.py'
spec = importlib.util.spec_from_file_location('pca_existing_oof_compare', SCRIPT)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def bacc(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    oof_dir, shard_dir = tmp_path / 'oof', tmp_path / 'shards'
    oof_dir.mkdir()
    shard_dir.mkdir()
    reports, arrays, shards = {}, {}, {}
    complete = {'cells': {}}
    subjects = np.repeat([f'u{i}' for i in range(8)], 4)
    rows = np.arange(len(subjects), dtype=np.int64)
    folds = []
    for i in range(4):
        held = [f'u{2*i}', f'u{2*i+1}']
        other = [f'u{j}' for j in range(8) if f'u{j}' not in held]
        folds.append({'fold': i, 'train': other[1:], 'val': other[:1], 'held': held, 'n_held': 8})
    fold_of = np.asarray([int(s[1:]) // 2 for s in subjects], dtype=np.int64)
    for model in M.MODELS:
        ckpt = ('a' if model == 'B' else 'b') * 64
        for task, weights in M.WEIGHTS.items():
            key = (model, task)
            report = {'complete': True, 'phase': 'B', 'encoder': model, 'task': task,
                      'weights': weights, 'readout': {'layer': 'pool', 'k_requested': 0}, 'max_iter': 12000,
                      'checkpoint_sha256': ckpt, 'solver_recipe': {'c_grid': [.001, .01, .1, 1., 10., 100.]},
                      'datasets': {}, 'cache_pins': {},
                      'oof': f'/remote/{model}_{task}_pool.oof.npz'}
            arrays[key] = {'datasets': np.asarray(list(weights))}
            for ds in weights:
                y = np.tile(['open', 'closed', 'open', 'closed'] if ds == 'epn612' else [1, 2, 1, 2], 8)
                # Vary accuracy by subject; equal B/R3c effects expose independent-draw bugs.
                old_pred = y.copy()
                change = np.asarray([int(s[1:]) % 2 == 0 for s in subjects])
                other = np.where(y == y[0], y[1], y[0])
                old_pred[change] = other[change]
                for k, value in {'row': rows, 'y': y, 's': subjects, 'pred': old_pred, 'fold': fold_of}.items():
                    arrays[key][f'{ds}_{k}'] = value.copy()
                score = bacc(y, old_pred)
                report['datasets'][ds] = {'subjects': [f'u{i}' for i in range(8)], 'folds': copy.deepcopy(folds),
                                         'input_sha256': 'c' * 64, 'n_oof_windows': len(y), 's': score}
                report['cache_pins'][ds] = {'sha256': 'd' * 64}
                sfolds = [{'fold': f['fold'], 'roles': {r: f[r] for r in ('train', 'val', 'held')},
                           'logistic': {'classes': np.unique(y).tolist()}} for f in copy.deepcopy(folds)]
                shard = {'status': 'completed', 'model': model, 'dataset': ds, 'task': task, 'view': 'pool',
                         'checkpoint_sha256': ckpt, 'input_sha256': 'c' * 64,
                         'recipe_sha256': 'e' * 64, 'recipe': {'features': {'k': 128},
                                                           'logistic': {'c_grid': report['solver_recipe']['c_grid']}},
                         'provenance': {'sha256': 'd' * 64}, 'folds': sfolds, 'rows': rows.tolist(),
                         'y': y.tolist(), 'subjects': subjects.tolist(), 'predictions': {'logistic': {'0': y.tolist()}}}
                shards[(model, ds)] = shard
                complete['cells'][f'{model}/{ds}/pool'] = {'bacc_oof': {'logistic': {'mean': 1.}}}
            reports[key] = report
    state = SimpleNamespace(root=tmp_path, manifest=tmp_path / 'manifest.json', oof_dir=oof_dir,
                            shard_dir=shard_dir, reports=reports, arrays=arrays, shards=shards,
                            complete=complete, git={}, manifest_data=None)

    def save():
        sources, missing, refs = [], [], []
        git = {}
        commit = 'f' * 40
        for key, report in reports.items():
            model, task = key
            for ds in M.WEIGHTS[task]:
                report['datasets'][ds]['oof_rows_sha256'] = M.fingerprint(arrays[key][f'{ds}_row'])
            buf = io.BytesIO()
            np.savez(buf, **arrays[key])
            raw = buf.getvalue()
            (oof_dir / f'{model}_{task}_pool.oof.npz').write_bytes(raw)
            report['oof_sha256'] = M.digest(raw)
            missing.append({'model': model, 'task': task, 'remote_path_from_report': report['oof'],
                            'sha256_from_report': report['oof_sha256']})
            git[f'{M.REPORT_BASE}/{model}_{task}_pool.json'] = json.dumps(report).encode()
        git[M.PSD3_REPORT] = json.dumps(complete).encode()
        for path, raw in git.items():
            sources.append({'kind': 'git_blob', 'ref': commit, 'path': path, 'sha256': M.digest(raw), 'bytes': len(raw)})
        for (model, ds), shard in shards.items():
            shard['roles_sha256'] = M.fingerprint([f['roles'] for f in shard['folds']])
            shard['provenance']['oof_data_sha256'] = M.fingerprint(shard['rows'], shard['y'], shard['subjects'])
            path = shard_dir / f'{model}_{ds}_pool.json'
            raw = json.dumps(shard).encode()
            path.write_bytes(raw)
            sources.append({'kind': 'local_file', 'path': str(path), 'sha256': M.digest(raw), 'bytes': len(raw)})
            task = shard['task']
            refs.append({'model': model, 'dataset': ds, 'pool768_bacc': reports[(model, task)]['datasets'][ds]['s'],
                         'pool128_bacc': complete['cells'][f'{model}/{ds}/pool']['bacc_oof']['logistic']['mean']})
        manifest = {'protocol': 'exploratory-existing-pool-pipelines-20261102-v1', 'source_commit': commit,
                    'sources': sources, 'local_oof_inventory': {'missing': missing}, 'dataset_scores': refs,
                    'actual_pipeline_differences': ['PCA, arithmetic and threads differ']}
        state.git, state.manifest_data = git, manifest
        state.manifest.write_text(json.dumps(manifest))

    state.save = save
    monkeypatch.setattr(M, '_git_blob', lambda repo, source: state.git[source['path']])
    save()
    return state


def load(bundle):
    return M.load_and_validate(bundle.manifest, bundle.oof_dir, bundle.shard_dir, bundle.root)


def test_native_string_numeric_labels_and_pins(bundle):
    grid, provenance = load(bundle)
    assert grid['epn612']['B/pool768']['label_family'] == 'text'
    assert grid['ninapro_db2']['B/pool128']['label_family'] == 'numeric'
    assert len(provenance['oof']) == 4 and len(provenance['shards']) == 10
    result = M.compare_cells(grid, n_boot=100)
    for task in M.WEIGHTS:
        assert result['tasks'][task]['cells']['B/pool768']['bacc'] == .5
        assert result['tasks'][task]['contrasts']['pool128_minus_pool768/B']['delta_pp'] == pytest.approx(50)


def test_row_permutations_are_paired_after_sorting(bundle):
    reference, _ = load(bundle)
    for arrays in bundle.arrays.values():
        for key, value in list(arrays.items()):
            if key != 'datasets':
                arrays[key] = value[::-1]
    for shard in bundle.shards.values():
        for key in ('rows', 'y', 'subjects'):
            shard[key] = shard[key][5:] + shard[key][:5]
        pred = shard['predictions']['logistic']['0']
        shard['predictions']['logistic']['0'] = pred[5:] + pred[:5]
    bundle.save()
    current, _ = load(bundle)
    assert M.compare_cells(reference, n_boot=20) == M.compare_cells(current, n_boot=20)


def test_utf8_npz_labels_are_mapped_to_native_json_labels(bundle):
    for key, arrays in bundle.arrays.items():
        if key[1] == 'p2':
            arrays['epn612_y'] = arrays['epn612_y'].astype('S')
            arrays['epn612_pred'] = arrays['epn612_pred'].astype('S')
    bundle.save()
    grid, _ = load(bundle)
    assert grid['epn612']['B/pool768']['y'].dtype.kind == 'U'


@pytest.mark.parametrize('mutation,match', [
    ('duplicate_row', 'duplicate row'), ('missing_user', 'missing or additional OOF subject'),
    ('duplicate_role_user', 'duplicate subject'), ('wrong_fold', 'row/subject/fold mismatch'),
    ('target', 'paired y mismatch'), ('new_subject', 'missing or additional PSD3 subject'),
    ('nonfinite', 'nonfinite numeric'), ('mixed_labels', 'mixed or invalid labels'),
])
def test_identity_failures_are_rejected(bundle, mutation, match):
    ds = 'ninapro_db2'
    old = bundle.arrays[('B', 'p1')]
    shard = bundle.shards[('B', ds)]
    if mutation == 'duplicate_row':
        old[f'{ds}_row'][0] = old[f'{ds}_row'][1]
    elif mutation == 'missing_user':
        old[f'{ds}_s'][old[f'{ds}_s'] == 'u0'] = 'u1'
    elif mutation == 'duplicate_role_user':
        bundle.reports[('B', 'p1')]['datasets'][ds]['folds'][0]['train'].append('u3')
    elif mutation == 'wrong_fold':
        old[f'{ds}_fold'][0] = 1
    elif mutation == 'target':
        shard['y'][0] = 2
    elif mutation == 'new_subject':
        shard['subjects'][0] = 'outsider'
    elif mutation == 'nonfinite':
        old[f'{ds}_pred'] = old[f'{ds}_pred'].astype(float)
        old[f'{ds}_pred'][0] = np.nan
    elif mutation == 'mixed_labels':
        shard['y'][0] = '1'
    bundle.save()
    with pytest.raises(ValueError, match=match):
        load(bundle)


@pytest.mark.parametrize('which', ['oof', 'shard', 'report'])
def test_hash_pin_mismatch_fails_before_analysis(bundle, which):
    if which == 'oof':
        p = bundle.oof_dir / 'B_p1_pool.oof.npz'
        p.write_bytes(p.read_bytes() + b'extra')
    elif which == 'shard':
        p = bundle.shard_dir / 'B_ninapro_db2_pool.json'
        p.write_bytes(p.read_bytes() + b' ')
    else:
        p = f'{M.REPORT_BASE}/B_p1_pool.json'
        bundle.git[p] += b' '
    with pytest.raises(ValueError, match='SHA-256 pin mismatch'):
        load(bundle)


def test_missing_oof_file_does_not_create_output(bundle):
    (bundle.oof_dir / 'B_p1_pool.oof.npz').unlink()
    output = bundle.root / 'absent.json'
    with pytest.raises(FileNotFoundError):
        M.main(['--manifest', str(bundle.manifest), '--oof-dir', str(bundle.oof_dir),
                '--shard-dir', str(bundle.shard_dir), '--output', str(output)])
    assert not output.exists()


def test_same_subject_weights_across_all_cells_and_batches(bundle):
    grid, _ = load(bundle)
    a = M.compare_cells(grid, n_boot=250, seed=17, batch_size=1)
    b = M.compare_cells(grid, n_boot=250, seed=17, batch_size=73)
    assert a == b
    for block in list(a['datasets'].values()) + list(a['tasks'].values()):
        # Identical checkpoint-specific patterns must cancel on every shared draw.
        for name in ('B_minus_R3c/pool768', 'B_minus_R3c/pool128', 'difference_of_B_minus_R3c'):
            assert block['contrasts'][name]['delta_pp'] == 0
            assert block['contrasts'][name]['ci95_pp'] == [0., 0.]
        assert block['contrasts']['pool128_minus_pool768/B']['ci95_pp'] == block['contrasts']['pool128_minus_pool768/R3c']['ci95_pp']


def test_fixed_aggregate_weights_on_points(bundle):
    grid, _ = load(bundle)
    for ds in ('ninapro_db3', 'uci_emg'):
        cell = grid[ds]['B/pool128']
        cell['pred'] = np.where(cell['y'] == 1, 2, 1)
        cell.pop('reported_bacc')
    result = M.compare_cells(grid, n_boot=50)
    assert result['tasks']['p1']['cells']['B/pool128']['bacc'] == pytest.approx(10 / 13)
    assert result['tasks']['p2']['cells']['B/pool128']['bacc'] == .5


def test_single_class_bootstrap_draws_are_retained():
    grid = {}
    for ds in M.DATASETS:
        grid[ds] = {}
        for name in M.CELLS:
            grid[ds][name] = {'rows': np.array([0, 1]), 'y': np.array([0, 1]),
                              'subjects': np.array(['one', 'two']), 'fold': np.array([0, 1]),
                              'pred': np.array([0, 1]), 'label_family': 'numeric'}
    result = M.compare_cells(grid, n_boot=100, seed=4)
    assert all(d['bootstrap_single_class_draws'] > 0 for d in result['datasets'].values())
    assert result['tasks']['p1']['cells']['B/pool768']['ci95'] == [1., 1.]


def test_incomplete_grid_and_changed_report_metric_fail(bundle):
    grid, _ = load(bundle)
    changed = copy.deepcopy(grid)
    changed['ninapro_db2'].pop('B/pool128')
    with pytest.raises(ValueError, match='incomplete model/pipeline'):
        M.compare_cells(changed, n_boot=5)
    grid['ninapro_db2']['B/pool768']['reported_bacc'] = .123
    with pytest.raises(ValueError, match='differs from pinned report'):
        M.compare_cells(grid, n_boot=5)


def test_cli_fixed_protocol_and_no_overwrite(bundle):
    output = bundle.root / 'comparison.json'
    args = ['--manifest', str(bundle.manifest), '--oof-dir', str(bundle.oof_dir),
            '--shard-dir', str(bundle.shard_dir), '--output', str(output)]
    assert M.main(args) == 0
    result = json.loads(output.read_text())
    assert result['n_boot'] == 2000 and result['seed'] == 20261102
    assert result['descriptive_only'] is True
    assert 'NOT a causal' in result['comparison']
    assert len(result['provenance']['oof']) == 4
    with pytest.raises(ValueError, match='already exists'):
        M.main(args)
