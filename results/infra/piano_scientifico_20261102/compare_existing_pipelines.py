#!/usr/bin/env python3
"""Read existing reports only: no fitting, bootstrap, network access or job submission."""
import argparse
import hashlib
import json
import pathlib
import subprocess

WEIGHTS = {
    'p1': {'ninapro_db2': 8 / 13, 'ninapro_db3': 3 / 13, 'ninapro_db6': 2 / 13},
    'p2': {'epn612': .5, 'uci_emg': .5},
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3])
    parser.add_argument('--ref', default='f69741dad4f644350aa1230c41df8ca7fc262208')
    parser.add_argument('--local-root', type=pathlib.Path, default=pathlib.Path.home() / 'wearusfm_local')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path(__file__).with_name('comparison.json'))
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.ref + '^{commit}'], cwd=args.repo, text=True).strip()
    sources = []

    def git_bytes(path):
        raw = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=args.repo)
        sources.append({'kind': 'git_blob', 'ref': commit, 'path': path, 'bytes': len(raw), 'sha256': sha(raw)})
        return raw

    def local_json(path):
        raw = path.read_bytes()
        sources.append({'kind': 'local_file', 'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)})
        return json.loads(raw)

    complete = json.loads(git_bytes('results/infra/psd3_20261029/report_completo.json'))
    records, aggregates, missing_oof, costs = [], [], [], []
    checks = {'role_arrays_checked': 0, 'folds_checked': 0, 'cache_hashes_checked': 0,
              'input_hashes_checked': 0, 'checkpoint_hashes_checked': 0, 'oof_counts_checked': 0,
              'same_c_grids_checked': 0, 'same_estimator_params_checked': 0,
              'same_dropped_window_counts_checked': 0, 'same_library_versions_checked': 0}
    observed = {'original_fit_seconds': 0., 'original_report_elapsed_seconds': 0.,
                'original_fit_count': 0, 'psd3_pool_logistic_seconds': 0., 'psd3_pool_pca_seconds': 0.}
    datasets = {}
    for model in ('B', 'R3c'):
        for task, weights in WEIGHTS.items():
            base = f'results/infra/pilot_posa/sonde_convergenza_20261029/B/{model}_{task}_pool.json'
            old_raw = git_bytes(base)
            old = json.loads(old_raw)
            local_old = args.local_root / f'sonde_convergenza_20261029/B/{model}_{task}_pool.json'
            assert local_old.read_bytes() == old_raw, ('local/git report mismatch', model, task)
            audits = [json.loads(line) for line in git_bytes(base + '.audit.jsonl').splitlines() if line]
            assert len(audits) == len(weights) * 4 * 7
            assert all(a['dtype'] == 'float32' and a['n_features'] == 768 for a in audits)
            assert all(a['max_iter'] == 12000 for a in audits)
            seconds = sum(a['elapsed_s'] for a in audits)
            observed['original_fit_seconds'] += seconds
            observed['original_report_elapsed_seconds'] += old['elapsed_s']
            observed['original_fit_count'] += len(audits)
            costs.append({'model': model, 'task': task, 'fit_count': len(audits),
                          'fit_seconds': seconds, 'report_elapsed_seconds': old['elapsed_s'],
                          'allocated_cores_original': 8,
                          'fit_seconds_times_allocated_cores_div_3600': seconds * 8 / 3600})
            missing_oof.append({'model': model, 'task': task, 'remote_path_from_report': old['oof'],
                                'sha256_from_report': old['oof_sha256'], 'local_copy_found': False})
            old_sum = new_sum = 0.
            for ds, weight in weights.items():
                shard_path = args.local_root / f'psd3_20261029/shards/{model}_{ds}_pool.json'
                new = local_json(shard_path)
                cell = complete['cells'][f'{model}/{ds}/pool']
                od = old['datasets'][ds]
                assert new['status'] == 'completed' and new['view'] == 'pool'
                assert old['cache_pins'][ds]['sha256'] == new['provenance']['sha256']
                checks['cache_hashes_checked'] += 1
                assert od['input_sha256'] == new['input_sha256']
                checks['input_hashes_checked'] += 1
                assert old['checkpoint_sha256'] == new['checkpoint_sha256']
                checks['checkpoint_hashes_checked'] += 1
                assert od['n_oof_windows'] == cell['n_oof'] == len(new['rows'])
                checks['oof_counts_checked'] += 1
                assert old['solver_recipe']['c_grid'] == new['recipe']['logistic']['c_grid']
                checks['same_c_grids_checked'] += 1
                for lib in ('python', 'numpy', 'scipy', 'sklearn'):
                    assert old['numeric_context']['versions'][lib] == new['runtime']['versions'][lib]
                    checks['same_library_versions_checked'] += 1
                for of, nf in zip(od['folds'], new['folds']):
                    assert of['fold'] == nf['fold']
                    checks['folds_checked'] += 1
                    for role in ('train', 'val', 'held'):
                        assert of[role] == nf['roles'][role]
                        checks['role_arrays_checked'] += 1
                    for role in ('held', 'val'):
                        assert of[f'dropped_{role}_windows'] == nf['dropped_windows'][role]
                        checks['same_dropped_window_counts_checked'] += 1
                    for fit in nf['logistic']['audit']:
                        params = dict(fit['parameters'])
                        params.pop('C')
                        assert params == old['solver_recipe']['params']
                        checks['same_estimator_params_checked'] += 1
                old_score, new_score = od['s'], cell['bacc_oof']['logistic']['mean']
                old_sum += weight * old_score
                new_sum += weight * new_score
                records.append({'model': model, 'task': task, 'dataset': ds,
                                'pool768_bacc': old_score, 'pool128_bacc': new_score,
                                'delta_pool128_minus_pool768_pp': 100 * (new_score - old_score),
                                'n_oof': od['n_oof_windows'], 'weight': weight,
                                'cache_sha256': new['provenance']['sha256'],
                                'input_sha256': new['input_sha256'], 'checkpoint_sha256': new['checkpoint_sha256'],
                                'original_oof_rows_sha256': od['oof_rows_sha256'],
                                'psd3_ordered_rows_sha256': new['provenance']['ordered_rows_sha256'],
                                'psd3_recipe_sha256': new['recipe_sha256']})
                observed['psd3_pool_logistic_seconds'] += cell['totals']['logistic_s']
                observed['psd3_pool_pca_seconds'] += cell['totals']['pca_s']
            assert abs(old_sum - old['S']) < 1e-12
            aggregates.append({'model': model, 'task': task, 'pool768_bacc': old_sum,
                               'pool128_bacc': new_sum,
                               'delta_pool128_minus_pool768_pp': 100 * (new_sum - old_sum)})
    all_npz = list(args.local_root.rglob('*.npz'))
    names = {pathlib.Path(d['remote_path_from_report']).name for d in missing_oof}
    found = [str(p) for p in all_npz if p.name in names or 'sonde_convergenza_20261029' in str(p)]
    if found:
        raise RuntimeError('OOF inventory changed; inspect these before claiming unavailable: ' + repr(found))
    contrasts = []
    for task in WEIGHTS:
        cells = {a['model']: a for a in aggregates if a['task'] == task}
        old_delta = 100 * (cells['B']['pool768_bacc'] - cells['R3c']['pool768_bacc'])
        new_delta = 100 * (cells['B']['pool128_bacc'] - cells['R3c']['pool128_bacc'])
        contrasts.append({'task': task, 'B_minus_R3c_pool768_pp': old_delta,
                          'B_minus_R3c_pool128_pp': new_delta,
                          'difference_between_pipeline_contrasts_pp': new_delta - old_delta})
    observed['original_fit_core_hours_at_8_cores'] = observed['original_fit_seconds'] * 8 / 3600
    observed['original_report_core_hours_at_8_cores'] = observed['original_report_elapsed_seconds'] * 8 / 3600
    observed['psd3_pool_logistic_plus_pca_core_hours_at_4_cores'] = 4 * (observed['psd3_pool_logistic_seconds'] + observed['psd3_pool_pca_seconds']) / 3600
    report = {
        'protocol': 'exploratory-existing-pool-pipelines-20261102-v1',
        'purpose': 'Confronto esplorativo dei punti stimati da pipeline esistenti; non una ablazione causale della sola PCA.',
        'operation': 'Solo lettura di report e metadata; nessun fit, bootstrap, job, GPU o accesso remoto.',
        'source_commit': commit,
        'delta_convention': 'Punti percentuali: PSD3 pool/logistic PCA128 meno sonde fase B pool/logistic768.',
        'dataset_scores': records, 'aggregate_scores': aggregates, 'B_minus_R3c': contrasts,
        'pairing_checks': dict(checks, all_passed=True,
                              limitation='Identita cache/fold/numero OOF verificata; pairing riga/target/soggetto delle due predizioni non verificabile senza OOF fase B.'),
        'identical_recipe_components': {
            'c_grid': [.001, .01, .1, 1., 10., 100.], 'folds': 4, 'fold_seed': 20261015,
            'probe_seed': 0, 'max_iter': 12000, 'tol': .0001, 'solver': 'lbfgs',
            'class_weight': 'balanced', 'penalty': 'l2',
            'selection': 'Massima BAcc validation; pareggio al C minore; nuovo fit sul solo train.',
            'standardization': 'StandardScaler stimato sul solo train; nessuna seconda scala dopo PCA.',
            'versions': old['numeric_context']['versions']},
        'actual_pipeline_differences': [
            {'component': 'feature representation', 'original': '768 colonne standardizzate senza PCA',
             'psd3': 'PCA randomized train-only a 128 componenti; whiten=False, seed 20261029, iterated_power 7, oversamples 10'},
            {'component': 'numeric preprocessing', 'original': 'Input/uscita dello scaler float32',
             'psd3': 'Input/scaler/PCA float64, coordinate finali float32'},
            {'component': 'thread limits', 'original': 8, 'psd3': 4},
            {'component': 'implementation', 'original_analysis_sha': old['analysis_code_sha'],
             'psd3_code_sha': new['runtime']['code_sha'],
             'note': 'PSD3 ricodifica le etichette in interi ordinati e le riconverte; stesso problema di classificazione.'}],
        'uncertainty': {'paired_confidence_intervals_computed': False,
                        'reason': 'Mancano le quattro predizioni OOF fase B sul Mac; i soli report non bastano al bootstrap appaiato.',
                        'future_if_recovered': {'bootstrap_unit': 'subject', 'replicates': 2000, 'seed': 20261102,
                                               'fixed_weights': WEIGHTS,
                                               'condition': 'Verificare SHA e identita esatta righe/target/soggetti prima del calcolo.'}},
        'local_oof_inventory': {'root': str(args.local_root), 'npz_files_seen': len(all_npz),
                                'phase_B_oof_files_found': found, 'missing': missing_oof},
        'observed_compute': dict(observed, by_original_report=costs,
                                interpretation='Tempi osservati moltiplicati per core allocati, non nuova contabilita sacct. Il riferimento a 768 dimensioni è circa 27 core-ore, non un preventivo verificato di 10 core-ore. Nessuna promessa di scaling da 8 a 4 thread.'),
        'limits': ['Nessun intervallo di confidenza in questa analisi; non chiamare significativi i delta.',
                   'Il confronto include PCA, precisione numerica e parallelismo; non isola causalmente la sola riduzione dimensionale.',
                   'Le differenze di C eventualmente selezionato sono effetti delle pipeline, non griglie diverse.',
                   'Un solo checkpoint per encoder; non dimostra assenza di informazione nei latenti.',
                   'Non valuta MLP a 768 dimensioni, decoder senza PCA o recupero della dinamica temporale.'],
        'controlled_followup_if_needed': 'Stessa implementazione PSD3, cache, ruoli, filtro classi, scaler float64, griglia, solver, seme e 4 thread; produrre 768 colonne standardizzate float32 saltando soltanto la PCA. Nessun nuovo training encoder, MLP o estrazione. Preventivo separato dai tempi osservati; prima tentare recupero OOF e confronto esplorativo appaiato.',
        'sources': sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'aggregates': aggregates, 'checks': checks,
                      'original_report_core_hours': observed['original_report_core_hours_at_8_cores']}, indent=2))


if __name__ == '__main__':
    main()
