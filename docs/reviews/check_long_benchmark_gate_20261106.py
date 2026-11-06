"""Local timing-only proof. No cluster calls, imports of project runtime or edits.
Reproduce: python3 /private/tmp/long_benchmark_local_assessment_20261106.py
Reads two operator-collected receipts. Computes a LOWER BOUND of the frozen
resource gate by omitting the historical floor inside max(measured, historical).
This does not execute/admit the full fifteen-benchmark gate.
"""
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/simonebenatti/wearusfm_local/line2/bench_peek')
ROOT = Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-consolidation')
OUT = Path('/private/tmp/long_benchmark_local_assessment_20261106.json')
report = {
    'scope': 'local read only; no cluster/status calls, job actions, worktree edits or score analysis',
    'proof': 'actual projection = sum(max(measured_work*row_ratio*class_ratio, historical_fit_seconds*measured_work/pool_work)*3*1.5) + 600 + 2*(load+scale); values below omit historical max floor and therefore are lower bounds',
    'frozen_gate_seconds': 13800, 'fit_allocation_seconds': 14400,
    'benchmark_allocation_seconds': 1800,
    'limitation': 'SLURM elapsed 1422/595s are operator-reported, not from a local sacct benchmark export; internal receipt timing verified below. Only DB2/pre and DB2/post receipts available. Full resource gate still requires all15 receipts.',
    'sources': {}, 'views': {},
}
for rel in ('scripts/consolidation_long_common.py', 'scripts/consolidation_long_classification_runtime.py', 'scripts/slurm/consolidation_long.sbatch'):
    p = ROOT / rel
    report['sources'][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
for view, slurm_seconds in (('pre', 1422), ('post', 595)):
    p = BASE / f'ninapro_db2_{view}_report.json'
    r = json.loads(p.read_text()); b = r['benchmark']; policy = r['policy']
    assert r['complete'] is True and b['timing_only'] is True and b['validation_or_held_scored'] is False
    assert all(f['converged_by_criterion'] and not f['warnings'] and f['exception'] is None for f in b['fits'])
    times = [f['elapsed_s'] for f in b['fits']]
    work = sum(times) + max(times)
    factor = policy['safety_factor'] * policy['data_and_prediction_multiplier']
    overhead = policy['fixed_seconds_per_job'] + 2*(b['load_s'] + b['scale_s'])
    costs = [work*n/b['full_train_rows']*k/b['full_train_classes']*factor
             for n,k in zip(b['fold_train_rows'],b['fold_train_classes'])]
    pairs = {','.join(map(str, fs)): sum(costs[f] for f in fs)+overhead for fs in ([0,1], [2,3])}
    singles = [c+overhead for c in costs]
    required_pool = [f['fit_seconds']*work*factor/(13800-overhead) for f in r['historical_timing']['folds']]
    report['views'][view] = {
        'receipt': str(p), 'receipt_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        'source_sha256': r['source_sha256'], 'recipe_sha256': r['recipe_sha256'],
        'code_sha': r['code_sha'], 'numeric_sha256': r['historical_numeric_match']['numeric_sha256'],
        'numeric_compatible': r['historical_numeric_match']['compatible'],
        'benchmark_internal_elapsed_seconds': r['measured_resources']['elapsed_seconds'],
        'slurm_elapsed_seconds_operator_reported': slurm_seconds,
        'benchmark_unused_seconds': 1800-slurm_seconds,
        'benchmark_admissible_slowdown_factor': 1800/slurm_seconds,
        'train_rows': b['full_train_rows'], 'classes': b['full_train_classes'],
        'candidate_times_iterations': [{k:f[k] for k in ('C','elapsed_s','n_iter')} for f in b['fits']],
        'candidate_sum_seconds': sum(times), 'slowest_refit_proxy_seconds': max(times),
        'measured_work_seconds': work, 'projection_overhead_seconds': overhead,
        'pair_shard_projection_lower_bounds_seconds': pairs,
        'pair_shards_proven_fail': [k for k,v in pairs.items() if v>13800],
        'hypothetical_single_fold_lower_bounds_seconds': singles,
        'single_fold_rule_sufficient_if_pool_work_seconds_at_least': max(required_pool),
        'projected_ram_gib': r['peak_rss_bytes']*3/1024**3,
    }
report['conclusion'] = 'All six DB2/pre paired-fold fit shards (two per dose) necessarily fail the frozen gate. Because all48 fit shards must pass globally, no classification fit can be admitted unchanged. This is a resource-policy issue, not evidence of worse representations.'
report['amendment_options_not_authorized_or_implemented'] = {
    'only_pre_singlefold': {'extra_fit_jobs':6,'new_compute_job_count':88,'checkout_count':4,'extra_allocation_core_hours_at4h':192,'new_total_allocation_core_hours':1903.6666666666665,'extra_projection_overhead_core_hours':8,'requires':'new per-view shard policy/code, complete benchmark check; same40 logical cells and frozen scientific recipe'},
    'all_db2_singlefold': {'extra_fit_jobs':16,'new_compute_job_count':98,'checkout_count':4,'extra_allocation_core_hours_at4h':512,'new_total_allocation_core_hours':2223.6666666666665,'extra_projection_overhead_core_hours':21.333333333333332,'requires':'existing per-dataset policy changed; full pool/post benchmark checking'},
    'benchmark_30to60min': {'extra_allocation_core_hours_per_job':4,'all15_max_extra_core_hours':60,'requires':'source and resource-plan amendment; existing12 extraction compatibility invalid under current strict policy'},
    'strict_source_repeat': {'additional_extraction_cap_gpu_hours':9,'observed_prior_extraction_gpu_hours':3.373611,'observed_prior_extraction_and_checkout_core_hours':27.001667,'requires':'new reviewed campaign, IDs, outputs and full budget/user decision under existing R-2(i)'},
    'cache_preserving_policy_change': 'Possible only by explicitly replacing R-2(i) after review/approval: pin original extraction+benchmark source/recipe and immutable receipts, audit frozen extraction and fit math unchanged, separate operational from scientific identity. Never rewrite or relabel old hashes. Not possible as launch_plan-only edit, and explicitly forbidden by current rule.'
}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(OUT)
print(json.dumps({k:report[k] for k in ('conclusion','amendment_options_not_authorized_or_implemented')},indent=2))
