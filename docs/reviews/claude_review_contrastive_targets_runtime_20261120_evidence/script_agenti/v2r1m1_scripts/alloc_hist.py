"""Apply fbfde797 measured_allocation to the historical C0/CN/CR bf16 benchmarks, then inject observer overhead."""
import gzip, json, sys, copy
from pathlib import Path
clone = Path(sys.argv[1])
import contrastive_benchmark_readout as B
base = clone/'results/infra/contrastive_comparison_20261116/benchmarks_completed_bf16_20261118/benchmark'
for arm in ('C0', 'CN', 'CR'):
    s = json.loads((base/arm/'summary.json').read_text())
    recs = [json.loads(l) for l in gzip.open(base/arm/'metrics.jsonl.gz', 'rt') if l.strip()]
    a = B.measured_allocation(s, recs)
    n = sum('contrast_global_grad_base' not in r for r in recs[8:]); d = len(recs[8:]) - n
    print(arm, 'steps', len(recs), 'elapsed', round(s['elapsed_s'], 1), 'sum_t_step', round(sum(r['t_step_s'] for r in recs), 1),
          'normal/diag n', n, d, 'normal', round(a['normal_update_mean_s'], 4), 'diag', round(a['diagnostic_update_mean_s'], 4),
          'per_update', round(a['conservative_update_s'], 4), 'phases', a['phase_allocation_seconds'], 'GPUh', round(a['allocated_gpu_hours'], 3))
    if arm == 'C0':
        r128 = recs[127]; r256 = recs[255]
        print('  rec128 step', r128['step'], 't_step', r128['t_step_s'], 'diag?', 'contrast_global_grad_base' in r128,
              '| rec256 step', r256['step'], 't_step', r256['t_step_s'], 'diag?', 'contrast_global_grad_base' in r256)
        for obs in (20, 30, 40, 50, 63):
            # split as estimated by r1: 128 gets the normal-backward share, 256 the diagnostic share
            o128, o256 = obs*0.4, obs*0.6
            s2 = dict(s); s2['elapsed_s'] = s['elapsed_s'] + obs
            r2 = copy.deepcopy(recs); r2[127]['t_step_s'] += o128; r2[255]['t_step_s'] += o256
            b = B.measured_allocation(s2, r2)
            print(f'  +{obs:>3d}s observer: per_update {b["conservative_update_s"]:.4f} (+{100*(b["conservative_update_s"]/a["conservative_update_s"]-1):.2f}%)',
                  'phases', b['phase_allocation_seconds'], f'GPUh {b["allocated_gpu_hours"]:.3f} (+{b["allocated_gpu_hours"]-a["allocated_gpu_hours"]:.3f}, +{100*(b["allocated_gpu_hours"]/a["allocated_gpu_hours"]-1):.1f}%)',
                  'normal', round(b['normal_update_mean_s'], 4), 'diag', round(b['diagnostic_update_mean_s'], 4))
