"""Sensibilita' di measured_allocation (codice reale a fbfde797) al tempo dell'osservatore sul C0.

Usa i benchmark storici bf16 (576 update) come base; aggiunge X secondi ripartiti fra i record
128 e 256 e a elapsed_s (dove l'osservatore li mette), per X in una griglia.
"""
import gzip, json, sys, copy
from pathlib import Path
clone = Path(sys.argv[1])
import contrastive_benchmark_readout as B
assert Path(B.__file__).resolve().is_relative_to(clone.resolve()), B.__file__
base = clone/'results/infra/contrastive_comparison_20261116/benchmarks_completed_bf16_20261118/benchmark'
for arm in ('C0', 'CN', 'CR'):
    summary = json.loads((base/arm/'summary.json').read_text())
    records = [json.loads(l) for l in gzip.open(base/arm/'metrics.jsonl.gz', 'rt') if l.strip()]
    a = B.measured_allocation(summary, records)
    normal = [r for r in records[8:] if 'contrast_global_grad_base' not in r]
    diag = [r for r in records[8:] if 'contrast_global_grad_base' in r]
    print(arm, 'records', len(records), 'normal', len(normal), 'diag', len(diag),
          'r128 diag?', 'contrast_global_grad_base' in records[127], 'r256 diag?', 'contrast_global_grad_base' in records[255])
    print('  elapsed %.1f elapsed/576 %.4f mean_normal %.4f mean_diag %.4f conservative %.4f phases %s alloc_gpuh %.3f' % (
        a['observed_end_to_end_s'], a['observed_end_to_end_s']/576, a['normal_update_mean_s'], a['diagnostic_update_mean_s'],
        a['conservative_update_s'], a['phase_allocation_seconds'], a['allocated_gpu_hours']))
    if arm != 'C0':
        continue
    for extra in (10, 20, 30, 40, 50, 60):
        s2, r2 = copy.deepcopy(summary), copy.deepcopy(records)
        # osservatore: due punti; ~40% a 128 (backward normale) e ~60% a 256 (backward diagnostico)
        r2[127]['t_step_s'] += .4*extra
        r2[255]['t_step_s'] += .6*extra
        s2['elapsed_s'] += extra
        b = B.measured_allocation(s2, r2)
        print('  +%2ds osservatore: conservative %.4f (+%.2f%%) phases %s alloc_gpuh %.3f (+%.3f GPUh, +%.1f%%)' % (
            extra, b['conservative_update_s'], 100*(b['conservative_update_s']/a['conservative_update_s']-1), b['phase_allocation_seconds'],
            b['allocated_gpu_hours'], b['allocated_gpu_hours']-a['allocated_gpu_hours'], 100*(b['allocated_gpu_hours']/a['allocated_gpu_hours']-1)))
