"""Local arithmetic on published aggregates; no fitting, bootstrap or cluster I/O."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sources = {}


def read(path):
    raw = (ROOT / path).read_bytes()
    sources[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


prefix = 'results/infra/tokenizer_execution_20261103'
t5 = read(f'{prefix}/5000/T100-G/summary.json')
t19 = read(f'{prefix}/19618/T100-G/summary.json')
big = read('results/infra/scaling_300m_20261104/scientific5000_completion_20261105/summary.json')
segments = [read(f'results/infra/consolidation_long_20261105/continuation/{step}/receipt.json')['summary']
            for step in (30000, 39237, 49046)]
assert all(s['world_size'] == 4 and s['stopped'] == 'max_steps' for s in [t5, t19, big, *segments])
hours = lambda s: s['elapsed_s'] * s['world_size'] / 3600
h5 = hours(t5)
h19 = h5 + hours(t19)  # resume summary reports elapsed time for its own segment
h49 = h19 + sum(map(hours, segments))
hb5 = hours(big)
hb19_est = hb5 * 19618 / 5000

synth = read('results/infra/scaling_consolidation_synthesis_20261106/data.json')
cls = read('results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/final/comparison.json')
repeat = read('results/infra/tokenizer_evaluation_20261103/t25_b_repeatability_20261104/paired_report.json')
pose = read('results/infra/tokenizer_evaluation_20261103/cpu_19618/final/dose_comparison.json')['cells']['T100-G']['dynamic']
dose = {}
for key in ['epn612', 'p1', 'p2', 'uci_emg']:
    old = synth['scaling']['classification'][key]['scores_percent']['T100-G-30M-natural']
    current = cls['tasks' if key in ('p1', 'p2') else 'datasets'][key]['cells']['T100-G']['balanced_accuracy'] * 100
    dose[key] = {'30m_5000_percent': old, '30m_19618_percent': current, 'delta_pp': current - old}
assert abs(pose['E_5000'] - synth['scaling']['pose']['control_E_dyn']) < 1e-14
epn = synth['epn']
ft = epn['15']['cells']['finetune']['balanced_accuracy_percent']
scratch = epn['15']['cells']['scratch']['balanced_accuracy_percent']

result = {
    'claude_reviewed_commit': '9b38c2097eeaad34f7cf024c118cade3eee31a8d',
    'source_sha256': sources,
    'training_gpu_hours_cumulative': {'30m_5000_measured': h5, '30m_19618_measured': h19,
        '30m_49046_measured': h49, '300m_5000_measured': hb5,
        '300m_19618_linear_extrapolation_NOT_actual': hb19_est},
    'cost_scope': 'Trainer elapsed time ×4GPUs; excludes SLURM overhead, technical pilots, extraction, readouts and downstream fitting.',
    'actual_runtime_ratio_at_5000': hb5 / h5,
    '30m19618_cost_above_300m5000_percent': 100 * (h19 / hb5 - 1),
    '30m49046_saving_vs_projected300m19618_percent': 100 * (1 - h49 / hb19_est),
    'descriptive_dose_classification': dose,
    'descriptive_dose_pose': pose,
    'p1_scaling': synth['scaling']['classification']['p1']['contrasts']['natural_300m_minus_30m'],
    'p1_historical': repeat['tasks']['p1']['contrast'],
    'epn_ft15': {'finetune_percent': ft, 'scratch_percent': scratch,
        'contrast': epn['15']['contrasts']['finetune_minus_scratch'],
        'relative_balanced_error_reduction_percent': 100 * (ft-scratch)/(100-scratch),
        'change_in_gap': epn['recipe_comparison']['transfer_gap15_minus_gap5']},
    'scope': 'Post hoc descriptive review only; no new confidence intervals, thresholds or protocol decisions.'
}
(HERE / 'numbers.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:result[k] for k in ['training_gpu_hours_cumulative','actual_runtime_ratio_at_5000',
    '30m19618_cost_above_300m5000_percent','30m49046_saving_vs_projected300m19618_percent','descriptive_dose_classification']}, indent=2))
