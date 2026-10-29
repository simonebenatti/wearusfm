"""Summarize frozen aggregate JSON reports; no fit, tuning or remote operations."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'final' / 'reports'


def read(name):
    return json.loads((REPORTS / name).read_text())


def compact_pose(x):
    e = x['e1_dynamic_control']
    groups = {}
    for name, group in x['groups'].items():
        c = group['centered_support_n_ge_2']
        groups[name] = {
            k: c[k] for k in ('n_windows', 'n_valid_elements', 'centered_mse_ratio',
                              'variance_ratio', 'pooled_centered_correlation', 'user_centered_gain_signs')
        }
        groups[name]['ratio_ci95'] = group['bootstrap_centered']['ratio_ci95']
        groups[name]['means'] = c['means']
    return {'E1': {k: e[k] for k in ('mae', 'mae_const', 'ratio', 'ratio_ci95_users', 'median_user_ratio')},
            'groups': groups}


def main():
    p = read('pose_comparison.json')
    summary = {
        'scope': 'Exploratory single-seed intermediate endpoint at19618; pose on15 held-out users; P1/P2 internal calibration probes, not official tests.',
        'report_sha256': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in REPORTS.glob('*.json')},
        'head': {k: compact_pose(v) for k, v in p['head_dynamics'].items()},
        'ridge_common': {k: compact_pose(p['accessibility'][k]['analysis']) for k in ('B', 'R3c')},
        'head_evolution': p['head_evolution'],
        'ridge_B_minus_R3c': p['accessibility']['B_minus_R3c'],
        'E2': {k: p[v]['E2']['gap_with_ci'] for k, v in [('B5000', 'baseline_head'), ('B19618', 'candidate_head')]},
        'transfer': {},
    }
    for task in ('p1', 'p2'):
        r = read('compare_' + task + '.json')['comparison']
        summary['transfer'][task] = {k: r[k] for k in ('A', 'B', 'delta', 'ci95', 'datasets')}
    h0 = summary['head']['B5000']['groups']['dynamic']['means']
    h1 = summary['head']['B19618']['groups']['dynamic']['means']
    summary['head_dynamic_window_absolute_error_reduction'] = {
        'mean_level_component': h0['mean_error'] - h1['mean_error'],
        'centered_component': h0['centered'] - h1['centered'],
        'total': h0['total'] - h1['total'],
        'interpretation': 'Additive standardized MSE components on common support; not causal attribution.'}
    for name in ('E1', 'centered_dynamic'):
        if name == 'E1':
            before = summary['head']['B5000']['E1']['ratio']
            after = summary['head']['B19618']['E1']['ratio']
        else:
            before = summary['head']['B5000']['groups']['dynamic']['centered_mse_ratio']
            after = summary['head']['B19618']['groups']['dynamic']['centered_mse_ratio']
        summary['head_dynamic_window_absolute_error_reduction'][name + '_relative_to_B5000'] = 1 - after / before
    (ROOT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(summary)


def plot(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 5.6), gridspec_kw={'width_ratios': [1, 1.1]})
    blue, gray, orange = '#1764a1', '#737980', '#b65b28'
    rows = [('Testa B · 5.000', s['head']['B5000'], 4, gray),
            ('Testa B · 19.618', s['head']['B19618'], 3, blue),
            ('Ridge R3c · 19.618', s['ridge_common']['R3c'], 1, gray),
            ('Ridge B · 19.618', s['ridge_common']['B'], 0, blue)]
    for label, d, y, color in rows:
        g = d['groups']['dynamic']; x = g['centered_mse_ratio']; lo, hi = g['ratio_ci95']
        a.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt='o', color=color, capsize=4, lw=2, ms=7)
        a.text(x, y + .23, f'{x:.3f}', ha='center', color=color, fontsize=10)
    a.set_yticks([r[2] for r in rows], [r[0] for r in rows])
    a.axvline(1, color='#444444', lw=1, ls='--')
    a.axhline(2, color='#d8dce0', lw=.8)
    a.set_xlim(.82, 1.015); a.set_ylim(-.6, 4.7)
    a.set_xlabel('MSE centrata / varianza del bersaglio\nMeno è meglio; 1 = predizione temporalmente piatta')
    a.set_title('Posa: la dinamica migliora', loc='left', fontweight='bold', pad=18)
    a.xaxis.set_major_locator(MultipleLocator(.05)); a.grid(axis='x', alpha=.15)
    a.text(.82, 2.2, 'Ridge con identica normalizzazione', fontsize=9, color='#555555')
    transfer_rows = [('P1', s['transfer']['p1'], 7, True), ('P2', s['transfer']['p2'], 6, True)]
    labels = {'ninapro_db2':'DB2', 'ninapro_db3':'DB3', 'ninapro_db6':'DB6', 'epn612':'EPN612', 'uci_emg':'UCI'}
    y = 4
    for task in ('p1','p2'):
        for name, d in s['transfer'][task]['datasets'].items():
            transfer_rows.append((labels[name], d, y, False)); y -= 1
    for label, d, y, primary in transfer_rows:
        x = 100*d['delta']; lo, hi = [100*v for v in d['ci95']]
        b.errorbar(x, y, xerr=[[x-lo],[hi-x]], fmt='o', color=orange if primary else gray,
                   capsize=4, lw=2 if primary else 1.2, ms=7 if primary else 5)
    b.set_yticks([r[2] for r in transfer_rows], [r[0] for r in transfer_rows])
    b.axvline(0, color='#444444', lw=1)
    for x in (-1,1): b.axvline(x, color='#b8bdc2', lw=.8, ls=':')
    b.axhline(5, color='#d8dce0', lw=.8)
    b.set_xlim(-4,4); b.set_ylim(-.6,7.7); b.grid(axis='x', alpha=.15)
    b.set_xlabel('ΔS = B − R3c, punti percentuali\nPiù è meglio; linee puntinate = ±1 punto')
    b.set_title('Sonde P1/P2: nessun beneficio osservato', loc='left', fontweight='bold', pad=18)
    fig.suptitle('WearUsFM · confronto intermedio a 19.618 passi', x=.06, ha='left', fontsize=15, fontweight='bold')
    fig.text(.06, .035, 'Intervalli al 95% da bootstrap utenti; un solo seme di pretraining. Posa: student, 15 utenti esclusi.\n'
             'P1/P2: teacher, validazione interna. Nessuna correzione multipla; confronti esplorativi.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.19, right=.98, bottom=.24, top=.80, wspace=.40)
    for ext in ('png','svg','pdf'):
        fig.savefig(ROOT / ('b19618_results.'+ext), dpi=180, facecolor='white')
    svg = ROOT / 'b19618_results.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
