"""Pose: descriptives, prespecified contrasts and an independent paired-bootstrap recomputation for the T100-LQ final reading.

usage: python -I claude_review_logsquare_final_reading_20261107_pose.py <root>
where <root>/pose/5000/compare/report.json and <root>/pose/19618/compare/report.json are copies of the compare outputs
written by jobs 63786157 and 63786158 under $WORK/wearusfm_runs/results/logsquare_cpu_20261107/.
"""
import json, math, sys
import numpy as np
ARMS = ('T100-LQ', 'T100-L', 'T25-L', 'T100-G')
for step in ('5000', '19618'):
    rep = json.load(open(f'{sys.argv[1]}/pose/{step}/compare/report.json'))
    print(f'\n################ POSA, passo {step}: job {rep.get("slurm_job_id")}, status={rep["status"]}, complete={rep["complete"]}')
    print('candidate report pin:', rep['provenance']['candidate']['report'])
    print('historical report pin:', rep['provenance']['historical']['report'])
    print('\n-- selezione di alpha (fit.selection) per braccio')
    for arm in ARMS:
        sel = rep['cells'][arm]['fit']['selection']
        grid = sorted(sel['grid'], key=lambda g: g['alpha'])
        mses = ' '.join(f"{g['alpha']:.0e}:{g['mse']:.5f}" for g in grid)
        print(f"  {arm:8s} alpha={sel['alpha']:.0e} cv_mse={sel['cv_mse']:.6f} | griglia: {mses}")
        extra = {k: v for k, v in sel.items() if k not in ('alpha', 'cv_mse', 'grid')}
        if extra: print('          altri campi:', json.dumps(extra)[:300])
    for group in ('dynamic', 'all', 'static'):
        print(f'\n-- gruppo {group}: finestre selezionate per braccio:', {a: rep['cells'][a]['analysis']['groups'][group]['n_selected_windows'] for a in ARMS})
        print(f'  {"braccio":8s} {"E=cent/vary":>12s} {"IC95 (bootstrap)":>24s} {"corr":>8s} {"varp/vary":>10s} {"MAE rad":>9s} {"gain_mean":>10s} {"gain_cent":>10s} {"n_win":>6s}')
        for arm in ARMS:
            g = rep['cells'][arm]['analysis']['groups'][group]
            s = g['centered']['sums']
            E = s['centered'] / s['vary']
            corr = s['cov'] / math.sqrt(s['varp'] * s['vary'])
            vr = s['varp'] / s['vary']
            mae = g['mae_rad'] if not isinstance(g['mae_rad'], dict) else g['mae_rad']
            ci = rep['paired'][group]['cell_ratio_ci95'][arm]
            print(f"  {arm:8s} {E:12.6f} [{ci[0]:10.6f}, {ci[1]:10.6f}] {corr:8.4f} {vr:10.4f} {str(mae)[:9]:>9s} {s['gain_mean']/s['vary']:10.4f} {s['gain_centered']/s['vary']:10.4f} {g['centered']['n_windows']:6d}")
        print('  contrasti prespecificati (delta_ratio = (SSE_LQ - SSE_ctrl)/sum vary; negativo = LQ meglio):')
        for name, c in rep['paired'][group]['contrasts'].items():
            print(f"    {name:22s} {c['role']:9s} delta={c['delta_ratio']:+.6f} IC95=[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] riduzione relativa={c['relative_error_reduction']:+.4%} segnale2%={c['signal_2percent_and_ci_below_zero']}")
        # independent recomputation of the paired bootstrap from the per-user sums
        rows = {a: rep['cells'][a]['analysis']['groups'][group]['centered']['per_user'] for a in ARMS}
        users = sorted(rows['T100-L'])
        den_u = np.array([rows['T100-L'][u]['sums']['vary'] for u in users])
        err = {a: np.array([rows[a][u]['sums']['centered'] for u in users]) for a in ARMS}
        draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
        den = den_u[draws].sum(1)
        mism = []
        for ctrl in ('T100-L', 'T25-L', 'T100-G'):
            num = err['T100-LQ'] - err[ctrl]
            vals = num[draws].sum(1) / den
            ci = np.percentile(vals, [2.5, 97.5])
            c = rep['paired'][group]['contrasts'][f'T100-LQ_minus_{ctrl}']
            ok = math.isclose(num.sum()/den_u.sum(), c['delta_ratio'], rel_tol=1e-9) and np.allclose(ci, c['ci95'], rtol=1e-9) and math.isclose(-num.sum()/err[ctrl].sum(), c['relative_error_reduction'], rel_tol=1e-9)
            mism.append(not ok)
        # my own extra contrasts among controls, same draws (context: repeatability T25-L vs T100-L, and T100-G vs T100-L)
        extra = {}
        for a, b in (('T25-L', 'T100-L'), ('T100-G', 'T100-L')):
            num = err[a] - err[b]; vals = num[draws].sum(1)/den; ci = np.percentile(vals, [2.5, 97.5])
            extra[f'{a}-{b}'] = (num.sum()/den_u.sum(), ci[0], ci[1], -num.sum()/err[b].sum())
        print('  ricalcolo indipendente dei tre contrasti:', 'IDENTICO' if not any(mism) else f'DIFFERENZE {mism}')
        for k, (d, lo, hi, rel) in extra.items():
            print(f'    contesto (mio, stessi draw) {k:14s} delta={d:+.6f} IC95=[{lo:+.6f}, {hi:+.6f}] riduzione relativa={rel:+.4%}')
        # per-user sign count for the primary contrast
        num = err['T100-LQ'] - err['T100-L']
        print(f'  primario per utente: LQ meglio in {int((num < 0).sum())} su {len(users)}, peggio in {int((num > 0).sum())}; mediana delta/vary per utente = {np.median(num/den_u):+.5f}')
    d = rep['log_weight_diagnostics']
    print('\n-- W_log (descrittivo R1-b), passo', d['checkpoint_step'])
    for who in ('teacher', 'student'):
        fam = d[who]['families']; tot = sum(f['sum_squared_norms'] for f in fam.values())
        norms = np.array(d[who]['column_l2_norms'])
        print(f"  {who:8s} colonne={len(norms)} min={norms.min():.4f} max={norms.max():.4f} | " + ' '.join(f"{k}: n={f['n']} media={f['mean']:.4f} quota_norme2={f['sum_squared_norms']/tot:.2%}" for k, f in fam.items()))
    others = {k: v for k, v in d.items() if k not in ('teacher', 'student', 'source_receipts', 'checkpoint_step', 'checkpoint_sha256')}
    print('  altri campi:', json.dumps(others)[:400])
