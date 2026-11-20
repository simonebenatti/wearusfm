# Verifica indipendente dalle somme per utente della ricevuta FIT 49046 (2930d7d).
import json, sys
import numpy as np
r = json.load(open(sys.argv[1]))
cells = r['cells']
out = {}
for cell in ('post_ridge', 'trained_head', 'pre_ridge'):
    a = cells[cell]['analysis']
    for grp in ('dynamic', 'all', 'static'):
        g = a['groups'][grp]
        c = g['centered']
        users = sorted(c['per_user'])
        S = {k: sum(c['per_user'][u]['sums'][k] for u in users) for k in ('centered', 'vary', 'mean_error', 'total', 'varp', 'cov')}
        n = sum(c['per_user'][u]['n_valid_elements'] for u in users)
        nw = sum(c['per_user'][u]['n_windows'] for u in users)
        E = S['centered'] / S['vary']
        lvl = S['mean_error'] / S['total']
        chk = abs(S['total'] - S['mean_error'] - S['centered']) / S['total']
        out[(cell, grp)] = dict(E=E, level_share=lvl, n=n, nw=nw, nsel=g['n_selected_windows'], users=len(users),
                                decomp_rel_err=chk, receipt_E=c['centered_mse_ratio'],
                                corr=S['cov'] / np.sqrt(S['varp'] * S['vary']), vr=S['varp'] / S['vary'])
for k, v in out.items():
    print(k, {kk: (round(vv, 10) if isinstance(vv, float) else vv) for kk, vv in v.items()})
# bootstrap: stessa estrazione del codice storico (default_rng(seed).integers(0, n_users, (B, n_users))) ma calcolo mio
def boot(cell_a, cell_b, grp='dynamic', B=2000, seed=20261102, level=95):
    ca = cells[cell_a]['analysis']['groups'][grp]['centered']['per_user']
    cb = cells[cell_b]['analysis']['groups'][grp]['centered']['per_user']
    users = sorted(ca)
    assert users == sorted(cb)
    Ca = np.array([ca[u]['sums']['centered'] for u in users]); Va = np.array([ca[u]['sums']['vary'] for u in users])
    Cb = np.array([cb[u]['sums']['centered'] for u in users]); Vb = np.array([cb[u]['sums']['vary'] for u in users])
    assert np.array_equal(Va, Vb)
    d = np.random.default_rng(seed).integers(0, len(users), (B, len(users)))
    Ea = Ca[d].sum(1) / Va[d].sum(1); Eb = Cb[d].sum(1) / Vb[d].sum(1)
    rel = 100 * (1 - Ea / Eb)
    q = (100 - level) / 2
    pa = Ca.sum() / Va.sum(); pb = Cb.sum() / Vb.sum()
    return dict(Ea=pa, Eb=pb, Ea_ci=np.percentile(Ea, [q, 100 - q]).tolist(), rel=100 * (1 - pa / pb),
                rel_ci=np.percentile(rel, [q, 100 - q]).tolist(), rel_sd=float(rel.std(ddof=1)))
print('trained_head vs post_ridge, 95:', boot('trained_head', 'post_ridge'))
print('trained_head vs post_ridge, 99:', boot('trained_head', 'post_ridge', level=99))
print('post_ridge alone CI95:', boot('post_ridge', 'trained_head')['Ea_ci'])
print('trained_head alone CI95:', boot('trained_head', 'post_ridge')['Ea_ci'])
# pesi per utente e numero efficace
ca = cells['post_ridge']['analysis']['groups']['dynamic']['centered']['per_user']
users = sorted(ca)
w = np.array([ca[u]['sums']['centered'] for u in users]); w = w / w.sum()
nwin = [ca[u]['n_windows'] for u in users]
print('n_eff (SSE centrata ridge post)', 1 / np.sum(w ** 2), 'finestre per utente min/max', min(nwin), max(nwin))
