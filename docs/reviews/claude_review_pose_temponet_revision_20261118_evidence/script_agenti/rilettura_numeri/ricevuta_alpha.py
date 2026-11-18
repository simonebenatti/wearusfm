# Percorso alpha della ridge post nella ricevuta storica pinnata e numeri del gate R-1.
import json, hashlib
P = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/work_costo/LONG-FIT-G49046-pose-20261105.json'
raw = open(P,'rb').read()
print('sha256', hashlib.sha256(raw).hexdigest())
d = json.loads(raw)
sel = d['cells']['post_ridge']['fit']['selection']
print('chiavi selection:', list(sel.keys()))
print('alpha scelto', sel['alpha'], 'cv_mse', sel['cv_mse'])
print('utenti per fold:', [len(x) for x in sel['fold_users']])
for g in sel['grid']:
    print('  grid', {k: v for k, v in g.items() if not isinstance(v, (list, dict))})
for k, v in sel.items():
    if k not in ('grid', 'fold_users'):
        print('  ', k, v)
g = {x['alpha']: x for x in sel['grid']}
keyname = [k for k in sel['grid'][0].keys() if 'mse' in k.lower()]
print('chiavi mse nella griglia:', keyname)
k = keyname[0]
a001 = g[0.01][k]; best = min(x[k] for x in sel['grid'])
print('alpha 0.01:', a001, ' migliore:', best, ' rapporto-1 = %.4f%%' % (100*(a001/best-1)))
fit = d['cells']['post_ridge']['fit']
print('chiavi fit:', [k for k in fit.keys()])
for kk in fit:
    if kk not in ('selection',):
        v = fit[kk]
        print('  fit.', kk, (json.dumps(v)[:300] if isinstance(v,(dict,list)) else v))
