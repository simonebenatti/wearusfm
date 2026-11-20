# Riproduce E_dyn e conteggi del gate R-1 dalla ricevuta FIT storica (somme per utente), codice proprio.
import json, sys
from fractions import Fraction
p = sys.argv[1]
d = json.load(open(p))
for cell in ['post_ridge', 'trained_head', 'pre_ridge']:
    g = d['cells'][cell]['analysis']['groups']['dynamic']
    cc = g['centered']
    pu = cc['per_user']
    C = sum(v['sums']['centered'] for v in pu.values())
    V = sum(v['sums']['vary'] for v in pu.values())
    nw = sum(v['n_windows'] for v in pu.values())
    ne = sum(v['n_valid_elements'] for v in pu.values())
    nwj = sum(v['n_window_joints'] for v in pu.values())
    print(cell, 'E_dyn=%r' % (C / V), 'ratio_stored=%r' % cc.get('centered_mse_ratio'),
          'users=%d' % len(pu), 'windows=%d' % nw, 'elements=%d' % ne, 'window_joints=%d' % nwj,
          'n_selected=%d' % g['n_selected_windows'], 'support_sha=%s' % d['cells'][cell]['analysis'].get('support_sha256'))
    # aggregate group stats
    print('   stored n_windows=%d n_valid_elements=%d' % (cc['n_windows'], cc['n_valid_elements']))
for grp in ['all', 'static']:
    g = d['cells']['post_ridge']['analysis']['groups'][grp]
    print(grp, 'selected', g['n_selected_windows'], 'supported', g['centered']['n_windows'], 'elements', g['centered']['n_valid_elements'])
