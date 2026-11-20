import json, sys, glob, numpy as np
for f in sorted(glob.glob(sys.argv[1] + '/runs/*.json')):
    r = json.load(open(f)); h = r['held']
    m = np.array([x['mse'] for x in h]); e = np.array([x['E_dyn'] for x in h]); lv = np.array([x['level_share'] for x in h])
    im, ie = int(m.argmin()), int(e.argmin())
    print(f"{f.split('/')[-1]:28s} ep {len(h)}  minMSE {m[im]:.4f}@{im+1} (E_dyn {e[im]:.4f}, livello {lv[im]:.2f})  minE_dyn {e[ie]:.4f}@{ie+1} (MSE {m[ie]:.4f})  "
          f"ultima: MSE {m[-1]:.4f} E_dyn {e[-1]:.4f} train {r['train'][-1]:.4f}  finiti {bool(np.isfinite(m).all() and np.isfinite(e).all())}")
