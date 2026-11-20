# Ricalcolo indipendente: hash di key_time_valid (N,160) tutto True, con la definizione di arrays_hash
# (pose_temporal_readout.py:41-47: sha256 di json [dtype.str, shape] + byte dell'array contiguo).
import hashlib, json, numpy as np
def h(a):
    a = np.ascontiguousarray(a); d = hashlib.sha256()
    d.update(json.dumps([a.dtype.str, a.shape]).encode()); d.update(memoryview(a).cast('B')); return d.hexdigest()
for n, want in ((1907, '0065b939fb665ae3ff7cd475d9090a90834ca2e3fea311bbbf8c1fbcd000246f'),
                (18548, '04749cb00edae4861a7ab70cb5e7cb3363b18f28d61a818dbb4d105631ab485c')):
    full = np.ones((n, 160), bool)
    print(n, 'tutto True:', h(full) == want, h(full)[:12])
    # controprove: una finestra da 159 patch; tutte da 160 ma con forma (n,161) e ultima colonna False (max_patches diverso)
    a = full.copy(); a[0, 159] = False
    b = np.zeros((n, 161), bool); b[:, :160] = True
    print('   una da 159:', h(a) == want, '| forma (n,161):', h(b) == want)
