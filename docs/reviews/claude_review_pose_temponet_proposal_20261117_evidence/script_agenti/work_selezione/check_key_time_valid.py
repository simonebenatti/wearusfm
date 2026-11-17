"""Se tutte le finestre hanno 160 patch reali, key_time_valid e' una matrice (N,160) tutta True.
Ricalcolo arrays_hash (stessa funzione di scripts/pose_temporal_readout.py:41-47) e confronto con le ricevute di estrazione."""
import hashlib, json
import numpy as np

def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()

expected = {"train": (1907, "0065b939fb665ae3ff7cd475d9090a90834ca2e3fea311bbbf8c1fbcd000246f"),
            "val": (18548, "04749cb00edae4861a7ab70cb5e7cb3363b18f28d61a818dbb4d105631ab485c")}
for split, (n, sha) in expected.items():
    got = arrays_hash(np.ones((n, 160), dtype=bool))
    print(split, n, "all-True(N,160) hash =", got, "| ricevuta =", sha, "| uguali:", got == sha)
    # controprova: una sola finestra da 159 patch cambierebbe l'hash
    a = np.ones((n, 160), dtype=bool); a[0, 159] = False
    print("   controprova (una finestra da 159 patch):", arrays_hash(a) == sha)
