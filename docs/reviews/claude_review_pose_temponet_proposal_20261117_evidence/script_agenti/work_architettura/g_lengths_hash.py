"""Le finestre delle cache hanno tutte 160 patch? Confronto l'hash di key_time_valid tutto vero con quello delle ricevute.
arrays_hash e' ricopiata alla lettera da scripts/pose_temporal_readout.py dello snapshot (righe 41-47), per non importare lo script."""
import hashlib, json, sys
import numpy as np

def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()

R = sys.argv[1]
for step in (19618, 39237, 49046):
    for split, n in (("train", 1907), ("val", 18548)):
        rec = json.load(open(f"{R}/LONG-G{step}-pose-{split}-20261105.json"))
        want = rec["details"]["key_time_valid_sha256"]
        got = arrays_hash(np.ones((n, 160), dtype=bool))
        # controprova: una sola finestra piu' corta di una patch cambia l'hash
        alt = np.ones((n, 160), dtype=bool); alt[0, 159] = False
        print(step, split, "n_windows", rec["details"]["n_windows"], "| ricevuta", want[:16], "| (n,160) tutto vero", got[:16],
              "| uguali:", want == got, "| controprova (una patch falsa) uguale:", arrays_hash(alt) == want)
