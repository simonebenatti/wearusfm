"""Is resources.percentile95 (np.quantile(values, .95, method="linear") on 10 samples) bit-identical across numpy versions?
Samples are generated from integers with the standard library only, so both interpreters see the same doubles."""
import hashlib, random, struct, sys
import numpy as np
random.seed(20261120)
digest = hashlib.sha256(); rows = []
for case in range(200000):
    scale = 10.0 ** random.randint(-4, 3)
    values = [random.getrandbits(53) / 2**53 * scale for _ in range(10)]
    q = float(np.quantile(np.asarray(values, dtype=float), .95, method="linear"))
    m = float(np.median(values))
    digest.update(struct.pack("<dd", q, m))
    if case < 3:
        rows.append((q.hex(), m.hex()))
print("numpy", np.__version__, "python", sys.version.split()[0], "| sha256 of 200000 (p95, median) pairs:", digest.hexdigest(), "| first", rows)
