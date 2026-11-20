"""Reviewer demo (synthetic data, historical metric code read-only from the snapshot).

Question: if a new 160-patch prediction is compared with the 148-patch reference layout using a wrong
crop (off by 6 patches), do the historical shape/support checks notice? Only the metric value changes.
"""
import sys
from pathlib import Path
import numpy as np
snap = Path(sys.argv[1])
sys.path[:0] = [str(snap / 'src'), str(snap / 'scripts')]
import tokenizer_pose_readout as R   # historical summarize(), THRESHOLD, OFFSET
import pose_temporal_readout as T    # historical central_mask()

rng = np.random.default_rng(0)
N, P, J, U = 240, 160, 20, 6
users = [f'u{i % U}' for i in range(N)]
lengths = np.full(N, P, dtype=np.int64)
# smooth targets (random walk, standardized-like), prediction = attenuated + lagged copy + noise
steps = rng.normal(0, 0.12, (N, P, J)).astype(np.float32)
target = np.cumsum(steps, axis=1).astype(np.float32)
pred160 = (0.5 * target + rng.normal(0, 0.25, (N, P, J))).astype(np.float32)
mask = np.ones((N, P, J), dtype=bool)
std = np.ones((N, J), dtype=np.float32)
pose_var = rng.uniform(0.1, 0.6, N).astype(np.float32)
data = {'target': target, 'mask': mask, 'std': std, 'pose_var': pose_var,
        'meta': [{'user': u, 'n_patches': P} for u in users], 'n_recordings': N,
        'lengths': lengths, 'central_mask': T.central_mask(mask, lengths, offset=R.OFFSET)}

def edyn(prediction148):
    s = R.summarize(data, prediction148.astype(np.float64))
    c = s['groups']['dynamic']['centered']
    return c['sums']['centered'] / c['sums']['vary'], s['support_sha256'], c['n_windows'], c['n_valid_elements']

cases = {'corretto  pred[:, 6:154]': pred160[:, 6:154], 'sbagliato pred[:, 0:148]': pred160[:, 0:148],
         'sbagliato pred[:, 12:160]': pred160[:, 12:160]}
base = None
for name, arr in cases.items():
    e, sha, nw, ne = edyn(arr)
    base = base or (sha, nw, ne)
    print(f'{name}: shape {arr.shape}  E_dyn dinamiche = {e:.6f}  support_sha256 = {sha[:12]}  finestre {nw}  elementi {ne}  '
          f'supporto/conteggi uguali al caso corretto: {(sha, nw, ne) == base}')
