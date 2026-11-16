"""Save tiny synthetic forward fixtures from the actual PyTorch loss code."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.io import savemat
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.model.contrastive import info_nce
from wearusfm.model.rank_contrast import pose_rank_contrast


def tensor(a):
    return torch.from_numpy(np.asarray(a))


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(20261116)
    nce, rnc = [], []
    for case in range(5):
        q, kp, kn = unit(rng.normal(size=(4, 5))), unit(rng.normal(size=(4, 5))), unit(rng.normal(size=(7, 5)))
        mask, qvalid = rng.random((4, 7)) > .25, np.ones(4, dtype=bool)
        minimum, temperature = 2, .2
        if case == 1: q[:] = q[0]; kp[:] = q[0]; kn[:] = q[0]
        if case == 2: qvalid[1] = False; mask[2:] = False
        if case == 3: kn, mask = kn[:0], mask[:, :0]
        if case == 4: temperature = 1e-4
        values, eligible, _ = info_nce(tensor(q), tensor(kp), tensor(kn), tensor(mask),
            temperature=temperature, min_negatives=minimum, query_valid=tensor(qvalid))
        nce.append(dict(Q=q, Kpositive=kp, Knegative=kn, negativeMask=mask, queryValid=qvalid[:, None],
            temperature=temperature, minNegatives=minimum, expectedLoss=values.mean().item(),
            expectedPerQuery=values.numpy()[:, None], expectedEligible=eligible.numpy()[:, None]))

    for case in range(6):
        h, y = rng.normal(size=(6, 5)), rng.normal(size=(6, 3))
        ids = np.array([1, 1, 2, 2, 2, 3])
        valid, allowed, temperature = np.ones_like(y, dtype=bool), ~np.eye(6, dtype=bool), .1
        if case == 1: h[:] = h[0]; y[:] = 0  # exact ties and uniform similarity
        if case == 2:
            valid[0] = [True, False, False]; valid[1] = [False, True, True]
            y[~valid] = np.nan
        if case == 3:
            y[1] = y[0]; allowed[0, 1] = allowed[1, 0] = False
        if case == 4:
            allowed[:] = False; allowed[0, 1] = allowed[1, 0] = True
        if case == 5: temperature = 1e-4
        # Pack the same selected instants into the production S x P x D API.
        # stride=1 selects every fixture point; temporal subsampling is upstream.
        z, target = np.zeros((3, 3, 5)), np.zeros((3, 3, 3))
        v, tv = np.zeros((3, 3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)
        for window in range(1, 4):
            rows = np.flatnonzero(ids == window); n = len(rows)
            z[window-1, :n], target[window-1, :n] = h[rows], y[rows]
            v[window-1, :n], tv[window-1, :n] = valid[rows], True
        loss, _, count, logs = pose_rank_contrast(tensor(z), (tensor(target), tensor(v)), tensor(tv),
            temperature=temperature, stride=1, pair_allowed=tensor(allowed))
        rnc.append(dict(H=h, Y=y, windowId=ids[:, None], valid=valid, pairAllowed=allowed,
            temperature=temperature, expectedLoss=loss.item(), expectedWindows=count.item(),
            expectedAnchors=logs['anchors'].item(), expectedPairs=logs['pairs'].item()))

    savemat(HERE / "python_reference.mat", {"nceCases": np.array(nce, dtype=object),
        "rncCases": np.array(rnc, dtype=object)}, do_compression=True)
    sources = [ROOT/'src/wearusfm/model/contrastive.py', ROOT/'src/wearusfm/model/rank_contrast.py']
    manifest = {'kind': 'synthetic forward reference, not training data', 'seed': 20261116,
        'torch': torch.__version__, 'dtype': 'float64', 'nce_cases': len(nce), 'rnc_cases': len(rnc),
        'source_pins': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'fixture_sha256': hashlib.sha256((HERE/'python_reference.mat').read_bytes()).hexdigest()}
    (HERE/'reference_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'nce_cases': len(nce), 'rnc_cases': len(rnc)}))


if __name__ == '__main__':
    main()
