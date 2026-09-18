"""Geometria sintetica dei canali per il proxy GPU (asse L) - solo numpy, testabile
sul Mac senza torch. Separato da proxy.py perche' quel modulo importa torch/flash_attn
e non e' importabile senza CUDA.
"""

from __future__ import annotations

import math

import numpy as np

from wearusfm.data.manifest import ChannelGroup


def build_neighbors(
    groups: tuple[ChannelGroup, ...], k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Posizioni sintetiche per gruppo (v10 §3.4) e k vicini piu' vicini per canale.

    anello -> cerchio; griglia -> reticolo 2D; sparso/mirato -> posizioni casuali (nessuna
    simmetria, v10 §3.4 - i vicini restano ben definiti ma senza significato geometrico
    reale, irrilevante per un proxy di costo).
    """
    positions: list[np.ndarray] = []
    for g in groups:
        n = g.n_channels
        if g.topology == "ring":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            positions.append(np.stack([np.cos(angles), np.sin(angles)], axis=1))
        elif g.topology == "grid":
            side = max(1, round(math.sqrt(n)))
            rows = np.arange(n) // side
            cols = np.arange(n) % side
            positions.append(np.stack([rows, cols], axis=1).astype(np.float64))
        else:  # targeted / virtual_bipolar / altro: nessuna geometria reale
            positions.append(rng.uniform(-1, 1, size=(n, 2)))
    pos = np.concatenate(positions, axis=0)
    c = pos.shape[0]
    k_eff = min(k, c - 1) if c > 1 else 1

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    idx = np.argsort(dist, axis=1)[:, :k_eff]
    if k_eff < k:
        idx = np.pad(idx, ((0, 0), (0, k - k_eff)), mode="edge")
    nn_dist = np.take_along_axis(dist, idx, axis=1)
    nn_dist = np.where(np.isinf(nn_dist), 0.0, nn_dist)
    return idx.astype(np.int64), nn_dist.astype(np.float32)
