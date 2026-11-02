"""Fixed, target-independent slot sketches; never preserve the full slot cache."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

K, D, Q, CHANNELS = 64, 384, 7, 48
EPS = 1e-5
SLOT_SEED, CHANNEL_SEED = 20261102, 20261103
READERS = ('M384', 'M48', 'S384', 'S720')
WIDTHS = {'M384': D, 'M48': CHANNELS, 'S384': (Q + 1) * CHANNELS, 'S720': D + Q * CHANNELS}
PROJECTION_PATH = Path(__file__).resolve().parents[1] / 'results/infra/pose_slots_20261102/projections.npz'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([value.dtype.str, value.shape]).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def _slot_basis():
    h = np.ones((1, 1), dtype=np.float64)
    while len(h) < K:
        h = np.block([[h, h], [h, -h]])
    rows = np.random.Generator(np.random.PCG64(SLOT_SEED)).choice(np.arange(1, K), size=Q, replace=False)
    return (h[rows] / K).astype(np.float32), rows.tolist()


def generate_projection_arrays():
    """One-time artifact construction; never called by runtime projection loading."""
    slot, _ = _slot_basis()
    gaussian = np.random.Generator(np.random.PCG64(CHANNEL_SEED)).standard_normal((D, CHANNELS))
    channel, triangular = np.linalg.qr(gaussian, mode='reduced')
    signs = np.where(np.diag(triangular) < 0, -1., 1.)
    channel = (channel * signs[None, :]).astype(np.float32)
    return {'slot_weights': slot, 'channel_projection': channel}


@lru_cache(maxsize=1)
def _load_projection_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        require(set(z.files) == {'slot_weights', 'channel_projection'}, 'unexpected projection arrays')
        slot, channel = z['slot_weights'], z['channel_projection']
    require(slot.dtype == channel.dtype == np.float32 and slot.shape == (Q, K)
            and channel.shape == (D, CHANNELS), 'invalid projection shape/dtype')
    require(np.isfinite(slot).all() and np.isfinite(channel).all(), 'nonfinite projection artifact')
    expected, _ = _slot_basis()
    require(np.array_equal(slot, expected), 'slot rows differ from frozen Hadamard recipe')
    require(np.allclose(channel.astype(np.float64).T @ channel, np.eye(CHANNELS), rtol=0, atol=2e-7),
            'channel projection is not orthonormal')
    slot.setflags(write=False)
    channel.setflags(write=False)
    return slot, channel


def projection_arrays():
    """Load the frozen artifact; missing/corrupt artifacts fail, never regenerate QR."""
    slot, channel = _load_projection_bytes(PROJECTION_PATH.read_bytes())
    return {'slot_weights': slot.copy(), 'channel_projection': channel.copy()}


def recipe():
    arrays = projection_arrays()
    slot, channel = arrays['slot_weights'], arrays['channel_projection']
    _, rows = _slot_basis()
    return {'slot_count': K, 'embedding_dim': D, 'residual_slot_modes': Q, 'channel_dim': CHANNELS,
            'slot_basis': 'Sylvester Hadamard64; no slot permutation; row0 mean is implicit',
            'slot_seed': SLOT_SEED, 'slot_rng': 'numpy.Generator(PCG64)', 'non_dc_rows_in_order': rows,
            'slot_weight_scale': 'each coefficient +/-1/64, same row norm as mean weights',
            'channel_seed': CHANNEL_SEED, 'channel_rng': 'numpy.Generator(PCG64)',
            'channel_basis': 'float64 Gaussian384x48 reduced QR, positive diagonal R; cast float32',
            'slot_weights_sha256': array_sha256(slot), 'channel_projection_sha256': array_sha256(channel),
            'residual_extraction': 'FP32: center z by its slot mean, apply slot weights then shared channel projection',
            'residual_storage': 'float32 raw contrasts, shape(window,patch,336); no normalization in extraction',
            'mean_source': 'original validated float16 mean cache',
            'normalization': 'M is historical FP32 torch LayerNorm(mean_FP16); residuals divided by sqrt(var_feature(mean_FP16_FP32)+eps); no separate residual LayerNorm, PCA or scaler',
            'normalization_eps': EPS, 'readers': list(READERS), 'widths': dict(WIDTHS),
            'M48_control': 'normalized mean projected to the same 48 channels; no residual modes',
            'S384_order': 'mean@channel_projection, residual slot modes in frozen row order',
            'S720_order': 'full normalized mean, residual slot modes in frozen row order'}


def projected_residuals(z):
    """(S,P,64,384) -> (S,P,336), FP32 raw contrasts; input is never mutated."""
    require(isinstance(z, torch.Tensor) and z.ndim == 4 and tuple(z.shape[-2:]) == (K, D)
            and z.is_floating_point(), 'expected floating slot tensor (S,P,64,384)')
    require(bool(torch.isfinite(z).all()), 'nonfinite slot input')
    arrays = projection_arrays()
    with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
        values = z.float()
        centered = values - values.mean(dim=-2, keepdim=True)
        slot = torch.as_tensor(arrays['slot_weights'], device=z.device, dtype=torch.float32)
        channel = torch.as_tensor(arrays['channel_projection'], device=z.device, dtype=torch.float32)
        modes = torch.einsum('qk,spkd->spqd', slot, centered)
        residuals = (modes @ channel).flatten(-2).contiguous()
    require(residuals.dtype == torch.float32 and bool(torch.isfinite(residuals).all()), 'nonfinite projected residuals')
    return residuals


def reader_features(mean_float16, residual_float32, reader, lengths):
    """Pure input transform; target availability is deliberately not an argument."""
    mean, residual, lengths = map(np.asarray, (mean_float16, residual_float32, lengths))
    require(reader in READERS, 'unknown slot reader')
    require(mean.ndim == 3 and mean.shape[-1] == D and mean.dtype == np.float16, 'expected original float16 mean cache')
    require(residual.shape == (*mean.shape[:2], Q * CHANNELS) and residual.dtype == np.float32,
            'expected raw float32 projected residuals')
    require(lengths.shape == (len(mean),) and lengths.dtype.kind in 'iu'
            and np.all((lengths > 0) & (lengths <= mean.shape[1])), 'invalid real patch lengths')
    real = np.arange(mean.shape[1])[None, :] < lengths[:, None]
    require(np.isfinite(mean[real]).all() and np.isfinite(residual[real]).all(), 'nonfinite real mean/residual input')
    safe_mean = np.where(real[..., None], mean, 0).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(safe_mean)
        normalized = torch.nn.functional.layer_norm(tensor, (D,), eps=EPS).numpy()
        denominator = torch.sqrt(torch.var(tensor, dim=-1, unbiased=False, keepdim=True) + EPS).numpy()
    if reader == 'M384':
        result = normalized
    elif reader == 'M48':
        result = normalized @ projection_arrays()['channel_projection']
    else:
        safe_residual = np.where(real[..., None], residual, 0).astype(np.float32)
        residual_norm = safe_residual / denominator
        first = normalized @ projection_arrays()['channel_projection'] if reader == 'S384' else normalized
        result = np.concatenate((first, residual_norm), axis=-1)
    result[~real] = 0
    require(result.dtype == np.float32 and result.shape[-1] == WIDTHS[reader]
            and np.isfinite(result).all(), 'nonfinite reader features')
    return result
