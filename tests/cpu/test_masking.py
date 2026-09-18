import numpy as np

from wearusfm.data.manifest import ChannelGroup
from wearusfm.data.masking import rvq_slab_mask, spatial_mask, temporal_mask


def test_temporal_mask_respects_budget_roughly():
    rng = np.random.default_rng(0)
    mask = temporal_mask(4000, fs_hz=1000.0, rng=rng, budget_fraction=0.5)
    assert mask.shape == (4000,)
    frac = mask.mean()
    assert 0.45 <= frac <= 0.55  # lo span finale e' troncato al budget residuo (masking.py)


def test_temporal_mask_short_window_does_not_crash():
    rng = np.random.default_rng(1)
    mask = temporal_mask(10, fs_hz=200.0, rng=rng, budget_fraction=0.5)
    assert mask.shape == (10,)


def test_spatial_mask_respects_total_channels():
    groups = (ChannelGroup("ring", 8), ChannelGroup("targeted", 4))
    rng = np.random.default_rng(2)
    mask = spatial_mask(groups, rng, budget_fraction=0.4)
    assert mask.shape == (12,)
    assert mask.dtype == bool


def test_spatial_mask_all_modes_produce_valid_mask():
    groups = (ChannelGroup("grid", 16), ChannelGroup("grid", 16), ChannelGroup("grid", 16))
    for seed in range(30):
        rng = np.random.default_rng(seed)
        mask = spatial_mask(groups, rng)
        assert mask.shape == (48,)
        assert mask.any()


def test_rvq_slab_mask_is_off_by_default_usage():
    rng = np.random.default_rng(3)
    mask = rvq_slab_mask(4000, fs_hz=1000.0, rng=rng, slab_ms=200.0, n_slabs=1)
    assert mask.shape == (4000,)
    assert mask.sum() == 200  # 200ms @ 1kHz = 200 campioni


def test_rvq_slab_mask_aligned_to_grid():
    rng = np.random.default_rng(4)
    fs = 1000.0
    slab_ms = 200.0
    slab_samples = round(slab_ms * fs / 1000.0)
    mask = rvq_slab_mask(4000, fs_hz=fs, rng=rng, slab_ms=slab_ms, n_slabs=1)
    (idx,) = np.where(mask)
    assert idx[0] % slab_samples == 0
