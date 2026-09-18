import numpy as np

from wearusfm.data.manifest import ChannelGroup
from wearusfm.model.geometry import build_neighbors


def test_ring_neighbors_shape_and_range():
    groups = (ChannelGroup("ring", 16),)
    rng = np.random.default_rng(0)
    idx, dist = build_neighbors(groups, k=4, rng=rng)
    assert idx.shape == (16, 4)
    assert dist.shape == (16, 4)
    assert idx.min() >= 0
    assert idx.max() < 16
    assert (dist >= 0).all()


def test_grid_neighbors_no_self_loop():
    groups = (ChannelGroup("grid", 64),)
    rng = np.random.default_rng(1)
    idx, dist = build_neighbors(groups, k=8, rng=rng)
    for c in range(64):
        assert c not in idx[c]


def test_mixed_groups_total_channels():
    groups = (ChannelGroup("ring", 8), ChannelGroup("targeted", 4))
    rng = np.random.default_rng(2)
    idx, dist = build_neighbors(groups, k=3, rng=rng)
    assert idx.shape == (12, 3)


def test_k_larger_than_available_channels_falls_back_gracefully():
    groups = (ChannelGroup("targeted", 3),)
    rng = np.random.default_rng(3)
    idx, dist = build_neighbors(groups, k=8, rng=rng)
    assert idx.shape == (3, 8)
    assert idx.min() >= 0
    assert idx.max() < 3


def test_single_channel_does_not_crash():
    groups = (ChannelGroup("targeted", 1),)
    rng = np.random.default_rng(4)
    idx, dist = build_neighbors(groups, k=8, rng=rng)
    assert idx.shape == (1, 8)
