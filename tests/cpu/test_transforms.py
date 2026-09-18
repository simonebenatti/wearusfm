import numpy as np
import pytest

from wearusfm.data.manifest import ChannelGroup, QuotaClass
from wearusfm.data.transforms import (
    add_realistic_noise,
    apply_montage_dropout,
    hd_virtual_bipolar_montage,
    maybe_hd_subsample,
    presented_class_for_virtual_montage,
    time_warp,
)


def test_montage_dropout_never_empties_the_sample():
    groups = (ChannelGroup("ring", 8), ChannelGroup("targeted", 4))
    rng = np.random.default_rng(0)
    for seed in range(200):
        rng = np.random.default_rng(seed)
        out = apply_montage_dropout(groups, rng, group_drop_p=0.9, channel_drop_p=0.9)
        assert sum(g.n_channels for g in out) >= 1


def test_montage_dropout_never_increases_channels():
    groups = (ChannelGroup("ring", 8), ChannelGroup("grid", 64))
    rng = np.random.default_rng(1)
    out = apply_montage_dropout(groups, rng, group_drop_p=0.0, channel_drop_p=0.0)
    assert sum(g.n_channels for g in out) == 72


def test_hd_virtual_bipolar_respects_c_max():
    groups = (ChannelGroup("grid", 64), ChannelGroup("grid", 64))
    rng = np.random.default_rng(2)
    for _ in range(50):
        out = hd_virtual_bipolar_montage(groups, rng, c_min=8, c_max=32)
        assert len(out) == 1
        assert 8 <= out[0].n_channels <= 32


def test_presented_class_threshold():
    assert presented_class_for_virtual_montage(8) == QuotaClass.A_RADI
    assert presented_class_for_virtual_montage(30) == QuotaClass.B_ANELLI


class _FakeMontage:
    def __init__(self, quota_class):
        self.quota_class = quota_class


def test_maybe_hd_subsample_noop_outside_class_c():
    groups = (ChannelGroup("ring", 8),)
    rng = np.random.default_rng(3)
    montage = _FakeMontage(QuotaClass.B_ANELLI)
    out_groups, out_class = maybe_hd_subsample(montage, groups, rng, p_piena=0.0)
    assert out_groups == groups
    assert out_class == QuotaClass.B_ANELLI


def test_maybe_hd_subsample_full_grid_when_p_piena_is_one():
    groups = (ChannelGroup("grid", 64),)
    rng = np.random.default_rng(4)
    montage = _FakeMontage(QuotaClass.C_GRIGLIE)
    out_groups, out_class = maybe_hd_subsample(montage, groups, rng, p_piena=1.0)
    assert out_groups == groups
    assert out_class == QuotaClass.C_GRIGLIE


def test_add_realistic_noise_preserves_shape():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(400, 16)).astype(np.float64)
    out = add_realistic_noise(data, fs_hz=1000.0, rng=rng)
    assert out.shape == data.shape


def test_time_warp_preserves_shape():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(400, 16)).astype(np.float64)
    out = time_warp(data, rng)
    assert out.shape == data.shape


def test_time_warp_handles_short_windows():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(2, 4)).astype(np.float64)
    out = time_warp(data, rng)
    assert out.shape == data.shape
