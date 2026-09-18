import numpy as np

from wearusfm.data.collate import (
    collate_bucketing,
    collate_packing,
    collate_padding,
    group_by_frequency,
    padding_fraction_analytic,
)
from wearusfm.data.manifest import MONTAGES_BY_NAME
from wearusfm.data.synthetic import generate_sample


def _samples(names, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    out = []
    for name in names:
        out.append(generate_sample(MONTAGES_BY_NAME[name], rng, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0, **kwargs))
    return out


def test_group_by_frequency_splits_correctly():
    samples = _samples(["emg2pose", "hyser", "ninapro_db5"])  # 1kHz, 2048Hz nativi, 200Hz
    groups = group_by_frequency(samples)
    assert set(groups.keys()) == {1000.0, 2048.0, 200.0}
    assert len(groups[1000.0]) == 1


def test_collate_padding_pads_to_max_channels():
    samples = _samples(["ninapro_standard", "capgmyo"], p_piena=1.0)  # 1kHz entrambi -> stesso T
    batch = collate_padding(samples)
    assert batch.data.shape[0] == 2
    c_max = max(s.c_presented for s in samples)
    assert batch.data.shape[1] == c_max
    assert batch.channel_mask.sum() == sum(s.c_presented for s in samples)
    assert batch.padding_fraction > 0  # 12 canali contro 128: c'e' padding vero


def test_collate_padding_channel_mask_is_correct_per_sample():
    samples = _samples(["ninapro_standard", "capgmyo"], p_piena=1.0)
    batch = collate_padding(samples)
    for i, s in enumerate(samples):
        assert batch.channel_mask[i].sum() == s.c_presented
        assert batch.channel_mask[i, : s.c_presented].all()
        assert not batch.channel_mask[i, s.c_presented :].any()


def test_collate_packing_total_tokens_equals_sum_of_channels():
    samples = _samples(["ninapro_standard", "capgmyo"], p_piena=1.0)
    packed = collate_packing(samples)
    total_c = sum(s.c_presented for s in samples)
    assert packed.n_tokens == total_c
    assert packed.cu_seqlens.tolist() == [0, samples[0].c_presented, total_c]
    assert packed.sample_ids.tolist() == [0] * samples[0].c_presented + [1] * samples[1].c_presented


def test_collate_bucketing_groups_by_exact_channel_count():
    samples = _samples(["emg2pose", "emg2pose", "ninapro_standard"], p_piena=1.0)
    # forza gli emg2pose a rimanere a 16 canali (nessun dropout, come sopra)
    buckets = collate_bucketing(samples)
    assert set(buckets.keys()) == {16, 12}
    assert buckets[16].data.shape[0] == 2
    assert buckets[12].data.shape[0] == 1
    # niente padding dentro un bucket a C uniforme
    assert buckets[16].padding_fraction == 0.0


def test_padding_fraction_analytic_matches_measured_order_of_magnitude():
    c_values = [12, 128]
    analytic = padding_fraction_analytic(c_values)
    samples = _samples(["ninapro_standard", "capgmyo"], p_piena=1.0)
    measured = collate_padding(samples).padding_fraction
    assert abs(analytic - measured) < 1e-9  # con un solo campione per C sono identiche
