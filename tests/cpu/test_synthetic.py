import numpy as np

from wearusfm.data.manifest import MONTAGES, MONTAGES_BY_NAME, QuotaClass, class_weights
from wearusfm.data.synthetic import generate_sample, sample_montage


def test_generate_sample_shape_matches_front_end_fs():
    montage = MONTAGES_BY_NAME["emg2pose"]
    rng = np.random.default_rng(0)
    sample = generate_sample(montage, rng, context_s=4.0, p_piena=1.0, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)
    assert sample.n_samples == round(1000.0 * 4.0)  # front-end a 1kHz per emg2pose
    assert sample.c_presented == 16
    assert sample.data.shape == (sample.n_samples, 16)
    assert sample.data.dtype == np.int16


def test_generate_sample_hd_full_grid_when_p_piena_one():
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(1)
    sample = generate_sample(montage, rng, p_piena=1.0, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)
    assert sample.quota_class_presented == QuotaClass.C_GRIGLIE
    assert sample.c_presented == 256


def test_generate_sample_hd_virtual_montage_when_p_piena_zero():
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(2)
    sample = generate_sample(montage, rng, p_piena=0.0, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)
    assert sample.c_presented <= 32
    assert sample.quota_class_presented in (QuotaClass.A_RADI, QuotaClass.B_ANELLI)


def test_sample_montage_respects_weights_in_expectation():
    montages = MONTAGES
    weights = class_weights(QuotaClass.C_GRIGLIE, c_pessimistic=True)
    class_c_only = tuple(m for m in montages if m.quota_class == QuotaClass.C_GRIGLIE)
    rng = np.random.default_rng(3)
    counts = {m.name: 0 for m in class_c_only}
    for _ in range(500):
        m = sample_montage(rng, class_c_only, weights)
        counts[m.name] += 1
    assert counts["hyser"] == 500  # pessimistico: solo Hyser ha peso > 0
