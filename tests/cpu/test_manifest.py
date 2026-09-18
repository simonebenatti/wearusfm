from wearusfm.data.manifest import (
    MONTAGES,
    MONTAGES_BY_CLASS,
    QuotaClass,
    class_weights,
    front_end_fs_hz,
    full_montage_weights,
)


def test_montage_channel_counts_match_spec():
    by_name = {m.name: m for m in MONTAGES}
    assert by_name["emg2pose"].n_channels == 16
    assert by_name["emg2qwerty"].n_channels == 32
    assert by_name["ninapro_db5"].n_channels == 16
    assert by_name["grabmyo"].n_channels == 28
    assert by_name["putemg"].n_channels == 24
    assert by_name["ninapro_standard"].n_channels == 12
    assert by_name["camargo2021"].n_channels == 11
    assert by_name["capgmyo"].n_channels == 128
    assert by_name["csl_hdemg"].n_channels == 168
    assert by_name["hyser"].n_channels == 256


def test_every_montage_assigned_to_exactly_one_class():
    classes = {c: {m.name for m in ms} for c, ms in MONTAGES_BY_CLASS.items()}
    all_names = set()
    for names in classes.values():
        assert not (all_names & names), "un montaggio non puo' stare in due classi"
        all_names |= names
    assert all_names == {m.name for m in MONTAGES}


def test_front_end_fs_hd_stays_native():
    hyser = next(m for m in MONTAGES if m.name == "hyser")
    assert front_end_fs_hz(hyser) == hyser.fs_native_hz == 2048.0


def test_front_end_fs_myo_stays_native_below_1khz():
    db5 = next(m for m in MONTAGES if m.name == "ninapro_db5")
    assert front_end_fs_hz(db5) == 200.0


def test_front_end_fs_default_1khz_for_conventional():
    emg2pose = next(m for m in MONTAGES if m.name == "emg2pose")
    assert front_end_fs_hz(emg2pose) == 1000.0  # 2000 Hz nativi -> 1 kHz al front-end


def test_front_end_fs_all_native_variant():
    emg2pose = next(m for m in MONTAGES if m.name == "emg2pose")
    assert front_end_fs_hz(emg2pose, all_native=True) == 2000.0


def test_class_weights_sum_to_one():
    for quota_class in QuotaClass:
        w = class_weights(quota_class)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w.keys()) == {m.name for m in MONTAGES_BY_CLASS[quota_class]}


def test_class_c_pessimistic_scenario_is_hyser_only():
    w = class_weights(QuotaClass.C_GRIGLIE, c_pessimistic=True)
    assert w["hyser"] == 1.0
    assert w["capgmyo"] == 0.0
    assert w["csl_hdemg"] == 0.0


def test_class_c_mixed_scenario_is_uniform():
    w = class_weights(QuotaClass.C_GRIGLIE, c_pessimistic=False)
    assert abs(w["hyser"] - 1 / 3) < 1e-9


def test_full_montage_weights_sums_to_one_and_respects_class_quota():
    quotas = {QuotaClass.A_RADI: 40.0, QuotaClass.B_ANELLI: 35.0, QuotaClass.C_GRIGLIE: 25.0}
    w = full_montage_weights(quotas, c_pessimistic=True)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    class_a_total = sum(w[m.name] for m in MONTAGES_BY_CLASS[QuotaClass.A_RADI])
    assert abs(class_a_total - 0.40) < 1e-9


def test_full_montage_weights_class_c_pessimistic_all_on_hyser():
    quotas = {QuotaClass.A_RADI: 40.0, QuotaClass.B_ANELLI: 35.0, QuotaClass.C_GRIGLIE: 25.0}
    w = full_montage_weights(quotas, c_pessimistic=True)
    assert abs(w["hyser"] - 0.25) < 1e-9
    assert w["capgmyo"] == 0.0
