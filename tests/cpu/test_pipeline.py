import numpy as np

from wearusfm.data.manifest import MONTAGES_BY_NAME
from wearusfm.data.pipeline import (
    generate_precomputed_sample_for_shard,
    generate_raw_sample_for_shard,
    run_pipeline_on_the_fly,
    run_pipeline_precompute,
)
from wearusfm.data.shards import write_shard


def test_raw_shard_has_full_native_channel_count(tmp_path):
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(0)
    raw = generate_raw_sample_for_shard(montage, rng, context_s=4.0, all_native_fs=False)
    assert raw.c_presented == 256  # nessun sottocampionamento sullo shard raw
    assert raw.quota_class_presented == raw.quota_class_origin


def test_on_the_fly_pipeline_can_reduce_hd_grid(tmp_path):
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(1)
    raw = generate_raw_sample_for_shard(montage, rng, context_s=1.0, all_native_fs=False)
    write_shard(tmp_path, "shard_0000", raw)

    pipe_rng = np.random.default_rng(2)
    sample, stages = run_pipeline_on_the_fly(
        tmp_path, "shard_0000", rng=pipe_rng, p_piena=0.0,
        montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0,
    )
    assert sample.c_presented <= 32  # p_piena=0 -> sempre montaggio virtuale
    assert set(stages.keys()) == {"read", "subsample_dropout", "augment", "masking"}
    assert all(v >= 0 for v in stages.values())


def test_on_the_fly_pipeline_keeps_full_grid_when_p_piena_one(tmp_path):
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(3)
    raw = generate_raw_sample_for_shard(montage, rng, context_s=1.0, all_native_fs=False)
    write_shard(tmp_path, "shard_0001", raw)

    pipe_rng = np.random.default_rng(4)
    sample, _ = run_pipeline_on_the_fly(
        tmp_path, "shard_0001", rng=pipe_rng, p_piena=1.0,
        montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0,
    )
    assert sample.c_presented == 256


def test_precompute_pipeline_reads_smaller_shard(tmp_path):
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(5)
    precomputed = generate_precomputed_sample_for_shard(
        montage, rng, context_s=1.0, all_native_fs=False, p_piena=0.0,
        montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0,
    )
    write_shard(tmp_path, "shard_0002", precomputed)
    assert precomputed.c_presented <= 32

    pipe_rng = np.random.default_rng(6)
    sample, stages = run_pipeline_precompute(tmp_path, "shard_0002", rng=pipe_rng)
    assert sample.c_presented == precomputed.c_presented
    assert stages["subsample_dropout"] == 0.0  # bakato nello shard


def test_precompute_shard_is_smaller_on_disk_than_raw(tmp_path):
    montage = MONTAGES_BY_NAME["hyser"]
    rng = np.random.default_rng(7)
    raw = generate_raw_sample_for_shard(montage, rng, context_s=1.0, all_native_fs=False)
    write_shard(tmp_path, "raw", raw)

    rng2 = np.random.default_rng(8)
    precomputed = generate_precomputed_sample_for_shard(
        montage, rng2, context_s=1.0, all_native_fs=False, p_piena=0.0,
        montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0,
    )
    write_shard(tmp_path, "pre", precomputed)

    raw_bytes = (tmp_path / "raw.bin").stat().st_size
    pre_bytes = (tmp_path / "pre.bin").stat().st_size
    assert pre_bytes < raw_bytes  # e' il vantaggio di I/O che il precompute promette
