import numpy as np

from wearusfm.data.manifest import MONTAGES_BY_NAME
from wearusfm.data.shards import read_shard, write_shard
from wearusfm.data.synthetic import generate_sample


def test_write_then_read_roundtrip(tmp_path):
    montage = MONTAGES_BY_NAME["ninapro_standard"]
    rng = np.random.default_rng(0)
    sample = generate_sample(montage, rng, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)

    write_shard(tmp_path, "shard_0000", sample)
    back = read_shard(tmp_path, "shard_0000")

    assert back.montage_name == sample.montage_name
    assert back.quota_class_origin == sample.quota_class_origin
    assert back.quota_class_presented == sample.quota_class_presented
    assert back.c_presented == sample.c_presented
    assert back.n_samples == sample.n_samples
    assert back.fs_native_hz == sample.fs_native_hz
    assert back.front_end_fs_hz == sample.front_end_fs_hz
    assert np.array_equal(np.asarray(back.data), sample.data)


def test_read_shard_is_memory_mapped_by_default(tmp_path):
    montage = MONTAGES_BY_NAME["capgmyo"]
    rng = np.random.default_rng(1)
    sample = generate_sample(montage, rng, p_piena=1.0, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)

    write_shard(tmp_path, "shard_0001", sample)
    back = read_shard(tmp_path, "shard_0001")

    assert isinstance(back.data, np.memmap)


def test_write_shard_creates_both_files(tmp_path):
    montage = MONTAGES_BY_NAME["camargo2021"]
    rng = np.random.default_rng(2)
    sample = generate_sample(montage, rng, montage_dropout_group_p=0.0, montage_dropout_channel_p=0.0)

    write_shard(tmp_path, "shard_0002", sample)
    assert (tmp_path / "shard_0002.bin").exists()
    assert (tmp_path / "shard_0002.json").exists()
