"""Historical producer identity, target support and label-independent feature path."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import data as D


def fixture(users=("u0", "u1", "u2", "u3"), windows=2):
    rng = np.random.default_rng(19)
    n = len(users)*windows
    y = rng.normal(size=(n, 160, 20)).astype(np.float32)
    mask = rng.random(y.shape) > .1
    producer = {"which":"teacher","split":"train","checkpoint_step":49046,
                "checkpoint_sha256":"a"*64,"code_sha":"b"*40,"source_sha256":"c"*64,
                "recipe_sha256":"d"*64,"pose_stats_sha256":"e"*64,"checkpoint_pose_stats_sha256":"e"*64}
    config = {**producer,"n_windows":10,"window_s":4.,"seed":20261026,"per_side":False}
    result = {"target":y,"mask":mask,"std":np.ones((n,20),dtype=np.float32),
              "pose_var":np.ones(n,dtype=np.float32),"pred":y*.5,
              "lat":rng.normal(size=(n,160,384)).astype(np.float16),
              "pre":rng.normal(size=(n,160,384)).astype(np.float16),
              "key_time_valid":np.ones((n,160),dtype=bool),"dim":384,"has_head":True,
              "n_recordings":n,"config":config,
              "meta":[{"user":u,"n_patches":160,"recording":i,"start":0}
                      for i,u in enumerate(np.repeat(users,windows))]}
    return result, producer


def test_producer_is_pinned_separately_from_reader_and_changed_hash_rejected():
    data, producer = fixture()
    assert D.validate_cache(data, producer=producer)["historical_producer"] == producer
    # New reader source hash is intentionally not accepted as the old producer.
    changed = {**producer,"source_sha256":"f"*64}
    with pytest.raises(ValueError, match="historical producer mismatch: source_sha256"):
        D.validate_producer(data, changed)
    with pytest.raises(ValueError, match="missing pinned historical"):
        D.validate_producer(data, {"source_sha256":"f"*64})


def test_feature_path_ignores_all_pose_values_and_masks():
    data, _ = fixture()
    expected = D.view_inputs(data, "post", np.array([0,2]))
    data["target"][:] = np.nan
    data["mask"][:] = False
    data["pose_var"][:] = np.nan
    np.testing.assert_array_equal(D.view_inputs(data, "post", np.array([0,2])), expected)
    assert expected.dtype == np.float32


def test_crop_retains_central_label_when_neighbor_label_missing():
    mask = np.ones((2,160,20),dtype=bool)
    mask[0,5] = False
    mask[1,100:] = False
    lengths = np.array([160,100])
    valid = np.arange(160)[None,:] < lengths[:,None]
    result = D.central_support(mask,lengths,valid)
    assert result.shape == (2,148,20) and result[0,0].all()
    assert result[1,:88].all() and not result[1,88:].any()
    invalid = valid.copy(); invalid[0,8] = False
    with pytest.raises(ValueError,match="real context"):
        D.central_support(mask,lengths,invalid)


def test_padded_values_are_not_features_and_cannot_be_targets():
    data, _ = fixture()
    data["meta"][0]["n_patches"] = 80
    data["lat"][0,80:] = np.nan
    data["mask"][0,80:] = False
    data["key_time_valid"][0,80:] = False
    assert (D.view_inputs(data)[0,80:] == 0).all()
    D.central_support(data["mask"],D.lengths_for(data),data["key_time_valid"])
    data["mask"][0,100,0] = True
    with pytest.raises(ValueError,match="padded"):
        D.central_support(data["mask"],D.lengths_for(data))


def test_sorted_user_folds_and_subset_preserve_metadata_alignment():
    data, _ = fixture(users=("z","a","m","b"))
    assert D.alternating_user_folds(data) == [["a","m"],["b","z"]]
    with pytest.raises(ValueError,match="frozen manifest"):
        D.alternating_user_folds(data, [["z","a"],["m","b"]])
    selected = D.subset(data, np.array([6,0]))
    np.testing.assert_array_equal(selected["target"],data["target"][[6,0]])
    assert [m["user"] for m in selected["meta"]] == ["b","z"]
    with pytest.raises(ValueError,match="duplicate"):
        D.subset(data,np.array([0,0]))


def test_read_pinned_cache_checks_reference_and_can_drop_unused_view(tmp_path):
    data, producer = fixture()
    cache, reference = tmp_path/"cache.npz", tmp_path/"reference.npz"
    D.historical.save_cache(cache,data)
    with reference.open("wb") as stream:
        np.savez(stream,**{k:data[k] for k in ("target","mask","std","pose_var","n_recordings")},
                 meta_json=np.array(json.dumps(data["meta"])))
    pin = {"path":str(cache),"sha256":D.file_sha256(cache),"bytes":cache.stat().st_size,"users":4,"windows":8}
    refpin = {"path":str(reference),"sha256":D.file_sha256(reference),"bytes":reference.stat().st_size,
              "n_users":4,"n_windows":8,"n_recordings":8}
    got = D.read_cache_pin(pin,producer=producer,reference_pin=refpin,depth="pre")
    assert "lat" not in got and "pre" not in got
    np.testing.assert_array_equal(D.view_inputs(got,"pre"),data["pre"].astype(np.float32))
    with pytest.raises(ValueError,match="not retained"):
        D.view_inputs(got,"post")
    badpin = {**pin,"sha256":"f"*64}
    with pytest.raises(ValueError,match="SHA"):
        D.read_cache_pin(badpin,producer=producer,reference_pin=refpin)


def test_wrong_geometry_and_prediction_storage_dtype_rejected(tmp_path):
    data, producer = fixture()
    data["dim"] = 383
    with pytest.raises(ValueError,match="depth sequence"):
        D.validate_cache(data,producer=producer)
    path = tmp_path/"prediction.npz"
    np.savez(path,prediction=np.zeros((1,148,20),np.float32))
    with pytest.raises(ValueError,match="float64"):
        D.read_predictions({"path":str(path),"sha256":D.file_sha256(path)})
