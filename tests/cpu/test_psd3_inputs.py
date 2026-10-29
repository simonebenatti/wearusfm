"""Pinned cache inventory/load and subject-fold isolation for PSD-3."""

import copy
import io
import json
import zipfile

import numpy as np
import pytest


def test_byte_string_labels_are_json_safe_unicode(tmp_path):
    path = tmp_path / "bytes.npz"
    np.savez(path, x_pool=np.ones((2, 4)), y=np.asarray([b"rest", b"move"]),
             s=np.asarray([b"a", b"b"]))
    _, y, subjects, provenance = PI.load_cache_view(path,
        {"bytes": path.stat().st_size, "sha256": PI.hash_file(path)}, "pool")
    assert y.tolist() == ["rest", "move"] and subjects.tolist() == ["a", "b"]
    assert len(provenance["canonical_rows_sha256"]) == 64

from wearusfm.harness import psd3_inputs as PI


def _cache(tmp_path, **changes):
    values = dict(x_pool=np.arange(24, dtype=np.float32).reshape(6, 4),
                  x_local=np.ones((6, 6), dtype=np.float16),
                  x_dec=np.arange(36, dtype=np.float16).reshape(6, 6),
                  y=np.array(["noGesture", "wave", "noGesture", "wave", "wave", "noGesture"]),
                  s=np.array(["a", "a", "b", "b", "c", "c"]),
                  subjects=np.array(["a", "b", "c"]),
                  encoder_dim=np.array(2), channels=np.array(3),
                  checkpoint_sha256=np.array("a" * 64), input_sha256=np.array("b" * 64),
                  splits_sha256=np.array("c" * 64), code_sha=np.array("d" * 40),
                  extraction_recipe=np.array(json.dumps({"encoder": "teacher", "layers": ["pool", "local", "dec"]})))
    values.update(changes)
    path = tmp_path / "cache.npz"
    np.savez(path, **values)
    entry = {"sha256": PI.hash_file(path), "bytes": path.stat().st_size,
             "checkpoint_sha256": "a" * 64, "input_sha256": "b" * 64,
             "splits_sha256": "c" * 64, "code_sha": "d" * 40,
             "extraction_recipe": {"encoder": "teacher", "layers": ["pool", "local", "dec"]},
             "n_windows": 6, "n_subjects": 3, "encoder_dim": 2, "channels": 3}
    return path, entry


def test_inventory_reads_headers_not_feature_payloads_or_hash_by_default(tmp_path, monkeypatch):
    path, entry = _cache(tmp_path)
    reads = []
    original = np.lib.format.read_array

    def read_small(stream, *args, **kwargs):
        reads.append(stream.name)
        assert stream.name not in ("x_pool.npy", "x_local.npy", "x_dec.npy", "y.npy", "s.npy", "subjects.npy")
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", read_small)
    monkeypatch.setattr(PI, "hash_file", lambda p: pytest.fail("unrequested full-file hash"))
    result = PI.inspect_cache(path)
    assert result["sha256"] is None and result["bytes"] == entry["bytes"]
    assert result["arrays"]["x_dec"] == {"shape": [6, 6], "dtype": "<f2", "fortran_order": False,
                                         "uncompressed_bytes": 72}
    assert result["metadata"]["extraction_recipe"] == entry["extraction_recipe"]
    assert "encoder_dim.npy" in reads


@pytest.mark.parametrize("view,width,dtype", [("pool", 4, np.float32), ("dec", 6, np.float16)])
def test_load_string_labels_and_only_requested_feature_view(tmp_path, monkeypatch, view, width, dtype):
    path, entry = _cache(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def tracked(cache, key):
        reads.append(key)
        return original(cache, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    x, y, s, provenance = PI.load_cache_view(path, entry, view)
    assert reads == ["x_" + view, "y", "s"]
    assert x.shape == (6, width) and x.dtype == dtype
    assert y[0] == "noGesture" and s.tolist() == ["a", "a", "b", "b", "c", "c"]
    assert provenance["sha256"] == entry["sha256"]
    assert len(provenance["ordered_rows_sha256"]) == 64
    assert PI.inspect_cache(path, compute_sha=True)["sha256"] == entry["sha256"]


def test_numeric_labels_and_subjects_are_valid_cache_inputs(tmp_path):
    path, entry = _cache(tmp_path, y=np.array([0, 1, 0, 1, 1, 0]), s=np.array([1, 1, 2, 2, 3, 3]))
    _, y, s, _ = PI.load_cache_view(path, entry, "pool")
    assert y.dtype.kind == s.dtype.kind == "i"


@pytest.mark.parametrize("field,value,message", [
    ("bytes", 1, "byte count"), ("sha256", "A" * 64, "lowercase SHA256"),
    ("n_windows", 7, "n_windows"), ("n_subjects", 4, "n_subjects"),
])
def test_rejects_invalid_size_hash_and_count_pins(tmp_path, field, value, message):
    path, entry = _cache(tmp_path)
    entry[field] = value
    with pytest.raises(ValueError, match=message):
        PI.load_cache_view(path, entry, "pool")


def test_does_not_enable_unplanned_feature_view(tmp_path):
    path, entry = _cache(tmp_path)
    with pytest.raises(ValueError, match="view must be pool or dec"):
        PI.load_cache_view(path, entry, "local")


def test_rejects_same_size_archive_corruption_before_loading_arrays(tmp_path, monkeypatch):
    path, entry = _cache(tmp_path)
    content = path.read_bytes()
    # Change one float payload byte, preserving the ZIP directory and size.
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("x_pool.npy")
        offset = member.header_offset + 30 + len(member.filename) + len(member.extra) + 140
    changed = bytearray(content)
    changed[offset] ^= 1
    path.write_bytes(changed)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("must fail before array load"))
    with pytest.raises(ValueError, match="SHA256"):
        PI.load_cache_view(path, entry, "pool")


@pytest.mark.parametrize("pin", ["checkpoint_sha256", "input_sha256", "splits_sha256", "code_sha", "extraction_recipe"])
def test_rejects_incorrect_metadata_pins(tmp_path, pin):
    path, entry = _cache(tmp_path)
    entry[pin] = {} if pin == "extraction_recipe" else "0" * 64
    with pytest.raises(ValueError, match="metadata pin differs"):
        PI.load_cache_view(path, entry, "pool")


@pytest.mark.parametrize("key,value,message", [
    ("y", np.array([0, 1, 0, 1, np.nan, 0]), "nonfinite"),
    ("s", np.array([1, 1, 2, 2, np.inf, 3]), "nonfinite"),
    ("y", np.array(["a", "a", "a", "a", " ", "b"]), "empty/blank"),
    ("s", np.array(["a", "a", "b", "b", "", "c"]), "empty/blank"),
    ("y", np.array([0, 1, 0, 1, 1, 0], dtype=object), "object dtype"),
    ("y", np.array([0, 1, 0, 1, 1, 0], dtype=complex), "unsupported y dtype"),
    ("s", np.array([["a"], ["a"], ["b"], ["b"], ["c"], ["c"]]), "vector"),
    ("x_pool", np.full((6, 4), np.nan), "nonfinite"),
    ("x_pool", np.zeros((6, 3)), "pool width"),
    ("x_dec", np.zeros((6, 5)), "dec width"),
    ("x_pool", np.zeros((5, 4)), "row counts"),
])
def test_rejects_invalid_real_cache_contents(tmp_path, key, value, message):
    path, entry = _cache(tmp_path, **{key: value})
    with pytest.raises(ValueError, match=message):
        PI.load_cache_view(path, entry, "dec" if key == "x_dec" else "pool")


def test_rejects_duplicate_zip_members_and_truncated_npy(tmp_path):
    stream = io.BytesIO()
    np.save(stream, np.zeros((2, 2), dtype=np.float32))
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x_pool.npy", stream.getvalue()[:-1])
    with pytest.raises(ValueError, match="payload size"):
        PI.inspect_cache(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x_pool.npy", stream.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("x_pool.npy", stream.getvalue())
    with pytest.raises(ValueError, match="duplicate"):
        PI.inspect_cache(path)


def test_rejects_file_change_during_load(tmp_path, monkeypatch):
    path, entry = _cache(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__

    def mutate(cache, key):
        result = original(cache, key)
        if key == "s":
            # Append without invalidating the still-open archive's contents.
            with path.open("ab") as stream:
                stream.write(b"changed")
        return result

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", mutate)
    with pytest.raises(ValueError, match="cache changed"):
        PI.load_cache_view(path, entry, "pool")


def _roles():
    return [{"train": ["b"], "val": ["c"], "held": ["a"]},
            {"train": ["c"], "val": ["a"], "held": ["b"]},
            {"train": ["a"], "val": ["b"], "held": ["c"]}]


def test_roles_accept_repeated_row_subjects_without_mutating_inputs():
    roles = _roles()
    before = copy.deepcopy(roles)
    PI.validate_roles(roles, ["a", "a", "b", "c"])
    assert roles == before


@pytest.mark.parametrize("mixed_location", ["subjects", "role"])
def test_roles_reject_mixed_ids_before_numpy_string_coercion(mixed_location):
    subjects = ["1", "a", "b", "c"]
    roles = []
    for held in subjects:
        others = [s for s in subjects if s != held]
        roles.append({"train": others[:2], "val": others[2:], "held": [held]})
    # Both variants would become a valid partition after np.asarray silently
    # converts integer 1 into string "1". They must fail before that conversion.
    if mixed_location == "subjects":
        subjects[0] = 1
    else:
        roles[3]["train"][0] = 1  # [1, "a"] must not become ["1", "a"].
    with pytest.raises(ValueError, match="ID types"):
        PI.validate_roles(roles, subjects)


def test_roles_reject_text_bytes_mixture_before_coercion():
    with pytest.raises(ValueError, match="ID types"):
        PI.validate_roles(_roles(), [b"a", "b", "c"])


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r[0].update(train=["a", "b"]), "overlap"),
    (lambda r: r[0].update(train=["b", "b"]), "repeats"),
    (lambda r: r[0].update(train=["outside"]), "partition"),
    (lambda r: r.pop(), "exactly once"),
    (lambda r: r.append(copy.deepcopy(r[0])), "exactly once"),
    (lambda r: r[0].update(val=[]), "nonempty"),
    (lambda r: r[0].update(held=[1]), "ID types"),
])
def test_roles_reject_overlap_missing_extra_or_repeated_held_subjects(mutation, message):
    roles = _roles()
    mutation(roles)
    with pytest.raises(ValueError, match=message):
        PI.validate_roles(roles, ["a", "b", "c"])
