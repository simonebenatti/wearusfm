"""`scripts/fix_capgmyo_grid.py` (opzione A del 07/10/2026): sidecar di CapgMyo da 8 x 16 a 16 x 8, manifest v1.1 col nuovo sha256 che il
dataloader accetta, scale bipolari di CapgMyo tolte, backup, prova a vuoto e rilancio senza effetti."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_pretraining_loader import _tree
from wearusfm.data import pretraining_loader as L
from wearusfm.ingest import capgmyo

_spec = importlib.util.spec_from_file_location("fix_capgmyo_grid", Path(__file__).resolve().parents[2] / "scripts" / "fix_capgmyo_grid.py")
FIX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIX)


def _old_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(capgmyo, "grid_position", lambda i: divmod(i, 16))  # la griglia vecchia, 8 x 16
    root, mpath = _tree(tmp_path)
    monkeypatch.undo()
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps({"scales": {"capgmyo/s01/s": 1.0, "capgmyo/s01/s|bip:row:1": 2.0, "emg2pose/u1/sessA": 3.0}}))
    return root, mpath, scales


def _coords(meta):
    return [(c["sensor_coords"]["grid_row"], c["sensor_coords"]["grid_col"]) for c in meta["montage"]["groups"][0]["channels"]]


def test_fix_rewrites_the_grid_and_the_manifest(tmp_path, monkeypatch):
    root, mpath, scales = _old_tree(tmp_path, monkeypatch)
    side = root / "capgmyo" / "s01" / "metadata.json"
    before = side.read_bytes()
    assert _coords(json.loads(before))[16] == (1, 0)  # vecchia: canale 16 sulla seconda riga
    out = tmp_path / "fix"
    assert FIX.main(["--manifest", str(mpath), "--root", str(root), "--scales", str(scales), "--out-dir", str(out), "--dry-run"]) == 0
    assert side.read_bytes() == before and not out.exists()  # prova: nulla scritto
    assert FIX.main(["--manifest", str(mpath), "--root", str(root), "--scales", str(scales), "--out-dir", str(out)]) == 0
    meta = json.loads(side.read_text())
    assert _coords(meta)[8] == (1, 0) and _coords(meta)[127] == (15, 7) and len(set(_coords(meta))) == 128
    assert (out / "backup" / "s01" / "s" / "metadata.json").read_bytes() == before
    rep = json.loads((out / "report.json").read_text())
    assert rep["rows_updated"] == 1 and rep["bipolar_scales_dropped"] == 1 and rep["sessions"][0]["changed"]
    sc = json.loads((out / "session_scales_without_capgmyo_bipolar.json").read_text())["scales"]
    assert set(sc) == {"capgmyo/s01/s", "emg2pose/u1/sessA"}
    new = out / "manifest_v1_1.json.gz"
    old_doc, new_doc = (json.loads(gzip.decompress(p.read_bytes())) for p in (mpath, new))
    assert new_doc["params"]["revision"]["version"] == "v1.1"
    diff = [(a, b) for a, b in zip(old_doc["rows"], new_doc["rows"]) if a != b]
    assert len(diff) == 1 and diff[0][0][0] == "capgmyo"  # cambia solo la riga di CapgMyo, solo lo sha256
    idx = L.ManifestIndex.load(new, [root])
    row = next(r for r in idx.rows if r["dataset"] == "capgmyo")
    view = L.PretrainLoader(idx, L.signed_config(None)).view(row)  # lo sha256 del manifest v1.1 combacia col sidecar nuovo
    assert view.arr.shape == (10, 1000, 128)
    old_idx = L.ManifestIndex.load(mpath, [root])
    old_row = next(r for r in old_idx.rows if r["dataset"] == "capgmyo")
    with pytest.raises(ValueError):
        L.PretrainLoader(old_idx, L.signed_config(None)).view(old_row)  # il manifest vecchio lo rifiuta
    sha_after = rep["sessions"][0]["sha256_new"]
    assert FIX.main(["--manifest", str(new), "--root", str(root), "--scales", str(scales), "--out-dir", str(tmp_path / "again")]) == 0
    again = json.loads((tmp_path / "again" / "report.json").read_text())
    assert not again["sessions"][0]["changed"] and again["sessions"][0]["sha256_new"] == sha_after  # rilancio: nulla cambia


def test_unexpected_montage_is_refused():
    with pytest.raises(ValueError):
        FIX.fixed_meta({"montage": {"groups": [{"topology": "ring", "channels": []}]}})
    meta = {"montage": {"groups": [{"topology": "grid_2d", "channels": [{"sensor_coords": {"channel_index": (i + 1) % 128, "grid_row": 0,
                                                                                             "grid_col": 0}} for i in range(128)]}]}}
    with pytest.raises(ValueError):
        FIX.fixed_meta(meta)
    assert np.array_equal(np.array(capgmyo.grid_position(9)), [1, 1])
