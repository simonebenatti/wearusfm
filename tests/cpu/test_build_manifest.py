"""scripts/build_manifest.py e src/wearusfm/data/pretraining_manifest.py su un albero processato sintetico (solo sidecar e array piccoli)."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wearusfm.data import pretraining_manifest as M

_spec = importlib.util.spec_from_file_location("build_manifest", Path(__file__).resolve().parents[2] / "scripts" / "build_manifest.py")
BM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BM)


def _session(d: Path, n: int, c: int, fs: float, qc=None, runs=None, trials=None):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", np.zeros((n, c), np.int16))
    chans = [{"qc_valid": True if qc is None else bool(qc[i])} for i in range(c)]
    meta = {"montage": {"groups": [{"channels": chans}]}, "native_fs_hz": fs}
    if runs:
        meta["constant_runs"] = runs
    if trials:
        meta["trials"] = trials
    (d / "metadata.json").write_text(json.dumps(meta))


def _tree(tmp_path):
    a, b = tmp_path / "w", tmp_path / "s"
    # NinaPro DB2 (classe A): due soggetti in pretraining, uno in test; un canale scartato e un buco su un canale valido in s01
    _session(a / "ninapro_db2" / "s01" / "session1", 7200, 4, 2000.0, qc=[1, 1, 1, 0], runs=[{"channel": 0, "start": 0, "n_samples": 2000},
                                                                                            {"channel": 3, "start": 0, "n_samples": 500}],
             trials=[{"offset": 0, "n_samples": 3600}, {"offset": 3600, "n_samples": 3600}])
    _session(a / "ninapro_db2" / "s02" / "session1", 3600, 4, 2000.0)
    _session(a / "ninapro_db2" / "s03" / "session1", 3600, 4, 2000.0)
    # emg2qwerty (classe B) e Zhang (anatomical A, random B) e Kaifosh (benchmark)
    _session(b / "emg2qwerty" / "u1" / "sessA", 72000, 2, 2000.0)
    _session(b / "zhang2026" / "HG_1" / "anatomical", 4000, 8, 2000.0)
    _session(b / "zhang2026" / "HG_1" / "random", 4000, 8, 2000.0)
    _session(b / "kaifosh" / "u000" / "dataset000", 2000, 16, 2000.0)
    splits = {"datasets": {
        "ninapro_db2": {"pretraining": ["s01", "s02"], "test": ["s03"], "nested": {"0.5": ["s01"]}},
        "emg2qwerty": {"pretraining": ["u1"], "test": [], "nested": {"0.5": ["u1"]}},
        "zhang2026": {"pretraining": ["HG_1"], "test": [], "nested": {}},
        "kaifosh": {"benchmark": ["u000"], "pretraining": [], "test": []}}}
    return [a, b], splits


def test_rows_accounting_split_and_units(tmp_path):
    roots, splits = _tree(tmp_path)
    rows, summary = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 1e9, 25.0)
    by = {(r.dataset, r.subject, r.session): r for r in rows}
    s01 = by[("ninapro_db2", "s01", "session1")]
    assert (s01.n_valid, s01.excluded_channel_samples, s01.n_segments, s01.nested) == (3, 2000, 2, ["0.5"])  # il buco sul canale scartato non conta
    assert s01.d_c(0.025) == pytest.approx((7200 * 3 - 2000) / 2000 / 0.025) and s01.d_t(0.025) == pytest.approx(7200 / 2000 / 0.025)
    assert by[("ninapro_db2", "s03", "session1")].split == "test" and by[("kaifosh", "u000", "dataset000")].split == "benchmark"
    assert by[("zhang2026", "HG_1", "anatomical")].quota_class == "A" and by[("zhang2026", "HG_1", "random")].quota_class == "B"
    assert M.rvq_status("ninapro_db8") == "on" and M.rvq_status("zhang2026") == "on" and M.rvq_status("ninapro_db10") == "pending"
    assert by[("kaifosh", "u000", "dataset000")].rvq == "on" and M.rvq_status("putemg") == "off"
    assert summary["totals"]["pretraining_sessions"] == 5


def test_weights_follow_quota_and_hours_and_sum_to_one(tmp_path):
    roots, splits = _tree(tmp_path)
    rows, summary = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 1.0, 1e9, 25.0)
    w = {(r.dataset, r.session if r.dataset == "zhang2026" else r.subject): r.weight for r in rows}
    assert sum(r.weight for r in rows) == pytest.approx(1.0)
    assert w[("ninapro_db2", "s03")] == 0.0 and w[("kaifosh", "u000")] == 0.0  # test e benchmark non si campionano
    a_total = w[("ninapro_db2", "s01")] + w[("ninapro_db2", "s02")] + w[("zhang2026", "anatomical")]
    assert a_total == pytest.approx(0.3)
    assert w[("ninapro_db2", "s01")] / w[("ninapro_db2", "s02")] == pytest.approx(2.0)  # dentro un dataset, in proporzione alle ore


def test_hash_ignores_discovery_order_and_changes_with_params(tmp_path):
    roots, splits = _tree(tmp_path)
    rows, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    p = {"quota": {"A": 0.3, "B": 0.7}}
    assert M.manifest_hash(p, rows) == M.manifest_hash(p, list(reversed(rows)))
    assert M.manifest_hash(p, rows) != M.manifest_hash({"quota": {"A": 0.2, "B": 0.8}}, rows)


def test_missing_subject_or_dataset_root_is_an_error(tmp_path):
    roots, splits = _tree(tmp_path)
    splits["datasets"]["ninapro_db2"]["pretraining"] = ["s01"]  # s02 non e' in nessuno split
    with pytest.raises(KeyError, match="s02"):
        BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    roots2, splits2 = _tree(tmp_path / "x")
    splits2["datasets"]["hyser"] = {"pretraining": [], "test": []}
    with pytest.raises(FileNotFoundError, match="hyser"):
        BM.build(roots2, splits2, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)


def test_main_writes_gzipped_manifest(tmp_path, capsys):
    roots, splits = _tree(tmp_path)
    sp = tmp_path / "splits.json"
    sp.write_text(json.dumps(splits))
    out = tmp_path / "m.json.gz"
    BM.main(["--root", str(roots[0]), "--root", str(roots[1]), "--splits", str(sp), "--out", str(out), "--quota", "A=0.3,B=0.7"])
    doc = json.loads(gzip.open(out).read())
    assert doc["params"]["version"] == "draft" and len(doc["rows"]) == 7 and len(doc["hash"]) == 64
    assert doc["columns"][:4] == ["dataset", "subject", "session", "split"]
    assert "manifest draft: 7 sessioni" in capsys.readouterr().out
