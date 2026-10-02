"""scripts/build_manifest.py e src/wearusfm/data/pretraining_manifest.py su un albero processato sintetico (solo sidecar e array piccoli)."""

import gzip
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from wearusfm.data import pretraining_manifest as M

_spec = importlib.util.spec_from_file_location("build_manifest", Path(__file__).resolve().parents[2] / "scripts" / "build_manifest.py")
BM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BM)


def _session(d: Path, n: int, c: int, fs: float, qc=None, runs=None, trials=None, time_axis=None):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", np.zeros((n, c), np.int16))
    chans = [{"qc_valid": True if qc is None else bool(qc[i])} for i in range(c)]
    meta = {"montage": {"groups": [{"channels": chans}]}, "native_fs_hz": fs}
    if runs:
        meta["constant_runs"] = runs
    if trials:
        meta["trials"] = trials
    if time_axis:
        meta["time_axis"] = time_axis
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
    # 72000 campioni a 2 kHz = 36 s di campioni, ma i timestamp coprono 36,5 s: 0,5 s mancanti in 2 salti
    _session(b / "emg2qwerty" / "u1" / "sessA", 72000, 2, 2000.0,
             time_axis={"n_gaps": 2, "duration_s": 71999 / 2000.0 + 0.5, "dt_max_s": 0.4005, "gaps_truncated": False,
                        "gaps": [{"index": 1000, "dt_s": 0.1}, {"index": 70000, "dt_s": 0.4005}]})  # tratti di 0,5 s, 34,5 s e 1 s
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
    assert M.rvq_status("ninapro_db8") == "on" and M.rvq_status("zhang2026") == "on" and M.rvq_status("ninapro_db10") == "on"
    assert by[("kaifosh", "u000", "dataset000")].rvq == "on" and M.rvq_status("putemg") == "off" and M.rvq_status("nuovo") == "pending"
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
    import copy

    roots, splits = _tree(tmp_path)
    rows, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    p = {"quota": {"A": 0.3, "B": 0.7}}
    shuffled = copy.deepcopy(rows)[::-1]  # stesse sessioni scoperte in un altro ordine: si ripesano dopo l'ordine canonico
    M.assign_weights(M.sort_rows(shuffled), {"A": 0.3, "B": 0.7}, 0.5, 8.0)
    assert M.manifest_hash(p, shuffled) == M.manifest_hash(p, rows)
    assert [r.weight for r in M.sort_rows(shuffled)] == [r.weight for r in rows]  # pesi identici bit per bit
    assert M.manifest_hash(p, rows) != M.manifest_hash({"quota": {"A": 0.2, "B": 0.8}}, rows)


def test_session_level_test_split_and_guards(tmp_path, capsys):
    roots, splits = _tree(tmp_path)
    _session(roots[1] / "emg2qwerty" / "u1" / "sessB", 2000, 2, 2000.0)
    splits["datasets"]["emg2qwerty"]["test_sessions"] = ["sessB", "sessX"]  # sessX non esiste: va segnalata
    splits["datasets"]["ninapro_db2"]["test"].append("s09")  # soggetto senza sessioni: va segnalato
    rows, summary = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    by = {(r.dataset, r.session): r for r in rows}
    assert by[("emg2qwerty", "sessB")].split == "test" and by[("emg2qwerty", "sessB")].weight == 0.0
    assert by[("emg2qwerty", "sessA")].split == "pretraining"
    assert summary["subjects_without_sessions"] == {"ninapro_db2": ["s09"]} and summary["test_sessions_not_found"] == {"emg2qwerty": 1}
    splits["datasets"]["ninapro_db2"]["test"].append("s01")  # s01 anche in pretraining
    with pytest.raises(ValueError, match="s01"):
        BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)


def test_frozen_version_refuses_draft_splits_and_missing_subjects(tmp_path):
    roots, splits = _tree(tmp_path)
    sp = tmp_path / "splits.json"
    base = ["--root", str(roots[0]), "--root", str(roots[1]), "--splits", str(sp), "--out", str(tmp_path / "m.json"), "--quota", "A=0.3,B=0.7"]
    sp.write_text(json.dumps({**splits, "draft": True}))
    with pytest.raises(SystemExit, match="bozza"):
        BM.main(base + ["--version", "manifest-v1"])
    splits["datasets"]["ninapro_db2"]["test"].append("s09")
    sp.write_text(json.dumps(splits))
    with pytest.raises(SystemExit, match="senza sessioni"):
        BM.main(base + ["--version", "manifest-v1"])
    BM.main(base)  # in bozza si scrive comunque, con l'avviso


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


def test_build_is_canonical_even_if_sessions_are_discovered_in_reverse(tmp_path, monkeypatch):
    roots, splits = _tree(tmp_path)
    rows, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    real = BM.discover_sessions
    monkeypatch.setattr(BM, "discover_sessions", lambda root, ds: list(reversed(real(root, ds))))
    rows_rev, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    assert [(r.dataset, r.subject, r.session, r.weight) for r in rows_rev] == [(r.dataset, r.subject, r.session, r.weight) for r in rows]



def test_time_axis_gaps_are_counted_per_session_and_unit(tmp_path):
    roots, splits = _tree(tmp_path)
    rows, summary = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    r = next(x for x in rows if x.dataset == "emg2qwerty")
    assert r.n_time_gaps == 2 and r.missing_time_s == pytest.approx(0.5) and r.max_gap_s == pytest.approx(0.4005)
    g = summary["units"]["emg2qwerty"]["time_gaps"]
    assert g == {"sessions": 1, "gaps": 2, "missing_s": pytest.approx(0.5), "max_gap_s": pytest.approx(0.4005), "truncated_sessions": 0,
                 "out_of_range": 0}
    assert summary["units"]["ninapro_db2"]["time_gaps"]["sessions"] == 0


def test_segment_lengths_split_at_gaps_inside_trials_only():
    trials = [{"offset": 0, "n_samples": 100}, {"offset": 100, "n_samples": 50}]
    assert M.segment_lengths(150, trials) == [100, 50]
    assert M.segment_lengths(150, trials, [30, 100, 120, 999]) == [30, 70, 20, 30]  # 100 e' gia' un bordo, 999 e' fuori
    assert M.segment_lengths(150, [], [30, 30]) == [30, 120] and M.segment_lengths(150, None) == [150]
    assert M.short_time_s([30, 70, 20, 30], 10.0, (3.0, 7.5)) == [pytest.approx(2.0), pytest.approx(15.0)]  # solo 20 < 30; 30 ci sta


def test_short_segments_per_session_and_unit(tmp_path):
    roots, splits = _tree(tmp_path)
    rows, summary = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0)
    by = {(r.dataset, r.session): r for r in rows}
    q = by[("emg2qwerty", "sessA")]  # 36 s senza salti; con i salti: 0,5 s + 34,5 s + 1 s
    assert q.short_segments_s == [0.0] * 4 and q.short_segments_gaps_s == [pytest.approx(x) for x in (0.5, 1.5, 1.5, 1.5)]
    assert q.gaps_out_of_range == 0 and q.outside_trials_s == 0.0
    z = next(r for r in rows if r.unit == "zhang2026:anatomical")
    assert z.short_segments_s == [0.0, 0.0, pytest.approx(2.0), pytest.approx(2.0)]  # 2 s esatti: ci sta una finestra da 2 s
    s = summary["units"]["ninapro_db2"]["short_segments"]  # s01 (due prove da 1,8 s) e s02 (1,8 s) in pretraining, s03 in test: non conta
    assert s["pretraining_s"] == pytest.approx(5.4) and s["trials_s"] == [0.0, pytest.approx(5.4), pytest.approx(5.4), pytest.approx(5.4)]
    assert summary["units"]["emg2qwerty"]["short_segments"]["trials_and_gaps_s"][0] == pytest.approx(0.5)


def test_short_segments_3d_arrays_partial_trials_and_bad_gap_positions(tmp_path):
    splits = {"datasets": {"ninapro_db2": {"pretraining": ["s01"], "test": []}}}
    d = tmp_path / "ninapro_db2" / "s01" / "a"
    d.mkdir(parents=True)
    np.save(d / "data_int16.npy", np.zeros((3, 1000, 2), np.int16))  # tre prove da 0,5 s
    (d / "metadata.json").write_text(json.dumps({"montage": {"groups": [{"channels": [{"qc_valid": True}] * 2}]}, "native_fs_hz": 2000.0}))
    r = BM.session_row(d, "ninapro_db2", "s01", "a", splits)
    assert r.n_segments == 3 and r.short_segments_s == [pytest.approx(1.5)] * 4 and r.outside_trials_s == 0.0
    _session(tmp_path / "x", 6000, 2, 2000.0, trials=[{"offset": 0, "n_samples": 4000}],
             time_axis={"n_gaps": 2, "duration_s": 3.0, "dt_max_s": 0.01, "gaps": [{"index": 2000, "dt_s": 0.01}, {"index": 9000, "dt_s": 0.01}]})
    r = BM.session_row(tmp_path / "x", "ninapro_db2", "s01", "x", splits)
    assert r.outside_trials_s == pytest.approx(1.0) and r.gaps_out_of_range == 1  # 2000 campioni fuori dalle prove; 9000 oltre la fine
    assert r.short_segments_s == [0.0, 0.0, pytest.approx(2.0), pytest.approx(2.0)]
    assert r.short_segments_gaps_s == [0.0, pytest.approx(2.0), pytest.approx(2.0), pytest.approx(2.0)]  # 1 s + 1 s


def test_parallel_reading_gives_identical_rows_and_logs_progress(tmp_path, capsys):
    roots, splits = _tree(tmp_path)
    serial, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0, io_threads=1)
    parallel, _ = BM.build(roots, splits, {"A": 0.3, "B": 0.7}, 0.5, 8.0, 25.0, io_threads=4)
    assert [asdict(r) for r in parallel] == [asdict(r) for r in serial]
    out = capsys.readouterr().out
    assert "ninapro_db2: 3 sessioni" in out and "emg2qwerty: 1 sessioni" in out  # una riga per dataset, scritta subito (flush)
