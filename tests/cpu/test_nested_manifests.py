"""Manifest annidati per soggetto (D16): il trainer allena solo sui soggetti della frazione (`RunConfig.nested`, `--nested`), la frazione
fa parte della ricetta (una ripresa con un'altra frazione e' rifiutata); `scripts/nested_manifest_report.py` verifica annidamento ed
esclusione di test/benchmark e conta quote, passaggi e passi per epoca."""

import gzip
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from wearusfm.training import run as R

from test_pretraining_loader import _tree

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nested_manifest_report", REPO / "scripts" / "nested_manifest_report.py")
NR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NR)


def _tagged(tmp_path, tags):
    """L'albero di prova con i sottoinsiemi annidati `tags` (lista per riga) nel manifest."""
    root, mpath = _tree(tmp_path)
    doc = json.loads(gzip.decompress(mpath.read_bytes()))
    cols = doc["columns"]
    rows = [dict(zip(cols, r)) for r in doc["rows"]]
    for r, t in zip(rows, tags):
        r["nested"] = t
    doc["rows"] = [[r[c] for c in cols] for r in rows]
    path = tmp_path / "nested.json.gz"
    path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    return root, path, rows


def test_training_on_a_nested_fraction_uses_only_its_subjects(tmp_path):
    root, mpath, rows = _tagged(tmp_path, [[], ["0.25", "0.5"], ["0.5"]])
    cfg = replace(R.small_config(datasets=None, max_steps=2), nested="0.25", log_batch_origin=True)
    out = tmp_path / "run"
    R.train(cfg, mpath, [root], out, log=lambda _: None)
    sessions = json.loads((out / "origin_sessions.json").read_text())
    assert sessions == [R._session_key(rows[1])]  # soltanto la riga del sottoinsieme 0.25
    assert json.loads((out / "config.json").read_text())["nested"] == "0.25"
    with pytest.raises(ValueError, match="nested"):  # la frazione e' nella ricetta: niente ripresa con un'altra
        R.train(replace(cfg, nested="0.5", max_steps=4), mpath, [root], out, log=lambda _: None)
    with pytest.raises(ValueError, match="D16"):
        R.train(replace(cfg, nested="0.3"), mpath, [root], tmp_path / "bad", log=lambda _: None)
    idx = R.load_index(mpath, [root], None, "0.5")
    assert sorted(R._session_key(r) for r in idx.rows) == sorted(R._session_key(r) for r in rows[1:])


def _manifest(tmp_path, rows, ws):
    cols = ["dataset", "subject", "session", "split", "nested", "n_samples", "fs_hz", "unit", "quota_class", "weight"]
    doc = {"columns": cols, "rows": [[r[c] for c in cols] for r in rows], "hash": "h",
           "params": {"version": "manifest-v1", "quota": {"A": 0.5, "B": 0.5}, "max_passes": 8.0, "splits_sha256": "s"}}
    m, w = tmp_path / "m.json.gz", tmp_path / "w.json"
    m.write_bytes(gzip.compress(json.dumps(doc).encode()))
    w.write_text(json.dumps({"manifest": "m", "window_s": ws}))
    return m, w


def _row(ds, subj, sess, nested, hours, weight, cls="A", split="pretraining"):
    return {"dataset": ds, "subject": subj, "session": sess, "split": split, "nested": nested, "n_samples": int(hours * 3600 * 1000),
            "fs_hz": 1000.0, "unit": ds, "quota_class": cls, "weight": weight}


def test_report_counts_fractions_and_steps(tmp_path):
    rows = [_row("a", "s1", "x", ["0.125", "0.25", "0.5"], 1.0, 0.25), _row("a", "s2", "x", ["0.5"], 1.0, 0.25),
            _row("b", "u1", "y", ["0.125", "0.25", "0.5"], 2.0, 0.5, cls="B"), _row("a", "t1", "x", [], 1.0, 0.0, split="test")]
    ws = {"a/s1/x": 2.0, "a/s2/x": 2.0, "b/u1/y": 4.0, "a/t1/x": 2.0}
    m, w = _manifest(tmp_path, rows, ws)
    rep = NR.report(m, w, batch=2, epochs=4.0)
    assert rep["problems"] == []
    half, full = rep["fractions"]["0.5"], rep["fractions"]["1"]
    assert (full["subjects"], full["sessions"], full["hours"]) == (3, 3, 4.0)  # il soggetto di test non conta
    assert rep["fractions"]["0.25"]["sessions"] == 2 and half["sessions"] == 3
    # finestre per epoca = ore x 3600 x somma(peso / durata): 4 h x 3600 x (0,25/2 + 0,25/2 + 0,5/4) = 5400; passi = 2700 col batch 2
    assert full["steps_per_epoch"] == 2700 and full["steps_at_E"] == 10800 and full["steps_per_epoch_4s_windows"] == 1800
    sub = rep["fractions"]["0.125"]  # a perde meta' delle ore: la sua quota scende a 1/3 (pesi rinormalizzati)
    assert sub["units"]["a"]["share"] == pytest.approx(1 / 3, abs=1e-5) and sub["units"]["a"]["share_full"] == 0.5
    assert sub["max_share_shift"] == pytest.approx(1 / 6, abs=1e-5)
    assert "| 0.125 |" in NR.table(rep)


def test_report_flags_broken_nesting_and_tagged_test_rows(tmp_path):
    rows = [_row("a", "s1", "x", ["0.125"], 1.0, 0.5), _row("a", "s2", "x", ["0.25", "0.5"], 1.0, 0.5),
            _row("a", "t1", "x", ["0.5"], 1.0, 0.0, split="test")]
    m, w = _manifest(tmp_path, rows, {"a/s1/x": 2.0, "a/s2/x": 2.0, "a/t1/x": 2.0})
    problems = NR.report(m, w)["problems"]
    assert any("test/benchmark" in p for p in problems) and any("non e' contenuto" in p for p in problems)
    assert NR.main([str(m), str(w)]) == 1
    rows = [_row("a", "s1", "x", ["0.125", "0.25", "0.5"], 1.0, 0.5), _row("a", "s1", "y", ["0.5"], 1.0, 0.5)]  # s1 a meta' nel 12,5%
    m, w = _manifest(tmp_path, rows, {"a/s1/x": 2.0, "a/s1/y": 2.0})
    assert any("solo una parte delle sessioni" in p for p in NR.report(m, w)["problems"])
