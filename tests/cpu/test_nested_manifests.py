"""Manifest annidati per soggetto (D16) come file congelati (GPT e Claude, 15/10/2026): pesi rifatti con D9 sul sottoinsieme e residuo di
classe gestito senza superare il tetto (`pretraining_manifest.assign_weights_nested`, `nested_manifest`); costruttore
`scripts/build_nested_manifests.py` (autoverifiche, file deterministici, nessuna sovrascrittura); guardia in due sensi fra `--nested` e la
frazione dichiarata dal manifest (loader e trainer); rapporto `scripts/nested_manifest_report.py` sulla distribuzione campionata."""

import gzip
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wearusfm.data import pretraining_loader as L
from wearusfm.data import pretraining_manifest as M
from wearusfm.training import run as R

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico
from test_pretraining_loader import _tree

REPO = Path(__file__).resolve().parents[2]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NR, BN = _script("nested_manifest_report"), _script("build_nested_manifests")
PARAMS = {"version": "test", "quota": {"A": 0.5, "B": 0.4, "C": 0.1}, "alpha": 0.5, "max_passes": 8.0, "epochs": 4.0}


def _row(ds, subj, sess, hours, cls, nested=(), split="pretraining", unit=None):
    return M.SessionRow(ds, subj, sess, split, list(nested), int(hours * 3600 * 1000), 1000.0, 4, 4, 0, 1, unit or ds, cls, "off", "x")


def _doc(rows, params=PARAMS):
    rows = M.sort_rows(rows)
    M.assign_weights(rows, params["quota"], params["alpha"], params["max_passes"], params["epochs"])
    return {"params": params, "hash": M.manifest_hash(params, rows), "columns": list(asdict(rows[0]).keys()),
            "rows": [list(asdict(r).values()) for r in rows]}


ALL = ("0.125", "0.25", "0.5")


def _corpus():
    """Nel completo la classe C riempie la sua quota (nessun residuo, come nel manifest vero); nel sottoinsieme del 12,5% (e del 25%) le resta
    un soggetto da 1 h, la sua unita' arriva al tetto e lascia quota non assegnabile (il caso del 25% del manifest vero)."""
    rows = [_row("a", f"s{i}", "x", 10.0, "A", ALL if i < 2 else ("0.5",) if i < 4 else ()) for i in range(8)]
    rows += [_row("b", f"u{i}", "y", 20.0, "B", ALL if i < 2 else ("0.5",) if i < 4 else ()) for i in range(8)]
    rows += [_row("c", "c0", "z", 1.0, "C", ALL), _row("c", "c1", "z", 10.0, "C", ("0.5",)), _row("c", "c2", "z", 10.0, "C")]
    rows += [_row("a", "t0", "x", 5.0, "A", split="test"), _row("b", "u0", "test_sess", 3.0, "B", split="test")]
    return rows


def test_residual_goes_only_to_units_under_the_cap():
    rows = [r for r in _corpus() if r.split != "pretraining" or "0.125" in r.nested]
    plain = [M.SessionRow(**asdict(r)) for r in rows]
    alloc = M.allocate({"a": ("A", 20.0), "b": ("B", 40.0), "c": ("C", 1.0)}, PARAMS["quota"], 0.5, 8.0, 4.0)
    assert alloc["unused_quota"]["C"] > 0  # il caso del 25% del manifest vero: tutta la classe C al tetto
    out = M.assign_weights_nested(rows, PARAMS["quota"], 0.5, 8.0, 4.0)
    dist = M.sampled_distribution(rows, 4.0)
    assert dist["max_passes"] <= 8.0 * (1 + 1e-12) and abs(dist["weight_sum"] - 1) < 1e-12
    assert out["residual"]["C"] == pytest.approx(alloc["unused_quota"]["C"]) and out["receivers"] == ["a", "b"]
    assert dist["shares"]["c"] == pytest.approx(alloc["per_unit"]["c"][0])  # l'unita' al tetto non cresce
    assert dist["classes"]["C"] < 0.1 and dist["classes"]["A"] > 0.5 and dist["classes"]["B"] > 0.4
    # con la normalizzazione del loader sui pesi di allocate (senza gestione del residuo) l'unita' al tetto lo supererebbe
    M.assign_weights(plain, PARAMS["quota"], 0.5, 8.0, 4.0)
    assert M.sampled_distribution(plain, 4.0)["passes"]["c"] > 8.0


def test_without_residual_the_weights_are_those_of_assign_weights():
    rows = M.sort_rows(_corpus())
    ref = [M.SessionRow(**asdict(r)) for r in rows]
    M.assign_weights(ref, PARAMS["quota"], 0.5, 8.0, 4.0)
    out = M.assign_weights_nested(rows, PARAMS["quota"], 0.5, 8.0, 4.0)
    assert out["residual"] == {} and [r.weight for r in rows] == [r.weight for r in ref]


def test_nested_manifest_keeps_subset_sessions_and_test_rows():
    doc = _doc(_corpus())
    nd = M.nested_manifest(doc, "0.125", "sha", "commit", "content")
    rows = [dict(zip(nd["columns"], r)) for r in nd["rows"]]
    pre = [r for r in rows if r["split"] == "pretraining"]
    assert {(r["dataset"], r["subject"]) for r in pre} == {("a", "s0"), ("a", "s1"), ("b", "u0"), ("b", "u1"), ("c", "c0")}
    assert sum(r["split"] == "test" for r in rows) == 2  # test e benchmark invariati, peso 0
    p = nd["params"]
    assert p["nested"] == "0.125" and p["derived_from"]["file_sha256"] == "sha" and p["derived_from"]["content_hash"] == "content"
    assert p["nested_residual"]["per_class"] and nd["summary"]["sampled"]["max_passes"] <= 8.0 * (1 + 1e-9)
    assert nd["hash"] == M.manifest_hash(p, [M.SessionRow(**r) for r in rows])
    with pytest.raises(ValueError, match="frazione"):
        M.nested_manifest(doc, "0.3", "sha")
    with pytest.raises(ValueError, match="gia' annidato"):
        M.nested_manifest(nd, "0.125", "sha")


def _write(path, doc):
    path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    return path


def test_builder_checks_writes_deterministic_files_and_never_overwrites(tmp_path, capsys):
    src = _write(tmp_path / "m.json.gz", _doc(_corpus()))
    out = tmp_path / "out"
    assert BN.main([str(src), str(out)]) == 0
    rep = json.loads((out / "nested_manifests.json").read_text())
    assert rep["checks"] == {"full_weights_reproduced_bitwise": True, "nested_subjects": True, "within_cap": True}
    assert rep["source"]["hash"]["equal"] is True
    files = sorted(p.name for p in out.glob("*.json.gz"))
    assert files == ["m_nested_0.125.json.gz", "m_nested_0.25.json.gz", "m_nested_0.5.json.gz"]
    first = (out / files[0]).read_bytes()
    assert BN.main([str(src), str(out)]) == 0 and (out / files[0]).read_bytes() == first  # stessa costruzione, stessi byte
    (out / files[0]).write_bytes(b"altro")
    assert BN.main([str(src), str(out)]) == 1 and "non lo sovrascrivo" in capsys.readouterr().out


def test_loader_and_trainer_guard_the_declared_fraction(tmp_path):
    root, mpath = _tree(tmp_path)
    doc = json.loads(gzip.decompress(mpath.read_bytes()))
    cols = doc["columns"]
    rows = [dict(zip(cols, r)) for r in doc["rows"]]
    for r, tags in zip(rows, ([], ["0.25", "0.5"], ["0.5"])):
        r["nested"] = tags
    doc = {**doc, "params": {**PARAMS, "quota": doc["params"]["quota"]}, "rows": [[r[c] for c in cols] for r in rows]}
    full = _write(tmp_path / "full.json.gz", doc)
    nested = _write(tmp_path / "n025.json.gz", M.nested_manifest(doc, "0.25", "sha"))
    assert len(L.ManifestIndex.load(nested, [root], "0.25").rows) == 1
    with pytest.raises(ValueError, match="dichiara la frazione"):  # frazione chiesta sul manifest completo
        L.ManifestIndex.load(full, [root], "0.25")
    with pytest.raises(ValueError, match="dichiara la frazione"):  # manifest annidato senza --nested
        L.ManifestIndex.load(nested, [root], None)
    cfg = replace(R.small_config(datasets=None, max_steps=2), nested="0.25", log_batch_origin=True)
    out = tmp_path / "run"
    R.train(cfg, nested, [root], out, log=lambda _: None)
    assert json.loads((out / "origin_sessions.json").read_text()) == [R._session_key(rows[1])]
    with pytest.raises(ValueError, match="nested"):  # la frazione e' nella ricetta
        R.train(replace(cfg, nested="0.5", max_steps=4), nested, [root], out, log=lambda _: None)
    with pytest.raises(ValueError, match="D16"):
        R.train(replace(cfg, nested="0.3"), nested, [root], tmp_path / "bad", log=lambda _: None)


def _ws(rows):
    return {f"{r.dataset}/{r.subject}/{r.session}": 2.0 for r in rows}


def test_report_on_files_checks_the_sampled_distribution(tmp_path):
    corpus = _corpus()
    doc = _doc(corpus)
    full = _write(tmp_path / "full.json.gz", doc)
    files = [_write(tmp_path / f"n{f}.json.gz", M.nested_manifest(doc, f, NR._load(full)["sha256"])) for f in ALL]
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"window_s": _ws(corpus)}))
    rep = NR.report(ws, [full, *files], batch=2, epochs=4.0)
    assert rep["problems"] == [] and list(rep["fractions"]) == ["0.125", "0.25", "0.5", "1"]
    f1 = rep["fractions"]["1"]
    assert f1["hours"] == pytest.approx(8 * 10 + 8 * 20 + 21)  # test esclusi
    # finestre da 2 s per ogni sessione: passi per epoca = ore x 3600 / 2 / batch
    assert f1["steps_per_epoch"] == round(f1["hours"] * 3600 / 2 / 2)
    assert all(r["max_passes_at_E"] <= 8.0 * (1 + 1e-9) for r in rep["fractions"].values())
    # un manifest annidato costruito da un altro completo, o con sessioni fuori dal completo, si vede
    other = _doc([r for r in corpus if r.subject != "s0"])
    bad = _write(tmp_path / "bad.json.gz", M.nested_manifest(other, "0.125", "altro"))
    problems = NR.report(ws, [full, bad], batch=2)["problems"]
    assert any("costruita da" in p for p in problems)
    assert NR.main([str(ws), str(full), str(bad)]) == 1


def test_cluster_launcher_refuses_output_inside_the_code(work):  # noqa: F811
    import os
    import subprocess

    tmp = work[0]
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    launcher = REPO / "scripts" / "slurm" / "build_nested_manifests.sbatch"
    for bad in ("relativa", str(tmp / "wearusfm" / "out"), str(tmp / "wearusfm_campaigns" / "c" / "out")):
        r = subprocess.run(["bash", str(launcher), "m.json.gz", bad, "w.json"], env={**env, "WORK": str(tmp)}, capture_output=True, text=True)
        assert r.returncode == 2 and "rifiutata" in r.stdout
