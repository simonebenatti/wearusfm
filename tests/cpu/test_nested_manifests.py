"""Manifest derivati per l'asse dei soggetti (D16; accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`):
base temporale sulle ore campionabili secondo window_s congelato, D9 sul sottoinsieme, residuo di C solo verso B con A alla sua quota, errori
espliciti di capacita'; durate (chiave mancante = errore di input, null dichiarato = peso 0, valori non validi = errore); verifiche dopo le
trasformazioni del sampler (coerenti col loader vero) e endpoint col floor sul totale; costruttore (file deterministici, nessuna
sovrascrittura), guardia `--nested` in due sensi, ripresa rifiutata con un manifest diverso, rapporto sui file."""

import gzip
import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
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
PARAMS = {"version": "test", "quota": {"A": 0.2, "B": 0.75, "C": 0.05}, "alpha": 0.5, "max_passes": 8.0, "epochs": 4.0}
ALL = ("0.125", "0.25", "0.5")


def _row(ds, subj, sess, hours, cls, nested=(), split="pretraining"):
    return M.SessionRow(ds, subj, sess, split, list(nested), int(hours * 3600 * 1000), 1000.0, 4, 4, 0, 1, ds, cls, "off", "x")


def _corpus():
    """Nel completo la classe C riempie la sua quota; nel 12,5% (e nel 25%) le resta un soggetto da 1 h, la sua unita' arriva al tetto e
    lascia quota non assegnabile (il caso del 25% del manifest vero). A: 2 unita', B: 2 unita'."""
    rows = []
    for ds, cls, h in (("a1", "A", 8.0), ("a2", "A", 6.0), ("b1", "B", 30.0), ("b2", "B", 20.0)):
        rows += [_row(ds, f"{ds}s{i}", "x", h, cls, ALL if i < 2 else ("0.5",) if i < 4 else ()) for i in range(8)]
    rows += [_row("c", "c0", "z", 1.0, "C", ALL), _row("c", "c1", "z", 30.0, "C", ("0.5",)), _row("c", "c2", "z", 30.0, "C")]
    rows += [_row("a1", "t0", "x", 5.0, "A", split="test"), _row("b1", "b1s0", "test_sess", 3.0, "B", split="test")]
    return M.sort_rows(rows)


def _ws(rows, value=2.0, **special):
    out = {M.session_key(r.dataset, r.subject, r.session): value for r in rows}
    out.update(special)
    return out


def _doc(rows, params=PARAMS):
    rows = M.sort_rows([M.SessionRow(**asdict(r)) for r in rows])
    M.assign_weights(rows, params["quota"], params["alpha"], params["max_passes"], params["epochs"])
    return {"params": params, "hash": M.manifest_hash(params, rows), "columns": list(asdict(rows[0]).keys()),
            "rows": [list(asdict(r).values()) for r in rows]}


def _units(rows):
    units = {}
    for r in rows:
        if r.split == "pretraining":
            c, h = units.get(r.unit, (r.quota_class, 0.0))
            units[r.unit] = (c, h + r.hours)
    return units


# --- durate --------------------------------------------------------------------------------------------------------------------------

def test_durations_missing_key_is_an_input_error_null_means_no_windows():
    rows = _corpus()
    key = M.session_key("b1", "b1s3", "x")
    ok = M.samplable_seconds(rows, _ws(rows, **{key: None}))
    assert ok[key] is False and sum(ok.values()) == len(ok) - 1
    missing = _ws(rows)
    del missing[key]
    with pytest.raises(ValueError, match="chiave assente"):
        M.samplable_seconds(rows, missing)
    for bad in (0.0, -1.0, float("nan"), float("inf"), "2", True):
        with pytest.raises(ValueError, match="non valida"):
            M.samplable_seconds(rows, _ws(rows, **{key: bad}))


def test_unsamplable_sessions_get_weight_zero_and_their_hours_are_reported_apart():
    rows = _corpus()
    key = M.session_key("b1", "b1s3", "x")
    ws = _ws(rows, **{key: None})
    out = M.assign_weights_derived(rows, ws, PARAMS["quota"], 0.5, 8.0, 4.0)
    excluded = next(r for r in rows if M.session_key(r.dataset, r.subject, r.session) == key)
    assert excluded.weight == 0.0 and out["sessions_excluded"] == 1 and out["hours_excluded"] == pytest.approx(30.0)
    assert out["hours_nominal"] == pytest.approx(out["hours_samplable"] + out["hours_excluded"])
    d = M.sampled_distribution(rows, ws, 4.0)
    assert abs(d["mass"] - 1) < 1e-12 and d["max_passes"] <= 8.0 * (1 + 1e-12)
    assert d["classes"]["A"] == pytest.approx(0.2, abs=1e-12) and d["classes"]["B"] == pytest.approx(0.75, abs=1e-12)
    assert d["hours_samplable"] == pytest.approx(out["hours_samplable"])  # passaggi e tetto sulle ore campionabili


# --- pesi e capacita' ----------------------------------------------------------------------------------------------------------------

def test_without_exclusions_or_residual_the_weights_are_those_of_assign_weights():
    rows = _corpus()
    ref = [M.SessionRow(**asdict(r)) for r in rows]
    M.assign_weights(ref, PARAMS["quota"], 0.5, 8.0, 4.0)
    out = M.assign_weights_derived(rows, _ws(rows), PARAMS["quota"], 0.5, 8.0, 4.0)
    assert out["residual_c_to_b"] == 0.0 and out["sessions_excluded"] == 0
    assert [r.weight for r in rows] == [r.weight for r in ref]  # bit per bit


def test_residual_of_c_goes_only_to_b_and_a_keeps_its_quota():
    rows = [r for r in _corpus() if r.split != "pretraining" or "0.125" in r.nested]
    ws = _ws(rows)
    first = M.allocate(_units(rows), PARAMS["quota"], 0.5, 8.0, 4.0)
    res = first["unused_quota"]["C"]
    assert res > 0
    out = M.assign_weights_derived(rows, ws, PARAMS["quota"], 0.5, 8.0, 4.0)
    d = M.sampled_distribution(rows, ws, 4.0)
    assert out["residual_c_to_b"] == pytest.approx(res)
    assert d["classes"]["A"] == pytest.approx(0.2, abs=1e-12)  # A invariata
    assert d["classes"]["B"] == pytest.approx(0.75 + res, abs=1e-12) and d["classes"]["C"] == pytest.approx(0.05 - res, abs=1e-12)
    assert d["shares"]["a1"] == pytest.approx(first["per_unit"]["a1"][0], abs=1e-15)  # le unita' di A non cambiano
    assert d["passes"]["c"] == pytest.approx(8.0) and d["max_passes"] <= 8.0 * (1 + 1e-12)  # C al tetto, nessuna unita' oltre
    # senza la regola, la normalizzazione del loader sui pesi di allocate porterebbe C oltre il tetto
    plain = [M.SessionRow(**asdict(r)) for r in rows]
    M.assign_weights(plain, PARAMS["quota"], 0.5, 8.0, 4.0)
    assert M.sampled_distribution(plain, ws, 4.0)["passes"]["c"] > 8.0


def test_insufficient_capacity_fails_explicitly():
    rows = [r for r in _corpus() if r.split != "pretraining" or "0.125" in r.nested]
    ws = _ws(rows)
    with pytest.raises(ValueError, match="classe A non sostiene"):  # A troppo piccola per il 60%
        M.assign_weights_derived(rows, ws, {"A": 0.6, "B": 0.35, "C": 0.05}, 0.5, 8.0, 4.0)
    with pytest.raises(ValueError, match="classe B non sostiene"):  # tetto 3,8: A arriva al 20,6%, B solo al 73,6%
        M.assign_weights_derived(rows, ws, PARAMS["quota"], 0.5, 3.8, 4.0)
    # B sostiene la sua quota ma non il residuo di C. Ore del 12,5%: A 28, B 100, C 1 su 129; col tetto di 4 passaggi le capacita' sono A
    # 0,217, B 0,775, C 0,0078: con quote 0,2 / 0,77 / 0,03 il residuo di C (0,022) supera il margine di B (0,005)
    with pytest.raises(ValueError, match="non assorbe il residuo"):
        M.assign_weights_derived(rows, ws, {"A": 0.2, "B": 0.77, "C": 0.03}, 0.5, 4.0, 4.0)


def test_weights_are_finite_non_negative_and_sum_to_one_after_the_sampler():
    doc = _doc(_corpus())
    for f in M.DERIVED_FRACTIONS:
        nd = M.derived_manifest(doc, f, _ws(_corpus()), {"parent_file_sha256": "sha"})
        rows = [M.SessionRow(**dict(zip(nd["columns"], r))) for r in nd["rows"]]
        assert all(math.isfinite(r.weight) and r.weight >= 0 for r in rows)
        assert sum(r.weight for r in rows if r.split == "pretraining") == pytest.approx(1.0, abs=1e-12)
        assert abs(nd["summary"]["sampled"]["mass"] - 1) < 1e-12


# --- sampler ed endpoint -------------------------------------------------------------------------------------------------------------

def test_time_shares_match_the_real_loader():
    rows = _corpus()
    ws = {k: (1.0 + (i % 4)) for i, k in enumerate(_ws(rows))}
    ws[M.session_key("a2", "a2s5", "x")] = None
    M.assign_weights_derived(rows, ws, PARAMS["quota"], 0.5, 8.0, 4.0)
    pre = [asdict(r) for r in rows if r.split == "pretraining"]
    w = np.asarray([r["weight"] for r in pre])
    probs = L.time_weighted_probs(L.ManifestIndex(pre, w / w.sum(), []), ws)
    t = probs * np.asarray([ws[M.session_key(r["dataset"], r["subject"], r["session"])] or 0.0 for r in pre])
    assert np.allclose(t / t.sum(), M.time_shares(rows, ws), rtol=0, atol=1e-15)


def test_endpoint_uses_a_single_floor_on_the_total():
    rows = [_row("a1", "s0", "x", 1.0, "A"), _row("b1", "u0", "y", 1.0, "B"), _row("c", "c0", "z", 1.0, "C")]
    q = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    ws = _ws(rows, 3.0)
    M.assign_weights_derived(rows, ws, q, 0.5, 8.0, 4.0)
    d = M.sampled_distribution(rows, ws, epochs=4.0, batch=7)
    per_epoch = 3 * 3600 / 3.0 / 7  # 514,29 passi per epoca
    assert d["steps_at_epochs"] == math.floor(4 * per_epoch) == 2057 and 4 * math.floor(per_epoch) == 2056
    assert d["effective_epochs"] <= 4.0 and d["effective_epochs"] == pytest.approx(2057 / per_epoch)


# --- manifest derivati e costruttore -------------------------------------------------------------------------------------------------

def test_derived_manifest_keeps_subset_sessions_test_rows_and_provenance():
    doc = _doc(_corpus())
    nd = M.derived_manifest(doc, "0.125", _ws(_corpus()), {"parent_file_sha256": "sha", "window_seconds_sha256": "ws"}, "commit")
    rows = [dict(zip(nd["columns"], r)) for r in nd["rows"]]
    pre = [r for r in rows if r["split"] == "pretraining"]
    assert {r["subject"] for r in pre} == {"a1s0", "a1s1", "a2s0", "a2s1", "b1s0", "b1s1", "b2s0", "b2s1", "c0"}
    assert sum(r["split"] == "test" for r in rows) == 2
    p = nd["params"]
    assert p["nested"] == "0.125" and p["derived_inputs"]["parent_file_sha256"] == "sha" and p["residual_c_to_b"] > 0
    assert set(p["derived_hours"]) == {"hours_nominal", "hours_samplable", "hours_excluded", "sessions_excluded"}
    assert nd["hash"] == M.manifest_hash(p, [M.SessionRow(**r) for r in rows])
    full = M.derived_manifest(doc, "1", _ws(_corpus()), {})
    assert len(full["rows"]) == len(doc["rows"]) and full["params"]["residual_c_to_b"] == 0.0
    with pytest.raises(ValueError, match="frazione"):
        M.derived_manifest(doc, "0.3", _ws(_corpus()), {})
    with pytest.raises(ValueError, match="gia' derivato"):
        M.derived_manifest(nd, "0.125", _ws(_corpus()), {})


def _write(path, doc):
    path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    return path


def _ws_file(path, rows, **special):
    path.write_text(json.dumps({"manifest": "m", "window_s": _ws(rows, **special)}))
    return path


def test_builder_checks_writes_deterministic_files_and_never_overwrites(tmp_path, capsys):
    src = _write(tmp_path / "m.json.gz", _doc(_corpus()))
    ws = _ws_file(tmp_path / "ws.json", _corpus(), **{M.session_key("b1", "b1s7", "x"): None})
    out = tmp_path / "out"
    assert BN.main([str(src), str(ws), str(out)]) == 0
    rep = json.loads((out / "nested_manifests.json").read_text())
    assert all(v is True for v in rep["checks"].values()) and {"window_config_same_as_training", "sampler_matches_loader"} <= set(rep["checks"])
    assert rep["derived_100_vs_parent"]["sessions_excluded"] == 1 and "window_s null" in rep["derived_100_vs_parent"]["explanation"]
    files = sorted(p.name for p in out.glob("*.json.gz"))
    assert files == [f"m_nested_{f}.json.gz" for f in ("0.125", "0.25", "0.5", "1")]
    first = (out / files[0]).read_bytes()
    assert BN.main([str(src), str(ws), str(out)]) == 0 and (out / files[0]).read_bytes() == first
    (out / files[0]).write_bytes(b"altro")
    assert BN.main([str(src), str(ws), str(out)]) == 1 and "non lo sovrascrivo" in capsys.readouterr().out
    missing = tmp_path / "ws_missing.json"
    data = json.loads(ws.read_text())
    del data["window_s"][M.session_key("a1", "a1s0", "x")]
    missing.write_text(json.dumps(data))
    assert BN.main([str(src), str(missing), str(tmp_path / "out2")]) == 1 and not (tmp_path / "out2").exists()


# --- loader, trainer, ripresa --------------------------------------------------------------------------------------------------------

def _tree_derived(tmp_path):
    root, mpath = _tree(tmp_path)
    doc = json.loads(gzip.decompress(mpath.read_bytes()))
    cols = doc["columns"]
    rows = [dict(zip(cols, r)) for r in doc["rows"]]
    for r, tags in zip(rows, ([], ["0.25", "0.5"], ["0.5"])):
        r["nested"] = tags
    # un dataset per classe e il 25% con la sola riga di B: quote e tetto adatti all'albero di prova (con le quote di D9 la regola
    # rifiuterebbe giustamente il sottoinsieme: A senza unita')
    doc = {**doc, "params": {**PARAMS, "quota": {"A": 0.0, "B": 1.0, "C": 0.0}, "max_passes": 1000.0},
           "rows": [[r[c] for c in cols] for r in rows]}
    ws = {M.session_key(r["dataset"], r["subject"], r["session"]): 2.0 for r in rows}
    return root, doc, rows, ws


def test_loader_and_trainer_guard_the_declared_fraction_and_resume_refuses_another_manifest(tmp_path):
    root, doc, rows, ws = _tree_derived(tmp_path)
    full = _write(tmp_path / "full.json.gz", doc)
    n025 = _write(tmp_path / "n025.json.gz", M.derived_manifest(doc, "0.25", ws, {"parent_file_sha256": "a"}))
    assert len(L.ManifestIndex.load(n025, [root], "0.25").rows) == 1
    with pytest.raises(ValueError, match="dichiara la frazione"):
        L.ManifestIndex.load(full, [root], "0.25")
    with pytest.raises(ValueError, match="dichiara la frazione"):
        L.ManifestIndex.load(n025, [root], None)
    cfg = replace(R.small_config(datasets=None, max_steps=2), nested="0.25", log_batch_origin=True)
    out = tmp_path / "run"
    R.train(cfg, n025, [root], out, log=lambda _: None)
    assert json.loads((out / "origin_sessions.json").read_text()) == [R._session_key(rows[1])]
    # stessa frazione, manifest diverso (altra provenienza, altro contenuto): la firma dei dati della ripresa lo rifiuta
    other = _write(tmp_path / "n025b.json.gz", M.derived_manifest(doc, "0.25", ws, {"parent_file_sha256": "b"}))
    with pytest.raises(ValueError, match="dati diversi"):
        R.train(replace(cfg, max_steps=4), other, [root], out, log=lambda _: None)
    with pytest.raises(ValueError, match="nested"):
        R.train(replace(cfg, nested="0.5", max_steps=4), n025, [root], out, log=lambda _: None)
    with pytest.raises(ValueError, match="D16"):
        R.train(replace(cfg, nested="0.3"), n025, [root], tmp_path / "bad", log=lambda _: None)
    n1 = _write(tmp_path / "n1.json.gz", M.derived_manifest(doc, "1", ws, {}))
    assert R.train(replace(cfg, nested="1"), n1, [root], tmp_path / "run1", log=lambda _: None)["steps"] == 2


# --- rapporto ------------------------------------------------------------------------------------------------------------------------

def test_report_on_files_checks_the_sampled_distribution_and_provenance(tmp_path):
    corpus = _corpus()
    doc = _doc(corpus)
    parent = _write(tmp_path / "full.json.gz", doc)
    ws = _ws_file(tmp_path / "ws.json", corpus)
    inputs = {"parent_file_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
              "window_seconds_sha256": hashlib.sha256(ws.read_bytes()).hexdigest()}
    wsd = json.loads(ws.read_text())["window_s"]
    files = [_write(tmp_path / f"n{f}.json.gz", M.derived_manifest(doc, f, wsd, inputs)) for f in M.DERIVED_FRACTIONS]
    rep = NR.report(ws, parent, files, batch=2, epochs=4.0)
    assert rep["problems"] == [] and list(rep["manifests"]) == ["genitore", "0.125", "0.25", "0.5", "1"]
    one = rep["manifests"]["1"]
    assert one["hours_samplable"] == pytest.approx(8 * (8 + 6 + 30 + 20) + 61)
    assert one["steps_at_epochs"] == math.floor(4 * one["hours_samplable"] * 3600 / 2.0 / 2)
    bad = _write(tmp_path / "bad.json.gz", M.derived_manifest(doc, "0.125", wsd, {**inputs, "parent_file_sha256": "altro"}))
    assert any("costruita da" in p for p in NR.report(ws, parent, [bad], batch=2)["problems"])
    assert NR.main([str(ws), str(parent), str(bad)]) == 1


def test_cluster_launcher_refuses_output_inside_the_code(work):  # noqa: F811
    import os
    import subprocess

    tmp = work[0]
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    launcher = REPO / "scripts" / "slurm" / "build_nested_manifests.sbatch"
    for bad in ("relativa", str(tmp / "wearusfm" / "out"), str(tmp / "wearusfm_campaigns" / "c" / "out")):
        r = subprocess.run(["bash", str(launcher), "m.json.gz", bad, "w.json"], env={**env, "WORK": str(tmp)}, capture_output=True, text=True)
        assert r.returncode == 2 and "rifiutata" in r.stdout
