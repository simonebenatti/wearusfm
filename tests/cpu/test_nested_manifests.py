"""Manifest derivati per l'asse dei soggetti (D16; accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`):
base temporale sulle ore campionabili secondo window_s congelato, D9 sul sottoinsieme, residuo di C solo verso B con A alla sua quota, errori
espliciti di capacita'; durate (chiave mancante = errore di input, null dichiarato = peso 0, valori non validi = errore); verifiche dopo le
trasformazioni del sampler (coerenti col loader vero; quote di A, B e C e somma dei pesi) e endpoint col floor sul totale; costruttore
(file deterministici anche al variare di PYTHONHASHSEED, nessuna sovrascrittura), guardia `--nested` in due sensi, training legato a sampler,
durate e configurazione delle finestre della costruzione, identita' dei file d'ingresso, ripresa rifiutata con un manifest diverso,
rapporto sui file con un caso che fallisce per ogni controllo (review del critico del 16/10/2026)."""

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
    assert BN.main([str(src), str(ws), str(out)]) == 1 and "non sovrascrivo" in capsys.readouterr().out
    # un conflitto su una frazione successiva: nessun file scritto, nemmeno quelli delle frazioni precedenti (ultrareview del 15/10)
    out3 = tmp_path / "out3"
    out3.mkdir()
    (out3 / "m_nested_0.5.json.gz").write_bytes(b"vecchio")
    assert BN.main([str(src), str(ws), str(out3)]) == 1
    assert sorted(p.name for p in out3.iterdir()) == ["m_nested_0.5.json.gz"]
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


def _tw(cfg):
    """La configurazione con il sampler pesato nel tempo (come --window1-rules) che i derivati richiedono."""
    return replace(cfg, loader=replace(cfg.loader, time_weighted=True))


WCFG = {f: getattr(R.small_config().loader, f) for f in M.WINDOW_CONFIG_FIELDS}


def test_loader_and_trainer_guard_the_declared_fraction_and_resume_refuses_another_manifest(tmp_path):
    root, doc, rows, ws = _tree_derived(tmp_path)
    full = _write(tmp_path / "full.json.gz", doc)
    n025 = _write(tmp_path / "n025.json.gz", M.derived_manifest(doc, "0.25", ws, {"parent_file_sha256": "a", "window_config": WCFG}))
    assert len(L.ManifestIndex.load(n025, [root], "0.25").rows) == 1
    with pytest.raises(ValueError, match="dichiara la frazione"):
        L.ManifestIndex.load(full, [root], "0.25")
    with pytest.raises(ValueError, match="dichiara la frazione"):
        L.ManifestIndex.load(n025, [root], None)
    # gli strumenti che indicizzano o misurano (default "*") accettano la frazione dichiarata; il training no (ultrareview del 15/10)
    assert len(L.ManifestIndex.load(n025, [root]).rows) == 1 and len(R.load_index(n025, [root], None).rows) == 1
    with pytest.raises(ValueError, match="va dichiarata"):
        R.train(replace(R.small_config(datasets=None, max_steps=1), nested="*"), n025, [root], tmp_path / "star", log=lambda _: None)
    cfg = _tw(replace(R.small_config(datasets=None, max_steps=2), nested="0.25", log_batch_origin=True))
    out = tmp_path / "run"
    R.train(cfg, n025, [root], out, log=lambda _: None, window_s=ws)
    assert json.loads((out / "origin_sessions.json").read_text()) == [R._session_key(rows[1])]
    # stessa frazione, manifest diverso (altra provenienza, altro contenuto): la firma dei dati della ripresa lo rifiuta
    other = _write(tmp_path / "n025b.json.gz", M.derived_manifest(doc, "0.25", ws, {"parent_file_sha256": "b", "window_config": WCFG}))
    with pytest.raises(ValueError, match="dati diversi"):
        R.train(replace(cfg, max_steps=4), other, [root], out, log=lambda _: None, window_s=ws)
    with pytest.raises(ValueError, match="nested"):
        R.train(replace(cfg, nested="0.5", max_steps=4), n025, [root], out, log=lambda _: None, window_s=ws)
    with pytest.raises(ValueError, match="D16"):
        R.train(replace(cfg, nested="0.3"), n025, [root], tmp_path / "bad", log=lambda _: None, window_s=ws)
    n1 = _write(tmp_path / "n1.json.gz", M.derived_manifest(doc, "1", ws, {"window_config": WCFG}))
    assert R.train(replace(cfg, nested="1"), n1, [root], tmp_path / "run1", log=lambda _: None, window_s=ws)["steps"] == 2


@pytest.mark.parametrize("change,message", [
    (lambda cfg, ws: (replace(cfg, loader=replace(cfg.loader, time_weighted=False)), ws), "pesato nel tempo"),
    (lambda cfg, ws: (cfg, None), "durate delle finestre"),
    (lambda cfg, ws: (replace(cfg, loader=replace(cfg.loader, min_window_s=2.0)), ws), "configurazione delle finestre"),
    (lambda cfg, ws: (replace(cfg, datasets=("emg2pose",)), ws), "tutti i suoi dataset"),
])
def test_trainer_refuses_a_derived_manifest_without_its_sampler_and_durations(tmp_path, change, message):
    """Rilievo 3 del critico (16/10): quote e tetto di un derivato valgono col sampler pesato nel tempo, con le durate e con la
    configurazione delle finestre della costruzione; altrimenti il training si rifiuta di partire."""
    root, doc, rows, ws = _tree_derived(tmp_path)
    n1 = _write(tmp_path / "n1.json.gz", M.derived_manifest(doc, "1", ws, {"window_config": WCFG}))
    cfg, w = change(_tw(replace(R.small_config(datasets=None, max_steps=1), nested="1")), ws)
    with pytest.raises(ValueError, match=message):
        R.train(cfg, n1, [root], tmp_path / "run", log=lambda _: None, window_s=w)
    assert not (tmp_path / "run" / "checkpoint.pt").exists()


def test_input_files_are_checked_against_the_plan_and_the_build(tmp_path):
    """Il file del manifest e' quello del piano (--manifest-sha256) e, per un derivato, le durate sono quelle della costruzione."""
    root, doc, rows, ws = _tree_derived(tmp_path)
    wfile = tmp_path / "ws.json"
    wfile.write_text(json.dumps({"window_s": ws}))
    wsha = hashlib.sha256(wfile.read_bytes()).hexdigest()
    n1 = _write(tmp_path / "n1.json.gz", M.derived_manifest(doc, "1", ws, {"window_config": WCFG, "window_seconds_sha256": wsha}))
    msha = hashlib.sha256(n1.read_bytes()).hexdigest()
    R.check_input_files(n1, wfile, "1", msha)
    with pytest.raises(ValueError, match="diverso da quello del piano"):
        R.check_input_files(n1, wfile, "1", "0" * 64)
    with pytest.raises(ValueError, match="servono le durate"):
        R.check_input_files(n1, None, "1")
    other = tmp_path / "ws2.json"
    other.write_text(json.dumps({"window_s": {**ws, "extra/x/y": 1.0}}))
    with pytest.raises(ValueError, match="con cui e' costruito"):
        R.check_input_files(n1, other, "1")
    R.check_input_files(_write(tmp_path / "full.json.gz", doc), None, None)  # manifest non derivato: niente durate obbligatorie


# sei unita' al tetto nella stessa iterazione: col codice di prima (set) 3 risultati diversi su 8 valori di PYTHONHASHSEED (verificato il 16/10)
ORDER_UNITS = {"s0": ("B", 1.445), "s1": ("B", 0.739), "s2": ("B", 2.887), "s3": ("B", 2.082), "s4": ("B", 1.716), "s5": ("B", 2.565),
               "big": ("B", 400.0)}


def test_allocation_does_not_depend_on_the_hash_seed():
    """Rilievo 1 del critico (16/10, bloccante): con piu' unita' al tetto nella stessa iterazione l'ordine delle sottrazioni cambia l'ultima
    cifra; prima era l'ordine di un set di stringhe (PYTHONHASHSEED) e il derivato al 25% aveva due contenuti possibili."""
    import itertools
    import os
    import subprocess
    import sys

    a = M.allocate(ORDER_UNITS, {"B": 1.0}, 0.5, 8.0, 4.0)
    capped = [k for k, (_, passes) in a["per_unit"].items() if abs(passes - 8.0) < 1e-9]
    assert sorted(capped) == ["s0", "s1", "s2", "s3", "s4", "s5"]  # sei unita' al tetto nella stessa iterazione
    budget = 4.0 * sum(h for _, h in ORDER_UNITS.values())
    sums = set()
    for perm in itertools.permutations(8.0 * ORDER_UNITS[k][1] / budget for k in capped):
        x = 1.0
        for c in perm:
            x -= c
        sums.add(x)
    assert len(sums) > 1  # il caso e' sensibile all'ordine
    code = ("import sys; sys.path.insert(0, %r); from wearusfm.data import pretraining_manifest as M; "
            "print(repr(M.allocate(%r, {'B': 1.0}, 0.5, 8.0, 4.0)))") % (str(REPO / "src"), ORDER_UNITS)
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}, capture_output=True, text=True,
                           check=True).stdout for seed in range(8)}
    assert len(outs) == 1


def test_derived_checks_catch_wrong_class_shares_and_inputs():
    """Rilievo 2 del critico (16/10): dopo il sampler si verificano anche B (quota + residuo) e C (quota - residuo); la somma dei pesi
    prima della normalizzazione; peso 0 per le righe senza finestre; una durata mancante e' un problema, non una rinormalizzazione."""
    corpus = _corpus()
    doc = _doc(corpus)
    ws = _ws(corpus)
    d = M.derived_manifest(doc, "1", ws, {})
    rows = [M.SessionRow(**dict(zip(d["columns"], r))) for r in d["rows"]]
    p, res = d["params"], d["params"]["residual_c_to_b"]
    assert M.derived_checks(rows, ws, p, res) == []
    moved = [replace(r, weight=r.weight * (1.2 if r.quota_class == "B" else 1.0)) for r in rows]
    tot = sum(r.weight for r in moved if r.split == "pretraining")
    moved = [replace(r, weight=r.weight / tot) for r in moved]  # somma 1, ma B sopra la sua quota e C sotto
    probs = M.derived_checks(moved, ws, p, res)
    assert any("quota di B" in x for x in probs) and any("quota di A" in x for x in probs)
    assert any("somma dei pesi" in x for x in M.derived_checks([replace(r, weight=2 * r.weight) for r in rows], ws, p, res))
    first = next(r for r in rows if r.split == "pretraining")
    nul = _ws(corpus, **{M.session_key(first.dataset, first.subject, first.session): None})
    assert any("senza finestre" in x for x in M.derived_checks(rows, nul, p, res))
    missing = {k: v for k, v in ws.items() if k != M.session_key(first.dataset, first.subject, first.session)}
    assert any("chiave assente" in x for x in M.derived_checks(rows, missing, p, res))


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


def _tamper(path, fn):
    d = json.loads(gzip.decompress(path.read_bytes()))
    fn(d)
    _write(path, d)


def _drop_subject(d, subject):
    cols = d["columns"]
    d["rows"] = [r for r in d["rows"] if dict(zip(cols, r))["subject"] != subject]


def _set_field(d, pred, field, value):
    cols = d["columns"]
    for r in d["rows"]:
        if pred(dict(zip(cols, r))):
            r[cols.index(field)] = value


def _scale_class(d, cls, factor):
    cols = d["columns"]
    for r in d["rows"]:
        x = dict(zip(cols, r))
        if x["split"] == "pretraining" and x["quota_class"] == cls:
            r[cols.index("weight")] *= factor
    tot = sum(dict(zip(cols, r))["weight"] for r in d["rows"] if dict(zip(cols, r))["split"] == "pretraining")
    for r in d["rows"]:
        r[cols.index("weight")] /= tot


@pytest.mark.parametrize("fraction,tamper,message", [
    ("0.125", lambda d: d["rows"].append(list(d["rows"][-1][:2]) + ["altra"] + list(d["rows"][-1][3:])), "non e' contenuto"),
    ("0.25", lambda d: _set_field(d, lambda r: r["subject"] == "a1s0", "session", "altra"), "solo una parte delle sessioni"),
    ("1", lambda d: _drop_subject(d, "c2"), "stesse sessioni di pretraining"),
    ("0.5", lambda d: _set_field(d, lambda r: r["split"] == "test", "n_samples", 7), "test/benchmark diverse"),
    ("0.5", lambda d: d["params"]["derived_inputs"].update(window_seconds_sha256="altro"), "window_s"),
    ("0.5", lambda d: _scale_class(d, "A", 1.3), "quota di A"),
    ("1", lambda d: d["params"].update(max_passes=1.0), "oltre il tetto"),
])
def test_report_flags_each_nesting_and_identity_problem(tmp_path, fraction, tamper, message):
    """Rilievo 4 del critico (16/10): un caso che fallisce, con un file manomesso, per i controlli di annidamento, sessioni, righe di
    test/benchmark, provenienza, quota e tetto (gli altri quattro in `test_report_flags_wrong_fractions_and_parent`)."""
    corpus = _corpus()
    doc = _doc(corpus)
    parent = _write(tmp_path / "full.json.gz", doc)
    ws = _ws_file(tmp_path / "ws.json", corpus)
    inputs = {"parent_file_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
              "window_seconds_sha256": hashlib.sha256(ws.read_bytes()).hexdigest()}
    wsd = json.loads(ws.read_text())["window_s"]
    files = {f: _write(tmp_path / f"n{f}.json.gz", M.derived_manifest(doc, f, wsd, inputs)) for f in M.DERIVED_FRACTIONS}
    assert NR.report(ws, parent, list(files.values()), batch=2)["problems"] == []
    _tamper(files[fraction], tamper)
    probs = NR.report(ws, parent, list(files.values()), batch=2)["problems"]
    assert any(message in x for x in probs), probs


@pytest.mark.parametrize("subject,session", [
    ("a1s7", None),  # un soggetto fuori dal 50%: fallisce l'annidamento dei soggetti (e delle sessioni)
    ("a1s0", "extra"),  # una sessione nuova di un soggetto gia' nel 50%: i soggetti restano annidati, fallisce solo quello delle sessioni
])
def test_builder_refuses_to_write_when_nesting_fails(tmp_path, monkeypatch, capsys, subject, session):
    """Il controllo `nested_subjects_and_sessions` del costruttore, da solo e nelle sue due meta': il 25% riceve a peso 0 una sessione senza
    finestre (pesi, quote e tetto restano giusti, quindi `derived_checks` passa) e nessun file viene scritto (seconda e terza review del
    critico)."""
    corpus = _corpus()
    doc = _doc(corpus)
    src = _write(tmp_path / "full.json.gz", doc)
    base = next(r for r in doc["rows"] if dict(zip(doc["columns"], r))["subject"] == subject)
    extra = list(base)
    if session is not None:
        extra[doc["columns"].index("session")] = session
    o = dict(zip(doc["columns"], extra))
    # senza finestre: a peso 0 non entra ne' nelle ore campionabili ne' nelle quote, quindi solo l'annidamento puo' accorgersene
    ws = _ws_file(tmp_path / "ws.json", corpus, **{M.session_key(o["dataset"], o["subject"], o["session"]): None})
    real = M.derived_manifest

    def broken(d0, fraction, window_s, inputs, code_commit=""):
        d = real(d0, fraction, window_s, inputs, code_commit)
        if fraction == "0.25":
            row = list(extra)
            row[d["columns"].index("weight")] = 0.0
            d["rows"].append(row)
        return d

    monkeypatch.setattr(BN.M, "derived_manifest", broken)
    report, _ = BN.build(src, ws, 32)
    assert report["checks"]["nested_subjects_and_sessions"] is False and report["checks"]["derived_checks"] is True
    assert BN.main([str(src), str(ws), str(tmp_path / "out")]) == 1 and not (tmp_path / "out").exists()
    assert "VERIFICHE FALLITE: ['nested_subjects_and_sessions']" in capsys.readouterr().out


def test_report_flags_wrong_fractions_and_parent(tmp_path):
    """Gli altri quattro controlli del rapporto, ciascuno con un caso che fallisce: frazione fuori elenco, due file per la stessa frazione,
    genitore che dichiara una frazione, sessioni di pretraining che nel genitore non lo sono."""
    corpus = _corpus()
    doc = _doc(corpus)
    parent = _write(tmp_path / "full.json.gz", doc)
    ws = _ws_file(tmp_path / "ws.json", corpus)
    wsd = json.loads(ws.read_text())["window_s"]
    files = {f: _write(tmp_path / f"n{f}.json.gz", M.derived_manifest(doc, f, wsd, {})) for f in M.DERIVED_FRACTIONS}
    odd = _write(tmp_path / "odd.json.gz", M.derived_manifest(doc, "0.5", wsd, {}))
    _tamper(odd, lambda d: d["params"].update(nested="0.3"))
    assert any("non fra" in x for x in NR.report(ws, parent, [odd], batch=2)["problems"])
    twin = _write(tmp_path / "twin.json.gz", M.derived_manifest(doc, "0.5", wsd, {}))
    assert any("due derivati per la frazione 0.5" in x for x in NR.report(ws, parent, [files["0.5"], twin], batch=2)["problems"])
    bad_parent = _write(tmp_path / "parent_nested.json.gz", {**doc, "params": {**doc["params"], "nested": "1"}})
    assert any("il genitore dichiara" in x for x in NR.report(ws, bad_parent, list(files.values()), batch=2)["problems"])
    _tamper(files["0.5"], lambda d: _set_field(d, lambda r: r["subject"] == "a1s2", "session", "nuova"))
    probs = NR.report(ws, parent, list(files.values()), batch=2)["problems"]
    assert any("nel genitore non lo sono" in x for x in probs), probs


def test_cluster_launcher_refuses_output_inside_the_code(work):  # noqa: F811
    import os
    import subprocess

    tmp = work[0]
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    launcher = REPO / "scripts" / "slurm" / "build_nested_manifests.sbatch"
    for bad in ("relativa", str(tmp / "wearusfm" / "out"), str(tmp / "wearusfm_campaigns" / "c" / "out")):
        r = subprocess.run(["bash", str(launcher), "m.json.gz", bad, "w.json"], env={**env, "WORK": str(tmp)}, capture_output=True, text=True)
        assert r.returncode == 2 and "rifiutata" in r.stdout
