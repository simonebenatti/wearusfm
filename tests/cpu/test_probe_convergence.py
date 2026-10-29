"""Controllo della convergenza delle sonde P1/P2 di B19618 contro R3c (29/10/2026; prompt di GPT `docs/prompt_claude_convergenza_sonde_20261029.md`,
protocollo `docs/sonde_convergenza_20261029.md`): audit per fit in `harness.probe.linear_probe` (acceso/spento identico, warning attribuiti al fit
giusto e separati da quelli della metrica, eccezioni propagate, nessuna fuga di informazione, parametri propagati a tutti i fit), shard e
aggregazione di `scripts/probe_convergence.py` uguali alla sonda intera di `channel_readout.py`, parita' esatta con un rapporto storico, cancello
della fase B, bootstrap congiunto uguale a quello ufficiale, guardie della ricetta del solver (etichette testuali di EPN612 comprese), piano dei
comandi e launcher con la regola dei checkout."""

import importlib.util
import inspect
import json
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico
from test_probe_launchers import _env as _launcher_env  # noqa: E402
from wearusfm.harness import downstream_ab as DA
from wearusfm.harness import p1_cross_subject as P1
from wearusfm.harness.probe import C_GRID, MAX_ITER, ProbeResult, linear_probe, solver_recipe

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("probe_convergence_script", REPO / "scripts" / "probe_convergence.py")
PC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PC)
SC, CP, AC, CS = PC.SC, PC.CP, PC.AC, PC.CS
CACHE_CODE = "24139ab901c052357a46a65bc20466f2eb1e8358"
SLURM = REPO / "scripts" / "slurm"


# ------------------------------------------------------------------------------------------------------------------------------- dati finti

def _data(rng, subjects, per_sub, n_cls, width, labels=None, sep=1.5):
    centers = rng.normal(size=(n_cls, width)) * sep
    xs, ys, ss = [], [], []
    for s in subjects:
        shift = rng.normal(size=width) * 0.3
        c = rng.integers(n_cls, size=per_sub)
        c[:n_cls] = np.arange(n_cls)  # ogni soggetto ha tutte le classi
        xs.append(centers[c] + shift + rng.normal(size=(per_sub, width)))
        ys.append(c)
        ss.append(np.full(per_sub, s))
    x, y, s = np.concatenate(xs).astype(np.float32), np.concatenate(ys), np.concatenate(ss)
    if labels is not None:
        y = np.asarray(labels)[y]
    return x, y, s


def _write_cache(path, x, y, s, subjects, task, splits_sha, ckpt, ckpt_sha, all_subjects=None):
    extra = {"all_subjects": np.asarray(all_subjects)} if all_subjects is not None else {}
    np.savez(path, x_pool=x, x_local=x[:, :4], x_dec=x[:, :4], y=y, s=s, subjects=np.asarray(subjects),
             input_sha256=np.asarray(DA.fingerprint(y, s)), checkpoint=np.asarray(ckpt), checkpoint_sha256=np.asarray(ckpt_sha),
             encoder_dim=np.asarray(x.shape[1] // 2), channels=np.asarray(2), protocol=np.asarray(SC.PROTOCOL),
             calibration_protocol=np.asarray(CP.PROTOCOL), task=np.asarray(task), splits_sha256=np.asarray(splits_sha),
             extraction_recipe=np.asarray(json.dumps(SC.recipe(task), sort_keys=True)), feature_set=np.asarray(SC.CR.FEATURE_SET),
             code_sha=np.asarray(CACHE_CODE), montage_signature=np.asarray("[]"), **extra)


def _campaign(tmp_path, seed=0, width=12):
    """Due encoder (B, R3c) con le stesse finestre e feature diverse, cinque dataset (P1 con etichette intere, EPN612 con etichette testuali),
    cache nel formato del readout per canale con ricevute, provenienze e pin; rapporti «storici» con la sonda intera di channel_readout.py."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "campagna"
    subjects_p1 = {"ninapro_db2": [f"s{i:02d}" for i in range(1, 13)], "ninapro_db3": [f"s{i:02d}" for i in range(1, 11)],
                   "ninapro_db6": [f"s{i:02d}" for i in range(1, 11)]}
    splits = {"datasets": {d: {"pretraining": subs[:8], "test": subs[8:]} for d, subs in subjects_p1.items()}}
    splits_path = root / "splits.json"
    splits_path.parent.mkdir(parents=True)
    splits_path.write_text(json.dumps(splits))
    splits_sha = SC._sha256(splits_path)
    all_p2 = [f"user{i:02d}" for i in range(1, 25)]
    internal = CP.p2_internal_subjects(all_p2)
    assert len(internal) >= 8
    base = {}
    for ds, subs in subjects_p1.items():
        base[ds] = _data(rng, splits["datasets"][ds]["pretraining"], 24, 3, width)
    base["epn612"] = _data(rng, internal, 20, 3, width, labels=["fist", "open", "waveIn"])
    base["uci_emg"] = _data(rng, internal, 20, 3, width)
    ckpts = {"B": ("/finto/B/checkpoint_step0019618.pt", "b" * 64), "R3c": ("/finto/R3c/checkpoint_step0019618.pt", "c" * 64)}
    pins = {"checkpoints": {e: {"path": p, "sha256": h, "step": 19618} for e, (p, h) in ckpts.items()}, "reports": {}, "caches": {}}
    features = {}
    for enc, (ckpt, ckpt_sha) in ckpts.items():
        mix = 0.6 if enc == "B" else 0.0
        for task, dss in PC.TASKS.items():
            d = root / "features" / enc / task
            d.mkdir(parents=True)
            features[enc, task] = d
            outputs = {}
            for ds in dss:
                x, y, s = base[ds]
                x = (x + mix * rng.normal(size=x.shape)).astype(np.float32)
                if task == "p1":
                    _write_cache(d / f"{ds}.npz", x, y, s, splits["datasets"][ds]["pretraining"], task, splits_sha, ckpt, ckpt_sha)
                    ssha = splits_sha
                else:
                    _write_cache(d / f"{ds}.npz", x, y, s, internal, task, CP.P2_SPLITS_SHA256, ckpt, ckpt_sha, all_subjects=all_p2)
                    ssha = CP.P2_SPLITS_SHA256
                csha = PC.sha256(d / f"{ds}.npz")
                prov = {"model": enc, "task": task, "dataset": ds, "which": "teacher", "checkpoint_step": 19618, "checkpoint_sha256": ckpt_sha,
                        "code_sha": CACHE_CODE, "cache": str(d / f"{ds}.npz"), "cache_sha256": csha, "input_sha256": DA.fingerprint(y, s),
                        "splits_sha256": ssha, "n_windows": int(len(y))}
                (d / f"{ds}.npz.provenance.json").write_text(json.dumps(prov))
                outputs[ds] = {"cache": str(d / f"{ds}.npz"), "sha256": csha, "provenance_sha256": PC.sha256(d / f"{ds}.npz.provenance.json")}
                pins["caches"][f"{enc}/{task}/{ds}"] = {**prov, "provenance_sha256": outputs[ds]["provenance_sha256"]}
            (d / "extraction_receipt.json").write_text(json.dumps({"complete": True, "checkpoint_unchanged": True, "model": enc, "mode": "channel",
                                                                   "checkpoint_step": 19618, "checkpoint": ckpt, "checkpoint_sha256": ckpt_sha,
                                                                   "code_sha": CACHE_CODE, "outputs": outputs}))
            pins["reports"][f"{enc}_{task}"] = {"extraction_receipt": str(d / "extraction_receipt.json"),
                                                "extraction_receipt_sha256": PC.sha256(d / "extraction_receipt.json")}
    originals = {}
    for enc in PC.ENCODERS:
        for task in PC.TASKS:
            out = root / "originali" / f"{enc}_{task}_pool.json"
            assert SC.main(["--stage", "probe", "--task", task, "--features", str(features[enc, task]), "--splits", str(splits_path),
                            "--layer", "pool", "--k", "0", "--seed", "0", "--out", str(out)]) == 0
            rep = json.loads(out.read_text())
            originals[enc, task] = out
            pins["reports"][f"{enc}_{task}"].update({"report": str(out), "report_sha256": PC.sha256(out), "oof": rep["oof"], "oof_sha256": rep["oof_sha256"]})
    pins_path = root / "pin_ingressi.json"
    pins_path.write_text(json.dumps(pins))
    return {"root": root, "splits": splits_path, "features": features, "pins": pins_path, "pins_dict": pins, "originals": originals, "base": base}


def _probe_split(rng, n_sub=9, per=20, n_cls=3, width=6, labels=None, sep=1.5):
    x, y, s = _data(rng, [f"u{i}" for i in range(n_sub)], per, n_cls, width, labels=labels, sep=sep)
    tr, va, te = np.isin(s, [f"u{i}" for i in range(6)]), np.isin(s, ["u6"]), np.isin(s, ["u7", "u8"])
    return x, y, s, tr, va, te


def _run(stage, *args):
    return PC.main(["--stage", stage, "--allow-unfrozen", *map(str, args)])


def _shards(camp, phase, out_root, encoders=PC.ENCODERS, gate=None):
    paths = {}
    for enc in encoders:
        for task, shards in PC.SHARDS.items():
            for name, datasets, folds in shards:
                out = out_root / phase / f"{enc}_{task}" / f"shard_{name}"
                args = ["--out", out, "--phase", phase, "--encoder", enc, "--task", task, "--datasets", *datasets, "--folds", *folds,
                        "--features", camp["features"][enc, task], "--pins", camp["pins"], "--splits", camp["splits"]]
                if gate is not None:
                    args += ["--gate", gate]
                assert _run("probe", *args) == 0
                paths.setdefault((enc, task), []).append(out / "shard.json")
    return paths


# ------------------------------------------------------------------------------------------------------------------------------ audit per fit

@pytest.mark.parametrize("labels", [None, ["fist", "open", "waveIn"]])
def test_audit_on_and_off_give_identical_scaler_selection_predictions_and_scores(labels):
    x, y, s, tr, va, te = _probe_split(np.random.default_rng(1), labels=labels)
    off = linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=50)
    audit: list = []
    on = linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=50, audit=audit, context={"dataset": "d", "fold": 0})
    assert np.array_equal(off.test_pred, on.test_pred) and off.c == on.c and off.val_scores == on.val_scores and off.test_bacc == on.test_bacc
    assert off.test_bacc_se == on.test_bacc_se and off.train_bacc == on.train_bacc and off.solver_params == on.solver_params
    assert off.bootstrap_warnings is None and isinstance(on.bootstrap_warnings, int)
    assert len(audit) == len(C_GRID) + 1 and [r["fit_phase"] for r in audit] == ["candidate"] * len(C_GRID) + ["refit"]
    assert [r["C"] for r in audit[:-1]] == [float(c) for c in C_GRID] and audit[-1]["C"] == on.c
    assert [r["selected"] for r in audit[:-1]].count(True) == 1 and audit[[r["selected"] for r in audit[:-1]].index(True)]["C"] == on.c
    sc = StandardScaler().fit(x[tr])
    assert all(r["scaler_sha256"] == DA.fingerprint(sc.mean_.astype(np.float64), sc.scale_.astype(np.float64)) for r in audit)
    assert all(r["n_train"] == int(tr.sum()) and r["n_features"] == x.shape[1] and r["dataset"] == "d" and r["fold"] == 0 for r in audit)
    assert all(r["solver_params"]["max_iter"] == MAX_ITER == 3000 and r["max_iter"] == MAX_ITER for r in audit)
    assert all(r["exception"] is None and isinstance(r["n_iter"], int) and r["elapsed_s"] >= 0 for r in audit)
    assert all(sorted(r["classes"]) == sorted(np.unique(y[tr]).tolist()) for r in audit) and all("val_bacc" in r for r in audit[:-1])
    json.dumps(audit)  # record serializzabili (etichette testuali comprese)
    sig = inspect.signature(linear_probe).parameters
    assert sig["max_iter"].default == 3000 and sig["audit"].default is None and sig["scale"].default is True  # default invariati


def test_induced_convergence_warnings_are_attributed_to_each_fit_and_scoring_warnings_stay_apart():
    x, y, s, tr, va, te = _probe_split(np.random.default_rng(2), n_sub=12, per=30, n_cls=4, width=8, labels=None)
    y = y.copy()
    keep = y[va] != 3  # la validazione non ha la classe 3: la metrica avvisa (classi predette assenti), ma non e' un fit fallito
    audit: list = []
    with pytest.warns(ConvergenceWarning):  # i warning catturati vengono riemessi: nessuna soppressione
        res = linear_probe(x[tr], y[tr], x[va][keep], y[va][keep], x[te], y[te], s[te], n_boot=20, max_iter=1, audit=audit)
    assert len(audit) == len(C_GRID) + 1 and all(r["max_iter"] == 1 and r["solver_params"]["max_iter"] == 1 for r in audit)
    assert all(r["reached_limit"] and r["convergence_warnings"] >= 1 and not r["converged_by_criterion"] for r in audit)
    assert all(w["category"] == "ConvergenceWarning" for r in audit for w in r["warnings"])
    assert all("UndefinedMetricWarning" not in {w["category"] for w in r["warnings"]} for r in audit)  # mai confusi con la metrica
    scoring = [w["category"] for r in audit[:-1] for w in r.get("scoring_warnings", [])]
    assert "UndefinedMetricWarning" in scoring or res.bootstrap_warnings is not None  # la metrica avvisa a parte (validazione o bootstrap)
    assert res.c in C_GRID and audit[[r["selected"] for r in audit[:-1]].index(True)]["reached_limit"]  # un candidato non convergente non si scarta

    # un'eccezione dentro il terzo fit: registrata e rilanciata, i record restano tre
    audit2: list = []
    calls = {"n": 0}
    real = LogisticRegression.fit

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("fit rotto")
        return real(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LogisticRegression, "fit", boom)
        with pytest.raises(RuntimeError, match="fit rotto"):
            linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=20, audit=audit2)
    assert len(audit2) == 3 and audit2[2]["exception"] == "RuntimeError: fit rotto" and audit2[1]["exception"] is None


def test_scaler_and_c_come_from_train_and_val_only_and_folds_are_subject_disjoint():
    x, y, s, tr, va, te = _probe_split(np.random.default_rng(3))
    a: list = []
    r1 = linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=20, audit=a)
    b: list = []
    r2 = linear_probe(x[tr], y[tr], x[va], y[va], x[te] * 7 + 3, y[te], s[te], n_boot=20, audit=b)  # held-out perturbato
    assert r1.c == r2.c and r1.val_scores == r2.val_scores and a[0]["scaler_sha256"] == b[0]["scaler_sha256"]
    assert a[-1]["fit_phase"] == "refit" and a[-1]["n_train"] == int(tr.sum())  # il riaddestramento usa le righe di train, non train + validazione
    assert r1.n_val == int(va.sum()) and r1.n_test == int(te.sum())
    subjects = [f"S{i:02d}" for i in range(9)]
    roles = CP.fold_roles(subjects)
    for r in roles:
        assert not (set(r["train"]) & set(r["val"])) and not (set(r["train"]) & set(r["held"])) and not (set(r["val"]) & set(r["held"]))
        assert sorted(r["train"] + r["val"] + r["held"]) == sorted(subjects)
    # uno shard di fold: le righe OOF sono solo quelle dei soggetti held dei fold scelti, nell'ordine dei fold
    s_all = np.repeat(subjects, 20)
    y_all = np.tile(np.arange(3), len(s_all) // 3 + 1)[:len(s_all)]
    x_all = (np.random.default_rng(4).normal(size=(3, 6))[y_all] + np.random.default_rng(5).normal(size=(len(s_all), 6))).astype(np.float32)
    audit: list = []
    part = SC.oof_predictions(x_all, y_all, s_all, roles, 0, None, audit=audit, folds=[1, 3], context={"dataset": "d"})
    assert sorted(set(part["s"].tolist())) == sorted(roles[1]["held"] + roles[3]["held"]) and [f["fold"] for f in part["folds"]] == [1, 3]
    assert len(audit) == 2 * (len(C_GRID) + 1) and {r["fold"] for r in audit} == {1, 3}
    whole = SC.oof_predictions(x_all, y_all, s_all, roles, 0, None)
    assert np.array_equal(whole["row"][np.isin(whole["s"], roles[1]["held"] + roles[3]["held"])], part["row"])  # stesse righe, stesso ordine
    assert np.array_equal(whole["pred"][np.isin(whole["s"], roles[1]["held"] + roles[3]["held"])], part["pred"])
    with pytest.raises(ValueError, match="nessun fold"):
        SC.oof_predictions(x_all, y_all, s_all, roles, 0, None, folds=[])


def test_max_iter_reaches_every_candidate_and_the_refit_and_the_tie_break_is_unchanged():
    x, y, s, tr, va, te = _probe_split(np.random.default_rng(6), sep=6.0)  # separabile: tutti i C a pari punteggio di validazione
    audit: list = []
    res = linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=20, max_iter=250, audit=audit)
    assert [r["solver_params"]["max_iter"] for r in audit] == [250] * (len(C_GRID) + 1)
    assert len(set(res.val_scores.values())) == 1 and res.c == min(C_GRID)  # a parita' vince la regolarizzazione piu' forte
    assert res.solver_params["class_weight"] == "balanced" and res.solver_params["solver"] == "lbfgs" and res.solver_params["tol"] == 1e-4
    recipe = solver_recipe(12000)
    assert recipe["max_iter"] == recipe["params"]["max_iter"] == 12000 and "C" not in recipe["params"] and recipe["c_grid"] == list(C_GRID)
    assert PC._recipe_without_max_iter(recipe) == PC._recipe_without_max_iter(solver_recipe(3000))
    assert DA.fingerprint(recipe) != DA.fingerprint(solver_recipe(3000))


# ------------------------------------------------------------------------------------------------- shard, aggregato, parita', cancello, fasi

def test_shards_aggregate_to_the_whole_probe_the_parity_is_exact_and_the_two_phases_compare(tmp_path):
    camp = _campaign(tmp_path)
    out_root = tmp_path / "risultati"
    shards = _shards(camp, "A", out_root)
    reports, parities = {}, {}
    for (enc, task), paths in shards.items():
        pin = camp["pins_dict"]["reports"][f"{enc}_{task}"]
        rep_path, par_path = out_root / "A" / f"{enc}_{task}_pool.json", out_root / "A" / f"parita_{enc}_{task}.json"
        assert _run("aggregate", "--out", rep_path, "--phase", "A", "--encoder", enc, "--task", task, "--pins", camp["pins"],
                    "--splits", camp["splits"], "--shards", *paths, "--original", camp["originals"][enc, task],
                    "--original-sha256", pin["report_sha256"], "--original-oof-sha256", pin["oof_sha256"], "--parity-out", par_path) == 0
        reports[enc, task], parities[enc, task] = rep_path, par_path
        rep, orig, par = (json.loads(p.read_text()) for p in (rep_path, camp["originals"][enc, task], par_path))
        assert par["identical"] and par["predictions_differing_total"] == 0 and par["folds_with_different_c"] == 0 and par["S_abs_diff"] == 0
        assert par["original_legacy_solver_recipe"] and not par["problems"]
        assert rep["S"] == orig["S"] and rep["S_se"] == orig["S_se"] and rep["weights"] == orig["weights"]
        for ds in orig["datasets"]:
            do, dn = orig["datasets"][ds], rep["datasets"][ds]
            assert do["s"] == dn["s"] and do["oof_rows_sha256"] == dn["oof_rows_sha256"] and do["readout"] == dn["readout"] and do["features"] == dn["features"]
            assert [f["c"] for f in do["folds"]] == [f["c"] for f in dn["folds"]] and [f["fold"] for f in dn["folds"]] == [0, 1, 2, 3]
            assert all(f["n_fits"] == len(C_GRID) + 1 and "val_scores" in f and f["max_iter"] == 3000 for f in dn["folds"])
        oo, on = CS.load_oof(orig), CS.load_oof(rep)
        assert all(np.array_equal(oo[d][i], on[d][i]) for d in oo for i in range(3))
        assert rep["audit"]["n_records"] == len(PC.TASKS[task]) * 4 * 7 == rep["audit_summary"]["n_records"]
        assert rep["solver_recipe_sha256"] == DA.fingerprint(solver_recipe(3000)) and rep["cache_code_sha"] == CACHE_CODE and rep["phase"] == "A"
        SC.check_readout_pairing(orig, camp["originals"][("R3c" if enc == "B" else "B", task)] and json.loads(camp["originals"][("R3c" if enc == "B" else "B", task)].read_text()))
        CS.check_pairing([orig, rep])
        with pytest.raises(ValueError, match="ricetta del solver"):  # legacy contro nuovo: non si accoppiano nel confronto ufficiale
            SC.check_readout_pairing(orig, rep)
    for task in PC.TASKS:
        SC.check_readout_pairing(json.loads(reports["B", task].read_text()), json.loads(reports["R3c", task].read_text()))
    # aggregazione incompleta, doppia o manomessa: rifiutata
    b_p1 = shards["B", "p1"]
    with pytest.raises(PC.Problem, match="fold coperti"):
        PC.aggregate([Path(p) for p in b_p1[1:]], "A", "B", "p1", camp["splits"], camp["pins_dict"])
    with pytest.raises(PC.Problem, match="presente in due shard"):
        PC.aggregate([Path(p) for p in b_p1 + b_p1[:1]], "A", "B", "p1", camp["splits"], camp["pins_dict"])
    with pytest.raises(PC.Problem, match="altra fase"):
        PC.aggregate([Path(p) for p in b_p1], "B", "B", "p1", camp["splits"], camp["pins_dict"])
    tampered = tmp_path / "manomesso"
    tampered.mkdir()
    for p in b_p1:
        d = tampered / Path(p).parent.name
        d.mkdir()
        for f in Path(p).parent.iterdir():
            (d / f.name).write_bytes(f.read_bytes())
    first = json.loads((tampered / Path(b_p1[0]).parent.name / "shard.json").read_text())
    audit_path = Path(first["audit"]["path"])
    (tampered / Path(b_p1[0]).parent.name / "shard.json").write_text(json.dumps({**first, "audit": {**first["audit"], "path": str(tampered / Path(b_p1[0]).parent.name / "audit.jsonl")},
                                                                                  "oof": {**first["oof"], "path": str(tampered / Path(b_p1[0]).parent.name / "shard.oof.npz")}}))
    (tampered / Path(b_p1[0]).parent.name / "audit.jsonl").write_text(audit_path.read_text() + "{}\n")
    with pytest.raises(PC.Problem, match="cambiati"):
        PC.aggregate([tampered / Path(p).parent.name / "shard.json" for p in b_p1], "A", "B", "p1", camp["splits"], camp["pins_dict"])
    # cancello della fase B: passa con le quattro parita' identiche; una parita' non identica lo chiude e la fase B non parte
    gate = out_root / "A" / "gate.json"
    assert _run("gate", "--out", gate, "--reports", *reports.values(), "--parities", *parities.values()) == 0
    g = json.loads(gate.read_text())
    assert g["passed"] and g["n_records"] == g["expected_records"] == 280 and len(g["reports"]) == 4
    broken = tmp_path / "parita_rotta.json"
    par = json.loads(parities["B", "p1"].read_text())
    broken.write_text(json.dumps({**par, "identical": False, "predictions_differing_total": 3}))
    bad_gate = tmp_path / "gate_rotto.json"
    assert _run("gate", "--out", bad_gate, "--reports", *reports.values(), "--parities", *[broken if k == ("B", "p1") else v for k, v in parities.items()]) == 1
    assert not json.loads(bad_gate.read_text())["passed"]
    with pytest.raises(PC.Problem, match="cancello"):
        _run("probe", "--out", tmp_path / "no", "--phase", "B", "--encoder", "B", "--task", "p2", "--features", camp["features"]["B", "p2"],
             "--pins", camp["pins"], "--gate", bad_gate)
    with pytest.raises(PC.Problem, match="richiede --gate"):
        _run("probe", "--out", tmp_path / "no2", "--phase", "B", "--encoder", "B", "--task", "p2", "--features", camp["features"]["B", "p2"],
             "--pins", camp["pins"])
    # fase B (12.000 iterazioni) con il cancello, aggregati e confronto fra le fasi col bootstrap congiunto e il controllo ufficiale incrociato
    shards_b = _shards(camp, "B", out_root, gate=gate)
    reports_b = {}
    for (enc, task), paths in shards_b.items():
        rep_path = out_root / "B" / f"{enc}_{task}_pool.json"
        assert _run("aggregate", "--out", rep_path, "--phase", "B", "--encoder", enc, "--task", task, "--pins", camp["pins"],
                    "--splits", camp["splits"], "--shards", *paths) == 0
        reports_b[enc, task] = rep_path
        rep = json.loads(rep_path.read_text())
        assert rep["max_iter"] == 12000 and rep["solver_recipe"]["params"]["max_iter"] == 12000 and rep["gate"]["sha256"] == PC.sha256(gate)
    confronto = out_root / "confronto"
    assert _run("compare-phases", "--out", confronto, "--gate", gate, "--reports", *reports.values(), *reports_b.values()) == 0
    for task in PC.TASKS:
        c = json.loads((confronto / f"confronto_{task}.json").read_text())
        s, k = c["S"], c["contrasts"]
        assert c["order"] == ["A_R3c", "A_B", "B_R3c", "B_B"] and all(c["official"][p]["agrees_with_joint_bootstrap"] for p in ("A", "B"))
        assert k["B_minus_R3c_A"]["delta"] == pytest.approx(s["A_B"] - s["A_R3c"], abs=1e-12)
        assert k["delta_of_deltas"]["delta"] == pytest.approx(k["B_minus_R3c_B"]["delta"] - k["B_minus_R3c_A"]["delta"], abs=1e-12)
        assert k["B_B_minus_A"]["delta"] == pytest.approx(s["B_B"] - s["A_B"], abs=1e-12) and all(v["ci95"][0] <= v["ci95"][1] for v in k.values())
        assert s["A_B"] == json.loads(reports["B", task].read_text())["S"] and s["B_R3c"] == json.loads(reports_b["R3c", task].read_text())["S"]
        for enc in PC.ENCODERS:
            for d, ch in c["changes_12000_vs_3000"][enc].items():
                assert 0 <= ch["fraction_changed"] <= 1 and len(ch["c_3000"]) == len(ch["c_12000"]) == 4
        assert set(c["warnings"]) == set(c["order"]) and set(c["residual_warnings_at_12000"]) == set(PC.ENCODERS)
        assert sum(v["weight"] for v in c["datasets"].values()) == pytest.approx(1.0)
    riepilogo = json.loads((confronto / "riepilogo.json").read_text())
    assert set(riepilogo["tasks"]) == {"p1", "p2"} and riepilogo["gate"]["sha256"] == PC.sha256(gate)


def test_recipe_and_identity_guards_reject_mismatches_including_textual_labels(tmp_path):
    rep_a = {"task": "p1", "splits_sha256": "x", **{k: 1 for k in SC.PROBE_KEYS}, "weights": {"d": 1.0}, "datasets": {"d": {"input_sha256": "i", "oof_rows_sha256": "r"}},
             "solver_recipe_sha256": "s1"}
    rep_b = {**rep_a, "solver_recipe_sha256": "s2"}
    legacy = {k: v for k, v in rep_a.items() if k != "solver_recipe_sha256"}
    SC.check_readout_pairing(rep_a, dict(rep_a))
    SC.check_readout_pairing(legacy, dict(legacy))  # due rapporti storici si accoppiano come prima
    for other in (rep_b, legacy):
        with pytest.raises(ValueError, match="ricetta del solver"):
            SC.check_readout_pairing(rep_a, other)
    good = {"max_iter": 3000, "solver_recipe": solver_recipe(3000), "solver_recipe_sha256": DA.fingerprint(solver_recipe(3000)), "checkpoint_sha256": "b",
            "numeric_sha256": "n", "weights": {"d": 1.0}, "splits_sha256": "x", **{k: 1 for k in SC.PROBE_KEYS}, "task": "p1",
            "datasets": {"d": {"input_sha256": "i", "oof_rows_sha256": "r"}}}
    reps = {("A", "R3c"): {**good, "checkpoint_sha256": "c"}, ("A", "B"): dict(good),
            ("B", "R3c"): {**good, "checkpoint_sha256": "c", "max_iter": 12000, "solver_recipe": solver_recipe(12000), "solver_recipe_sha256": DA.fingerprint(solver_recipe(12000))},
            ("B", "B"): {**good, "max_iter": 12000, "solver_recipe": solver_recipe(12000), "solver_recipe_sha256": DA.fingerprint(solver_recipe(12000))}}
    PC.phase_guard(reps)
    with pytest.raises(PC.Problem, match="max_iter diverso dalla fase"):
        PC.phase_guard({**reps, ("B", "B"): {**reps["B", "B"], "max_iter": 3000}})
    with pytest.raises(PC.Problem, match="legacy"):
        PC.phase_guard({**reps, ("A", "B"): {k: v for k, v in reps["A", "B"].items() if k != "solver_recipe_sha256"}})
    with pytest.raises(ValueError, match="ricetta del solver"):  # dentro una fase i due encoder devono avere la stessa firma
        PC.phase_guard({**reps, ("A", "B"): {**reps["A", "B"], "solver_recipe_sha256": "altro"}})
    other = solver_recipe(12000)
    other["params"]["tol"] = 1e-3
    with pytest.raises(PC.Problem, match="oltre a max_iter"):  # fra le fasi cambia solo max_iter
        PC.phase_guard({**reps, ("B", "B"): {**reps["B", "B"], "solver_recipe": other, "solver_recipe_sha256": DA.fingerprint(other)},
                        ("B", "R3c"): {**reps["B", "R3c"], "solver_recipe": other, "solver_recipe_sha256": DA.fingerprint(other)}})
    with pytest.raises(PC.Problem, match="stesso checkpoint"):
        PC.phase_guard({**reps, ("A", "R3c"): dict(good)})
    with pytest.raises(PC.Problem, match="contesto numerico"):  # dtype, backend o thread diversi fra i rapporti
        PC.phase_guard({**reps, ("B", "B"): {**reps["B", "B"], "numeric_sha256": "altro"}})
    # etichette testuali (EPN612) e numeriche nel bootstrap congiunto; finitezza controllata sulle feature, non sulle etichette
    rng = np.random.default_rng(7)
    s = np.repeat([f"u{i}" for i in range(6)], 10)
    y = np.asarray(["fist", "open"] * 30)
    oofs = [{"e": (y, np.where(rng.random(60) < p, y, np.asarray(["open", "fist"] * 30)), s)} for p in (0.7, 0.75, 0.72, 0.8)]
    jb = PC.joint_bootstrap(oofs, {"e": 1.0}, n_boot=50)
    assert jb["S_point"].shape == (4,) and jb["S_draws"].shape == (50, 4) and np.all((0 <= jb["S_point"]) & (jb["S_point"] <= 1))
    with pytest.raises(PC.Problem, match="finestre o soggetti diversi"):
        PC.joint_bootstrap(oofs + [{"e": (y[::-1], y, s)}], {"e": 1.0}, n_boot=5)
    feats = tmp_path / "f"
    feats.mkdir()
    all_subjects = [f"user{i:02d}" for i in range(1, 13)]
    internal = CP.p2_internal_subjects(all_subjects)
    s2 = np.repeat(internal, 10)
    y2 = np.asarray(["fist", "open"] * (len(s2) // 2))
    x = rng.normal(size=(len(s2), 4)).astype(np.float32)
    x[3, 2] = np.nan
    _write_cache(feats / "epn612.npz", x, y2, s2, internal, "p2", CP.P2_SPLITS_SHA256, "/c.pt", "a" * 64, all_subjects=all_subjects)
    with pytest.raises(ValueError, match="feature non finite"):  # etichette testuali accettate, feature non finite rifiutate
        SC.load_cache(feats, "epn612", "p2", None, CP.P2_SPLITS_SHA256, "pool")


def test_joint_bootstrap_reproduces_the_official_paired_comparison_and_the_delta_of_deltas():
    rng = np.random.default_rng(11)
    subjects = [f"u{i}" for i in range(8)]
    s = np.repeat(subjects, 15)
    oofs, weights = [], {"a": 0.6, "b": 0.4}
    y = {d: np.tile(np.arange(3), 40) for d in weights}
    for p in (0.5, 0.56, 0.52, 0.6):  # R3c_A, B_A, R3c_B, B_B
        oofs.append({d: (y[d], np.where(rng.random(120) < p, y[d], (y[d] + 1) % 3), s) for d in weights})
    jb = PC.joint_bootstrap(oofs, weights, n_boot=300, seed=20261009)
    # il confronto ufficiale: `arms_compare.s_comparison` ricampiona i soggetti per dataset in ordine alfabetico con un solo generatore
    rng2 = np.random.default_rng(20261009)
    w = np.array([weights[d] for d in sorted(weights)])
    rows, draws = {}, []
    for d in sorted(weights):
        stats = [DA.subject_counts(y[d], oofs[i][d][1], s) for i in (0, 1, 0, 1)]
        res, sample = DA.paired_component(stats, rng2, 300)
        rows[d] = res
        draws.append(sample)
    official_delta, official_ci = float(w @ [rows[d]["delta"] for d in sorted(weights)]), np.quantile(w @ np.asarray(draws), [.025, .975]).tolist()
    mine = PC.contrast(jb["S_point"], jb["S_draws"], PC.CONTRASTS["B_minus_R3c_A"])
    assert mine["delta"] == pytest.approx(official_delta, abs=1e-12) and mine["ci95"] == pytest.approx(official_ci, abs=1e-12)
    dd = PC.contrast(jb["S_point"], jb["S_draws"], PC.CONTRASTS["delta_of_deltas"])
    a, b = PC.contrast(jb["S_point"], jb["S_draws"], PC.CONTRASTS["B_minus_R3c_A"]), PC.contrast(jb["S_point"], jb["S_draws"], PC.CONTRASTS["B_minus_R3c_B"])
    assert dd["delta"] == pytest.approx(b["delta"] - a["delta"], abs=1e-12)
    assert jb["S_point"][1] == pytest.approx(float(w @ [rows[d]["B"] for d in sorted(weights)]), abs=1e-12)  # pesi degli aggregati rispettati
    per = {d: PC.contrast(jb["point"][i], jb["draws"][i], PC.CONTRASTS["B_minus_R3c_A"]) for i, d in enumerate(jb["datasets"])}
    assert all(per[d]["delta"] == pytest.approx(rows[d]["delta"], abs=1e-12) and per[d]["ci95"] == pytest.approx(rows[d]["ci95"], abs=1e-12) for d in weights)


# ------------------------------------------------------------------------------------------- regressioni della review GPT de06bca (R1, R2, R3)

def _minimal_pair(tmp_path):
    """Un rapporto storico minimo e la sua copia «nuova» (come la riproduzione indipendente di GPT): stesse righe, ruoli, C, S."""
    oof = tmp_path / "oof.npz"
    rows = np.arange(4)
    np.savez(oof, datasets=np.array(["d"]), d_y=[0, 1, 0, 1], d_pred=[0, 1, 1, 1], d_s=["u", "u", "v", "v"], d_row=rows)
    fold = dict(train=["t"], val=["w"], held=["u", "v"], c=1.0, held_bacc=.75, val_bacc=.75, train_bacc=.8, n_train=10, n_val=4, n_held=4)
    rep = {**{k: 0 for k in SC.PROBE_KEYS}, "kind": "aggregate", "phase": "A", "task": "p1", "weights": {"d": 1.0}, "splits_sha256": "split",
           "extraction_recipe": {}, "feature_set": "pool", "checkpoint_sha256": "checkpoint", "S": .75, "S_se": .1, "oof": str(oof),
           "oof_sha256": PC.sha256(oof), "datasets": {"d": {"input_sha256": "input", "oof_rows_sha256": DA.fingerprint(rows), "subjects": ["u", "v"],
                                                            "s": .75, "folds": [fold]}}}
    original = tmp_path / "original.json"
    original.write_text(json.dumps(rep))
    return rep, original, oof


@pytest.mark.parametrize("change", [("fold", "val_bacc", .5), ("fold", "train_bacc", .9), ("fold", "held_bacc", .5), ("fold", "c", 10.0),
                                    ("fold", "n_held", 3), ("fold", "val", ["q"]), ("dataset", "subjects", ["u", "w"]), ("dataset", "s", .74),
                                    ("report", "S_se", .2), ("report", "weights", {"d": .9}), ("report", "fold_seed", 1)])
def test_parity_rejects_any_single_scientific_difference(tmp_path, change):
    """R1: una differenza sola alla volta, in un campo scientifico confrontabile, chiude la parita' e viene elencata; nessuna tolleranza."""
    rep, original, oof = _minimal_pair(tmp_path)
    changed = json.loads(json.dumps(rep))
    level, key, value = change
    if level == "fold":
        changed["datasets"]["d"]["folds"][0][key] = value
    elif level == "dataset":
        changed["datasets"]["d"][key] = value
    else:
        changed[key] = value
    new = tmp_path / "new.json"
    new.write_text(json.dumps(changed))
    same = tmp_path / "same.json"
    same.write_text(json.dumps(rep))
    assert PC.parity_check(original, PC.sha256(original), PC.sha256(oof), same)["identical"]
    parity = PC.parity_check(original, PC.sha256(original), PC.sha256(oof), new)
    assert not parity["identical"] and parity["problems"] and parity["predictions_differing_total"] == 0
    assert any(key in p for p in parity["problems"]) or any(key in k for d in parity["differences"]["datasets"].values() for k in d)
    if key == "val_bacc":
        assert parity["datasets"]["d"]["val_bacc_abs_diff"] == [.25]


def test_fit_exception_persists_the_audit_and_the_failure_on_disk_and_the_aggregator_refuses_it(tmp_path):
    """R2: i record dei fit finiti e quello del fit fallito sono su disco prima del rilancio; nessun shard.json; l'aggregatore rifiuta lo shard."""
    camp = _campaign(tmp_path)
    out = tmp_path / "risultati" / "A" / "B_p2" / "shard_p2"
    calls = {"n": 0}
    real = LogisticRegression.fit

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("fit rotto iniettato")
        return real(self, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LogisticRegression, "fit", boom)
        with pytest.raises(RuntimeError, match="fit rotto iniettato"):
            _run("probe", "--out", out, "--phase", "A", "--encoder", "B", "--task", "p2", "--features", camp["features"]["B", "p2"],
                 "--pins", camp["pins"], "--splits", camp["splits"])
    assert not (out / "shard.json").exists() and not (out / "audit.jsonl").exists()
    progress = [json.loads(l) for l in (out / "audit_progress.jsonl").read_text().splitlines() if l.strip()]
    assert len(progress) == 3 and progress[2]["exception"] == "RuntimeError: fit rotto iniettato" and progress[0]["exception"] is None
    assert [r["fit_phase"] for r in progress] == ["candidate"] * 3 and all(r["experiment_phase"] == "A" for r in progress)  # due chiavi distinte
    assert "val_bacc" in progress[0] and progress[0]["dataset"] == "epn612" and progress[0]["fold"] == 0
    failed = json.loads((out / "shard_failed.json").read_text())
    assert failed["complete"] is False and "fit rotto iniettato" in failed["exception"] and failed["audit"]["n_records"] == 3 and failed["datasets_completed"] == []
    with pytest.raises(PC.Problem, match="interrotto o fallito"):
        PC.aggregate([out / "shard.json"], "A", "B", "p2", camp["splits"], camp["pins_dict"])
    with pytest.raises(PC.Problem, match="manca"):  # anche senza il marcatore: niente shard.json, niente aggregato
        (out / "shard_failed.json").unlink()
        PC.aggregate([out / "shard.json"], "A", "B", "p2", camp["splits"], camp["pins_dict"])
    # a shard completo, il file progressivo contiene tutti i fit (senza `selected`) e l'audit completo li ha tutti con `selected`
    ok = tmp_path / "risultati" / "A" / "R3c_p2" / "shard_p2"
    assert _run("probe", "--out", ok, "--phase", "A", "--encoder", "R3c", "--task", "p2", "--features", camp["features"]["R3c", "p2"],
                "--pins", camp["pins"], "--splits", camp["splits"]) == 0
    prog = [json.loads(l) for l in (ok / "audit_progress.jsonl").read_text().splitlines() if l.strip()]
    full = [json.loads(l) for l in (ok / "audit.jsonl").read_text().splitlines() if l.strip()]
    assert len(prog) == len(full) == 2 * 4 * 7 and sum(r["selected"] is True for r in full) == 8
    assert json.loads((ok / "shard.json").read_text())["complete"] is True


def test_phase_b_stops_before_any_fit_if_recipe_context_or_attested_reports_differ(tmp_path, monkeypatch):
    """R3: prima dei fit della fase B si confrontano la ricetta corrente (riportata a max_iter 3000) e il contesto numerico con il cancello, e i
    rapporti attestati devono esistere con gli SHA attestati; una differenza ferma il job senza spendere un fit."""
    reached = []

    def spy(*a, **k):
        reached.append(True)
        raise RuntimeError("FIT RAGGIUNTO")

    monkeypatch.setattr(SC, "oof_predictions", spy)
    env = PC.environment()
    recipe = solver_recipe(3000, seed=0)
    report = tmp_path / "A_B_p1_pool.json"
    report.write_text("{}")
    good_gate = {"protocol": PC.PROTOCOL, "phase": "A", "passed": True, "solver_recipe_sha256": DA.fingerprint(recipe), "solver_recipe": recipe,
                 "numeric_context": PC.numeric_context(env), "numeric_sha256": DA.fingerprint(PC.numeric_context(env)),
                 "reports": {f"{e}_{t}": {"path": str(report), "sha256": PC.sha256(report)} for e in PC.ENCODERS for t in PC.TASKS}}
    other_numeric = {**good_gate["numeric_context"], "thread_limits": {"OMP_NUM_THREADS": "99", "OPENBLAS_NUM_THREADS": "99", "MKL_NUM_THREADS": "99"}}
    cases = {"seed": ({**good_gate}, ["--seed", "1"], "ricetta del solver diversa"),
             "sha_only": ({**good_gate, "solver_recipe_sha256": DA.fingerprint(solver_recipe(3000, seed=1))}, [], "ricetta del solver diversa"),
             "content": ({**good_gate, "solver_recipe": {**recipe, "tie_break": "altro"}}, [], "per esteso"),
             "numeric": ({**good_gate, "numeric_context": other_numeric, "numeric_sha256": DA.fingerprint(other_numeric)}, [], "contesto numerico"),
             "reports": ({k: v for k, v in good_gate.items() if k != "reports"}, [], "quattro rapporti"),
             "closed": ({**good_gate, "passed": False}, [], "cancello")}
    for name, (gate, extra, msg) in cases.items():
        gate_path = tmp_path / f"gate_{name}.json"
        gate_path.write_text(json.dumps(gate))
        with pytest.raises(PC.Problem, match=msg):
            _run("probe", "--out", tmp_path / f"out_{name}", "--phase", "B", "--encoder", "B", "--task", "p2", "--features", tmp_path / "nessuna",
                 "--pins", tmp_path / "nessuno.json", "--gate", gate_path, *extra)
        assert not reached and not (tmp_path / f"out_{name}").exists()
    report.write_text("{ }")  # il rapporto attestato e' cambiato dopo il cancello: la fase B non parte
    gate_path = tmp_path / "gate_changed.json"
    gate_path.write_text(json.dumps(good_gate))
    with pytest.raises(PC.Problem, match="attestato dal cancello"):
        _run("probe", "--out", tmp_path / "out_changed", "--phase", "B", "--encoder", "B", "--task", "p2", "--features", tmp_path / "nessuna",
             "--pins", tmp_path / "nessuno.json", "--gate", gate_path)
    assert not reached


# ------------------------------------------------------------------------------------------------------------------------- piano e launcher

def test_plan_covers_every_shard_once_per_phase_with_blocks_under_the_queue_limit(tmp_path):
    camp = _campaign(tmp_path) if False else None  # il piano non ha bisogno delle cache: bastano i pin
    pins = {"checkpoints": {}, "reports": {f"{e}_{t}": {"report_sha256": "a" * 64, "oof_sha256": "b" * 64} for e in PC.ENCODERS for t in PC.TASKS},
            "caches": {"B/p1/ninapro_db2": {"code_sha": CACHE_CODE}}}
    plan = PC.build_plan("f" * 40, "pilot-posa-sonde-20261029", "/W/results", "/S/features", "/W/originali", "results/infra/x/pin.json", "gpt",
                         "pilot-posa-sonde-20261029", pins)
    jobs = plan["jobs"]
    assert len(jobs) == 30 and len({j["id"] for j in jobs}) == 30 and plan["gpu_hours_cap"] == 0 and plan["cache_code_sha"] == CACHE_CODE
    covered = {"A": [], "B": []}
    for j in jobs:
        if j["stage"] == "probe":
            argv = shlex.split(j["command"])
            phase = argv[argv.index("--phase") + 1]
            i, k = argv.index("--datasets"), argv.index("--folds")
            datasets, folds = argv[i + 1:k], argv[k + 1:argv.index("--features")]
            enc = argv[argv.index("--encoder") + 1]
            covered[phase] += [(enc, d, int(f)) for d in datasets for f in folds]
            assert ("--gate" in argv) == (phase == "B") and j["id"].startswith(f"PILOT-POSA-SONDE-{phase}-{enc}-")
    for phase in ("A", "B"):
        assert sorted(covered[phase]) == sorted((e, d, f) for e in PC.ENCODERS for t in PC.TASKS for d in PC.TASKS[t] for f in range(4))
    for j in jobs:
        argv = shlex.split(j["command"])
        assert argv[:2] == ["python3", "scripts/submit_job.py"] and argv[argv.index("--gpu-ore") + 1] == "0" and argv[argv.index("--agente") + 1] == "gpt"
        tail = argv[argv.index("--") + 1:]  # export, cpu, memoria, tempo, launcher, uscita, fase, argomenti
        assert f"EXPECTED_SHA={'f' * 40}" in tail[0] and tail[4].endswith("scripts/slurm/probe_convergence.sbatch") and tail[5:7] == [j["out"], j["stage"]]
        assert j["core_hours_cap"] > 0 and j["block"] in range(1, 6)
    for b in plan["blocks"]:
        assert b["n_jobs"] <= plan["serial_partition"]["max_queued_per_user"]
    compare = next(j for j in jobs if j["stage"] == "compare-phases")
    assert len(compare["depends_on"]) == 4 and all("aggregate" in d for d in compare["depends_on"]) and compare["block"] == 5
    assert all(j["time"] <= "04:00:00" for j in jobs)
    assert plan["core_hours_cap"]["A"] < 60 and plan["core_hours_cap"]["B"] < 130 and plan["core_hours_cap"]["total"] == pytest.approx(plan["core_hours_cap"]["A"] + plan["core_hours_cap"]["B"])
    assert plan["core_hours_cap"]["total_with_checkout"] == pytest.approx(plan["core_hours_cap"]["total"] + 1 / 6)  # il checkout: 1 core x 10 min
    assert plan["fits"] == {"per_fold": 7, "per_dataset": 28, "p1": 84, "p2": 56, "per_phase": 280}
    a_aggr = next(j for j in jobs if j["stage"] == "aggregate" and "-A-B-p1-" in j["id"])
    assert "--original /W/originali/B_p1_pool.json" in a_aggr["command"] and "--parity-out" in a_aggr["command"]
    # il comando scritto nel file e lo stesso piano
    pins_path = tmp_path / "pin.json"
    pins_path.write_text(json.dumps(pins))
    out_dir = tmp_path / "piano"
    assert PC.main(["--stage", "plan", "--code-sha", "f" * 40, "--pins", str(pins_path), "--out-dir", str(out_dir)]) == 0
    lines = [l for l in (out_dir / "comandi.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert lines == [j["command"] for j in json.loads((out_dir / "piano.json").read_text())["jobs"]]


def test_launcher_requires_a_frozen_checkout_and_outputs_outside_the_code(work):  # noqa: F811
    root, canon, first, second = work
    env = _launcher_env(root)
    launcher = SLURM / "probe_convergence.sbatch"
    out = f"{root}/out/A/B_p1/shard_db2a"
    args = [out, "probe", "--phase", "A", "--encoder", "B", "--task", "p1"]
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(canon), "EXPECTED_SHA": second}, capture_output=True, text=True)
    assert f"VENV_PYTHON scripts/probe_convergence.py --stage probe --out {out} --phase A --encoder B --task p1" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True)  # senza CODE_DIR: rifiutato
    assert r.returncode != 0 and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(canon), "EXPECTED_SHA": first}, capture_output=True, text=True)
    assert r.returncode == 2 and "atteso" in r.stderr and "VENV_PYTHON" not in r.stdout  # SHA diverso dal checkout
    for bad, msg in (("rel/out", "relativa"), (f"{root}/wearusfm/out", "dentro il codice"), (f"{root}/wearusfm_campaigns/x/out", "dentro il codice")):
        r = subprocess.run(["bash", str(launcher), bad, "probe"], env={**env, "CODE_DIR": str(canon), "EXPECTED_SHA": second}, capture_output=True, text=True)
        assert r.returncode == 2 and msg in r.stdout and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), out, "extract"], env={**env, "CODE_DIR": str(canon), "EXPECTED_SHA": second}, capture_output=True, text=True)
    assert r.returncode == 2 and "sconosciuta" in r.stdout
