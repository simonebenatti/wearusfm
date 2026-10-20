"""Diagnostica dei rami (`scripts/spectral_diagnostic.py`, protocollo `rami_diagnostica_v1`) su cache sintetiche con la struttura di quelle del
2C e della tranche downstream: righe di test della P1 mai lette, soggetti di test della P2 scartati, provenienza e righe verificate, sonda
della validation interna nel formato della calibrazione, confronti accoppiati per famiglia e lettura congelata."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("spectral_diagnostic", REPO / "scripts" / "spectral_diagnostic.py")
SDG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SDG)

from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

SPLITS = REPO / "splits" / "v1" / "splits_v1.json"
P2_USERS = {"epn612": [f"testingJSON/user{i}" for i in range(1, 25)], "uci_emg": [f"{i:02d}" for i in range(1, 21)]}


def _x(rng, y, dim, signal, centers):
    return (signal * centers[y][:, :dim] + rng.normal(size=(len(y), dim))).astype(np.float32)


def _rows(task, ds):
    """{prefisso: (y, s)}: P1 coi ruoli della cache (train_/val_/test_ come probe_p1), P2 una sola tabella con tutti i soggetti."""
    if task == "p1":
        roles = P1.subject_roles(json.loads(SPLITS.read_text()), ds)
        out = {}
        for role in ("train", "val", "test"):
            subs = roles[role]
            out[f"{role}_"] = (np.tile(np.arange(4), 2 * len(subs)), np.repeat(subs, 8))
        return out
    subs = P2_USERS[ds]
    return {"": (np.tile(np.arange(4), 2 * len(subs)), np.repeat(subs, 8))}


def build(tmp_path, task, arms=("A_seed0", "A_seed1", "B_seed0", "B_seed1"), signal=None, nan_test=True):
    """Cache del 2C (rami al passo 0 e allenati) e della tranche downstream con le loro provenienze. `signal`: intensita' per parte."""
    signal = signal or {"raw": 0.6, "init": 1.0, "trained": 1.0}
    ev, dn = tmp_path / "eval", tmp_path / "down"
    for k, arm in enumerate(arms):
        rng = np.random.default_rng(k)
        centers = rng.normal(size=(4, SDG.RAW_DIM))
        caches = {"raw": {}, "init": {}, "trained": {}}
        for ds in SDG.DATASETS[task]:
            rows = _rows(task, ds)
            for part in caches:
                folder = (dn / task / arm) if part == "raw" else (ev / task / arm / f"branches_{part}")
                folder.mkdir(parents=True, exist_ok=True)
                dim = SDG.RAW_DIM if part == "raw" else SDG.BRANCH_COLS
                arrays = {}
                for p, (y, s) in rows.items():
                    x = _x(rng, y, dim, signal[part], centers)
                    if p == "test_" and nan_test:
                        x[:] = np.nan  # se la diagnostica leggesse le righe di test, fallirebbe sul controllo dei valori finiti
                    arrays.update({p + "x": x, p + "y": y, p + "s": s})
                path = folder / f"{ds}.npz"
                np.savez(path, **arrays, input_sha256=np.asarray("i" * 64))
                entry = {"sha256": sha256_file(path), "input_sha256": "i" * 64,
                         "rows": {p: {"n": len(y), "labels_subjects_sha256": fingerprint(y, s), "subjects": sorted(set(s.tolist()))}
                                  for p, (y, s) in rows.items()}}
                if part != "raw":
                    entry.update(feature_set="branches", model_dim=SDG.BRANCH_COLS)
                caches[part][ds] = entry
        for part, cache in caches.items():
            if part == "raw":
                ident = {**SDG.DOWNSTREAM, "task": task, "arm": arm, "which": "teacher", "split_sha256": SDG.SPLIT_SHA,
                         "checkpoint_sha256": SDG.CHECKPOINTS[arm], "probe_seed": 0}
                (dn / task / arm / "provenance.json").write_text(json.dumps({"identity": ident, "cache": cache}))
            else:
                ident = {**SDG.STEP2C, "task": task, "arm": arm, "variant": f"branches_{part}", "which": "teacher", "probe_seed": 0,
                         "split_sha256": SDG.SPLIT_SHA, "source_checkpoint_sha256": SDG.CHECKPOINTS[arm]}
                (ev / task / arm / f"branches_{part}" / "provenance.json").write_text(json.dumps({"identity": ident, "cache": cache}))
    return ev, dn


def _probe(ev, dn, out, task, arm, sets, *extra):
    return SDG.main(["probe", "--task", task, "--arm", arm, "--sets", *sets, "--eval-root", str(ev), "--downstream-root", str(dn),
                     "--out-dir", str(out), *extra])


def test_p1_uses_only_pretraining_subjects_and_never_reads_test_rows(tmp_path):
    ev, dn = build(tmp_path, "p1", arms=("B_seed0",))
    out = tmp_path / "out"
    assert _probe(ev, dn, out, "p1", "B_seed0", ["R", "PW0", "PW", "RPW"], "--redundancy") == 0  # test_x e' NaN: mai letto
    splits = json.loads(SPLITS.read_text())
    for name in ("R", "PW0", "PW", "RPW"):
        rep = json.loads((out / "p1" / "B_seed0" / f"{name}.json").read_text())
        assert rep["protocol"] == SDG.PROTOCOL and 0 < rep["S"] <= 1 and rep["splits_sha256"] == SDG.SPLIT_SHA
        with np.load(rep["oof"]) as z:
            for ds in SDG.DATASETS["p1"]:
                pre = sorted(splits["datasets"][ds]["pretraining"])
                assert sorted(set(z[f"{ds}_s"].tolist())) == pre and not set(pre) & set(splits["datasets"][ds]["test"])
        width = {"R": 768, "PW0": 256, "PW": 256, "RPW": 1024}[name]
        assert all(v["features"]["columns"] == width for v in rep["datasets"].values())
    red = json.loads((out / "p1" / "B_seed0" / "redundancy.json").read_text())
    for row in red["datasets"].values():
        assert all(0 <= row[k] <= 1 for k in ("cka_R_PW0", "cka_R_PW", "cka_PW0_PW")) and row["r2_PW_from_R"] < 1
        assert len(row["alphas_PW"]) == 4 and set(row["alphas_PW"]) <= set(SDG.RIDGE_ALPHAS)


def test_p2_keeps_only_subjects_outside_the_three_official_tests(tmp_path):
    ev, dn = build(tmp_path, "p2", arms=("A_seed0",))
    out = tmp_path / "out"
    assert _probe(ev, dn, out, "p2", "A_seed0", ["PW0"]) == 0
    rep = json.loads((out / "p2" / "A_seed0" / "PW0.json").read_text())
    assert rep["weights"] == {"epn612": 0.5, "uci_emg": 0.5} and rep["window_s"] is None
    from wearusfm.harness.splits import split_subjects

    with np.load(rep["oof"]) as z:
        for ds, users in P2_USERS.items():
            test = set().union(*(split_subjects(users, (0.7, 0.1, 0.2), seed=k).test for k in (0, 1, 2)))
            seen = set(z[f"{ds}_s"].tolist())
            assert seen and not seen & test and seen == set(users) - test


def test_changed_files_foreign_provenance_and_misaligned_rows_are_refused(tmp_path):
    ev, dn = build(tmp_path, "p1", arms=("B_seed0",))
    out = tmp_path / "out"
    path = ev / "p1" / "B_seed0" / "branches_trained" / "ninapro_db3.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["train_x"] = d["train_x"] + 1  # file cambiato dopo la provenienza
    np.savez(path, **d)
    with pytest.raises(ValueError, match="SHA-256 diverso"):
        _probe(ev, dn, out, "p1", "B_seed0", ["PW"])
    d["train_x"] = d["train_x"] - 1
    d["train_s"] = d["train_s"][::-1].copy()  # righe permutate, provenienza aggiornata al nuovo file: l'impronta delle righe non torna
    np.savez(path, **d)
    prov_path = path.parent / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["cache"]["ninapro_db3"]["sha256"] = sha256_file(path)
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(ValueError, match="righe"):
        _probe(ev, dn, out, "p1", "B_seed0", ["PW"])
    prov["identity"]["source_checkpoint_sha256"] = SDG.CHECKPOINTS["A_seed0"]  # cache di un altro teacher
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(ValueError, match="provenienza diversa"):
        _probe(ev, dn, out, "p1", "B_seed0", ["PW"])


def test_compare_reads_the_contrasts_per_family_with_the_frozen_rule(tmp_path, capsys):
    """Rami allenati senza segnale e rami al passo 0 col segnale: i rami peggiorano da soli e nella concatenazione."""
    ev, dn = build(tmp_path, "p2", signal={"raw": 0.15, "init": 1.5, "trained": 0.0})
    out = tmp_path / "out"
    for arm in SDG.CHECKPOINTS:
        assert _probe(ev, dn, out, "p2", arm, list(SDG.SETS)) == 0
    res_path = tmp_path / "esito.json"
    assert SDG.main(["compare", "--out-dir", str(out), "--out", str(res_path), "--tasks", "p2"]) == 0
    res = json.loads(res_path.read_text())
    for fam in ("A", "B"):
        c = res["families"]["p2"][fam]["contrasts"]
        assert c["rami_da_soli"]["reading"] == "peggiora" and c["concatenazione"]["reading"] == "peggiora"
        assert c["apporto_passo0"]["reading"] == "migliora" and c["rami_da_soli"]["delta"] < -0.1
        assert res["families"]["p2"][fam]["answer"] == SDG.ANSWERS[(True, True)]
        assert len(c["rami_da_soli"]["delta_by_model_seed"]) == 2
    assert set(res["teachers"]["p2"]) == set(SDG.CHECKPOINTS)
    assert "rami_da_soli" in capsys.readouterr().out
    assert SDG.main(["compare", "--out-dir", str(out), "--out", str(tmp_path / "b.json"), "--tasks", "p2", "--families", "B"]) == 0
    assert set(json.loads((tmp_path / "b.json").read_text())["families"]["p2"]) == {"B"}
    rep = out / "p2" / "B_seed1" / "PW.json"
    r = json.loads(rep.read_text())
    rep.write_text(json.dumps({**r, "fold_seed": 1}))  # rapporti non accoppiati: nessun confronto
    with pytest.raises(ValueError, match="provenienza non accoppiata"):
        SDG.main(["compare", "--out-dir", str(out), "--out", str(res_path), "--tasks", "p2"])
    rep.write_text(json.dumps({**r, "set": "PW0"}))
    with pytest.raises(ValueError, match="altro protocollo"):
        SDG.main(["compare", "--out-dir", str(out), "--out", str(res_path), "--tasks", "p2"])


def test_linear_cka_and_reading_rule():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 20))
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert SDG.linear_cka(x, x) == pytest.approx(1.0) and SDG.linear_cka(x, 3 * x @ q) == pytest.approx(1.0)
    assert SDG.linear_cka(x, rng.normal(size=(500, 20))) < 0.1
    base = {"beats": False, "ci95": [-0.03, -0.005], "delta_by_model_seed": [-0.02, -0.01]}
    assert SDG.reading({**base, "delta": -0.015}) == "peggiora"
    assert SDG.reading({**base, "delta": -0.009}) == "entro le soglie"  # sotto la soglia di 1 punto
    assert SDG.reading({**base, "delta": -0.015, "delta_by_model_seed": [-0.03, 0.001]}) == "entro le soglie"  # un teacher contrario
    assert SDG.reading({**base, "delta": -0.015, "ci95": [-0.03, 0.001]}) == "entro le soglie"  # IC che tocca lo zero
    assert SDG.reading({**base, "beats": True, "delta": 0.02}) == "migliora"
    assert set(SDG.ANSWERS) == {(a, b) for a in (True, False) for b in (True, False)}
