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
                run = {**SDG.STEP2C, "arm": arm, "collaudo": False, "complete": True, "encoder_identical": True, "steps": 3000,
                       "source_checkpoint_sha256": SDG.CHECKPOINTS[arm], "checkpoint_sha256": f"{k}" * 64, "init_checkpoint_sha256": f"{k + 4}" * 64}
                (ev / task / arm / f"branches_{part}" / "provenance.json").write_text(json.dumps({"identity": ident, "run": run, "cache": cache}))
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
        assert res["families"]["p2"][fam]["answer"] == SDG.answer("peggiora", "peggiora")
        assert len(c["rami_da_soli"]["delta_by_model_seed"]) == 2
        dd = res["families"]["p2"][fam]["differenza_dei_delta"]  # descrittivo: entrambi i delta negativi, nessuna soglia
        assert len(dd["delta_by_model_seed"]) == 2 and dd["ci95"][0] <= dd["delta"] <= dd["ci95"][1] and set(dd["datasets"]) == {"epn612", "uci_emg"}
        assert dd["delta"] == pytest.approx(c["concatenazione"]["delta"] - c["rami_da_soli"]["delta"], abs=1e-9)
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
    for a in SDG.READINGS:  # la risposta riporta solo cosa ha superato la regola: nessun meccanismo, nessun «soprattutto»
        for b in SDG.READINGS:
            text = SDG.answer(a, b)
            assert "compensa" not in text and "soprattutto" not in text and "informazione" not in text
    assert "non esclude una differenza, neanche oltre 1 punto" in SDG.answer("entro le soglie", "peggiora")


def test_constant_branches_give_undefined_measures_not_a_failed_job(tmp_path, monkeypatch):
    """Review GPT del 20/10 (P2): rami allenati costanti -> R^2 e CKA non definiti (null con lo stato), rango 0, JSON scritto, nessun
    valore fittizio."""
    rng = np.random.default_rng(123)
    subjects = [f"s{i:02d}" for i in range(12)]
    s = np.repeat(subjects, 8)
    data = {"epn612": {"raw": rng.normal(size=(len(s), 5)), "init": rng.normal(size=(len(s), 3)), "trained": np.zeros((len(s), 3)),
                       "s": s, "subjects": subjects}}
    monkeypatch.setattr(SDG, "load", lambda *a: data)
    from types import SimpleNamespace

    args = SimpleNamespace(task="p2", arm="B_seed0", sets=[], redundancy=True, seed=0, eval_root=tmp_path / "e", downstream_root=tmp_path / "d",
                           out_dir=tmp_path / "out", splits=None)
    assert SDG.probe(args) == 0  # prima: «Out of range float values are not JSON compliant: nan» e job fallito
    doc = json.loads((args.out_dir / "p2" / "B_seed0" / "redundancy.json").read_text())
    assert doc["effective_rank"]["trained"]["epn612"]["effective_rank"] == 0
    red = doc["datasets"]["epn612"]
    assert red["r2_PW_from_R"] is None and red["r2_PW_status"] == SDG.UNDEFINED and red["cka_R_PW"] is None
    assert red["cka_status"]["cka_R_PW"] == SDG.UNDEFINED and red["r2_PW0_from_R"] is not None and red["r2_PW0_status"] == "definito"


def test_ridge_r2_recovers_known_cases():
    """Review del critico (20/10): R^2 ~ 1 per rami combinazione lineare del raw, ~ 0 per rami indipendenti."""
    rng = np.random.default_rng(0)
    s = np.repeat([f"s{i:02d}" for i in range(16)], 40)
    roles = SDG.CP.fold_roles(sorted(set(s.tolist())))
    x = rng.normal(size=(len(s), 30))
    lin = SDG.ridge_r2(x, x @ rng.normal(size=(30, 8)) + 0.01 * rng.normal(size=(len(s), 8)), s, roles)
    ind = SDG.ridge_r2(x, rng.normal(size=(len(s), 8)), s, roles)
    assert lin["r2"] > 0.99 and abs(ind["r2"]) < 0.05 and len(lin["alphas"]) == 4


def _fake_reports(out, task="p2"):
    """Rapporti minimi per `compare` (il confronto statistico e' sostituito nel test che li usa)."""
    for arm in SDG.CHECKPOINTS:
        for name in SDG.SETS:
            d = out / task / arm
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.json").write_text(json.dumps({"protocol": SDG.PROTOCOL, "task": task, "arm": arm, "set": name, "S": 0.5,
                                                        "S_se": 0.01, "datasets": {"epn612": {"s": 0.5, "features": {"effective_rank": 9.0}}}}))


CASES = {"peggiora": (-0.05, [-0.06, -0.04]), "entro le soglie": (-0.005, [-0.025, 0.015]), "migliora": (0.05, [0.04, 0.06]),
         # seconda review del critico: stima oltre 1 punto ma IC che tocca lo zero -> regola non superata
         "entro, oltre 1 punto": (-0.015, [-0.03, 0.001])}


@pytest.mark.parametrize("alone,concat", [(a, b) for a in CASES for b in CASES])
def test_compare_maps_every_combination_of_readings(tmp_path, monkeypatch, alone, concat):
    """Review del critico (20/10): tutte le combinazioni passano per `compare`, con le letture dei due confronti chiave."""

    def fake_s_comparison(ref, arm):
        key = (ref[0]["set"], arm[0]["set"])
        d, ci = CASES[alone] if key == ("PW0", "PW") else CASES[concat] if key == ("RPW0", "RPW") else (0.0, [-0.01, 0.01])
        return {"A": 0.5, "B": 0.5 + d, "delta": d, "ci95": ci, "delta_by_model_seed": [d, d], "beats": d >= 0.01 and ci[0] > 0}

    real = SDG._script

    def fake_script(name):
        if name == "arms_compare":
            return type("AC", (), {"s_comparison": staticmethod(fake_s_comparison)})
        return real(name)

    monkeypatch.setattr(SDG, "_script", fake_script)
    monkeypatch.setattr(SDG, "difference_of_deltas", lambda reps, arms: {"delta": 0.0, "ci95": [0.0, 0.0]})
    out = tmp_path / "out"
    _fake_reports(out)
    assert SDG.main(["compare", "--out-dir", str(out), "--out", str(tmp_path / "e.json"), "--tasks", "p2"]) == 0
    res = json.loads((tmp_path / "e.json").read_text())["families"]["p2"]
    expected = {k: ("entro le soglie" if k.startswith("entro") else k) for k in CASES}
    for fam in ("A", "B"):
        c = res[fam]["contrasts"]
        assert (c["rami_da_soli"]["reading"], c["concatenazione"]["reading"]) == (expected[alone], expected[concat])
        assert res[fam]["answer"] == SDG.answer(expected[alone], expected[concat])
        if alone == "entro, oltre 1 punto":
            assert "neanche oltre 1 punto" in res[fam]["answer"] and c["rami_da_soli"]["delta"] == -0.015


def test_difference_of_deltas_on_a_known_case(tmp_path):
    """Soli rami invariati (PW = PW0), concatenazione che perde: differenza dei delta = delta della concatenazione, IC sotto zero."""
    rng = np.random.default_rng(1)
    s = np.repeat([f"u{i:02d}" for i in range(30)], 20)
    y = np.tile(np.arange(4), len(s) // 4)
    rows = np.arange(len(y), dtype=np.int64)

    def noisy(acc):
        return np.where(rng.random(len(y)) < acc, y, (y + 1) % 4)

    reps = {}
    for arm in ("B_seed0", "B_seed1"):
        same = noisy(0.6)
        preds = {"PW0": same, "PW": same.copy(), "RPW0": noisy(0.9), "RPW": noisy(0.6)}
        reps[arm] = {}
        for name, pred in preds.items():
            path = tmp_path / f"{arm}_{name}.oof.npz"
            np.savez(path, datasets=np.asarray(["epn612"]), epn612_y=y, epn612_pred=pred, epn612_s=s, epn612_row=rows)
            reps[arm][name] = {"protocol": SDG.PROTOCOL, "splits_sha256": "x", "k_folds": 4, "fold_seed": 1, "val_frac": 0.1, "probe_seed": 0,
                               "feature_set": SDG.FEATURE_SET, "window_s": None, "extraction_recipe": {}, "weights": {"epn612": 1.0},
                               "oof": str(path), "oof_sha256": sha256_file(path),
                               "datasets": {"epn612": {"input_sha256": "i" * 64, "oof_rows_sha256": fingerprint(rows)}}}
    dd = SDG.difference_of_deltas(reps, ("B_seed0", "B_seed1"))
    assert dd["delta"] < -0.2 and dd["ci95"][1] < 0 and all(v < 0 for v in dd["delta_by_model_seed"])
    reps["B_seed1"]["RPW"] = {**reps["B_seed1"]["RPW"], "fold_seed": 2}
    with pytest.raises(ValueError, match="provenienza non accoppiata"):
        SDG.difference_of_deltas(reps, ("B_seed0", "B_seed1"))


def test_branch_caches_must_come_from_one_complete_2c_run(tmp_path):
    """Review del critico (20/10): passo 0 e allenati dallo stesso run del 2C, completo, non un collaudo."""
    for field, value, match in (("collaudo", True, "provenienza diversa"), ("complete", False, "provenienza diversa"),
                                ("checkpoint_sha256", "f" * 64, "stesso run")):
        d = tmp_path / field
        ev, dn = build(d, "p2", arms=("B_seed0",))
        for which in ("init", "trained"):
            p = ev / "p2" / "B_seed0" / f"branches_{which}" / "provenance.json"
            meta = json.loads(p.read_text())
            meta["run"] = {**SDG.STEP2C, "arm": "B_seed0", "collaudo": False, "complete": True, "encoder_identical": True,
                           "source_checkpoint_sha256": SDG.CHECKPOINTS["B_seed0"], "checkpoint_sha256": "a" * 64,
                           "init_checkpoint_sha256": "b" * 64}
            if which == "trained":
                meta["run"][field] = value
            p.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=match):
            _probe(ev, dn, d / "out", "p2", "B_seed0", ["PW0"])


def test_plan_feeds_the_evaluation_chain_in_tranches(tmp_path):
    """Review del critico (20/10): 16 sonde e il confronto via evaluation_chain.py, entro il limite dei job seriali dell'utente (finestra 1
    compresa), confronto dopo tutte le sonde, --nice positivo; uscite fuori dal codice."""
    out = tmp_path / "piano.json"
    assert SDG.main(["plan", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/leonardo_scratch/x/rami", "--out", str(out),
                     "--max-serial", "3", "--campaign", "rami-t"]) == 0
    default = tmp_path / "piano2.json"
    assert SDG.main(["plan", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/leonardo_scratch/x/rami", "--out", str(default),
                     "--campaign", "rami-t"]) == 0
    assert json.loads(default.read_text())["limits"] == {"lrd_all_serial": 2}  # uno in esecuzione e uno pronto
    plan = json.loads(out.read_text())
    EC = SDG._script("evaluation_chain")
    nodes = EC.validate(plan)
    assert len(nodes) == 17 and nodes[-1]["id"] == "RAMI-rami-t-compare" and len(nodes[-1]["after"]) == 16
    assert [n["id"] for n in nodes[:4]] == [f"RAMI-rami-t-B_seed0-{j}" for j in ("p1-a", "p1-b", "p1-c", "p2")]
    assert all(f"--nice={SDG.NICE}" in n["sbatch"] for n in nodes) and plan["limits"] == {"lrd_all_serial": 3}

    class Ledger:
        def __init__(self, states):
            self.states = states

        def read(self, i):
            return {"stato": self.states[i]} if i in self.states else None

    rows = EC.schedule(plan, Ledger({}).read, {"lrd_all_serial": 1})  # un job seriale della finestra 1 gia' in coda: ne entrano 2
    assert [r["id"] for r in rows if r["status"] == "sottomesso ora"] == ["RAMI-rami-t-B_seed0-p1-a", "RAMI-rami-t-B_seed0-p1-b"]
    assert next(r for r in rows if r["id"] == "RAMI-rami-t-compare")["status"] == "in attesa"
    with pytest.raises(ValueError, match="assoluta"):
        SDG.main(["plan", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "rel", "--out", str(out), "--campaign", "rami-t"])
    with pytest.raises(SystemExit):  # senza campagna non si pianifica (review GPT del 20/10: gli ID del registro sono per sempre)
        SDG.main(["plan", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/x", "--out", str(out)])
    # campagne diverse = ID diversi: lo stato sottomesso di una non fa saltare l'altra
    other = tmp_path / "piano3.json"
    SDG.main(["plan", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/leonardo_scratch/y/rami", "--out", str(other), "--campaign", "rami-u"])
    first = {n["id"] for n in json.loads(default.read_text())["nodes"]}
    rows = EC.schedule(json.loads(other.read_text()), Ledger({i: "sottomesso" for i in first}).read, {"lrd_all_serial": 0})
    assert not any(r["status"] == "gia' sottomesso" for r in rows) and sum(r["status"] == "sottomesso ora" for r in rows) == 2


def test_per_set_plan_skips_done_sets_and_keeps_redundancy_once(tmp_path):
    """Piano v2 (21/10): un job per insieme coi tempi misurati, insiemi gia' fatti saltati, ridondanza col primo job rimasto del compito,
    famiglie intere, confronto che dipende anche dai job di un'altra campagna e legge solo le famiglie pianificate."""
    out = tmp_path / "v2.json"
    assert SDG.main(["plan", "--per-set", "--arms", "B_seed0", "B_seed1", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/x/rami",
                     "--out", str(out), "--campaign", "v2", "--done", "p1/B_seed0/R", "p1/B_seed0/PW0", "p1/B_seed0/RPW0",
                     "--external", "RAMI-v1-B_seed0-p1-b"]) == 0
    plan = json.loads(out.read_text())
    probes = [n for n in plan["nodes"] if n["kind"] == "probe"]
    ids = [n["id"] for n in probes]
    assert "RAMI-v2-B_seed0-p1-R" not in ids and "RAMI-v2-B_seed0-p1-PW" in ids and len(ids) == 2 * 10 - 3
    red = {(n["args"][3], n["args"][5]): n["args"][7] for n in probes if "--redundancy" in n["args"]}
    assert red == {("p1", "B_seed0"): "PW", ("p2", "B_seed0"): "RPW0", ("p1", "B_seed1"): "RPW0", ("p2", "B_seed1"): "RPW0"}
    assert all(n["sbatch"][0] == f"--time={SDG.SET_TIME[n['args'][3]][n['args'][7]]}" for n in probes)
    compare = plan["nodes"][-1]
    assert compare["args"][1:] == ["compare", "--families", "B"] and "RAMI-v1-B_seed0-p1-b" in compare["after"]
    SDG._script("evaluation_chain").validate(plan)
    with pytest.raises(ValueError, match="famiglie intere"):
        SDG.main(["plan", "--per-set", "--arms", "B_seed0", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/x/rami",
                  "--out", str(out), "--campaign", "v2"])
    with pytest.raises(ValueError, match="nessun job del compito rimasto"):
        SDG.main(["plan", "--per-set", "--arms", "B_seed0", "B_seed1", "--code-dir", "/C", "--expected-sha", "a" * 40, "--out-dir", "/x/rami",
                  "--out", str(out), "--campaign", "v2", "--done", *[f"p1/B_seed0/{s}" for s in SDG.SETS]])
