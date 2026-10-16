"""Selezione della calibrazione C1-C4 (`scripts/calibration_probe.py`, `scripts/calibration_select.py`, protocollo
`docs/calibrazione_c1_c4_selezione.md`): fold di soli soggetti di pretraining, sonda fuori fold su feature sintetiche, ammissibilita' (STOP,
run incompleto, valori non finiti, allarmi e diagnostiche finali di collasso, pannello fisso), regole di C1-C4 con confronto accoppiato e
parita' al LR piu' basso, unione delle decisioni nella sequenza del pianificatore, launcher."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico

REPO = Path(__file__).resolve().parents[2]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CP, CS = _mod("calibration_probe"), _mod("calibration_select")
DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6")
N_PRE = {"ninapro_db2": 12, "ninapro_db3": 8, "ninapro_db6": 8}
N_TEST = {"ninapro_db2": 4, "ninapro_db3": 3, "ninapro_db6": 2}


def _splits(tmp_path):
    d = {"datasets": {ds: {"pretraining": [f"S{i:02d}" for i in range(N_PRE[ds])],
                           "test": [f"T{i:02d}" for i in range(N_TEST[ds])]} for ds in DATASETS}}
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(d))
    return p, d


def _features(tmp_path, splits, signal=2.0, seed=0, name="feat", ckpt="/R/run/checkpoint_postdecay.pt"):
    """Feature sintetiche dei soggetti di pretraining: 4 classi separabili con intensita' `signal`, 30 finestre per soggetto."""
    rng = np.random.default_rng(seed)
    out = tmp_path / name
    out.mkdir()
    for ds in DATASETS:
        subs = splits["datasets"][ds]["pretraining"]
        y = np.tile(np.arange(1, 5), 30 * len(subs) // 4)
        s = np.repeat(subs, 30)
        centers = rng.normal(size=(5, 16))
        x = (signal * centers[y] + rng.normal(size=(len(y), 16))).astype(np.float32)
        np.savez(out / f"{ds}.npz", x=x, y=y, s=s, subjects=np.asarray(subs), input_sha256=np.asarray("i" * 64),
                 checkpoint=np.asarray(ckpt), checkpoint_sha256=np.asarray("c" * 64), encoder_dim=np.asarray(384),
                 splits_sha256=np.asarray(CP._sha256(tmp_path / "splits.json")),
                 extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE, sort_keys=True)),
                 protocol=np.asarray(CP.PROTOCOL), **_cache(16))
    return out


def _cache(width):
    from wearusfm.harness.feature_sets import cache_record

    return cache_record("model", width)


def test_folds_cover_each_pretraining_subject_once_and_never_test():
    subs = [f"S{i:02d}" for i in range(8)]
    roles = CP.fold_roles(subs)
    assert roles == CP.fold_roles(subs)  # deterministici
    held = [s for r in roles for s in r["held"]]
    assert sorted(held) == subs and len(roles) == CP.K_FOLDS
    for r in roles:
        assert not (set(r["train"]) & set(r["val"]) or set(r["train"]) & set(r["held"]) or set(r["val"]) & set(r["held"]))
        assert len(r["val"]) >= 1 and set(r["train"]) | set(r["val"]) | set(r["held"]) == set(subs)
    with pytest.raises(ValueError, match="sia di pretraining sia di test"):
        CP.pretraining_subjects({"datasets": {"d": {"pretraining": ["a", "b"], "test": ["b"]}}}, "d")
    with pytest.raises(ValueError, match="soggetti per"):
        CP.fold_roles(["a", "b", "c"])


def test_probe_stage_scores_out_of_fold_on_pretraining_subjects_only(tmp_path):
    sp, splits = _splits(tmp_path)
    feat = _features(tmp_path, splits)
    out = tmp_path / "rep" / "C1-a.json"
    assert CP.main(["--stage", "probe", "--features", str(feat), "--splits", str(sp), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["protocol"] == CP.PROTOCOL and 0.8 < rep["S"] <= 1.0 and rep["S_se"] >= 0
    w = rep["weights"]
    assert abs(w["ninapro_db2"] - 4 / 9) < 1e-12 and abs(w["ninapro_db6"] - 2 / 9) < 1e-12  # pesi della P1: soggetti di test, solo i conteggi
    with np.load(rep["oof"]) as z:
        for ds in DATASETS:
            assert sorted(set(z[f"{ds}_s"].tolist())) == splits["datasets"][ds]["pretraining"]  # ognuno predetto, nessun soggetto di test
            assert len(z[f"{ds}_y"]) == 30 * N_PRE[ds]
    assert all(len(rep["datasets"][d]["folds"]) == CP.K_FOLDS for d in DATASETS)
    bad = json.loads(sp.read_text())
    bad["datasets"]["ninapro_db3"]["pretraining"] = bad["datasets"]["ninapro_db3"]["pretraining"][:-1]  # feature di altri soggetti
    sp.write_text(json.dumps(bad))
    # Il nuovo gate degli split intercetta il cambio gia' sul primo dataset.
    with pytest.raises(ValueError, match="SHA-256 degli split|altri soggetti"):
        CP.main(["--stage", "probe", "--features", str(feat), "--splits", str(sp), "--out", str(tmp_path / "x.json")])


def test_extract_reads_only_pretraining_subjects(tmp_path, monkeypatch):
    sp, splits = _splits(tmp_path)
    seen = []

    class PP:
        @staticmethod
        def dataset_features(model, roots, ds, roles, args, log):
            seen.append((ds, roles))
            subs = roles["pretraining"]
            args.input_sha256 = "f" * 64
            return {"pretraining": (np.zeros((len(subs), 4), np.float32), np.ones(len(subs), int), np.asarray(subs))}

    class Model:
        class cfg:
            dim = 384

    from wearusfm.harness import fm_features as FF

    monkeypatch.setattr(CP, "_probe_p1_module", lambda: PP)
    monkeypatch.setattr(FF, "load_model", lambda *a, **k: Model())
    monkeypatch.setattr(FF, "feature_width", lambda m, fs: 4)
    ckpt = tmp_path / "checkpoint_postdecay.pt"
    ckpt.write_bytes(b"pesi")
    assert CP.main(["--stage", "extract", "--checkpoint", str(ckpt), "--root", str(tmp_path), "--features", str(tmp_path / "f"),
                    "--splits", str(sp), "--device", "cpu"]) == 0
    assert [d for d, _ in seen] == list(DATASETS)
    assert all(r == {"pretraining": splits["datasets"][d]["pretraining"]} for d, r in seen)
    with np.load(tmp_path / "f" / "ninapro_db6.npz") as z:
        assert str(z["checkpoint_sha256"]) == CP._sha256(ckpt) and int(z["encoder_dim"]) == 384 and str(z["protocol"]) == CP.PROTOCOL
        assert int(z["model_dim"]) == 4 and str(z["feature_set"]) == "model"  # identita' della cache, come la P1


# ---- ammissibilita' e selezione ----

STEPS = 2000


def _run(tmp_path, name, *, stop=False, stopped="max_steps", steps=STEPS, nan_at=None, alarm=False, erank_last=100.0, panel=None):
    d = tmp_path / name
    d.mkdir()
    (d / CS.CHECKPOINT).write_bytes(b"synthetic checkpoint")
    (d / "summary.json").write_text(json.dumps({"stopped": stopped, "steps": steps, "elapsed_s": 1.0, "code": {"sha": "x"}}))
    if stop:
        (d / "STOP").write_text("allarme")
    lines = []
    for t in range(1, STEPS + 1):
        r = {"step": t, "total": 1.0 / t ** 0.1, "grad_norm": 0.5}
        if t == nan_at:
            r["total"] = float("nan")
        if t % 500 == 0 or t == STEPS:
            r.update({"student_collapse_ratio": 0.5, "teacher_collapse_ratio": 0.4, "student_erank": erank_last if t == STEPS else 100.0,
                      "teacher_erank": 100.0})
            if alarm and t == 1000:
                r["alarm"] = "collasso da query: prova"
        lines.append(json.dumps(r))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    panel = panel if panel is not None else [1.0, 0.8, 0.7, 0.65, 0.6]
    (d / "fixed_eval.jsonl").write_text("".join(json.dumps({"step": 500 * i, "total": v, "jepa": 0.01}) + "\n" for i, v in enumerate(panel)))
    return d


def _report(tmp_path, name, run_dir, quality, seed=0):
    """Rapporto di sonda con predizioni fuori fold: frazione `quality` di predizioni giuste per soggetto (stesse finestre per tutti)."""
    rng = np.random.default_rng(seed)
    from wearusfm.harness.downstream_ab import fingerprint
    oof, per = {}, {}
    for ds in DATASETS:
        subs = [f"S{i:02d}" for i in range(N_PRE[ds])]
        y = np.tile(np.arange(1, 5), 10 * len(subs))
        s = np.repeat(subs, 40)
        pred = np.where(rng.random(len(y)) < quality, y, (y % 4) + 1)
        rows = np.arange(len(y), dtype=np.int64)
        oof.update({f"{ds}_y": y, f"{ds}_pred": pred, f"{ds}_s": s, f"{ds}_row": rows})
        per[ds] = {"s": float(np.mean(pred == y)), "input_sha256": "a" * 64, "oof_rows_sha256": fingerprint(rows)}
    path = tmp_path / f"{name}.oof.npz"
    np.savez(path, datasets=np.asarray(DATASETS), **oof)
    w = {"ninapro_db2": 4 / 9, "ninapro_db3": 3 / 9, "ninapro_db6": 2 / 9}
    rep = {"protocol": CS.PROTOCOL, "checkpoint": str(run_dir / CS.CHECKPOINT), "checkpoint_sha256": CS._sha256(run_dir / CS.CHECKPOINT), "model_dim": 384,
           "splits_sha256": "d" * 64, "k_folds": CP.K_FOLDS, "fold_seed": CP.FOLD_SEED, "val_frac": CP.VAL_FRAC,
           "probe_seed": 0, "feature_set": CP.FEATURE_SET, "window_s": CP.WINDOW_S, "extraction_recipe": CP.EXTRACTION_RECIPE,
           "weights": w, "S": sum(w[d] * per[d]["s"] for d in DATASETS), "S_se": 0.01, "datasets": per, "oof": str(path),
           "oof_sha256": CS._sha256(path)}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(rep))
    return p


def _cand(tmp_path, lr, quality, seed=0, **run_kw):
    name = f"lr{lr:g}".replace(".", "p")
    rd = _run(tmp_path, name, **run_kw)
    return (lr, rd, _report(tmp_path, name, rd, quality, seed), STEPS)


@pytest.mark.parametrize("kw,problem", [
    ({"stop": True}, "STOP"),
    ({"stopped": "time_limit"}, "non completo"),
    ({"steps": STEPS - 1}, "non completo"),
    ({"nan_at": 7}, "non finiti"),
    ({"alarm": True}, "allarme di collasso"),
    ({"erank_last": 30.0}, "rango effettivo finale"),
    ({"panel": [1.0, 0.8, 0.95, 0.95, 0.95]}, "valutazioni consecutive"),
    ({"panel": [1.0, 0.95, 0.93, 0.92, 0.91]}, "apprendimento"),
    ({"panel": [1.0, 0.8, 0.7, 0.6]}, "pannello: valutazioni"),
    ({"panel": [1.0, 0.8, float("nan"), 0.6, 0.5]}, "non finiti"),
])
def test_inadmissible_runs(tmp_path, kw, problem):
    lr, rd, rp, steps = _cand(tmp_path, 3e-4, 0.8, **kw)
    adm = CS.admissibility(rd, json.loads(rp.read_text()), steps)
    assert not adm["admissible"] and any(problem in p for p in adm["problems"]), adm["problems"]


def test_admissible_run_and_tolerated_cases(tmp_path):
    lr, rd, rp, steps = _cand(tmp_path, 3e-4, 0.8, panel=[1.0, 0.7, 0.8, 0.8, 0.6])  # due valutazioni sopra soglia: registrate, non allarme
    adm = CS.admissibility(rd, json.loads(rp.read_text()), steps)
    assert adm["admissible"], adm["problems"]
    assert adm["panel"]["longest_run_above"] == 2 and adm["non_collapse"]["erank_min"] == pytest.approx(38.4)
    rep = json.loads(rp.read_text())
    rep["checkpoint"] = "/altro/checkpoint_postdecay.pt"
    assert any("la sonda viene da" in p for p in CS.admissibility(rd, rep, steps)["problems"])


def test_checkpoint_path_of_the_probe_is_compared_robustly(tmp_path):
    """Lo stesso checkpoint scritto in modi diversi (barre doppie o finali, `.`, link simbolico) non rende il run inammissibile."""
    lr, rd, rp, steps = _cand(tmp_path, 3e-4, 0.8)
    (rd / "checkpoint_postdecay.pt").write_bytes(b"ckpt")
    rep = json.loads(rp.read_text())
    rep["checkpoint_sha256"] = CS._sha256(rd / CS.CHECKPOINT)
    link = tmp_path / "collegamento"
    link.symlink_to(rd, target_is_directory=True)
    for ckpt in (f"{rd}//checkpoint_postdecay.pt", f"{rd}/./checkpoint_postdecay.pt", str(link / "checkpoint_postdecay.pt")):
        rep["checkpoint"] = ckpt
        assert CS.admissibility(rd, rep, steps)["admissible"], ckpt
    assert CS.admissibility(Path(f"{rd}/"), {**rep, "checkpoint": str(rd / "checkpoint_postdecay.pt")}, steps)["admissible"]
    for other in (str(tmp_path / "altro" / "checkpoint_postdecay.pt"), str(rd / "checkpoint_predecay.pt"), None):
        assert any("la sonda viene da" in p for p in CS.admissibility(rd, {**rep, "checkpoint": other}, steps)["problems"]), other
    assert CS.same_file(rd / "assente.pt", f"{rd}//assente.pt") and not CS.same_file(rd / "assente.pt", rd / "altro.pt")


def test_c1_reference_plus_best_alternative_with_ties_to_the_lower_lr(tmp_path):
    a, b, c = (tmp_path / x for x in "abc")
    for x in (a, b, c):
        x.mkdir()
    dec, rec = CS.select("C1", [_cand(a, 1e-4, 0.55), _cand(a, 3e-4, 0.7), _cand(a, 1e-3, 0.85)])
    assert dec == {"C1": {"finalists": [3e-4, 1e-3]}} and rec["comparisons"][0]["winner"] == 1e-3
    dec, rec = CS.select("C1", [_cand(b, 1e-4, 0.70, seed=1), _cand(b, 3e-4, 0.7), _cand(b, 1e-3, 0.70, seed=2)])
    assert dec == {"C1": {"finalists": [1e-4, 3e-4]}} and rec["comparisons"][0]["tie"]  # parita' entro 1 SE: il LR piu' basso
    # mai la training loss o un numero di test: la sola S fuori fold (qui 1e-4 e' migliore anche se la loss di 1e-3 fosse piu' bassa)
    dec, _ = CS.select("C1", [_cand(c, 1e-4, 0.9), _cand(c, 3e-4, 0.7), _cand(c, 1e-3, 0.5)])
    assert dec == {"C1": {"finalists": [1e-4, 3e-4]}}


@pytest.mark.parametrize("bad,expected", [
    ({3e-4: {"stop": True}}, {"finalists": [1e-4, 1e-3]}),
    ({3e-4: {"stop": True}, 1e-3: {"nan_at": 3}}, "stop"),
    ({1e-4: {"stop": True}, 1e-3: {"stop": True}}, "stop"),
    ({1e-3: {"alarm": True}}, {"finalists": [1e-4, 3e-4]}),
])
def test_c1_admissibility_rules(tmp_path, bad, expected):
    dec, rec = CS.select("C1", [_cand(tmp_path, lr, 0.8, seed=i, **bad.get(lr, {})) for i, lr in enumerate((1e-4, 3e-4, 1e-3))])
    if expected == "stop":
        assert "stop" in dec["C1"] and "meno di due" in dec["C1"]["stop"]
    else:
        assert dec["C1"] == expected


def test_c2_c3_c4_rules(tmp_path):
    dirs = [tmp_path / str(i) for i in range(5)]
    for d in dirs:
        d.mkdir()
    assert CS.select("C2", [_cand(dirs[0], 3e-4, 0.8), _cand(dirs[0], 1e-3, 0.8, seed=1)])[0] == {"C2": {"confirmed": [3e-4, 1e-3]}}
    assert CS.select("C2", [_cand(dirs[1], 3e-4, 0.8), _cand(dirs[1], 1e-3, 0.8, seed=1, stop=True)])[0] == {"C2": {"confirmed": [3e-4]}}
    assert CS.select("C3", [_cand(dirs[2], 3e-4, 0.6), _cand(dirs[2], 1e-3, 0.85, seed=1)])[0] == {"C3": {"proposed_300m_lr": 1e-3}}
    # accordo del 15/10: meno di due candidati ammissibili ferma il passaggio (anche con un solo LR ammissibile al 100M)
    dec = CS.select("C3", [_cand(dirs[3], 3e-4, 0.6), _cand(dirs[3], 1e-3, 0.85, seed=1, stop=True)])[0]
    assert set(dec["C3"]) == {"stop"} and "meno di due" in dec["C3"]["stop"]
    dirs.append(tmp_path / "5")
    dirs[-1].mkdir()
    dec = CS.select("C3", [_cand(dirs[5], 3e-4, 0.6, stop=True), _cand(dirs[5], 1e-3, 0.85, seed=1, stop=True)])[0]
    assert set(dec["C3"]) == {"stop"}
    assert CS.select("C4", [_cand(dirs[4], 3e-4, 0.6)])[0] == {"C4": {"lr": 3e-4, "passed": True}}
    with pytest.raises(ValueError, match="attesi"):
        CS.select("C1", [_cand(tmp_path, 1e-4, 0.6)])


def test_paired_comparison_refuses_different_windows(tmp_path):
    a = {"d": (np.array([1, 2, 1, 2]), np.array([1, 2, 2, 2]), np.array(["s1", "s1", "s2", "s2"]))}
    b = {"d": (np.array([1, 2, 2, 2]), np.array([1, 2, 2, 2]), np.array(["s1", "s1", "s2", "s2"]))}
    with pytest.raises(ValueError, match="non e' accoppiato"):
        CS.paired_difference(a, b, {"d": 1.0})


def test_merge_follows_the_planner_sequence(tmp_path):
    d = CS.merge({}, {"C1": {"finalists": [3e-4, 1e-3]}})
    with pytest.raises(ValueError, match="gia' decisa"):
        CS.merge(d, {"C1": {"finalists": [1e-4, 3e-4]}})
    with pytest.raises(ValueError, match="mancano le decisioni precedenti"):
        CS.merge(d, {"C3": {"proposed_300m_lr": 3e-4}})
    with pytest.raises(ValueError, match="devono essere finalisti"):
        CS.merge(d, {"C2": {"confirmed": [1e-4]}})
    d = CS.merge(d, {"C2": {"confirmed": [3e-4, 1e-3]}})
    with pytest.raises(ValueError, match="uno dei due LR di C3"):
        CS.merge(d, {"C3": {"proposed_300m_lr": 1e-4}})
    d = CS.merge(d, {"C3": {"proposed_300m_lr": 1e-3}})
    with pytest.raises(ValueError, match="esito di C4"):
        CS.merge(d, {"C4": {"lr": 3e-4, "passed": True}})
    d = CS.merge(d, {"C4": {"lr": 1e-3, "passed": False}})
    p, a = tmp_path / "dec.json", tmp_path / "add.json"
    a.write_text(json.dumps({"C1": {"stop": "prova"}}))
    assert CS.main(["merge", "--decisions", str(p), "--add", str(a)]) == 0 and json.loads(p.read_text()) == {"C1": {"stop": "prova"}}


def test_select_launcher_groups_candidates_and_uses_the_venv(work, tmp_path):  # noqa: F811
    root = work[0]
    venv = root / "venv_wearusfm" / "bin"
    venv.mkdir(parents=True)
    fakebin = root / "venvpy"
    fakebin.mkdir()
    (fakebin / "python3").write_text("#!/bin/sh\necho VENV_PYTHON \"$@\"\n")
    (fakebin / "python3").chmod(0o755)
    (venv / "activate").write_text(f"export PATH={fakebin}:$PATH\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    launcher = REPO / "scripts" / "slurm" / "calibration_select.sbatch"
    args = ["C1", f"{root}/out/d.json", f"{root}/out/r.json", "0.0001", "/R/a", "/E/a.json", "24523", "0.0003", "/R/b", "/E/b.json", "24523"]
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert ("VENV_PYTHON scripts/calibration_select.py select --phase C1 --out " + f"{root}/out/d.json --record {root}/out/r.json "
            "--candidate 0.0001 /R/a /E/a.json 24523 --candidate 0.0003 /R/b /E/b.json 24523") in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), *args[:-1]], env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert r.returncode == 2 and "gruppi di 4" in r.stdout
    r = subprocess.run(["bash", str(launcher), "C1", "rel/d.json", f"{root}/out/r.json", *args[3:7]], env={**env, "WORK": str(root)},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "relativa" in r.stdout
    for name in ("calibration_extract.sbatch", "calibration_probe.sbatch"):
        assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "slurm" / name)]).returncode == 0
