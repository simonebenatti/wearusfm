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
        # i campi che `--stage probe` controlla (review del critico sul merge, 20/10): una regressione si vedrebbe solo dopo l'estrazione GPU
        assert str(z["splits_sha256"]) == CP._sha256(sp) and json.loads(str(z["extraction_recipe"])) == CP.EXTRACTION_RECIPE


# ---- validation interna della P2 (bracci dell'obiettivo, decisione del 16/10: il default si sceglie su S) ----

def test_p2_internal_subjects_exclude_the_test_of_all_three_official_splits():
    from wearusfm.harness.splits import split_subjects

    ids = [f"u{i:03d}" for i in range(40)]
    internal = CP.p2_internal_subjects(ids + ids)  # finestre ripetute: conta il soggetto
    test = set().union(*(split_subjects(ids, (0.7, 0.1, 0.2), seed=s).test for s in (0, 1, 2)))
    assert internal == sorted(set(ids) - test) and not set(internal) & test and len(internal) >= CP.K_FOLDS
    assert internal == CP.p2_internal_subjects(list(reversed(ids)))


def test_p2_internal_subjects_are_outside_the_tests_that_probe_p2_really_uses(monkeypatch):
    """Legame col codice della P2 ufficiale, non con una sua copia (rilievo 6 del critico, 21/10): i soggetti di test che
    `probe_p2.probe_dataset` passa davvero alla sonda, con i suoi semi, sono esattamente il complemento di `p2_internal_subjects`."""
    from types import SimpleNamespace

    from wearusfm.harness import probe as HP

    PP = _mod("probe_p2")
    seen = []

    def fake_probe(xtr, ytr, xva, yva, xte, yte, subjects_test, seed=0):
        seen.append(set(subjects_test.tolist()))
        return SimpleNamespace(test_bacc=0.5, test_bacc_se=0.01, val_bacc=0.5, train_bacc=0.5, c=1.0, n_train=len(ytr), n_val=len(yva),
                               n_test=len(yte), n_test_subjects=len(set(subjects_test.tolist())), test_pred=yte)

    monkeypatch.setattr(HP, "linear_probe", fake_probe)
    for ids in ([f"trainingJSON/user{i}" for i in range(60)], [f"{i:02d}" for i in range(1, 37)]):  # come EPN-612 e UCI-EMG
        seen.clear()
        subjects = np.array([s for s in ids for _ in range(3)])
        labels = np.array([k % 3 for k in range(len(subjects))])
        PP.probe_dataset(np.zeros((len(subjects), 2)), labels, subjects, PP.SPLIT_SEEDS)
        assert len(seen) == len(PP.SPLIT_SEEDS) == len(CP.P2_SPLIT_SEEDS)
        assert CP.p2_internal_subjects(subjects) == sorted(set(ids) - set().union(*seen))


def test_p2_extraction_never_processes_test_subjects(tmp_path, monkeypatch):
    from wearusfm.harness import benchmarks as B
    from wearusfm.harness import fm_features as FF

    ids = [f"u{i:03d}" for i in range(30)]
    seen = []

    def fake_load(*a, **k):
        s = np.repeat(ids, 3)
        return np.zeros((len(s), 10, 8), np.float32), np.tile([1, 2, 3], len(ids)), s, 200.0

    class Model:
        class cfg:
            dim = 384

    def fake_extract(model, w, fs, montage, subjects, **k):
        seen.append(sorted(set(subjects.tolist())))
        return np.zeros((len(w), 4), np.float32)

    monkeypatch.setattr(B, "load_epn612", fake_load)
    monkeypatch.setattr(B, "load_uci_emg", fake_load)
    monkeypatch.setattr(FF, "load_model", lambda *a, **k: Model())
    monkeypatch.setattr(FF, "extract_features", fake_extract)
    monkeypatch.setattr(FF, "feature_width", lambda m, fs: 4)
    ckpt = tmp_path / "checkpoint_postdecay.pt"
    ckpt.write_bytes(b"pesi")
    assert CP.main(["--task", "p2", "--stage", "extract", "--checkpoint", str(ckpt), "--epn-root", str(tmp_path), "--uci-root", str(tmp_path),
                    "--features", str(tmp_path / "f"), "--device", "cpu"]) == 0
    internal = CP.p2_internal_subjects(ids)
    assert seen == [internal, internal]  # le finestre dei soggetti di test non arrivano mai all'estrazione
    with np.load(tmp_path / "f" / "epn612.npz") as z:
        assert str(z["task"]) == "p2" and z["subjects"].tolist() == internal and z["all_subjects"].tolist() == ids
        assert str(z["splits_sha256"]) == CP.P2_SPLITS_SHA256 and json.loads(str(z["extraction_recipe"])) == CP.EXTRACTION_RECIPE_P2


def test_p2_probe_scores_internal_subjects_with_equal_dataset_weights(tmp_path):
    ids = [f"u{i:03d}" for i in range(40)]
    internal = CP.p2_internal_subjects(ids)
    rng = np.random.default_rng(0)
    feat = tmp_path / "f2"
    feat.mkdir()
    for ds in CP.P2_DATASETS:
        y = np.tile(np.arange(1, 4), 10 * len(internal))
        s = np.repeat(internal, 30)
        x = (2.0 * rng.normal(size=(4, 8))[y] + rng.normal(size=(len(y), 8))).astype(np.float32)
        np.savez(feat / f"{ds}.npz", x=x, y=y, s=s, subjects=np.asarray(internal), all_subjects=np.asarray(ids),
                 input_sha256=np.asarray("i" * 64), checkpoint=np.asarray("/R/c.pt"), checkpoint_sha256=np.asarray("c" * 64),
                 encoder_dim=np.asarray(384), protocol=np.asarray(CP.PROTOCOL), task=np.asarray("p2"),
                 splits_sha256=np.asarray(CP.P2_SPLITS_SHA256), extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE_P2, sort_keys=True)),
                 **_cache(8))
    out = tmp_path / "s_p2.json"
    assert CP.main(["--task", "p2", "--stage", "probe", "--features", str(feat), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["task"] == "p2" and rep["weights"] == {"epn612": 0.5, "uci_emg": 0.5} and 0.5 < rep["S"] <= 1
    assert rep["splits_sha256"] == CP.P2_SPLITS_SHA256 and rep["extraction_recipe"] == CP.EXTRACTION_RECIPE_P2
    for key, bad in (("splits_sha256", "0" * 64), ("extraction_recipe", json.dumps(CP.EXTRACTION_RECIPE, sort_keys=True))):
        with np.load(feat / "epn612.npz") as z:  # merge con la review di GPT (41ce569): anche la cache della P2 registra split e ricetta
            d = {k: z[k] for k in z.files}
        good = d[key]
        d[key] = np.asarray(bad)
        np.savez(feat / "epn612.npz", **d)
        with pytest.raises(ValueError, match="ripetere l'estrazione"):
            CP.main(["--task", "p2", "--stage", "probe", "--features", str(feat), "--out", str(tmp_path / "z.json")])
        d[key] = good
        np.savez(feat / "epn612.npz", **d)
    with np.load(rep["oof"]) as z:
        assert sorted(set(z["epn612_s"].tolist())) == internal
    with pytest.raises(ValueError, match="altri soggetti"):  # un soggetto di test dei tre split dentro le feature: rifiutato
        test_subject = sorted(set(ids) - set(internal))[0]
        with np.load(feat / "uci_emg.npz") as z:
            d = {k: z[k] for k in z.files}
        d["subjects"] = np.asarray(sorted(internal + [test_subject]))
        np.savez(feat / "uci_emg.npz", **d)
        CP.main(["--task", "p2", "--stage", "probe", "--features", str(feat), "--out", str(tmp_path / "x.json")])
    with pytest.raises(ValueError, match="altro compito"):  # feature della P1 date alla P2
        sp, splits = _splits(tmp_path)
        p1 = _features(tmp_path, splits, name="f1")
        for ds in CP.P2_DATASETS:
            (p1 / "ninapro_db2.npz").rename(p1 / f"{ds}.npz") if ds == "epn612" else (p1 / "ninapro_db3.npz").rename(p1 / f"{ds}.npz")
        CP.main(["--task", "p2", "--stage", "probe", "--features", str(p1), "--out", str(tmp_path / "y.json")])


# ---- ammissibilita' e selezione ----

STEPS = 2000
STEPS_AT_D = 2000  # warmup = max(1000, 2%) = 1000, decay da 1600
CODE_SHA = "e" * 40


_TMP = Path(__import__("tempfile").mkdtemp())
(_TMP / "ws.json").write_text(json.dumps({"window_s": {}, "scales": {}}))


def _full_plan(runs, seed=0):
    """Piano di training completo (formato di wsd_chain.py) coi run [(id, lr)], come quelli di calibration_plan.py."""
    return {"campaign": "w1cal-test", "agent": "claude", "code_dir": "/C", "expected_sha": CODE_SHA, "manifest": "/M/m_nested_1.json.gz",
            "scales": str(_TMP / "ws.json"), "window_seconds": str(_TMP / "ws.json"), "run_root": "/R",
            "slurm": {"qos": "normal", "time": "24:00:00"}, "margin_min": 40, "startup_min": 15, "slack": 1.15, "spare_jobs": 0, "gpus": 4,
            "s_per_step": {"sanity/32": 0.7}, "ckpt_gb": {"sanity": 0.45},
            "common_args": "--window1-rules --datasets all --prefetch --fixed-eval-every 500 --manifest-sha256 " + "6" * 64,
            "runs": [{"id": name, "preset": "sanity", "seed": seed, "batch_size": 32, "accum_steps": 1, "steps_at_d": STEPS_AT_D,
                      "total_steps": STEPS, "keep_checkpoints": [], "args": f"--lr {lr:g} --nested 1"} for name, lr in runs]}


@__import__("functools").lru_cache(maxsize=None)
def _true_config_json(lr, seed=0):
    """`config.json` come lo scrive il trainer: `train_jepa.main` con gli argomenti di `train_ddp.sbatch` per un run del piano (COMMON,
    --max-steps, EXTRA_ARGS di wsd_chain), `R.train` sostituito per catturare la configurazione (ripristinato subito)."""
    import importlib.util
    import shlex

    def load(name):
        sp = importlib.util.spec_from_file_location(f"{name}_t", REPO / "scripts" / f"{name}.py")
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    W, TJ = load("wsd_chain"), load("train_jepa")
    plan = _full_plan([("x", lr)], seed)
    run = W.resolve(plan)[0]
    argv = ["--preset", "sanity", "--manifest", "m.json.gz", "--root", "r", "--scales", str(_TMP / "ws.json"), "--out-dir", str(_TMP / "out"),
            "--num-workers", "6", "--max-steps", str(STEPS), "--time-limit-s", "60"] + shlex.split(W.extra_args(plan, run))
    captured = {}
    orig = TJ.R.train, TJ.R.check_input_files
    try:
        TJ.R.train = lambda cfg, *a, **k: captured.__setitem__("cfg", cfg) or {"steps": 0}
        TJ.R.check_input_files = lambda *a, **k: None
        TJ.main(argv)
    finally:
        TJ.R.train, TJ.R.check_input_files = orig
    return json.dumps(TJ.R.config_to_dict(captured["cfg"]))


def _train_plan(cands, seed=0):
    """Il piano di training dei candidati: un run per cartella, col LR del candidato."""
    return _full_plan([(Path(c[1]).name, c[0]) for c in cands], seed)  # candidati da 4 o 5 elementi (evidenza di fallimento)


def _sel(phase, cands):
    return CS.select(phase, cands, _train_plan(cands))


def _run(tmp_path, name, *, stop=False, stopped="max_steps", steps=STEPS, nan_at=None, alarm=False, erank_last=100.0, panel=None, lr=3e-4):
    d = tmp_path / name
    d.mkdir()
    (d / CS.CHECKPOINT).write_bytes(b"synthetic checkpoint")
    (d / "summary.json").write_text(json.dumps({"stopped": stopped, "steps": steps, "elapsed_s": 1.0, "code": {"sha": CODE_SHA}}))
    (d / "config.json").write_text(_true_config_json(lr))
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


def _cand(tmp_path, lr, quality, seed=0, no_probe=False, **run_kw):
    name = f"lr{lr:g}".replace(".", "p")
    rd = _run(tmp_path, name, lr=lr, **run_kw)
    return (lr, rd, None if no_probe else _report(tmp_path, name, rd, quality, seed), STEPS)


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
    assert adm["panel"]["longest_run_above"] == 2 and adm["non_collapse"]["erank_floor"] == pytest.approx(38.4)
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
    dec, rec = _sel("C1", [_cand(a, 1e-4, 0.55), _cand(a, 3e-4, 0.7), _cand(a, 1e-3, 0.85)])
    assert dec == {"C1": {"finalists": [3e-4, 1e-3]}} and rec["comparisons"][0]["winner"] == 1e-3
    dec, rec = _sel("C1", [_cand(b, 1e-4, 0.70, seed=1), _cand(b, 3e-4, 0.7), _cand(b, 1e-3, 0.70, seed=2)])
    assert dec == {"C1": {"finalists": [1e-4, 3e-4]}} and rec["comparisons"][0]["tie"]  # parita' entro 1 SE: il LR piu' basso
    # mai la training loss o un numero di test: la sola S fuori fold (qui 1e-4 e' migliore anche se la loss di 1e-3 fosse piu' bassa)
    dec, _ = _sel("C1", [_cand(c, 1e-4, 0.9), _cand(c, 3e-4, 0.7), _cand(c, 1e-3, 0.5)])
    assert dec == {"C1": {"finalists": [1e-4, 3e-4]}}


@pytest.mark.parametrize("bad,expected", [
    ({3e-4: {"stop": True}}, {"finalists": [1e-4, 1e-3]}),
    ({3e-4: {"stop": True}, 1e-3: {"nan_at": 3}}, "stop"),
    ({1e-4: {"stop": True}, 1e-3: {"stop": True}}, "stop"),
    ({1e-3: {"alarm": True}}, {"finalists": [1e-4, 3e-4]}),
])
def test_c1_admissibility_rules(tmp_path, bad, expected):
    dec, rec = _sel("C1", [_cand(tmp_path, lr, 0.8, seed=i, **bad.get(lr, {})) for i, lr in enumerate((1e-4, 3e-4, 1e-3))])
    if expected == "stop":
        assert "stop" in dec["C1"] and "meno di due" in dec["C1"]["stop"]
    else:
        assert dec["C1"] == expected


def test_c2_c3_c4_rules(tmp_path):
    dirs = [tmp_path / str(i) for i in range(5)]
    for d in dirs:
        d.mkdir()
    assert _sel("C2", [_cand(dirs[0], 3e-4, 0.8), _cand(dirs[0], 1e-3, 0.8, seed=1)])[0] == {"C2": {"confirmed": [3e-4, 1e-3]}}
    assert _sel("C2", [_cand(dirs[1], 3e-4, 0.8), _cand(dirs[1], 1e-3, 0.8, seed=1, stop=True)])[0] == {"C2": {"confirmed": [3e-4]}}
    assert _sel("C3", [_cand(dirs[2], 3e-4, 0.6), _cand(dirs[2], 1e-3, 0.85, seed=1)])[0] == {"C3": {"proposed_300m_lr": 1e-3}}
    # accordo del 15/10: meno di due candidati ammissibili ferma il passaggio (anche con un solo LR ammissibile al 100M)
    dec = _sel("C3", [_cand(dirs[3], 3e-4, 0.6), _cand(dirs[3], 1e-3, 0.85, seed=1, stop=True)])[0]
    assert set(dec["C3"]) == {"stop"} and "meno di due" in dec["C3"]["stop"]
    dirs.append(tmp_path / "5")
    dirs[-1].mkdir()
    dec = _sel("C3", [_cand(dirs[5], 3e-4, 0.6, stop=True), _cand(dirs[5], 1e-3, 0.85, seed=1, stop=True)])[0]
    assert set(dec["C3"]) == {"stop"}
    assert _sel("C4", [_cand(dirs[4], 3e-4, 0.6)])[0] == {"C4": {"lr": 3e-4, "passed": True}}
    with pytest.raises(ValueError, match="attesi"):
        _sel("C1", [_cand(tmp_path, 1e-4, 0.6)])


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


def test_effective_configuration_must_match_the_plan(tmp_path):
    """Review di GPT (16/10): LR, seme e configurazione effettiva del run (config.json) e codice (summary.json) contro il piano, non le
    etichette: una differenza ferma la selezione."""
    cands = [_cand(tmp_path, lr, 0.6 + 0.1 * i, seed=i) for i, lr in enumerate((1e-4, 3e-4, 1e-3))]
    plan = _train_plan(cands)
    assert CS.select("C1", cands, plan)[0]["C1"]["finalists"]
    rec = CS.select("C1", cands, plan)[1]
    assert rec["candidates"]["0.0003"]["identity"]["config_checked"] is True and rec["candidates"]["0.0003"]["identity"]["fields"] > 50
    swapped = [(cands[0][0], cands[2][1], cands[0][2], STEPS), cands[1], (cands[2][0], cands[0][1], cands[2][2], STEPS)]
    with pytest.raises(ValueError, match="LR dichiarato"):  # etichette scambiate: la cartella dice un altro LR
        CS.select("C1", swapped, plan)
    rd = cands[1][1]
    good = (rd / "config.json").read_text()
    flat = CS._flatten(json.loads(good))
    for key in sorted(flat):  # OGNI campo della configurazione: alterarlo ferma la selezione (seconda review del critico)
        cfg = json.loads(good)
        node = cfg
        *path, leaf = key.split(".")
        for part in path:
            node = node[part]
        v = node[leaf]
        node[leaf] = (not v if isinstance(v, bool) else v + 1 if isinstance(v, int) else v * 2 + 1 if isinstance(v, float)
                      else v + "x" if isinstance(v, str) else list(v) + [0] if isinstance(v, list) else 0)
        (rd / "config.json").write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match=f"configurazione effettiva diversa dal piano: .*{key.replace('.', '[.]')}"):
            CS.check_identity(rd, plan, 3e-4)
    (rd / "config.json").write_text(json.dumps({**json.loads(good), "campo_nuovo": 1}))  # un campo in piu' (altro codice)
    with pytest.raises(ValueError, match="campo_nuovo"):
        CS.check_identity(rd, plan, 3e-4)
    (rd / "config.json").write_text(good)
    assert CS.check_identity(rd, plan, 3e-4)["fields"] == len(flat)
    (rd / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": STEPS, "code": {"sha": "f" * 40}}))
    with pytest.raises(ValueError, match="codice"):
        CS.select("C1", cands, plan)
    with pytest.raises(ValueError, match="serve il piano di training"):
        CS.select("C1", cands)


def test_a_run_stopped_before_the_probe_is_inadmissible_and_the_phase_goes_on(tmp_path):
    """Review di GPT (16/10): un candidato fermato prima della sonda non blocca la selezione della fase: entra inammissibile, senza S."""
    cands = [_cand(tmp_path, 1e-4, 0.6, no_probe=True, stop=True), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    dec, rec = _sel("C1", cands)
    assert dec == {"C1": {"finalists": [3e-4, 1e-3]}}
    c = rec["candidates"]["0.0001"]
    assert not c["admissibility"]["admissible"] and "S" not in c and any("STOP" in p for p in c["admissibility"]["problems"])
    assert [(d["a"], d["b"]) for d in rec["descriptive_comparisons"]] == [(3e-4, 1e-3)]  # solo fra chi ha la sonda
    gone = tmp_path / "gone"
    gone.mkdir()
    early = [(1e-4, gone / "lr0p0001", None, STEPS), _cand(gone, 3e-4, 0.7, seed=1), _cand(gone, 1e-3, 0.8, seed=2)]
    with pytest.raises(ValueError, match="senza prova di fallimento terminale"):  # R1 (GPT 3272398): cartella assente = stato ignoto
        _sel("C1", early)
    ev = _evidence(tmp_path, "lr0p0001", early, ["FAILED"])
    early[0] = (*early[0], ev)
    dec, rec = _sel("C1", early)  # fallimento provato da sacct, prima di scrivere config.json: identita' non verificabile, decide A1
    assert dec == {"C1": {"finalists": [3e-4, 1e-3]}} and rec["candidates"]["0.0001"]["identity"]["config_checked"] is False
    fe = rec["candidates"]["0.0001"]["failure_evidence"]
    assert fe["kind"] == "sacct" and fe["sha256"] == CS._sha256(ev) and fe["jobs"][0]["state"] == "FAILED"
    assert rec["candidates"]["0.0003"]["failure_evidence"] is None
    with pytest.raises(ValueError, match="LR dichiarato"):  # il LR dichiarato si controlla comunque contro il piano
        CS.select("C1", [(2e-4, gone / "lr0p0001", None, STEPS, ev)] + early[1:], _train_plan(early))
    sub = tmp_path / "ok"
    sub.mkdir()
    fine = [_cand(sub, 1e-4, 0.6, no_probe=True), _cand(sub, 3e-4, 0.7, seed=1), _cand(sub, 1e-3, 0.8, seed=2)]
    with pytest.raises(ValueError, match="senza prova di fallimento terminale"):  # un run finito bene non si salta
        _sel("C1", fine)
    fine[0] = (*fine[0], _evidence(sub, "lr0p0001", fine, ["FAILED"]))
    with pytest.raises(ValueError, match="ammissibile senza sonda"):  # anche con un'evidenza: un run che risulta sano e' un errore
        _sel("C1", fine)
    out, recf = tmp_path / "d.json", tmp_path / "r.json"
    args = ["select", "--phase", "C1", "--plan", str(tmp_path / "plan.json"), "--out", str(out), "--record", str(recf)]
    (tmp_path / "plan.json").write_text(json.dumps(_train_plan(cands)))
    for lr, rd, rp, st in cands:
        args += ["--candidate", repr(lr), str(rd), "-" if rp is None else str(rp), str(st)]
    assert CS.main(args) == 0 and json.loads(out.read_text()) == {"C1": {"finalists": [3e-4, 1e-3]}}
    assert json.loads(recf.read_text())["plan"]["sha256"] == CS._sha256(tmp_path / "plan.json")


def test_selection_accepts_only_p1_validation_reports(tmp_path):
    """Seconda review del critico (19/10): i rapporti di S della P2 hanno lo stesso protocollo, ma la calibrazione usa S della P1."""
    cands = [_cand(tmp_path, lr, 0.6 + 0.1 * i, seed=i) for i, lr in enumerate((1e-4, 3e-4, 1e-3))]
    rp = cands[1][2]
    rep = json.loads(rp.read_text())
    rp.write_text(json.dumps({**rep, "task": "p2"}))
    with pytest.raises(ValueError, match="S della P1"):
        _sel("C1", cands)
    rp.write_text(json.dumps({**rep, "weights": {"epn612": 0.5, "uci_emg": 0.5}}))
    with pytest.raises(ValueError, match="S della P1"):
        _sel("C1", cands)


def test_train_jepa_build_config_errors_and_their_order(tmp_path):
    """`build_config` senza parser solleva ValueError (la usa la selezione); in `main` l'identita' dei file viene prima degli argomenti,
    come prima dell'estrazione di build_config (seconda review del critico, 19/10)."""
    import importlib.util

    sp = importlib.util.spec_from_file_location("train_jepa_err", REPO / "scripts" / "train_jepa.py")
    TJ = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TJ)
    args = TJ.build_parser().parse_args(["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-", "--lr", "-1"])
    with pytest.raises(ValueError, match="--lr deve essere positivo"):
        TJ.build_config(args)
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    import contextlib
    import io

    err = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stderr(err):
        TJ.main(["--preset", "sanity", "--manifest", str(manifest), "--root", "-", "--out-dir", str(tmp_path / "o"), "--lr", "-1",
                 "--manifest-sha256", "0" * 64])
    assert "diverso da quello del piano" in err.getvalue() and "--lr deve essere positivo" not in err.getvalue()


def test_a_healthy_run_without_probe_and_without_checkpoint_is_an_error_not_a_verdict(tmp_path):
    """Review del critico sul merge (20/10): senza sonda, un checkpoint perso non rende inammissibile un run sano (la fase andrebbe avanti
    su un file mancante): resta l'errore operativo «run ammissibile senza sonda»."""
    cands = [_cand(tmp_path, 1e-4, 0.6, no_probe=True), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    (cands[0][1] / CS.CHECKPOINT).unlink()
    with pytest.raises(ValueError, match="senza prova di fallimento terminale"):  # R1 (GPT 3272398): prima ancora, nessuna prova
        _sel("C1", cands)
    cands[0] = (*cands[0], _evidence(tmp_path, "lr0p0001", cands, ["FAILED"]))
    with pytest.raises(ValueError, match="ammissibile senza sonda"):
        _sel("C1", cands)
    report = json.loads(cands[1][2].read_text())  # con la sonda il checkpoint mancante e' invece un problema di A1
    (cands[1][1] / CS.CHECKPOINT).unlink()
    assert "checkpoint post-decay assente o vuoto" in CS.admissibility(cands[1][1], report, STEPS)["problems"]


def test_collaudo_phase_lists_admissible_runs_and_records_descriptive_comparisons(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ca, cb = _cand(a, 3e-4, 0.8), _cand(b, 3e-4, 0.6, seed=1, stop=True)  # stesso LR: nel collaudo contano le etichette
    out, rec = tmp_path / "d.json", tmp_path / "r.json"
    args = ["select", "--phase", "collaudo", "--out", str(out), "--record", str(rec),
            "--candidate", "w1cat-T", str(ca[1]), str(ca[2]), str(STEPS), "--candidate", "w1cat-B", str(cb[1]), str(cb[2]), str(STEPS)]
    assert CS.main(args) == 0
    assert json.loads(out.read_text()) == {"collaudo": {"admissible": ["w1cat-T"]}}
    r = json.loads(rec.read_text())
    assert sorted(r["candidates"]) == ["w1cat-B", "w1cat-T"] and not r["candidates"]["w1cat-B"]["admissibility"]["admissible"]
    (d,) = r["descriptive_comparisons"]  # descrittivo anche con un candidato inammissibile: non decide nulla
    assert (d["a"], d["b"]) == ("w1cat-B", "w1cat-T") and d["delta"] < 0 and d["se"] > 0
    with pytest.raises(ValueError, match="un collaudo no"):
        CS.merge({}, {"collaudo": {"admissible": []}})


def test_real_phases_also_record_all_pairs_descriptively(tmp_path):
    _, rec = _sel("C1", [_cand(tmp_path, lr, 0.6 + 0.1 * i, seed=i) for i, lr in enumerate((1e-4, 3e-4, 1e-3))])
    assert [(d["a"], d["b"]) for d in rec["descriptive_comparisons"]] == [(1e-4, 3e-4), (1e-4, 1e-3), (3e-4, 1e-3)]
    assert sorted(rec["candidates"]) == ["0.0001", "0.0003", "0.001"]


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
    args = ["C1", f"{root}/out/d.json", f"{root}/out/r.json", "results/w1cal/calibrazione_C1.json", "0.0001", "/R/a", "/E/a.json", "24523",
            "0.0003", "/R/b", "-", "24523"]
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert ("VENV_PYTHON scripts/calibration_select.py select --phase C1 --plan results/w1cal/calibrazione_C1.json --out " +
            f"{root}/out/d.json --record {root}/out/r.json --candidate 0.0001 /R/a /E/a.json 24523 --candidate 0.0003 /R/b - 24523"
            ) in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), "collaudo", f"{root}/out/d.json", f"{root}/out/r.json", "-", *args[4:8]],
                       env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert "--plan" not in r.stdout and "--phase collaudo --out" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), *args[:-1]], env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert r.returncode == 2 and "gruppi di 4" in r.stdout
    r = subprocess.run(["bash", str(launcher), "C1", "rel/d.json", f"{root}/out/r.json", "-", *args[4:8]], env={**env, "WORK": str(root)},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "relativa" in r.stdout
    for name in ("calibration_extract.sbatch", "calibration_probe.sbatch"):
        assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "slurm" / name)]).returncode == 0


@pytest.mark.parametrize("run,lr", [("W1CAL-C1-30M-lr0p0001-s0", 1e-4), ("W1CAL-C1-30M-lr0p0003-s0", 3e-4), ("W1CAL-C1-30M-lr0p001-s0", 1e-3)])
def test_real_c1_configs_stay_verifiable_after_new_config_fields(run, lr):
    """I `config.json` veri di C1 (codice 72cf24a) contro il piano committato, col codice di oggi: i campi aggiunti dopo (pilot dei rami del
    20/10: `branch_objective`, `branch_freeze_output`) si confrontano coi loro default su tutte e due le parti, il resto resta identico."""
    plan = json.loads((REPO / "results/infra/w1cal_c1_20261016/calibrazione_C1.json").read_text())
    out = CS.check_identity(REPO / "results/infra/w1cal_c1_20261016/runs" / run, plan, lr)
    assert out["config_checked"] and out["code_sha"] == "72cf24abec5b8332011d83ce2b2fc7ae6bcc9cec"
    have = json.loads((REPO / "results/infra/w1cal_c1_20261016/runs" / run / "config.json").read_text())
    assert "branch_objective" not in have["model"]  # davvero scritti senza i campi nuovi


def _evidence(root, run_id, cands, states, logical=None, name=None, **extra):
    """Un'evidenza da sacct (`calibration_select.py failure-evidence`) per il run `run_id` del piano dei candidati."""
    ids = logical if logical is not None else CS.planned_run(_train_plan(cands), run_id)["job_ids"]
    doc = {"protocol": CS.EVIDENCE_PROTOCOL, "run_id": run_id, "source": "sacct", "read_at": "2026-10-21T00:00:00+02:00",
           "jobs": [{"logical_id": i, "job_id": str(100 + k), "state": s, "exit_code": "1:0"} for k, (i, s) in enumerate(zip(ids, states))], **extra}
    path = Path(root) / (name or f"{run_id}.failure.json")
    path.write_text(json.dumps(doc))
    return path


@pytest.mark.parametrize("states,ok", [(["FAILED"], True), (["OUT_OF_MEMORY"], True), (["CANCELLED by 123"], True), (["NODE_FAIL"], True),
                                       (["TIMEOUT"], False), (["RUNNING"], False), (["PENDING"], False), (["COMPLETED"], False),
                                       (["PREEMPTED"], False), (["REQUEUED"], False)])
def test_only_a_terminal_failure_from_sacct_excludes_a_candidate_without_probe(tmp_path, states, ok):
    """R1 della review GPT 3272398: in coda, in corso, timeout riprendibile o finito bene non sono un fallimento terminale."""
    cands = [(1e-4, tmp_path / "lr0p0001", None, STEPS), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    cands[0] = (*cands[0], _evidence(tmp_path, "lr0p0001", cands, states))
    if ok:
        assert _sel("C1", cands)[0] == {"C1": {"finalists": [3e-4, 1e-3]}}
    else:
        with pytest.raises(ValueError, match="non provano un fallimento terminale"):
            _sel("C1", cands)


def test_failure_evidence_must_be_of_that_run_and_of_its_planned_jobs(tmp_path):
    cands = [(1e-4, tmp_path / "lr0p0001", None, STEPS), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    for bad, match in ((_evidence(tmp_path, "lr0p0003", cands, ["FAILED"]), "un altro run"),
                       (_evidence(tmp_path, "lr0p0001", cands, ["FAILED"], logical=["altro-j01"], name="b.json"), "diversi da quelli del piano"),
                       (_evidence(tmp_path, "lr0p0001", cands, ["FAILED"], protocol="vecchio", name="c.json"), "altro protocollo")):
        with pytest.raises(ValueError, match=match):
            _sel("C1", [(*cands[0], bad), *cands[1:]])
    good = _evidence(tmp_path, "lr0p0001", cands, ["FAILED"])
    with pytest.raises(ValueError, match="insieme"):  # un candidato con la sonda non porta anche un'evidenza di fallimento
        _sel("C1", [(*cands[0], good), (*cands[1], good), cands[2]])
    stop = tmp_path / "lr0p0001"  # lo STOP del trainer basta da solo ed e' registrato
    stop.mkdir()
    (stop / "STOP").write_text("allarme: collasso\n")
    dec, rec = _sel("C1", cands)
    fe = rec["candidates"]["0.0001"]["failure_evidence"]
    assert dec == {"C1": {"finalists": [3e-4, 1e-3]}} and fe["kind"] == "STOP" and fe["text"] == "allarme: collasso"


def test_steps_come_from_the_frozen_plan(tmp_path):
    """R2 della review GPT 3272398: PASSI diversi dal piano sono un errore negli argomenti, non un run incompleto."""
    cands = [_cand(tmp_path, 1e-4, 0.6), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    plan = _train_plan(cands)
    assert CS.planned_run(plan, cands[0][1].name)["total_steps"] == STEPS
    for wrong in (STEPS + 1, STEPS - 1):
        with pytest.raises(ValueError, match="diverso dal piano"):
            CS.select("C1", [(*cands[0][:3], wrong), *cands[1:]], plan)
    with pytest.raises(ValueError, match="non e' un run del piano"):
        CS.planned_run(plan, "altro")


def test_failure_evidence_is_written_from_the_ledger_and_sacct(tmp_path):
    """Il sottocomando `failure-evidence`: ID SLURM dal registro del submitter, stati da sacct (cluster finto); scrive solo un fallimento
    terminale."""
    import sys as _sys

    S = CS._script("submit_job")
    cands = [_cand(tmp_path, 1e-4, 0.6), _cand(tmp_path, 3e-4, 0.7, seed=1), _cand(tmp_path, 1e-3, 0.8, seed=2)]
    plan = _train_plan(cands)
    run = CS.planned_run(plan, "lr0p0001")
    reg = tmp_path / "jobs"
    fake = tmp_path / "cluster.py"
    fake.write_text("import sys, json, pathlib\nst = json.loads(pathlib.Path(sys.argv[1]).read_text())\nr = sys.argv[2]\n"
                    "print(st['next']) if r.startswith('sbatch') else [print(f\"{j}|{s}\") for j, s in st['jobs'].items()]\n")
    state = tmp_path / "state.json"
    planp = tmp_path / "plan.json"
    planp.write_text(json.dumps(plan))
    for outcome, ok in (("FAILED|1:0", True), ("TIMEOUT|0:0", False)):
        state.write_text(json.dumps({"next": 4242, "jobs": {"4242": outcome}}))
        ssh = f"{_sys.executable} {fake} {state}"
        if not (reg / run["job_ids"][0]).exists():
            assert S.main(["--registro", str(reg), "--ssh", ssh, "--id", run["job_ids"][0], "--agente", "claude", "--campagna", "w1",
                           "--gpu-ore", "1", "--", "/C/train.sbatch"]) == 0
        out = tmp_path / f"ev_{ok}.json"
        if ok:
            assert CS.main(["failure-evidence", "--plan", str(planp), "--run-id", "lr0p0001", "--out", str(out), "--registro", str(reg),
                            "--ssh", ssh]) == 0
            doc = json.loads(out.read_text())
            assert doc["jobs"] == [{"logical_id": run["job_ids"][0], "job_id": "4242", "state": "FAILED", "exit_code": "1:0"}]
            assert CS.check_failure_evidence(out, "lr0p0001", run["job_ids"])["kind"] == "sacct"
        else:
            with pytest.raises(ValueError, match="non provano un fallimento terminale"):
                CS.main(["failure-evidence", "--plan", str(planp), "--run-id", "lr0p0001", "--out", str(out), "--registro", str(reg),
                         "--ssh", ssh])
            assert not out.exists()


def test_a3_uses_the_absolute_rank_floor_of_the_run_configuration(tmp_path):
    """Firma di Simone del 23/10/2026: A3 usa la stessa soglia dell'allarme del run. Con `alarm.erank_min` 38,4 nella configurazione un rango
    finale di 66,7 al 300M (dim 896) e' ammissibile; senza, vale la regola firmata (16), il 10% della dimensione (89,6)."""
    lr, rd, rp, steps = _cand(tmp_path, 3e-4, 0.8, erank_last=66.7)
    rep = json.loads(rp.read_text())
    rep["model_dim"] = 896
    old = CS.admissibility(rd, rep, steps)
    assert not old["admissible"] and any("sotto 10% di 896" in p for p in old["problems"])
    cfg = json.loads((rd / "config.json").read_text())
    cfg["alarm"]["erank_min"] = 38.4
    (rd / "config.json").write_text(json.dumps(cfg))
    new = CS.admissibility(rd, rep, steps)
    assert not any("rango effettivo" in p for p in new["problems"]) and new["non_collapse"]["erank_floor"] == 38.4
    assert CS.erank_floor(tmp_path / "assente", 896) == (pytest.approx(89.6), "10% di 896")
