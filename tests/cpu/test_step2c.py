"""Step 2C dello step 2 (`scripts/step2c.py`, protocollo `docs/step2c_frozen_branches.md`): training dei soli rami sopra un teacher dell'A/B
congelato (encoder bit per bit uguale, controllo al passo 0, ingressi e teacher verificati), verdetto congelato del collaudo, uscite dei soli
rami concatenate alle cache downstream con l'allineamento verificato, sonde P2 ufficiali da capo a fondo su dati sintetici, confronto accoppiato
e criteri congelati."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch  # prima di sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import step2c as S2C  # noqa: E402

from test_downstream_ab import _reports  # noqa: E402
from test_probe_p2 import _epn, _uci  # noqa: E402
from test_spectral_step1 import _calibration  # noqa: E402
from test_spectral_step1 import _setup as _step1_setup  # noqa: E402
from wearusfm.model import spectral_branches as SB  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

SD, S2, sha256_file = S2C.SD, S2C.S2, S2C.sha256_file


def _setup(tmp_path, monkeypatch, arms=("B_seed0",)):
    """Teacher «dell'A/B» minuscoli (un passo, con la testa dello Step 1, peso 0 per A e 0,05 per B) con la loro provenienza, e gli ingressi
    del training registrati come quelli dell'A/B."""
    root, manifest, cfg, _, y, m = _step1_setup(tmp_path)
    cal = tmp_path / "cal.npz"
    _calibration(cal, cfg, manifest, y, m)
    scales, window = tmp_path / "scales.json", tmp_path / "window.json"
    scales.write_text(json.dumps({"scales": {}}))
    window.write_text(json.dumps({"window_s": None}))

    def ab(arm, calibration):
        family, seed = arm.split("_seed")
        return replace(R.with_spectral_keep(cfg, calibration, 0.0 if family == "A" else 0.05), seed=int(seed))

    monkeypatch.setattr(S2C, "_ab_config", ab)
    for arm in arms:
        out = tmp_path / "runs" / arm
        R.train(ab(arm, cal), manifest, [root], out, log=lambda _: None)
        monkeypatch.setitem(SD.CHECKPOINTS, arm, sha256_file(out / "checkpoint.pt"))
        (out / "run_provenance.json").write_text(json.dumps({"source_commit": SD.TRAIN_COMMIT, "checkpoint_sha256": SD.CHECKPOINTS[arm]}))
    monkeypatch.setattr(S2C, "INPUT_SHA256", {k: sha256_file(p) for k, p in zip(S2C.INPUTS, (manifest, scales, window, cal))})
    monkeypatch.setattr(S2C, "STEPS", 3)
    monkeypatch.setattr(S2C, "WARMUP", 1)
    monkeypatch.setattr(SD, "check_source", lambda _: None)
    return SimpleNamespace(expected_commit="test", arm=arms[0] if arms else None, collaudo=False, runs_root=tmp_path / "runs", branch_runs=tmp_path / "branch",
                           manifest=manifest, scales=scales, window_seconds=window, calibration=cal, root=[root], time_limit_s=None,
                           device="cpu", num_workers=0)


def test_config_is_the_ab_arm_with_frozen_encoder_and_only_the_branch_term(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch, arms=())
    for arm in ("A_seed1", "B_seed0"):
        ab, cfg = S2C._ab_config(arm, args.calibration), S2C.build_config(arm, args.calibration)
        assert cfg.freeze_encoder and cfg.jepa.keep_weight == 0 and cfg.jepa.branch_weight == 1.0 and cfg.model.branch_alpha == 0.5
        assert cfg.model.spectral_branches == ("psd", "wavelet") and not cfg.model.wavelet_learnable and cfg.branch_calibration_batches == 16
        assert (cfg.max_steps, cfg.lr, cfg.warmup_steps, cfg.weight_decay, cfg.grad_clip) == (3, 1e-3, 1, 0.05, 1.0)
        assert cfg.seed == ab.seed == int(arm[-1]) and cfg.loader == ab.loader and cfg.datasets == ab.datasets
        assert replace(cfg.model, spectral_branches=()) == ab.model  # stesso encoder del teacher, solo i rami in piu'


def test_train_keeps_the_encoder_and_saves_the_step0_control(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    S2C.train(args)
    run = args.branch_runs / "B_seed0"
    prov = json.loads((run / "run_provenance.json").read_text())
    assert prov["complete"] and prov["encoder_identical"] and prov["steps"] == 3 and not prov["collaudo"]
    assert prov["raw_check"]["final"]["encoder_tensors"] > 0 and prov["raw_check"]["init"]["teacher"]
    init, final = (torch.load(run / n, weights_only=False) for n in ("checkpoint_init.pt", "checkpoint.pt"))
    assert init["step"] == 0 and final["step"] == 3 and init["config"] == final["config"]
    heads = [k for k in final["teacher"] if k.startswith("branch_heads.") and k.endswith("linear.weight")]
    assert heads and all(not torch.equal(init["teacher"][k], final["teacher"][k]) for k in heads)  # i rami si sono allenati
    assert all(torch.equal(init["student"][k], init["teacher"][k]) for k in init["student"])  # passo 0: teacher = studente
    key = "spectral_branches.branches.wavelet.scales.slow.input_mean"
    assert torch.equal(init["teacher"][key], final["teacher"][key])  # stessa calibrazione dei rami
    assert json.loads((run / "spectral_branches.json").read_text())["init_checkpoint_sha256"] == prov["init_checkpoint_sha256"]
    assert S2C._run(args, "B_seed0")[1] == prov
    with pytest.raises(ValueError, match="NEW"):
        S2C.train(args)  # niente sovrascritture
    (run / "checkpoint_init.pt").write_bytes((run / "checkpoint_init.pt").read_bytes() + b"x")
    with pytest.raises(ValueError, match="cambiato"):
        S2C._run(args, "B_seed0")


def test_train_refuses_other_inputs_or_teachers(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    monkeypatch.setitem(S2C.INPUT_SHA256, "calibration", "0" * 64)
    with pytest.raises(ValueError, match="ingressi"):
        S2C.train(args)
    args = SimpleNamespace(**{**vars(args), "branch_runs": tmp_path / "other"})
    prov = args.runs_root / "B_seed0" / "run_provenance.json"
    prov.write_text(json.dumps({"source_commit": "altro", "checkpoint_sha256": SD.CHECKPOINTS["B_seed0"]}))
    with pytest.raises(ValueError, match="provenienza del training"):
        S2C.train(args)
    prov.write_text(json.dumps({"source_commit": SD.TRAIN_COMMIT, "checkpoint_sha256": SD.CHECKPOINTS["B_seed0"]}))
    ckpt = args.runs_root / "B_seed0" / "checkpoint.pt"
    ckpt.write_bytes(ckpt.read_bytes() + b"x")
    with pytest.raises(ValueError, match="teacher congelato"):
        S2C.train(args)


def test_collaudo_writes_its_verdict_in_its_own_folder(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="dbg"):
        S2C.train(SimpleNamespace(**{**vars(args), "collaudo": True, "time_limit_s": 3600.0}))
    S2C.train(SimpleNamespace(**{**vars(args), "collaudo": True, "time_limit_s": 600.0}))
    prov = json.loads((args.branch_runs / "collaudo" / "B_seed0" / "run_provenance.json").read_text())
    verdict = prov["collaudo_verdict"]
    assert prov["collaudo"] and prov["encoder_identical"] and verdict["updates"] == 3 and verdict["integrity"]
    assert not verdict["enough_updates"] and not verdict["passed"]  # 3 aggiornamenti: troppo pochi per giudicare la curva
    assert not (args.branch_runs / "B_seed0").exists()
    with pytest.raises(FileNotFoundError):
        S2C._run(args, "B_seed0")  # un collaudo non e' un run del protocollo


def test_raw_check_sees_any_encoder_change():
    torch.manual_seed(0)
    from wearusfm.model.fm import WearUsFM
    cfg = replace(R.small_config().model, keep_readout=True)
    src = WearUsFM(cfg).state_dict()
    model = WearUsFM(replace(cfg, spectral_branches=("psd", "wavelet")))
    model.load_state_dict(src, strict=False)
    state = {"student": model.state_dict(), "teacher": model.state_dict()}
    assert S2C.raw_check(src, state)["encoder_identical"]
    teacher = dict(state["teacher"])
    teacher["spectral_keep.target_mean"] = teacher["spectral_keep.target_mean"] + 1
    out = S2C.raw_check(src, {"student": state["student"], "teacher": teacher})
    assert out["encoder_identical"] and not out["teacher_step1_head_identical"]  # la testa dello Step 1 si registra soltanto
    key = next(k for k in src if k.startswith("backbone."))
    teacher[key] = teacher[key] + 1e-6
    assert not S2C.raw_check(src, {"student": state["student"], "teacher": teacher})["encoder_identical"]
    del teacher[key]
    assert not S2C.raw_check(src, {"student": state["student"], "teacher": teacher})["teacher"]


def _metrics(loss, t=0.7):
    return [{"keep_branches": float(v), "t_step_s": t, "t_data_s": 0.5} for v in loss]


@pytest.mark.parametrize("case,passed", [("decreasing", True), ("flat", False), ("spike", False), ("slow", False), ("short", False),
                                         ("broken", False)])
def test_frozen_collaudo_verdict(case, passed):
    n = 300 if case == "short" else 1500
    loss = 1 + np.exp(-np.arange(n) / 300)
    if case == "flat":
        loss = np.ones(n) + 0.01 * np.sin(np.arange(n))
    if case == "spike":
        loss[1000:1100] *= 3
    t = 4.0 if case == "slow" else 0.7
    v = S2C.collaudo_verdict(_metrics(loss, t), {"elapsed_s": t * n + 60}, integrity=case != "broken")
    assert v["passed"] is passed
    if case == "decreasing":
        assert v["learned"] and v["stable"] and v["time_ok"] and v["median_step_s"] == pytest.approx(0.7)
        assert v["run_allocation_h"] == 1.0  # ((60 + 3000 x 0,7) x 1,25 + 600) s = 55 min -> 1 h
    if case == "slow":
        assert v["learned"] and not v["time_ok"]
    if case == "spike":
        assert v["learned"] and not v["stable"] and v["max_rise"] > 1.5


# --- uscite dei rami, varianti e sonde (P2 sintetico) ------------------------------------------------------------------------------------

def _downstream(args, arm):
    """La cache dell'embedding raw del teacher come la scrive la tranche downstream (estrazione con audit + provenance.json)."""
    folder = args.downstream_root / "p2" / arm
    argv = ["--checkpoint", str(args.runs_root / arm / "checkpoint.pt"), "--features", str(folder), "--stage", "extract",
            "--epn-root", str(args.epn_root[0]), "--uci-root", str(args.uci_root), "--audit-inputs", "--device", "cpu", "--batch", "8"]
    assert S2.probe_p2.main(argv) == 0
    identity = {"protocol": SD.PROTOCOL, "source_commit": S2.DOWNSTREAM_COMMIT, "training_commit": SD.TRAIN_COMMIT,
                "checkpoint_sha256": SD.CHECKPOINTS[arm], "arm": arm, "task": "p2", "which": "teacher", "probe_seed": 0,
                "split_sha256": SD.SPLIT_SHA}
    (folder / "provenance.json").write_text(json.dumps({"identity": identity, "cache": S2.inventory(folder, "p2", S2.DOWN_DIM)}))


def _extract_setup(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    S2C.train(args)
    _epn(tmp_path / "epn" / "trainingJSON", 10)
    _uci(tmp_path / "uci", 10)
    monkeypatch.setattr(S2, "DOWN_DIM", 2 * R.small_config().model.dim)
    args = SimpleNamespace(**vars(args), task="p2", downstream_root=tmp_path / "down", eval_root=tmp_path / "eval",
                           epn_root=[tmp_path / "epn" / "trainingJSON"], uci_root=tmp_path / "uci", variant=None, compare_after=False)
    _downstream(args, "B_seed0")
    return args


def test_p2_branch_outputs_variants_and_probes_end_to_end(tmp_path, monkeypatch):
    args = _extract_setup(tmp_path, monkeypatch)
    S2C.extract(args)
    d = S2.DOWN_DIM
    for ds in SD.DATASETS["p2"]:
        load = lambda *p: np.load(args.eval_root.joinpath("p2", "B_seed0", *p, f"{ds}.npz"))  # noqa: E731
        with np.load(args.downstream_root / "p2" / "B_seed0" / f"{ds}.npz") as m, load("branches_trained") as bt, load("branches_init") as b0, \
                load("RP") as rp, load("RW") as rw, load("RPW") as rpw, load("RPW0") as rpw0:
            assert str(bt["feature_set"]) == "branches" and int(bt["model_dim"]) == 256 and bt["x"].shape[1] == 256
            assert str(bt["input_sha256"]) == str(m["input_sha256"]) == str(rpw["input_sha256"])
            np.testing.assert_array_equal(rpw["x"], np.concatenate([m["x"], bt["x"]], 1))
            np.testing.assert_array_equal(rp["x"], np.concatenate([m["x"], bt["x"][:, :128]], 1))
            np.testing.assert_array_equal(rw["x"], np.concatenate([m["x"], bt["x"][:, 128:]], 1))
            np.testing.assert_array_equal(rpw0["x"], np.concatenate([m["x"], b0["x"]], 1))
            assert not np.array_equal(bt["x"], b0["x"])  # allenato contro passo 0
            assert str(rpw["feature_set"]) == "model" and int(rpw["model_dim"]) == d + 256 and np.array_equal(rpw["y"], m["y"])
    # coerenza: l'embedding completo del run allenato ([raw, rami], encode compreso) e' la variante RPW, bit per bit (il raw e' congelato)
    full = tmp_path / "full"
    argv = ["--checkpoint", str(args.branch_runs / "B_seed0" / "checkpoint.pt"), "--features", str(full), "--stage", "extract",
            "--epn-root", str(args.epn_root[0]), "--uci-root", str(args.uci_root), "--device", "cpu", "--batch", "8"]
    assert S2.probe_p2.main(argv) == 0
    for ds in SD.DATASETS["p2"]:
        with np.load(full / f"{ds}.npz") as f, np.load(args.eval_root / "p2" / "B_seed0" / "RPW" / f"{ds}.npz") as rpw:
            np.testing.assert_array_equal(f["x"], rpw["x"])
    # le finestre P2 lette una volta per i due modelli: cache identiche a quelle della sonda ufficiale (`probe_p2 --feature-set branches`)
    for which, name in (("trained", "checkpoint.pt"), ("init", "checkpoint_init.pt")):
        ref = tmp_path / f"ref_{which}"
        argv = ["--checkpoint", str(args.branch_runs / "B_seed0" / name), "--features", str(ref), "--stage", "extract", "--feature-set",
                "branches", "--epn-root", str(args.epn_root[0]), "--uci-root", str(args.uci_root), "--audit-inputs", "--device", "cpu",
                "--batch", "64"]
        assert S2.probe_p2.main(argv) == 0
        for ds in SD.DATASETS["p2"]:
            with np.load(ref / f"{ds}.npz") as a, np.load(args.eval_root / "p2" / "B_seed0" / f"branches_{which}" / f"{ds}.npz") as b:
                assert sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    with pytest.raises(ValueError, match="NEW"):
        S2C.extract(args)
    S2C.probe(SimpleNamespace(**{**vars(args), "arm": "B_seed0", "variant": ["RPW", "RPW0"]}))
    for v in ("RPW", "RPW0"):
        report = json.loads(S2C._report_path(args, "p2", "B_seed0", v).read_text())
        assert report["feature_set"] == "model" and report["provenance"]["identity"]["variant"] == v and 0 <= report["p2"] <= 1
        assert Path(report["checkpoint"]).name == ("checkpoint.pt" if v == "RPW" else "checkpoint_init.pt")
    path = args.downstream_root / "p2" / "B_seed0" / "uci_emg.npz"
    path.write_bytes(path.read_bytes() + b"x")  # la cache d'origine cambia dopo: ogni variante che ne deriva si ferma
    with pytest.raises(ValueError, match="cambiat"):
        S2C._variant_meta(args, "p2", "B_seed0", "RP")


def test_extract_refuses_branch_outputs_of_other_windows(tmp_path, monkeypatch):
    """Embedding raw estratti su finestre diverse (un utente EPN in meno): l'hash degli ingressi non coincide e niente viene concatenato."""
    args = _extract_setup(tmp_path, monkeypatch)
    _epn(tmp_path / "epn_small" / "trainingJSON", 9)
    with pytest.raises(ValueError, match="stesse finestre"):
        S2C.extract(SimpleNamespace(**{**vars(args), "epn_root": [tmp_path / "epn_small" / "trainingJSON"]}))
    assert not any((args.eval_root / "p2" / "B_seed0" / v / "epn612.npz").exists() for v in S2C.VARIANTS)


def test_extract_refuses_an_incomplete_or_changed_run(tmp_path, monkeypatch):
    args = _extract_setup(tmp_path, monkeypatch)
    prov_path = args.branch_runs / "B_seed0" / "run_provenance.json"
    prov = json.loads(prov_path.read_text())
    for field, value in (("encoder_identical", False), ("complete", False), ("source_commit", "altro")):
        prov_path.write_text(json.dumps({**prov, field: value}))
        with pytest.raises(ValueError, match="run non valido"):
            S2C.extract(args)


# --- confronto ------------------------------------------------------------------------------------------------------------------------

def _compare_setup(tmp_path, monkeypatch, rpw_a_low=False):
    monkeypatch.setattr(SD, "check_source", lambda _: None)
    base = _reports()  # «A»: 0,75, «B»: 1,0 per task e dataset (test_downstream_ab), stessi soggetti
    inv = {"data": {"sha256": "x", "input_sha256": "i", "rows": {"": {"n": 16}}}}
    args = SimpleNamespace(expected_commit="test", eval_root=tmp_path / "eval", downstream_root=tmp_path / "down", step2a_root=tmp_path / "s2a",
                           branch_runs=tmp_path / "branch")
    low, high = (lambda t, k: copy.deepcopy(base[t][f"A_seed{k}"])), (lambda t, k: copy.deepcopy(base[t][f"B_seed{k}"]))
    level = {"RP": high, "RW": low, "RPW": high, "RPW0": low}
    metas = {}
    for task in ("p1", "p2"):
        for arm in S2C.ARMS:
            k = int(arm[-1])
            for variant, make in level.items():
                if rpw_a_low and variant == "RPW" and arm.startswith("A"):
                    make = low
                meta = {"identity": S2C._identity("test", task, arm, variant), "cache": copy.deepcopy(inv)}
                metas[(task, arm, variant)] = meta
                path = S2C._report_path(args, task, arm, variant)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({**make(task, k), "feature_set": "model", "provenance": meta}))
    monkeypatch.setattr(S2C, "_variant_meta", lambda a, t, arm, v: (None, copy.deepcopy(metas[(t, arm, v)])))
    monkeypatch.setattr(S2C, "_reference", lambda a, t, arm: (low(t, int(arm[-1])), {"cache": copy.deepcopy(inv)}, "h"))
    monkeypatch.setattr(S2C, "_training_summary", lambda a, arm: {})
    hashes = {}
    for task in ("p1", "p2"):  # il 2A concluso: R+S dei teacher B, col confronto che ne registra gli hash
        for arm in S2C.FAMILIES["B"]:
            path = args.step2a_root / task / arm / "MS.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            identity = {"protocol": S2.PROTOCOL, "task": task, "variant": "MS", "arm": arm}
            path.write_text(json.dumps({**low(task, int(arm[-1])), "provenance": {"identity": identity, "cache": copy.deepcopy(inv)}}))
            hashes[f"{task}/{arm}/MS"] = sha256_file(path)
    (args.step2a_root / "comparison.json").write_text(json.dumps({"protocol": S2.PROTOCOL, "report_sha256": hashes}))
    return args


def test_compare_frozen_decisions(tmp_path, monkeypatch):
    args = _compare_setup(tmp_path, monkeypatch)
    S2C.compare(args)
    result = json.loads((args.eval_root / "comparison.json").read_text())
    d = result["decision"]
    assert d["A"] == d["B"] == {"primary_positive": True, "training_matters": True}
    assert d["better_family_with_branches"] == "nessuna differenza oltre le soglie" and d["better_than_fixed_stats_B"] is True
    fam = result["families"]["B"]
    assert fam["primary_RPW_vs_R"]["decision"]["p1"]["delta"] == pytest.approx(.25)
    assert fam["descriptive"]["RW"]["tasks"]["p2"]["delta"] == pytest.approx(0.)
    assert "practical_guard_pass" not in fam["primary_RPW_vs_R"]["paired"]["tasks"]["p1"]  # la guardia dell'A/B non vale qui
    assert result["descriptive"]["raw_A_vs_B"]["tasks"]["p1"]["delta"] == pytest.approx(0.)
    assert result["thresholds"] == {"useful_delta": .01, "harm_delta": -.01}


def test_compare_family_rule_and_missing_2a(tmp_path, monkeypatch):
    args = _compare_setup(tmp_path, monkeypatch, rpw_a_low=True)
    S2C.compare(SimpleNamespace(**{**vars(args), "step2a_root": None}))
    d = json.loads((args.eval_root / "comparison.json").read_text())["decision"]
    assert not d["A"]["primary_positive"] and d["B"]["primary_positive"] and d["better_family_with_branches"] == "B"
    assert d["better_than_fixed_stats_B"] == "2A non disponibile: rimandato"


def test_compare_refuses_a_2a_report_not_in_its_comparison(tmp_path, monkeypatch):
    args = _compare_setup(tmp_path, monkeypatch)
    path = args.step2a_root / "p2" / "B_seed1" / "MS.json"
    report = json.loads(path.read_text())
    report["p2"] = 0.99
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="confronto del 2A"):
        S2C.compare(args)


def test_family_rule_is_symmetric():
    row = lambda d, lo, hi, sd: {"A": .7, "B": .7 + d, "delta": d, "ci95": [lo, hi], "delta_by_model_seed": sd}  # noqa: E731
    b_better = {"tasks": {"p1": row(.03, .01, .05, [.02, .04]), "p2": row(0., -.01, .01, [0., 0.])}}
    a_better = {"tasks": {"p1": row(-.03, -.05, -.01, [-.02, -.04]), "p2": row(0., -.01, .01, [0., 0.])}}
    assert S2C.better_family(b_better)["better"] == "B" and S2C.better_family(a_better)["better"] == "A"
    harmful = {"tasks": {"p1": row(.03, .01, .05, [.02, .04]), "p2": row(-.02, -.04, 0., [-.02, -.02])}}
    assert S2C.better_family(harmful)["better"] == "nessuna differenza oltre le soglie"
