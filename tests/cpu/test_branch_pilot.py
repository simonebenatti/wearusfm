"""Pilot dei rami (`scripts/branch_pilot.py`, protocollo `rami_pilota_v1`): configurazione del 2C salvo il braccio, controlli del run (passo 0
caricato dal 2C, strati fissi, perdita del passo 1), collaudo (PWf a un passo, poi PWm), ibrido PWh, verdetto del collaudo, sonde del pilot
nella diagnostica, confronto con la lettura congelata (anche con un braccio collassato) e piano della catena."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import branch_pilot as BP  # noqa: E402
import step2c as S2C  # noqa: E402

from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

from test_branch_pilot_trainer import _source_and_cfg  # noqa: E402

CAL = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_0710/calibration.npz"
CONFIGS_2C = REPO / "results" / "passo6" / "step2c_configs_20261014"  # i config.json e le provenienze dei run del 2C (copiati il 23/10)


@pytest.fixture
def no_hash(monkeypatch):
    """`with_spectral_keep` legge l'hash della calibrazione del cluster: qui quello registrato nella configurazione del 2C."""
    sha = json.loads((CONFIGS_2C / "B_seed0_config.json").read_text())["keep_calibration_sha256"]
    monkeypatch.setattr(R, "sha256_file", lambda p: sha)


# --- configurazione ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("teacher", BP.TEACHERS)
def test_each_arm_differs_from_the_real_2c_config_only_where_the_protocol_says(no_hash, teacher):
    """Contro i `config.json` veri dei run del 2C (step2c_1410): col codice di oggi la configurazione del 2C si riproduce senza differenze e
    ogni braccio cambia solo le sue chiavi."""
    ref = json.loads((CONFIGS_2C / f"{teacher}_config.json").read_text())
    assert BP.config_check(R.config_to_dict(S2C.build_config(teacher, CAL)), ref, "PWf")["differences"] == {}
    for arm, spec in BP.ARMS.items():
        cfg = BP.build_config(arm, teacher, CAL)
        assert (cfg.lr, cfg.model.branch_freeze_output, cfg.model.branch_objective) == (spec["lr"], spec["freeze_output"], spec["objective"])
        assert cfg.fixed_eval_every == 0 and cfg.max_steps == S2C.STEPS and cfg.seed == int(teacher[-1])
        check = BP.config_check(R.config_to_dict(cfg), ref, arm)
        assert check["ok"] and set(check["differences"]) == {".".join(k) for k in BP.ALLOWED[arm]} - ({"branch_monitor"} if arm != "PWl" else set())


def test_the_2c_provenance_copied_in_results_is_the_frozen_one():
    for t in BP.TEACHERS:
        prov = json.loads((CONFIGS_2C / f"{t}_run_provenance.json").read_text())
        assert {k: prov[k] for k in BP.STEP2C_RUNS[t]} == BP.STEP2C_RUNS[t] and prov["source_commit"] == BP.STEP2C_COMMIT


def test_a_difference_outside_the_arm_is_refused(no_hash):
    ref = R.config_to_dict(S2C.build_config("B_seed0", CAL))
    lr = R.config_to_dict(BP.build_config("PWl", "B_seed0", CAL))
    assert not BP.config_check(lr, ref, "PWf")["ok"]  # l'LR cambia solo in PWl
    other = R.config_to_dict(replace(BP.build_config("PWm", "B_seed0", CAL), batch_size=16))
    assert BP.config_check(other, ref, "PWm")["unexpected"] == ["batch_size"]
    one = R.config_to_dict(BP.build_config("PWf", "B_seed0", CAL, max_steps=1))
    assert not BP.config_check(one, ref, "PWf")["ok"] and BP.config_check(one, ref, "PWf", extra=(("max_steps",),))["ok"]
    old = {k: v for k, v in ref.items() if k != "branch_monitor"}  # un campo aggiunto dopo il 2C, col suo default: nessuna differenza
    assert BP.config_check(R.config_to_dict(BP.build_config("PWf", "B_seed0", CAL)), old, "PWf")["ok"]


def test_the_probe_code_is_the_one_of_the_diagnostic_reports():
    """I rapporti PW0, PW, RPW0, RPW vengono dal codice della diagnostica (aba1f2e); quelli del pilot dal suo commit: la sonda deve essere la
    stessa (critico, 23/10). Hash di aba1f2e (`git show aba1f2e:<file>`): cambiare uno di questi file rompe l'accoppiamento coi rapporti della
    diagnostica, serve una decisione, non un aggiornamento dell'hash."""
    at_diagnostic = {
        "scripts/calibration_probe.py": "197d19d061aa188a5a6c87305e60087f25b346e1966ad949c1a6ab7f8522aac3",
        "src/wearusfm/harness/probe.py": "aaf3e9495edf307db4f08320b6121ae39af1144b8ab6cb7345b36430dfe8d3ae",
        "src/wearusfm/harness/p1_cross_subject.py": "7045b947920d314e7b25e85dc41e42de6bc5dc42aef0812c485b247bda31c4ce",
        "src/wearusfm/harness/downstream_ab.py": "ea3b91423fe6c12743acb9679e4a3e9e214d84e4135769aeb4bbd82ed97b3164",
        "src/wearusfm/harness/splits.py": "51039fe49c211b8670445e1f037faae4c08b6290a881757e779971f71adc501e"}
    assert {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in at_diagnostic} == at_diagnostic


# --- controlli del run, sul trainer vero (modello piccolo) -------------------------------------------------------------------------------

@pytest.fixture
def runs(tmp_path):
    """Un run come il 2C (obiettivo vecchio, nessuno strato fisso) e i bracci PWl, PWf e PWm dal suo passo 0."""
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    ref = tmp_path / "2c"
    R.train(replace(cfg_b, max_steps=2), manifest, [root], ref, log=lambda _: None, init_from=src)
    cfgs = {"PWl": replace(cfg_b, max_steps=2, lr=cfg_b.lr / 3, branch_monitor=True),
            "PWf": replace(cfg_b, max_steps=2, model=replace(cfg_b.model, branch_freeze_output=True)),
            "PWm": replace(cfg_b, max_steps=2, model=replace(cfg_b.model, branch_objective="masked", branch_freeze_output=True))}
    for arm, cfg in cfgs.items():
        R.train(cfg, manifest, [root], tmp_path / arm, log=lambda _: None, init_from=src, branch_state_from=ref / "checkpoint_init.pt")
    return SimpleNamespace(tmp=tmp_path, root=root, manifest=manifest, cfg_b=cfg_b, src=src, ref=ref, cfgs=cfgs)


def test_runs_from_the_2c_step_0_pass_the_checks(runs):
    ref_init = BP._load(runs.ref / "checkpoint_init.pt")
    for arm, cfg in runs.cfgs.items():
        out = runs.tmp / arm
        final, own = BP._load(out / "checkpoint.pt"), BP._load(out / "checkpoint_init.pt")
        assert BP.init_check(ref_init, own)["identical"]
        fixed = BP.fixed_check(BP.trainable_names(cfg), ref_init, final)  # PWl: EMA completa, strati fissi solo i buffer e i frontend
        assert fixed["identical"] and fixed["student_tokens_changed"] and fixed["teacher_tokens_changed"], arm
        dist = BP.token_distance(ref_init["teacher"], final["teacher"])
        assert dist["all"] > 0 and set(dist) == {"all", *S2C.KINDS}
    for arm in ("PWl", "PWf"):
        step1 = BP.step1_check(BP._records(runs.tmp / arm / "metrics.jsonl"), BP._records(runs.ref / "metrics.jsonl"))
        assert step1["ok"] and step1["run"] == pytest.approx(step1["step2c"], rel=1e-6)


def test_a_moved_fixed_layer_a_different_step_0_or_step_1_loss_fail(runs):
    ref_init = BP._load(runs.ref / "checkpoint_init.pt")
    final = BP._load(runs.tmp / "PWm" / "checkpoint.pt")
    key = next(k for k in final["teacher"] if k.startswith("spectral_branches.") and ".combine." in k)
    final["teacher"][key] = final["teacher"][key] + 1e-6
    out = BP.fixed_check(BP.trainable_names(runs.cfgs["PWm"]), ref_init, final)
    assert out["student"] and not out["teacher"] and not out["identical"]
    pwl = BP._load(runs.tmp / "PWl" / "checkpoint.pt")
    stat = next(k for k in pwl["student"] if k.startswith("spectral_branches.") and k.endswith("input_mean"))
    pwl["student"][stat] = pwl["student"][stat] + 1
    assert not BP.fixed_check(BP.trainable_names(runs.cfgs["PWl"]), ref_init, pwl)["identical"]
    own = BP._load(runs.tmp / "PWf" / "checkpoint_init.pt")
    head = next(k for k in own["student"] if k.startswith("branch_heads."))
    own["student"][head] = own["student"][head] + 1
    assert not BP.init_check(ref_init, own)["identical"]
    metrics = BP._records(runs.tmp / "PWf" / "metrics.jsonl")
    metrics[0]["keep_branches"] *= 1 + 2e-4
    assert not BP.step1_check(metrics, BP._records(runs.ref / "metrics.jsonl"))["ok"]
    assert not BP.step1_check([], BP._records(runs.ref / "metrics.jsonl"))["ok"]


def _env(runs, monkeypatch, collaudo=False):
    """Ingressi, teacher e run del 2C del test al posto di quelli congelati del cluster; `build_config` -> i bracci piccoli del test."""
    files = {k: runs.tmp / f"{k}.json" for k in S2C.INPUTS if k != "manifest"}  # il manifest e' quello vero della calibrazione del test
    files["scales"].write_text(json.dumps({"scales": None}))
    files["window_seconds"].write_text(json.dumps({"window_s": None}))
    files["calibration"].write_text("{}")
    files["manifest"] = runs.manifest
    monkeypatch.setattr(S2C, "INPUT_SHA256", {k: sha256_file(f) for k, f in files.items()})
    monkeypatch.setattr(BP.SD, "CHECKPOINTS", {**BP.SD.CHECKPOINTS, "B_seed0": sha256_file(runs.src)})
    monkeypatch.setattr(BP.SD, "check_source", lambda c: None)
    monkeypatch.setattr(BP, "STEP2C_RUNS", {"B_seed0": {"checkpoint_sha256": sha256_file(runs.ref / "checkpoint.pt"),
                                                        "init_checkpoint_sha256": sha256_file(runs.ref / "checkpoint_init.pt")}})
    monkeypatch.setattr(BP, "build_config", lambda arm, teacher, cal, max_steps=None: replace(runs.cfgs[arm], max_steps=max_steps or 2))
    return SimpleNamespace(**files, root=[runs.root], device="cpu", num_workers=0, expected_commit="e" * 40, collaudo=collaudo,
                           pilot_runs=runs.tmp / "pilot_runs", step2c_runs=runs.tmp, arm="PWm", teacher="B_seed0", time_limit_s=600.0)


def test_train_one_writes_the_provenance_with_every_check(runs, monkeypatch):
    """`_train_one` col trainer vero: configurazione del 2C, numeri finiti, checkpoint coerenti, raw, passo 0, strati fissi, (PWl, PWf) passo 1."""
    args = _env(runs, monkeypatch)
    for arm in BP.ARMS:
        prov, summary = BP._train_one(args, arm, "B_seed0", runs.tmp / f"run_{arm}", max_steps=None, time_limit_s=600, source=runs.src,
                                      run2c=runs.ref, scales=None, window_s=None)
        assert prov["complete"] and prov["checks_passed"], prov["checks_ok"]
        assert set(prov["checks_ok"]) == ({"config", "finite", "consistent", "raw", "init", "fixed", "branch_state"}
                                          | ({"step1"} if arm != "PWm" else set()))
        assert json.loads((runs.tmp / f"run_{arm}" / "run_provenance.json").read_text())["checks_passed"]
        assert summary["spectral_branches"]["collapse_monitor"] is not None
    real = BP._load
    monkeypatch.setattr(BP, "_load", lambda path: {**real(path), "config": {"altro": 1}} if path.parent.name == "run_odd"
                        and path.name == "checkpoint.pt" else real(path))
    prov, _ = BP._train_one(args, "PWf", "B_seed0", runs.tmp / "run_odd", max_steps=None, time_limit_s=600, source=runs.src, run2c=runs.ref,
                            scales=None, window_s=None)
    assert prov["checks_ok"]["consistent"] is False and not prov["checks_passed"]  # checkpoint con un'altra configurazione
    monkeypatch.setattr(BP, "_load", real)
    monkeypatch.setattr(BP, "build_config", lambda arm, teacher, cal, max_steps=None: replace(runs.cfgs[arm], batch_size=4, max_steps=2))
    with pytest.raises(ValueError, match="configurazione diversa dal 2C"):
        BP._train_one(args, "PWf", "B_seed0", runs.tmp / "bad", max_steps=None, time_limit_s=600, source=runs.src, run2c=runs.ref,
                      scales=None, window_s=None)


def test_collaudo_runs_pwf_for_one_step_then_pwm_and_writes_the_verdict(runs, monkeypatch):
    """`train --collaudo`: PWf per un passo (controllo del passo 1) e poi PWm nello stesso job; il verdetto entra nella provenienza del PWm.
    Due aggiornamenti non bastano per l'apprendimento: il collaudo non passa e il job esce con errore (nessun retry)."""
    args = _env(runs, monkeypatch, collaudo=True)
    monkeypatch.setattr(S2C, "_source", lambda a, t: runs.src)
    monkeypatch.setattr(BP, "step2c_run", lambda a, t: (runs.ref, {}))
    with pytest.raises(ValueError, match="collaudo non passato"):
        BP.train(args)
    first = json.loads((args.pilot_runs / "collaudo" / "PWf_B_seed0_passo1" / "run_provenance.json").read_text())
    assert first["steps"] == 1 and first["checks_passed"] and first["checks"]["step1"]["ok"]
    prov = json.loads((args.pilot_runs / "collaudo" / "PWm_B_seed0" / "run_provenance.json").read_text())
    v = prov["collaudo_verdict"]
    assert v["integrity_ok"] and v["integrity"]["PWf_passo1_step1"] and v["integrity"]["PWm_fixed"] and v["integrity"]["no_stop"]
    assert not v["passed"] and not v["learning"]["verifiable"]
    with pytest.raises(ValueError, match="PWm su B_seed0"):
        BP.train(SimpleNamespace(**{**vars(args), "arm": "PWf"}))


def test_hybrid_takes_tokens_from_pw_and_final_layers_from_pw0(runs, monkeypatch):
    trained, init = BP._load(runs.ref / "checkpoint.pt"), BP._load(runs.ref / "checkpoint_init.pt")
    h = BP.hybrid_state(trained, init)
    assert "optimizer" not in h and h["config"] == trained["config"]
    for part in ("student", "teacher"):
        for k, v in h[part].items():
            fixed = k.startswith("spectral_branches.") and any(f in k for f in BP.FIXED_LAYERS)
            assert torch.equal(v, init[part][k] if fixed else trained[part][k])
    moved = [k for k in trained["teacher"] if k.startswith("spectral_branches.") and ".output." in k
             and not torch.equal(trained["teacher"][k], init["teacher"][k])]
    assert moved  # nel run come il 2C `output` si allena: l'ibrido lo riporta al passo 0
    # la fase `hybrid` dal run del 2C (provenienza controllata), e `pilot_run` che la rilegge prima dell'estrazione
    args = _env(runs, monkeypatch)
    two_c = runs.tmp / "two_c"
    (two_c / "B_seed0").mkdir(parents=True)
    for name in ("checkpoint.pt", "checkpoint_init.pt"):
        (two_c / "B_seed0" / name).write_bytes((runs.ref / name).read_bytes())
    prov2c = {"protocol": S2C.PROTOCOL, "source_commit": BP.STEP2C_COMMIT, "arm": "B_seed0", "collaudo": False, "complete": True,
              "encoder_identical": True, "source_checkpoint_sha256": BP.SD.CHECKPOINTS["B_seed0"], **BP.STEP2C_RUNS["B_seed0"]}
    (two_c / "B_seed0" / "run_provenance.json").write_text(json.dumps(prov2c))
    hargs = SimpleNamespace(expected_commit=args.expected_commit, teacher="B_seed0", step2c_runs=two_c, pilot_runs=args.pilot_runs)
    BP.hybrid(hargs)
    folder, prov = BP.pilot_run(hargs, "PWh", "B_seed0")
    assert prov["checks_passed"] and prov["fixed_tensors"] > 0 and prov["checks"]["token_distance"]["teacher"]["all"] > 0
    with pytest.raises(ValueError, match="run non valido per il pilot"):  # estrazione da un altro codice
        BP.pilot_run(SimpleNamespace(**{**vars(hargs), "expected_commit": "0" * 40}), "PWh", "B_seed0")
    torch.save({"x": 1}, folder / "checkpoint.pt")
    with pytest.raises(ValueError, match="checkpoint cambiato"):
        BP.pilot_run(hargs, "PWh", "B_seed0")
    (two_c / "B_seed0" / "checkpoint_init.pt").write_bytes(b"altro")
    with pytest.raises(ValueError, match="checkpoint del 2C cambiato"):
        BP.step2c_run(hargs, "B_seed0")
    (two_c / "B_seed0" / "run_provenance.json").write_text(json.dumps({**prov2c, "complete": False}))
    with pytest.raises(ValueError, match="diverso da quello congelato"):
        BP.step2c_run(hargs, "B_seed0")


# --- apprendimento e collaudo -----------------------------------------------------------------------------------------------------------

def _metrics(n=300, norm=0.5, targets=3, step_s=0.8):
    keys = BP.branch_keys()
    return [{"step": i + 1, "t_step_s": step_s, "masked": 1.0 - 0.001 * i, "masked_normalized": norm,
             **{f"masked_{k}_normalized": norm for k in keys},
             **{f"masked_{k}_targets": targets for k in keys}, **{f"masked_{k}_context_per_band": [1, 2] for k in keys}}
            for i in range(n)]


def test_learned_needs_100_defined_values_at_most_0_9():
    assert BP.learned(_metrics())["learned"]
    assert not BP.learned(_metrics(norm=0.95))["learned"] and BP.learned(_metrics(norm=0.95))["verifiable"]
    m = _metrics()
    m[-3]["masked_normalized"] = None
    assert BP.learned(m) == {"verifiable": False, "updates": 100, "undefined": 1, "mean": None, "learned": False}
    assert not BP.learned(_metrics(n=50))["verifiable"]
    comp = _metrics()
    comp[-5]["masked_wavelet_slow_normalized"] = None  # componente a varianza nulla: la media resta definita, la misura no (critico)
    assert BP.learned(comp)["undefined"] == 1 and not BP.learned(comp)["verifiable"]
    absent = _metrics()
    del absent[-5]["masked_wavelet_slow_normalized"]  # nessun bersaglio di quella scala nel batch: non e' una misura
    assert BP.learned(absent)["learned"]


def _summary(n=300, step_s=0.8, verified=True):
    return {"elapsed_s": n * step_s + 120, "spectral_branches": {"collapse_monitor": {"verified": verified, "alarm": None, "checks": 20,
                                                                                    "reference_undefined": []}}}


def test_collaudo_verdict_passes_and_each_criterion_fails():
    ok = {"no_stop": True, "PWm_fixed": True}
    v = BP.collaudo_verdict(_metrics(), _summary(), ok)
    # ((120 + 3.000 x 0,8) x 1,25 + 600) s -> 1 h 15
    assert v["passed"] and v["time"]["run_allocation_h"] == 1.25 and v["coverage"]["context_per_band"]["psd_fast"] == [300, 600]
    assert not BP.collaudo_verdict(_metrics(), _summary(), {**ok, "PWf_passo1_step1": False})["passed"]
    assert not BP.collaudo_verdict(_metrics(norm=0.95), _summary(), ok)["passed"]
    assert not BP.collaudo_verdict(_metrics(), _summary(verified=False), ok)["passed"]
    sparse = _metrics()
    for r in sparse[:160]:
        r["masked_wavelet_slow_targets"] = 0
    v = BP.collaudo_verdict(sparse, _summary(), ok)
    assert not v["passed"] and v["coverage"]["fraction"]["wavelet_slow"] == pytest.approx(140 / 300)
    slow = BP.collaudo_verdict(_metrics(step_s=1.6), _summary(step_s=1.6), ok)
    assert not slow["passed"] and slow["time"]["run_allocation_h"] > 1.5
    assert not BP.collaudo_verdict([], _summary(), ok)["passed"]


# --- sonde del pilot nella diagnostica ----------------------------------------------------------------------------------------------------

def _pilot_caches(tmp_path, ev, task, arm, models=("PWf", "PWm"), **run):
    """Cache del pilot come quelle di `extract`, copiate dalle uscite allenate del 2C sintetico (stesse righe)."""
    SDG = BP._script("spectral_diagnostic")
    pilot = tmp_path / "pilot_cache"
    for i, m in enumerate(models):
        src = ev / task / arm / "branches_trained"
        dst = pilot / task / arm / f"branches_{m}"
        dst.mkdir(parents=True)
        meta = json.loads((src / "provenance.json").read_text())
        for ds in SDG.DATASETS[task]:
            (dst / f"{ds}.npz").write_bytes((src / f"{ds}.npz").read_bytes())
        prov = {"protocol": BP.PROTOCOL, "source_commit": "f" * 40, "arm": m, "teacher": arm, "collaudo": False, "complete": True,
                "checks_passed": True, "checkpoint_sha256": f"{i}" * 64, **run}
        (dst / "provenance.json").write_text(json.dumps({"identity": BP.cache_identity("f" * 40, task, arm, m), "run": prov,
                                                         "cache": meta["cache"]}))
    return pilot


def test_pilot_sets_are_probed_with_their_own_protocol_and_recipe(tmp_path):
    import test_spectral_diagnostic as TSD

    ev, dn = TSD.build(tmp_path, "p2", arms=("B_seed0",))
    pilot = _pilot_caches(tmp_path, ev, "p2", "B_seed0")
    out = tmp_path / "out"
    assert TSD._probe(ev, dn, out, "p2", "B_seed0", ["RPWm", "PWf", "RPW0"], "--pilot-root", str(pilot)) == 0
    rep = json.loads((out / "p2" / "B_seed0" / "RPWm.json").read_text())
    assert rep["protocol"] == BP.PROTOCOL and rep["set"] == "RPWm" and rep["feature_set"] == "cache_pilota"
    assert rep["extraction_recipe"]["branches"] == {"protocol": BP.PROTOCOL, "source_commit": "f" * 40, "device": "cpu",
                                                    "checkpoint_sha256": "1" * 64}
    assert json.loads((out / "p2" / "B_seed0" / "PWf.json").read_text())["extraction_recipe"]["branches"]["checkpoint_sha256"] == "0" * 64
    assert all(v["features"]["columns"] == 1024 for v in rep["datasets"].values())
    assert json.loads((out / "p2" / "B_seed0" / "RPW0.json").read_text())["protocol"] == "rami_diagnostica_v1"
    with pytest.raises(ValueError, match="pilot-root"):
        TSD._probe(ev, dn, out, "p2", "B_seed0", ["PWm"])


def test_a_pilot_cache_from_an_incomplete_run_another_code_or_other_rows_is_refused(tmp_path):
    import test_spectral_diagnostic as TSD

    ev, dn = TSD.build(tmp_path, "p2", arms=("B_seed0",))
    pilot = _pilot_caches(tmp_path, ev, "p2", "B_seed0")
    path = pilot / "p2" / "B_seed0" / "branches_PWm" / "provenance.json"
    meta = json.loads(path.read_text())
    meta["run"]["checks_passed"] = False
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenienza diversa"):
        TSD._probe(ev, dn, tmp_path / "o1", "p2", "B_seed0", ["PWm"], "--pilot-root", str(pilot))
    meta["run"]["checks_passed"] = True
    meta["identity"]["source_commit"] = meta["run"]["source_commit"] = "0" * 40
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="codici diversi"):
        TSD._probe(ev, dn, tmp_path / "o2", "p2", "B_seed0", ["PWm", "PWf"], "--pilot-root", str(pilot))
    meta["identity"]["source_commit"] = meta["run"]["source_commit"] = "f" * 40
    meta["cache"]["epn612"]["input_sha256"] = "x" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="non accoppiati alla cache raw"):
        TSD._probe(ev, dn, tmp_path / "o3", "p2", "B_seed0", ["PWm"], "--pilot-root", str(pilot))
    meta["cache"]["epn612"]["input_sha256"] = "i" * 64
    meta["identity"]["device"] = "cuda"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenienza diversa"):
        TSD._probe(ev, dn, tmp_path / "o4", "p2", "B_seed0", ["PWm"], "--pilot-root", str(pilot))


# --- confronto ----------------------------------------------------------------------------------------------------------------------------

DATASETS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}


def _report(folder: Path, task: str, teacher: str, name: str, quality: float, seed: int, commit="f" * 40, ckpt="k" * 64, raw="r" * 64):
    """Un rapporto nel formato della sonda, con predizioni fuori fold di accuratezza `quality` sulle stesse righe per tutti."""
    SDG = BP._script("spectral_diagnostic")
    diag = name in BP.DIAG_SETS
    rng = np.random.default_rng(seed)
    oof, datasets = {}, {}
    for d in DATASETS[task]:
        y = np.tile(np.arange(4), 40)
        s = np.repeat([f"{d}_s{i}" for i in range(20)], 8)
        row = np.arange(len(y), dtype=np.int64)
        pred = np.where(rng.random(len(y)) < quality, y, rng.integers(0, 4, len(y)))
        oof.update({f"{d}_y": y, f"{d}_pred": pred, f"{d}_s": s, f"{d}_row": row})
        files = {"raw": {"path": "raw", "sha256": raw}} if name.startswith("R") and raw else {}
        datasets[d] = {"input_sha256": "i" * 64, "oof_rows_sha256": fingerprint(row), "s": float((pred == y).mean()), "files": files}
    path = folder / task / teacher / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".oof.npz"), datasets=np.asarray(list(DATASETS[task])), **oof)
    w = {d: 1 / len(DATASETS[task]) for d in DATASETS[task]}
    branches = SDG.STEP2C if diag else {"protocol": BP.PROTOCOL, "source_commit": commit, "device": "cpu", "checkpoint_sha256": ckpt}
    rep = {"protocol": SDG.PROTOCOL if diag else BP.PROTOCOL, "task": task, "arm": teacher, "set": name, "k_folds": 4, "fold_seed": 1,
           "val_frac": 0.2, "probe_seed": 0, "window_s": 1.0 if task == "p1" else None, "splits_sha256": "s" * 64,
           "feature_set": "cache_2C" if diag else "cache_pilota",
           "extraction_recipe": {"branches": branches, "raw": SDG.DOWNSTREAM, "split_sha256": SDG.SPLIT_SHA},
           "weights": w, "datasets": datasets, "S": 0.0, "oof": str(path.with_suffix(".oof.npz")),
           "oof_sha256": sha256_file(path.with_suffix(".oof.npz"))}
    path.write_text(json.dumps(rep))


def _pilot_runs(tmp_path, norm=0.5, verified=True, commit="f" * 40, collapsed=()):
    runs = tmp_path / "runs"
    for arm in BP.MODELS:
        for t in BP.TEACHERS:
            d = runs / arm / t
            d.mkdir(parents=True)
            if f"{arm}/{t}" in collapsed:  # fermato dall'allarme: STOP e `stopped` nel formato del trainer (run.py), provenienza incompleta
                why = "allarme: allarme di collasso dei rami: psd_fast_erank 0.01 sotto 0.1 x riferimento 9 per 3 controlli consecutivi"
                (d / "STOP").write_text(why + "\n")
                (d / "run_provenance.json").write_text(json.dumps({"protocol": BP.PROTOCOL, "arm": arm, "teacher": t, "collaudo": False,
                                                                   "complete": False, "stopped": why, "source_commit": commit}))
                continue
            (d / "run_provenance.json").write_text(json.dumps({
                "protocol": BP.PROTOCOL, "arm": arm, "teacher": t, "complete": True, "checks_passed": True, "source_commit": commit,
                "checkpoint_sha256": "k" * 64, "checks": {"token_distance": {"teacher": {"all": 0.1}}}}))
            if arm != "PWh":
                (d / "summary.json").write_text(json.dumps(_summary(verified=verified)))
                (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _metrics(norm=norm)))
    return runs


def _reports(tmp_path, quality, skip=(), **kw):
    diag, pil = tmp_path / "diag", tmp_path / "sonde"
    for k, task in enumerate(DATASETS):
        for j, t in enumerate(BP.TEACHERS):
            for i, name in enumerate(BP.DIAG_SETS + BP.PILOT_SETS):
                if name not in skip:
                    _report(diag if name in BP.DIAG_SETS else pil, task, t, name, quality.get(name, 0.5), seed=100 * k + 10 * j + i, **kw)
    return diag, pil


def _compare(tmp_path, runs, *extra, out="esito.json"):
    diag, pil = tmp_path / "diag", tmp_path / "sonde"
    BP.main(["--stage", "compare", "--expected-commit", "c" * 40, "--diagnostic-dir", str(diag), "--reports-dir", str(pil), "--pilot-runs",
             str(runs), "--out", str(tmp_path / out), *extra])
    return json.loads((tmp_path / out).read_text())


@pytest.fixture
def no_source(monkeypatch):
    monkeypatch.setattr(BP.SD, "check_source", lambda c: None)


def test_primary_is_positive_when_pwm_beats_pwf_and_is_read_only_if_pwm_learned(tmp_path, no_source):
    good = {"RPWf": 0.4, "RPWm": 0.8}
    for name, q, kw in (("a", good, {}), ("b", {"RPWf": 0.8, "RPWm": 0.4}, {}), ("c", good, {"norm": 0.95}), ("d", good, {"verified": False})):
        _reports(tmp_path / name, q)
    res = _compare(tmp_path / "a", _pilot_runs(tmp_path / "a"))
    assert res["primary"]["readable"] and res["primary"]["reading"]["positive"] and res["compare_commit"] == "c" * 40
    assert res["primary"]["reading"]["readings"] == {"p1": "migliora", "p2": "migliora"} and "wavelet lente" in res["note"]
    assert set(res["descriptive"]) == set(BP.DESCRIPTIVE) and res["runs"]["PWm"]["B_seed0"]["learned"]["learned"]
    worse = _compare(tmp_path / "b", _pilot_runs(tmp_path / "b"))
    assert not worse["primary"]["reading"]["positive"] and worse["primary"]["reading"]["readings"]["p1"] == "peggiora"
    lazy = _compare(tmp_path / "c", _pilot_runs(tmp_path / "c", norm=0.95))
    assert not lazy["primary"]["readable"] and lazy["primary"]["reading"].startswith("non si legge")
    unverified = _compare(tmp_path / "d", _pilot_runs(tmp_path / "d", verified=False))
    assert not unverified["primary"]["readable"] and not unverified["gate"]["PWf_not_collapsed"]
    primary = _compare(tmp_path / "a", _pilot_runs(tmp_path / "e"), "--primary-only", out="esito_primario.json")
    assert "descriptive" not in primary and primary["primary"]["reading"]["positive"]


def test_a_collapsed_arm_gives_a_formal_outcome(tmp_path, no_source):
    """Un braccio fermato dall'allarme non ha uscite ne' sonde: per PWf o PWm l'esito e' «collassato = non batte il riferimento», per PWl i
    suoi descrittivi si saltano (critico, 23/10)."""
    res = _compare(tmp_path, _pilot_runs(tmp_path, collapsed=("PWm/B_seed1",)), "--primary-only", out="esito_primario.json")
    assert res["collapsed"] == ["PWm/B_seed1"] and not res["primary"]["readable"] and not res["primary"]["reading"]["positive"]
    _reports(tmp_path / "l", {"RPWf": 0.4, "RPWm": 0.8}, skip=("PWl", "RPWl"))
    res = _compare(tmp_path / "l", _pilot_runs(tmp_path / "l", collapsed=("PWl/B_seed0",)))
    assert res["primary"]["reading"]["positive"] and res["collapsed"] == ["PWl/B_seed0"]
    assert sorted(res["descriptive_skipped"]) == ["PW_PWl_su_PW", "PW_PWl_su_RPW"]


def test_a_run_really_stopped_by_the_branch_alarm_is_read_as_collapsed(runs, monkeypatch):
    """Col trainer vero (allarme forzato come in test_branch_pilot_trainer): STOP e provenienza scritti da `_train_one`; `run_state` lo legge
    come «collassato». Un run fermato per altro, o mancante, non ha esito (seconda review del critico, 23/10)."""
    from wearusfm.model import branch_masking as BM

    args = _env(runs, monkeypatch)
    monkeypatch.setattr(BM.CollapseAlarm, "update", lambda self, step, metrics: "allarme di collasso dei rami: prova" if step == 2 else None)
    out = BP._run_dir(args, "PWm", "B_seed0")
    prov, summary = BP._train_one(args, "PWm", "B_seed0", out, max_steps=4, time_limit_s=600, source=runs.src, run2c=runs.ref,
                                  scales=None, window_s=None, extra=(("max_steps",),))  # il run del test «2C» ha 2 passi
    assert not prov["complete"] and summary["stopped"].startswith("allarme: allarme di collasso dei rami")
    assert BP.run_state(args, "PWm", "B_seed0") == ("collassato", args.expected_commit)
    (out / "STOP").write_text("allarme: rango effettivo sotto 10% della dimensione per 3 valutazioni\n")
    with pytest.raises(ValueError, match="fermato per altro"):
        BP.run_state(args, "PWm", "B_seed0")
    with pytest.raises(ValueError, match="mancante"):
        BP.run_state(args, "PWh", "B_seed0")


def test_compare_refuses_reports_of_other_sources_codes_checkpoints_or_raw_files(tmp_path, no_source):
    runs = _pilot_runs(tmp_path)
    diag, pil = _reports(tmp_path, {})
    _report(pil, "p1", "B_seed0", "RPWl", 0.5, seed=0, commit="0" * 40)
    with pytest.raises(ValueError, match="codici diversi"):
        _compare(tmp_path, runs)
    _report(pil, "p1", "B_seed0", "RPWl", 0.5, seed=0, ckpt="z" * 64)
    with pytest.raises(ValueError, match="checkpoint del run"):
        _compare(tmp_path, runs)
    _report(pil, "p1", "B_seed0", "RPWl", 0.5, seed=0)
    _report(pil, "p2", "B_seed1", "RPWf", 0.5, seed=0, raw="q" * 64)
    with pytest.raises(ValueError, match="file raw diversi"):
        _compare(tmp_path, runs)
    _report(pil, "p2", "B_seed1", "RPWf", 0.5, seed=0, raw=None)
    with pytest.raises(ValueError, match="SHA del file raw mancante"):
        _compare(tmp_path, runs)
    _report(pil, "p2", "B_seed1", "RPWf", 0.5, seed=0)
    path = pil / "p1" / "B_seed1" / "PWm.json"
    for field, value in (("arm", "B_seed0"), ("task", "p2"), ("set", "PWf")):
        rep = json.loads(path.read_text())
        path.write_text(json.dumps({**rep, field: value}))
        with pytest.raises(ValueError, match="altro protocollo, compito, teacher o insieme"):
            _compare(tmp_path, runs)
        path.write_text(json.dumps(rep))
    rep = json.loads(path.read_text())
    path.write_text(json.dumps({**rep, "extraction_recipe": {**rep["extraction_recipe"], "raw": {"protocol": "altro"}}}))
    with pytest.raises(ValueError, match="cache raw o split diversi"):
        _compare(tmp_path, runs)
    path.write_text(json.dumps(rep))
    dpath = diag / "p2" / "B_seed1" / "PW.json"
    drep = json.loads(dpath.read_text())
    dpath.write_text(json.dumps({**drep, "extraction_recipe": {**drep["extraction_recipe"], "branches": {"protocol": "altro"}}}))
    with pytest.raises(ValueError, match="2C"):
        _compare(tmp_path, runs)
    dpath.write_text(json.dumps(drep))
    other = _pilot_runs(tmp_path / "other", commit="0" * 40)
    with pytest.raises(ValueError, match="run e uscite"):
        _compare(tmp_path, other)
    bad = _pilot_runs(tmp_path / "bad")
    prov = bad / "PWf" / "B_seed0" / "run_provenance.json"
    prov.write_text(json.dumps({**json.loads(prov.read_text()), "checks_passed": False}))
    with pytest.raises(ValueError, match="non valido"):
        _compare(tmp_path, bad)
    missing = _pilot_runs(tmp_path / "missing")
    (missing / "PWh" / "B_seed1" / "run_provenance.json").unlink()  # un ibrido mancante e' un fallimento, non un descrittivo saltato
    with pytest.raises(ValueError, match="mancante"):
        _compare(tmp_path, missing)


def test_primary_reading_rule():
    row = lambda r, d: {"reading": r, "delta": d}  # noqa: E731
    assert BP.primary_reading({"p1": row("migliora", 0.02), "p2": row("entro le soglie", -0.005)})["positive"]
    assert not BP.primary_reading({"p1": row("migliora", 0.02), "p2": row("entro le soglie", -0.02)})["positive"]
    assert BP.primary_reading({"p1": row("entro le soglie", 0.0), "p2": row("migliora", 0.03)})["positive"]
    assert not BP.primary_reading({"p1": row("entro le soglie", 0.0), "p2": row("entro le soglie", 0.0)})["positive"]


# --- piano ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("queue", ["serial", "gpu"])
def test_plan_puts_the_primary_first_and_validates(tmp_path, queue):
    out, cmds = tmp_path / "piano.json", tmp_path / "comandi.txt"
    BP.main(["--stage", "plan", "--code-dir", "/leonardo_work/x/code", "--expected-sha", "a" * 40, "--pilot-dir", "/scratch/pilota",
             "--campaign", "pilota-t", "--out", str(out), "--commands", str(cmds), "--eval-queue", queue])
    plan = json.loads(out.read_text())
    ids = [n["id"] for n in plan["nodes"]]
    assert len(ids) == 2 + 16 + 32 + 2 and ids.index("RAMIPILOT-pilota-t-compare-primario") < ids.index("RAMIPILOT-pilota-t-PWh-B_seed0")
    first = plan["nodes"][:ids.index("RAMIPILOT-pilota-t-compare-primario")]
    assert {n["args"][7] for n in first if n["kind"] == "probe"} == set(BP.PRIMARY)
    probes = [n for n in plan["nodes"] if n["kind"] == "probe"]
    assert all(n["exports"] == {"PILOT_ROOT": "/scratch/pilota/cache"} and n["after"][0].endswith("-ext") for n in probes)
    full = plan["nodes"][-1]
    assert full["args"] == ["/scratch/pilota", "compare"] and BP.DIAG_COMPARE in full["after"] and len(full["after"]) == 33
    exts = [n for n in plan["nodes"] if n["kind"] == "extract" and "extract" in n["args"]]
    assert exts and all("--extract-device" not in n["args"] for n in exts)  # sempre in CPU, come il 2C
    assert all(n["gpu_hours"] == 0 for n in plan["nodes"]) == (queue == "serial")
    cpu_h = sum(BP._hours(next(o for o in n["sbatch"] if o.startswith("--time="))[7:]) * int(next(
        (o.split("=")[1] for o in n["sbatch"] if o.startswith("--cpus-per-task=")), 8)) for n in plan["nodes"])
    assert plan["estimate"]["cpu_core_hours_max"] == pytest.approx(cpu_h) == pytest.approx(1060, abs=1)
    lines = [line for line in cmds.read_text().splitlines() if line]
    assert len(lines) == 7 and "boost_qos_dbg" in lines[0] and lines[0].endswith("PWm B_seed0 collaudo")
    assert all("--qos=boost_qos_lprod --time=01:30:00" in line for line in lines[1:])


# --- estrazione e sonda: stesso codice del 2C e della diagnostica -----------------------------------------------------------------------

def test_extract_p2_writes_the_same_branch_outputs_as_the_2c_for_the_same_checkpoint(tmp_path, monkeypatch):
    """`extract` (P2, in CPU) sul checkpoint di un run del 2C sintetico, presentato come run del pilot: cache uguale bit per bit a quella
    dell'estrazione del 2C (`step2c.extract`, branches_trained), identita' e provenienza del pilot, accoppiata alla cache raw."""
    import test_step2c as TS

    args = TS._extract_setup(tmp_path, monkeypatch)
    S2C.extract(args)
    two_c = args.branch_runs / "B_seed0"
    pilot_runs = tmp_path / "pilot_runs"
    run = pilot_runs / "PWm" / "B_seed0"
    run.mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes((two_c / "checkpoint.pt").read_bytes())
    prov = {"protocol": BP.PROTOCOL, "source_commit": args.expected_commit, "arm": "PWm", "teacher": "B_seed0", "collaudo": False,
            "complete": True, "checks_passed": True, "source_checkpoint_sha256": BP.SD.CHECKPOINTS["B_seed0"],
            "checkpoint_sha256": sha256_file(run / "checkpoint.pt")}
    (run / "run_provenance.json").write_text(json.dumps(prov))
    pargs = SimpleNamespace(**{**vars(args), "arm": "PWm", "teacher": "B_seed0", "pilot_runs": pilot_runs, "eval_root": tmp_path / "pilot_cache"})
    BP.extract(pargs)
    folder = pargs.eval_root / "p2" / "B_seed0" / "branches_PWm"
    meta = json.loads((folder / "provenance.json").read_text())
    assert meta["identity"] == BP.cache_identity(args.expected_commit, "p2", "B_seed0", "PWm") and meta["run"] == prov
    for ds in BP.SD.DATASETS["p2"]:
        with np.load(folder / f"{ds}.npz") as a, np.load(args.eval_root / "p2" / "B_seed0" / "branches_trained" / f"{ds}.npz") as b:
            assert sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    with pytest.raises(ValueError, match="NEW"):
        BP.extract(pargs)  # cartella gia' scritta: niente sovrascritture
    (run / "run_provenance.json").write_text(json.dumps({**prov, "checks_passed": False}))
    with pytest.raises(ValueError, match="run non valido per il pilot"):
        BP.extract(SimpleNamespace(**{**vars(pargs), "eval_root": tmp_path / "other"}))


def test_the_probe_path_gives_the_same_report_as_the_diagnostic_code(tmp_path):
    """Il percorso della sonda in `spectral_diagnostic.py` (caricamento, colonne, `probe_set`) da' lo stesso S e le stesse predizioni del codice
    con cui la diagnostica ha prodotto PW0, PW, RPW0, RPW (aba1f2e): l'accoppiamento coi rapporti del pilot regge (seconda review del critico)."""
    import subprocess
    import test_spectral_diagnostic as TSD

    try:
        old_src = subprocess.check_output(["git", "show", "aba1f2e:scripts/spectral_diagnostic.py"], cwd=REPO, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("storia git non disponibile")
    old = type(sys)("spectral_diagnostic_aba1f2e")
    old.__file__ = str(REPO / "scripts" / "spectral_diagnostic.py")  # stesse radici e stessi moduli della sonda
    exec(compile(old_src, old.__file__, "exec"), old.__dict__)
    ev, dn = TSD.build(tmp_path, "p2", arms=("B_seed0",))
    reports = {}
    for name, mod in (("new", BP._script("spectral_diagnostic")), ("old", old)):
        out = tmp_path / name
        assert mod.main(["probe", "--task", "p2", "--arm", "B_seed0", "--sets", "RPW0", "PW", "--eval-root", str(ev), "--downstream-root",
                         str(dn), "--out-dir", str(out)]) == 0
        reports[name] = {s: json.loads((out / "p2" / "B_seed0" / f"{s}.json").read_text()) for s in ("RPW0", "PW")}
    for s in ("RPW0", "PW"):
        a, b = reports["new"][s], reports["old"][s]
        skip = {"oof", "oof_sha256", "elapsed_s"}
        assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
        with np.load(a["oof"]) as za, np.load(b["oof"]) as zb:
            assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
