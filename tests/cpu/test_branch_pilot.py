"""Pilot dei rami (`scripts/branch_pilot.py`, protocollo `rami_pilota_v1`): configurazione del 2C salvo il braccio, controlli del run (passo 0
caricato dal 2C, strati fissi, perdita del passo 1), ibrido PWh, verdetto del collaudo, sonde del pilot nella diagnostica, confronto con la
lettura congelata e piano della catena."""

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


@pytest.fixture
def no_hash(monkeypatch):
    """`with_spectral_keep` legge l'hash della calibrazione del cluster: qui un valore fisso."""
    monkeypatch.setattr(R, "sha256_file", lambda p: "c" * 64)


# --- configurazione ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("teacher", BP.TEACHERS)
def test_each_arm_differs_from_the_2c_config_only_where_the_protocol_says(no_hash, teacher):
    ref = R.config_to_dict(S2C.build_config(teacher, CAL))
    for arm, spec in BP.ARMS.items():
        cfg = BP.build_config(arm, teacher, CAL)
        assert (cfg.lr, cfg.model.branch_freeze_output, cfg.model.branch_objective) == (spec["lr"], spec["freeze_output"], spec["objective"])
        assert cfg.fixed_eval_every == 0 and cfg.max_steps == S2C.STEPS and cfg.seed == int(teacher[-1])
        check = BP.config_check(R.config_to_dict(cfg), ref, arm)
        assert check["ok"] and {k for k in check["differences"]} <= {".".join(k) for k in BP.ALLOWED[arm]}
    assert set(BP.config_check(R.config_to_dict(BP.build_config("PWl", teacher, CAL)), ref, "PWl")["differences"]) == {"lr", "branch_monitor"}


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


# --- controlli del run, sul trainer vero (modello piccolo) -------------------------------------------------------------------------------

@pytest.fixture
def runs(tmp_path):
    """Un run come il 2C (obiettivo vecchio, nessuno strato fisso) e i bracci PWf e PWm dal suo passo 0."""
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    ref = tmp_path / "2c"
    R.train(replace(cfg_b, max_steps=2), manifest, [root], ref, log=lambda _: None, init_from=src)
    cfgs = {"PWf": replace(cfg_b, max_steps=2, model=replace(cfg_b.model, branch_freeze_output=True)),
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
        fixed = BP.fixed_check(BP.trainable_names(cfg), ref_init, final)
        assert fixed["identical"] and fixed["student_tokens_changed"] and fixed["teacher_tokens_changed"]
        dist = BP.token_distance(ref_init["teacher"], final["teacher"])
        assert dist["all"] > 0 and set(dist) == {"all", *S2C.KINDS}
        assert BP.config_check(R.config_to_dict(cfg), R.config_to_dict(runs.cfg_b) | {"max_steps": 2}, arm)["ok"]
    step1 = BP.step1_check(BP._records(runs.tmp / "PWf" / "metrics.jsonl"), BP._records(runs.ref / "metrics.jsonl"))
    assert step1["ok"] and step1["run"] == pytest.approx(step1["step2c"], rel=1e-6)


def test_a_moved_fixed_layer_a_different_step_0_or_step_1_loss_fail(runs):
    ref_init = BP._load(runs.ref / "checkpoint_init.pt")
    final = BP._load(runs.tmp / "PWm" / "checkpoint.pt")
    key = next(k for k in final["teacher"] if k.startswith("spectral_branches.") and ".combine." in k)
    final["teacher"][key] = final["teacher"][key] + 1e-6
    out = BP.fixed_check(BP.trainable_names(runs.cfgs["PWm"]), ref_init, final)
    assert out["student"] and not out["teacher"] and not out["identical"]
    own = BP._load(runs.tmp / "PWf" / "checkpoint_init.pt")
    head = next(k for k in own["student"] if k.startswith("branch_heads."))
    own["student"][head] = own["student"][head] + 1
    assert not BP.init_check(ref_init, own)["identical"]
    metrics = BP._records(runs.tmp / "PWf" / "metrics.jsonl")
    metrics[0]["keep_branches"] *= 1 + 2e-4
    assert not BP.step1_check(metrics, BP._records(runs.ref / "metrics.jsonl"))["ok"]
    assert not BP.step1_check([], BP._records(runs.ref / "metrics.jsonl"))["ok"]


def test_train_one_writes_the_provenance_with_every_check(runs, monkeypatch):
    """`_train_one` col trainer vero: configurazione del 2C, controlli del raw, del passo 0, degli strati fissi e (PWf) del passo 1."""
    files = {k: runs.tmp / f"{k}.bin" for k in S2C.INPUTS if k != "manifest"}  # il manifest e' quello vero della calibrazione del test
    for f in files.values():
        f.write_bytes(f.name.encode())
    files["manifest"] = runs.manifest
    (runs.ref / "run_provenance.json").write_text("{}")
    monkeypatch.setattr(S2C, "INPUT_SHA256", {k: sha256_file(f) for k, f in files.items()})
    monkeypatch.setattr(BP.SD, "CHECKPOINTS", {**BP.SD.CHECKPOINTS, "B_seed0": sha256_file(runs.src)})
    monkeypatch.setattr(BP.SD, "check_source", lambda c: None)
    monkeypatch.setattr(BP, "STEP2C_RUNS", {"B_seed0": {"checkpoint_sha256": sha256_file(runs.ref / "checkpoint.pt"),
                                                        "init_checkpoint_sha256": sha256_file(runs.ref / "checkpoint_init.pt")}})
    monkeypatch.setattr(BP, "build_config", lambda arm, teacher, cal, max_steps=None: replace(runs.cfgs[arm], max_steps=max_steps or 2))
    args = SimpleNamespace(**files, root=[runs.root], device="cpu", num_workers=0, expected_commit="e" * 40, collaudo=False)
    for arm in ("PWf", "PWm"):
        prov, summary = BP._train_one(args, arm, "B_seed0", runs.tmp / f"run_{arm}", max_steps=None, time_limit_s=600, source=runs.src,
                                      run2c=runs.ref, scales=None, window_s=None)
        assert prov["complete"] and prov["checks_passed"], prov["checks_ok"]
        assert set(prov["checks_ok"]) == {"config", "raw", "init", "fixed", "branch_state"} | ({"step1"} if arm == "PWf" else set())
        assert json.loads((runs.tmp / f"run_{arm}" / "run_provenance.json").read_text())["checks_passed"]
        assert summary["spectral_branches"]["collapse_monitor"] is not None
    monkeypatch.setattr(BP, "build_config", lambda arm, teacher, cal, max_steps=None: replace(runs.cfgs[arm], batch_size=4, max_steps=2))
    with pytest.raises(ValueError, match="configurazione diversa dal 2C"):
        BP._train_one(args, "PWf", "B_seed0", runs.tmp / "bad", max_steps=None, time_limit_s=600, source=runs.src, run2c=runs.ref,
                      scales=None, window_s=None)


def test_hybrid_takes_tokens_from_pw_and_final_layers_from_pw0(runs):
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


# --- apprendimento e collaudo -----------------------------------------------------------------------------------------------------------

def _metrics(n=300, norm=0.5, targets=3, step_s=0.8):
    keys = BP.branch_keys()
    return [{"step": i + 1, "t_step_s": step_s, "masked": 1.0 - 0.001 * i, "masked_normalized": norm,
             **{f"masked_{k}_targets": targets for k in keys}, **{f"masked_{k}_context_per_band": [1, 2] for k in keys}}
            for i in range(n)]


def test_learned_needs_100_defined_values_at_most_0_9():
    assert BP.learned(_metrics())["learned"]
    assert not BP.learned(_metrics(norm=0.95))["learned"] and BP.learned(_metrics(norm=0.95))["verifiable"]
    m = _metrics()
    m[-3]["masked_normalized"] = None
    assert BP.learned(m) == {"verifiable": False, "updates": 100, "undefined": 1, "mean": None, "learned": False}
    assert not BP.learned(_metrics(n=50))["verifiable"]


def _summary(n=300, step_s=0.8, verified=True):
    return {"elapsed_s": n * step_s + 120, "spectral_branches": {"collapse_monitor": {"verified": verified, "alarm": None, "checks": 20,
                                                                                    "reference_undefined": []}}}


def test_collaudo_verdict_passes_and_each_criterion_fails():
    ok = {"no_stop": True, "PWm_fixed": True}
    v = BP.collaudo_verdict(_metrics(), _summary(), ok)
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
    for m in models:
        src = ev / task / arm / "branches_trained"
        dst = pilot / task / arm / f"branches_{m}"
        dst.mkdir(parents=True)
        meta = json.loads((src / "provenance.json").read_text())
        for ds in SDG.DATASETS[task]:
            (dst / f"{ds}.npz").write_bytes((src / f"{ds}.npz").read_bytes())
        prov = {"protocol": BP.PROTOCOL, "source_commit": "f" * 40, "arm": m, "teacher": arm, "collaudo": False, "complete": True,
                "checks_passed": True, **run}
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
    assert rep["extraction_recipe"]["branches"] == {"protocol": BP.PROTOCOL, "source_commit": "f" * 40}
    assert all(v["features"]["columns"] == 1024 for v in rep["datasets"].values())
    assert json.loads((out / "p2" / "B_seed0" / "RPW0.json").read_text())["protocol"] == "rami_diagnostica_v1"
    with pytest.raises(ValueError, match="pilot-root"):
        TSD._probe(ev, dn, out, "p2", "B_seed0", ["PWm"])


def test_a_pilot_cache_from_an_incomplete_run_or_another_code_is_refused(tmp_path):
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


# --- confronto ----------------------------------------------------------------------------------------------------------------------------

DATASETS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}


def _report(folder: Path, task: str, teacher: str, name: str, quality: float, seed: int, commit="f" * 40):
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
        datasets[d] = {"input_sha256": "i" * 64, "oof_rows_sha256": fingerprint(row), "s": float((pred == y).mean())}
    path = folder / task / teacher / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".oof.npz"), datasets=np.asarray(list(DATASETS[task])), **oof)
    w = {d: 1 / len(DATASETS[task]) for d in DATASETS[task]}
    rep = {"protocol": SDG.PROTOCOL if diag else BP.PROTOCOL, "task": task, "arm": teacher, "set": name, "k_folds": 4, "fold_seed": 1,
           "val_frac": 0.2, "probe_seed": 0, "window_s": 1.0 if task == "p1" else None, "splits_sha256": "s" * 64,
           "feature_set": "cache_2C" if diag else "cache_pilota",
           "extraction_recipe": {"branches": SDG.STEP2C if diag else {"protocol": BP.PROTOCOL, "source_commit": commit}, "raw": SDG.DOWNSTREAM,
                                 "split_sha256": SDG.SPLIT_SHA},
           "weights": w, "datasets": datasets, "S": 0.0, "oof": str(path.with_suffix(".oof.npz")),
           "oof_sha256": sha256_file(path.with_suffix(".oof.npz"))}
    path.write_text(json.dumps(rep))


def _pilot_runs(tmp_path, norm=0.5, verified=True, commit="f" * 40):
    runs = tmp_path / "runs"
    for arm in BP.MODELS:
        for t in BP.TEACHERS:
            d = runs / arm / t
            d.mkdir(parents=True)
            (d / "run_provenance.json").write_text(json.dumps({
                "protocol": BP.PROTOCOL, "arm": arm, "teacher": t, "complete": True, "checks_passed": True, "source_commit": commit,
                "checkpoint_sha256": "k" * 64, "checks": {"token_distance": {"teacher": {"all": 0.1}}}}))
            if arm != "PWh":
                (d / "summary.json").write_text(json.dumps(_summary(verified=verified)))
                (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _metrics(norm=norm)))
    return runs


def _compare(tmp_path, quality, runs, primary_only=False):
    diag, pil = tmp_path / "diag", tmp_path / "sonde"
    for k, task in enumerate(DATASETS):
        for j, t in enumerate(BP.TEACHERS):
            for i, name in enumerate(BP.DIAG_SETS + BP.PILOT_SETS):
                _report(diag if name in BP.DIAG_SETS else pil, task, t, name, quality.get(name, 0.5), seed=100 * k + 10 * j + i)
    out = tmp_path / "esito.json"
    BP.main(["--stage", "compare", "--diagnostic-dir", str(diag), "--reports-dir", str(pil), "--pilot-runs", str(runs), "--out", str(out)]
            + (["--primary-only"] if primary_only else []))
    return json.loads(out.read_text())


def test_primary_is_positive_when_pwm_beats_pwf_and_is_read_only_if_pwm_learned(tmp_path):
    res = _compare(tmp_path / "a", {"RPWf": 0.4, "RPWm": 0.8}, _pilot_runs(tmp_path / "a"))
    assert res["primary"]["readable"] and res["primary"]["reading"]["positive"]
    assert res["primary"]["reading"]["readings"] == {"p1": "migliora", "p2": "migliora"}
    assert set(res["descriptive"]) == set(BP.DESCRIPTIVE) and res["runs"]["PWm"]["B_seed0"]["learned"]["learned"]
    worse = _compare(tmp_path / "b", {"RPWf": 0.8, "RPWm": 0.4}, _pilot_runs(tmp_path / "b"))
    assert not worse["primary"]["reading"]["positive"] and worse["primary"]["reading"]["readings"]["p1"] == "peggiora"
    lazy = _compare(tmp_path / "c", {"RPWf": 0.4, "RPWm": 0.8}, _pilot_runs(tmp_path / "c", norm=0.95))
    assert not lazy["primary"]["readable"] and lazy["primary"]["reading"].startswith("non si legge")
    unverified = _compare(tmp_path / "d", {"RPWf": 0.4, "RPWm": 0.8}, _pilot_runs(tmp_path / "d", verified=False))
    assert not unverified["primary"]["readable"] and not unverified["gate"]["PWf_not_collapsed"]
    primary = _compare(tmp_path / "e", {"RPWf": 0.4, "RPWm": 0.8}, _pilot_runs(tmp_path / "e"), primary_only=True)
    assert "descriptive" not in primary and primary["primary"]["reading"]["positive"]


def test_primary_reading_rule():
    row = lambda r, d: {"reading": r, "delta": d}  # noqa: E731
    assert BP.primary_reading({"p1": row("migliora", 0.02), "p2": row("entro le soglie", -0.005)})["positive"]
    assert not BP.primary_reading({"p1": row("migliora", 0.02), "p2": row("entro le soglie", -0.02)})["positive"]
    assert BP.primary_reading({"p1": row("entro le soglie", 0.0), "p2": row("migliora", 0.03)})["positive"]
    assert not BP.primary_reading({"p1": row("entro le soglie", 0.0), "p2": row("entro le soglie", 0.0)})["positive"]


def test_compare_refuses_reports_of_other_sources_or_codes(tmp_path):
    runs = _pilot_runs(tmp_path)
    diag, pil = tmp_path / "diag", tmp_path / "sonde"
    for task in DATASETS:
        for t in BP.TEACHERS:
            for i, name in enumerate(BP.DIAG_SETS + BP.PILOT_SETS):
                _report(diag if name in BP.DIAG_SETS else pil, task, t, name, 0.5, seed=i, commit="0" * 40 if name == "RPWl" else "f" * 40)
    args = ["--stage", "compare", "--diagnostic-dir", str(diag), "--reports-dir", str(pil), "--pilot-runs", str(runs), "--out",
            str(tmp_path / "o.json")]
    with pytest.raises(ValueError, match="codici diversi"):
        BP.main(args)
    _report(pil, "p1", "B_seed0", "RPWl", 0.5, seed=0)
    path = diag / "p2" / "B_seed1" / "PW.json"
    rep = json.loads(path.read_text())
    path.write_text(json.dumps({**rep, "extraction_recipe": {**rep["extraction_recipe"], "branches": {"protocol": "altro"}}}))
    with pytest.raises(ValueError, match="2C"):
        BP.main(args)


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
    assert all(("--extract-device" in n["args"]) == (queue == "gpu") for n in exts)
    assert all(n["gpu_hours"] == 0 for n in plan["nodes"]) == (queue == "serial")
    lines = [line for line in cmds.read_text().splitlines() if line]
    assert len(lines) == 7 and "boost_qos_dbg" in lines[0] and lines[0].endswith("PWm B_seed0 collaudo")
    assert all("--qos=boost_qos_lprod --time=01:30:00" in line for line in lines[1:])
