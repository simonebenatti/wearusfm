"""Pianificatore della calibrazione C1-C4 (`scripts/calibration_plan.py`, accordo del 15/10/2026): fasi bloccate senza le decisioni,
finalisti validati e mai scelti dal pianificatore, conferma fallita che ferma C3/C4, WSD col warmup di D e decay sul 20% di ogni endpoint,
passi con un solo floor dal rapporto dei manifest derivati, preventivo e piani validi per `wsd_chain.py`; `--lr` del comando di training."""

import importlib.util
import json
import math
import shlex
from pathlib import Path

import pytest

from wearusfm.training import run as R

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("calibration_plan", REPO / "scripts" / "calibration_plan.py")
CP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CP)

WPE = 784756.9792373349  # finestre per epoca del derivato al 100% (rapporto del 15/10)
SHA_100 = "6d6d8167b882eb4079fcc566748028dc6de81692859f220a0246291c198574ec"  # file del derivato al 100% costruito sul cluster
REPORT = {"fractions": {"1": {"windows_per_epoch": WPE, "hash": "h", "file_sha256": SHA_100, "file": "m_nested_1.json.gz"}},
          "code": {"sha": "11f868b8ad8a949381a5cbb8519733e7dbc538bf", "source": "launcher (lib_code_dir.sh: checkout di campagna pulito)"}}
DEC = {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [3e-4, 1e-3]}, "C3": {"proposed_300m_lr": 3e-4}}


def test_without_decisions_only_c1_is_planned():
    r = CP.plan(REPORT)
    assert [x["lr"] for x in r["runs"]] == [1e-4, 3e-4, 1e-3] and {x["seed"] for x in r["runs"]} == {0}
    assert set(r["blocked"]) == {"C2", "C3", "C4"} and all("finalisti di C1" in v for v in r["blocked"].values())
    assert r["dry_run"] is True


def test_full_plan_steps_warmup_decay_and_estimate():
    r = CP.plan(REPORT, DEC)
    steps_at_d = math.floor(4 * WPE / 32)
    assert r["steps_at_d"] == steps_at_d and r["warmup_steps"] == R.wsd_warmup_steps(steps_at_d) == 1962
    by = {x["id"]: x for x in r["runs"]}
    assert len(by) == 8 and r["blocked"] == {}
    for x in r["runs"]:
        assert x["steps"] == math.floor(CP.PHASES[x["phase"]]["epochs"] * WPE / 32)  # un solo floor
        assert x["warmup_steps"] == 1962 and x["decay_start"] == x["steps"] - round(0.2 * x["steps"])
    assert [x["seed"] for x in r["runs"] if x["phase"] == "C2"] == [1, 1]
    assert [x["lr"] for x in r["runs"] if x["phase"] == "C3"] == [3e-4, 1e-3] and [x["lr"] for x in r["runs"] if x["phase"] == "C4"] == [3e-4]
    assert {x["size"] for x in r["runs"]} == {"30M", "100M", "300M"}
    assert r["totals"]["gpu_hours_training_only"] == pytest.approx(152.1, abs=0.1)  # l'ordine di grandezza dell'accordo
    assert r["totals"]["gpu_hours_expected"] <= r["totals"]["gpu_hours_max"]


def test_lr_tags_are_lossless():
    r = CP.plan(REPORT, DEC)
    c4 = next(x for x in r["runs"] if x["phase"] == "C4")
    assert c4["id"] == "W1CAL-C4-300M-lr0p0003-s0" and CP._lr_tag(2.5e-4) == "0p00025" and CP._lr_tag(2.5e-4) != CP._lr_tag(2e-4)
    assert CP._lr_tag(1e-4) == "0p0001"


def test_failed_confirmation_stops_c3_and_c4_without_substitution():
    r = CP.plan(REPORT, {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [3e-4]}})
    assert {x["phase"] for x in r["runs"]} == {"C1", "C2"} and "nessuna sostituzione" in r["blocked"]["C3"]
    assert set(r["blocked"]) == {"C3", "C4"}


@pytest.mark.parametrize("bad", [{"C1": {"finalists": [3e-4]}}, {"C1": {"finalists": [3e-4, 3e-4]}}, {"C1": {"finalists": [3e-4, 5e-4]}},
                                 {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [1e-4]}},
                                 {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [3e-4, 1e-3]}, "C3": {"proposed_300m_lr": -1}},
                                 {**DEC, "C3": {"proposed_300m_lr": 2.5e-4}},  # nessuna sostituzione: uno dei due LR di C3
                                 {**DEC, "C4": {"lr": 1e-3, "passed": True}}, {**DEC, "C4": {"lr": 3e-4, "passed": "si"}}])
def test_malformed_decisions_are_refused(bad):
    with pytest.raises(ValueError):
        CP.plan(REPORT, bad)


def test_plans_feed_the_chain_planner_with_lr_and_the_derived_manifest(tmp_path):
    r = CP.plan(REPORT, DEC)
    W = CP._wsd_chain()
    for name, p in r["plans"].items():
        jobs = W.plan_jobs(p)
        assert jobs["totals"]["jobs"] == len(p["runs"])  # un job per run, nessuna scorta
        for cmd in jobs["commands"]:
            extra = next(t for t in shlex.split(cmd) if t.startswith("--export=")).split("EXTRA_ARGS=", 1)[1].split()
            assert "--lr" in extra and extra[extra.index("--nested") + 1] == "1" and "--prefetch" in extra
            assert extra[extra.index("--fixed-eval-every") + 1] == "500"  # il pannello del criterio di stabilita' A4
            assert extra[extra.index("--manifest-sha256") + 1] == SHA_100  # il training verifica il file del piano
    assert CP.main([str(_write(tmp_path / "rep.json", REPORT)), "--decisions", str(_write(tmp_path / "dec.json", DEC)), "--out",
                    str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["calibrazione_C1.json", "calibrazione_C2.json", "calibrazione_C3.json",
                                                                   "calibrazione_C4.json", "calibrazione_riepilogo.json", "valutazione_C1.json",
                                                                   "valutazione_C2.json", "valutazione_C3.json", "valutazione_C4.json"]


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_training_cli_accepts_the_learning_rate(tmp_path):
    import runpy
    import sys

    from test_pretraining_loader import _tree

    root, mpath = _tree(tmp_path)
    argv = sys.argv
    sys.argv = ["train_jepa.py", "--preset", "small", "--manifest", str(mpath), "--root", str(root), "--out-dir", str(tmp_path / "run"),
                "--device", "cpu", "--max-steps", "1", "--datasets", "all", "--lr", "0.001"]
    try:
        with pytest.raises(SystemExit) as e:
            runpy.run_path(str(REPO / "scripts" / "train_jepa.py"), run_name="__main__")
    finally:
        sys.argv = argv
    assert e.value.code in (0, None)
    assert json.loads((tmp_path / "run" / "config.json").read_text())["lr"] == 0.001


def test_stops_block_the_following_phases():
    r = CP.plan(REPORT, {"C1": {"stop": "meno di due candidati ammissibili"}})
    assert {x["phase"] for x in r["runs"]} == {"C1"} and all("C1 si e' fermata" in v for v in r["blocked"].values())
    r = CP.plan(REPORT, {**DEC, "C3": {"stop": "nessuno dei due LR ammissibile al 100M"}})
    assert {x["phase"] for x in r["runs"]} == {"C1", "C2", "C3"} and "C3 si e' fermata" in r["blocked"]["C4"]


def test_evaluation_plans_feed_the_evaluation_chain():
    EC = _load("evaluation_chain")
    camp = {"campaign": "w1cal", "agent": "claude", "code_dir": "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/w1cal", "expected_sha": "a" * 40,
            "manifest": "/M/m_nested_1.json.gz", "scales": "/M/s.json", "window_seconds": "/M/w.json", "run_root": "/leonardo_scratch/R",
            "eval": {"features_root": "/leonardo_scratch/F", "results_root": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/w1cal"}}
    r = CP.plan(REPORT, DEC, camp, "results/infra/w1cal")
    assert "eval" not in r["plans"]["C1"]  # il piano dei training resta quello di wsd_chain.py
    for name, e in r["eval_plans"].items():
        order = EC.validate(e)
        runs = [x for x in r["runs"] if x["phase"] == name]
        kinds = [n["kind"] for n in order]
        assert kinds.count("guard") == kinds.count("extract") == kinds.count("probe") == len(runs) and kinds[-1] == "compare"
        sel = order[-1]
        assert sorted(sel["after"]) == sorted(f"{x['id']}-probe" for x in runs) and sel["args"][0] == name
        assert sel["args"][3] == f"results/infra/w1cal/calibrazione_{name}.json"  # il piano di training per l'identita' dei candidati
        assert sel["args"][4::4] == [repr(float(x["lr"])) for x in runs] and sel["args"][7::4] == [str(x["steps"]) for x in runs]
        for g in (n for n in order if n["kind"] == "guard"):
            run = next(x for x in runs if g["args"][0].endswith(x["id"]))
            assert g["after"] == [f"{run['id']}-j01"] and g["args"][1] == "checkpoint_postdecay.pt" and str(run["steps"]) in g["args"]
        for n in order:
            assert ("boost_usr_prod" in EC.node_labels(n)) == (n["kind"] == "extract")  # solo le estrazioni sui nodi GPU
        ext = [n for n in order if n["kind"] == "extract"]
        assert e["estimate"]["gpu_hours_max"] == pytest.approx(sum(n["gpu_hours"] for n in ext))


def _load(name):
    s = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_the_plan_is_bound_to_the_cluster_build_report():
    """Rilievo 5 del critico (16/10): lo SHA del file viene dal rapporto della costruzione sul cluster; senza, o con un manifest che il
    rapporto non descrive, nessun piano."""
    bad = {**REPORT, "fractions": {"1": {**REPORT["fractions"]["1"], "file_sha256": None}}}
    with pytest.raises(ValueError, match="costruzione sul cluster"):
        CP.plan(bad)
    local = {**REPORT, "code": {"sha": "56cea75", "clean": True, "source": "git"}}  # come il rapporto della costruzione sul Mac
    with pytest.raises(ValueError, match="launcher sul cluster"):
        CP.plan(local)
    camp = {"campaign": "w1cal", "agent": "claude", "code_dir": "/C", "expected_sha": "a" * 40, "manifest": "/M/altro.json.gz",
            "scales": "/M/s.json", "window_seconds": "/M/w.json", "run_root": "/R"}
    with pytest.raises(ValueError, match="il rapporto descrive"):
        CP.plan(REPORT, None, camp)


def test_runs_stopped_before_the_probe_enter_the_selection_without_evaluation_nodes():
    """Review di GPT (16/10): un training fermato non blocca la selezione della fase; niente guardia, estrazione e sonda per lui, rapporto
    «-» nella selezione, che dipende solo dalle sonde degli altri. Codice della valutazione da un checkout diverso da quello dei training."""
    EC = _load("evaluation_chain")
    camp = {"campaign": "w1cal", "agent": "claude", "code_dir": "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/w1cal", "expected_sha": "a" * 40,
            "manifest": "/M/m_nested_1.json.gz", "scales": "/M/s.json", "window_seconds": "/M/w.json", "run_root": "/leonardo_scratch/R",
            "eval": {"features_root": "/leonardo_scratch/F", "results_root": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/w1cal",
                     "code_dir": "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/w1cal-eval", "expected_sha": "b" * 40}}
    failed = "W1CAL-C1-30M-lr0p0001-s0"
    r = CP.plan(REPORT, None, camp, "results/infra/w1cal", {failed: "STOP"})  # R1 (GPT 3272398): ogni run escluso ha la sua prova
    e = r["eval_plans"]["C1"]
    order = EC.validate(e)
    assert not any(n["id"].startswith(failed) for n in order if n["kind"] != "compare")
    sel = order[-1]
    assert sel["args"][4:8] == [repr(1e-4), f"/leonardo_scratch/R/{failed}", "-", str(r["runs"][0]["steps"])]
    assert sorted(sel["after"]) == ["W1CAL-C1-30M-lr0p0003-s0-probe", "W1CAL-C1-30M-lr0p001-s0-probe"]
    assert {n["expected_sha"] for n in order} == {"b" * 40} and e["guard"]["expected_sha"] == "b" * 40
    assert r["plans"]["C1"]["expected_sha"] == "a" * 40  # i training restano allo SHA del loro checkout
    with pytest.raises(ValueError, match="nessun run arrivato alla sonda"):
        CP.plan(REPORT, None, camp, "results/infra/w1cal", {x["id"]: "STOP" for x in r["runs"]})
    with pytest.raises(ValueError, match="non sono run della fase"):
        CP.plan(REPORT, None, camp, "results/infra/w1cal", {"W1CAL-C9-altro": "STOP"})
    # un'evidenza da sacct (relativa al repo): verificata dal pianificatore, poi passata alla selezione come «failed:<percorso>»
    CS = _load("calibration_select")
    ev_rel = "tests/cpu/_tmp_failure_evidence.json"
    ev = CP.ROOT / ev_rel
    try:
        ev.write_text(json.dumps({"protocol": CS.EVIDENCE_PROTOCOL, "run_id": failed, "source": "sacct",
                                  "jobs": [{"logical_id": f"{failed}-j01", "job_id": "1", "state": "FAILED", "exit_code": "1:0"}]}))
        r2 = CP.plan(REPORT, None, camp, "results/infra/w1cal", {failed: ev_rel})
        assert EC.validate(r2["eval_plans"]["C1"])[-1]["args"][6] == f"failed:{ev_rel}"
        ev.write_text(json.dumps({"protocol": CS.EVIDENCE_PROTOCOL, "run_id": failed, "source": "sacct",
                                  "jobs": [{"logical_id": f"{failed}-j01", "job_id": "1", "state": "TIMEOUT", "exit_code": "0:0"}]}))
        with pytest.raises(ValueError, match="non provano un fallimento terminale"):
            CP.plan(REPORT, None, camp, "results/infra/w1cal", {failed: ev_rel})
    finally:
        ev.unlink(missing_ok=True)
    assert CP._failed_arg([f"{failed}=STOP"]) == {failed: "STOP"}
    with pytest.raises(SystemExit, match="RUN=STOP"):
        CP._failed_arg([failed])
