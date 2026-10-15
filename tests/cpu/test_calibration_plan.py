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
REPORT = {"fractions": {"1": {"windows_per_epoch": WPE, "hash": "h", "file_sha256": "f", "file": "m_nested_1.json.gz"}}}
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
            "manifest": "/M/m.json.gz", "scales": "/M/s.json", "window_seconds": "/M/w.json", "run_root": "/leonardo_scratch/R",
            "eval": {"features_root": "/leonardo_scratch/F", "results_root": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/w1cal"}}
    r = CP.plan(REPORT, DEC, camp)
    assert "eval" not in r["plans"]["C1"]  # il piano dei training resta quello di wsd_chain.py
    for name, e in r["eval_plans"].items():
        order = EC.validate(e)
        runs = [x for x in r["runs"] if x["phase"] == name]
        kinds = [n["kind"] for n in order]
        assert kinds.count("guard") == kinds.count("extract") == kinds.count("probe") == len(runs) and kinds[-1] == "compare"
        sel = order[-1]
        assert sorted(sel["after"]) == sorted(f"{x['id']}-probe" for x in runs) and sel["args"][0] == name
        assert sel["args"][3::4] == [repr(float(x["lr"])) for x in runs] and sel["args"][6::4] == [str(x["steps"]) for x in runs]
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
