"""Pianificatore della calibrazione C1-C4 (`scripts/calibration_plan.py`, accordo del 15/10/2026): fasi bloccate senza le decisioni di GPT,
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
    r = CP.plan(REPORT, {**DEC, "C3": {"proposed_300m_lr": 2.5e-4}})
    c4 = next(x for x in r["runs"] if x["phase"] == "C4")
    assert c4["id"] == "W1CAL-C4-300M-lr0p00025-s0" and CP._lr_tag(2.5e-4) != CP._lr_tag(2e-4) and CP._lr_tag(1e-4) == "0p0001"


def test_failed_confirmation_stops_c3_and_c4_without_substitution():
    r = CP.plan(REPORT, {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [3e-4]}})
    assert {x["phase"] for x in r["runs"]} == {"C1", "C2"} and "nessuna sostituzione" in r["blocked"]["C3"]
    assert set(r["blocked"]) == {"C3", "C4"}


@pytest.mark.parametrize("bad", [{"C1": {"finalists": [3e-4]}}, {"C1": {"finalists": [3e-4, 3e-4]}}, {"C1": {"finalists": [3e-4, 5e-4]}},
                                 {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [1e-4]}},
                                 {"C1": {"finalists": [3e-4, 1e-3]}, "C2": {"confirmed": [3e-4, 1e-3]}, "C3": {"proposed_300m_lr": -1}}])
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
    assert CP.main([str(_write(tmp_path / "rep.json", REPORT)), "--decisions", str(_write(tmp_path / "dec.json", DEC)), "--out",
                    str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["calibrazione_C1.json", "calibrazione_C2.json", "calibrazione_C3.json",
                                                                   "calibrazione_C4.json", "calibrazione_riepilogo.json"]


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
