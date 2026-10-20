"""Regressioni GPT sulla seconda tranche c21ca40: comportamento atteso.

WEARUSFM_REVIEW_SOURCE deve puntare al worktree da verificare, con il suo
src/ nel PYTHONPATH. Nessun accesso al cluster; fixture sintetiche ufficiali.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


SOURCE = Path(os.environ["WEARUSFM_REVIEW_SOURCE"])
sys.path.insert(0, str(SOURCE / "tests/cpu"))


def load_fixture(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tests/cpu" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_fixture("test_calibration_selection")
A = load_fixture("test_arms_plan")


def candidates(root):
    return [T._cand(root, 1e-4, .9, no_probe=True),
            T._cand(root, 3e-4, .7, seed=1),
            T._cand(root, 1e-3, .8, seed=2)]


def test_no_probe_without_terminal_failure_must_not_decide(tmp_path, monkeypatch):
    """File parziali sono compatibili con running/unknown, non una prova di fallimento."""
    monkeypatch.setattr(T.CS, "N_BOOT", 10)
    cands = candidates(tmp_path)
    (cands[0][1] / "summary.json").unlink()
    assert not (cands[0][1] / "STOP").exists()
    with pytest.raises(ValueError):
        T.CS.select("C1", cands, T._train_plan(cands))


def test_candidate_steps_must_match_frozen_plan(tmp_path, monkeypatch):
    """Il piano e il run hanno 2000 passi: un PASSI diverso non deve escludere il run sano."""
    monkeypatch.setattr(T.CS, "N_BOOT", 10)
    cands = candidates(tmp_path)
    plan = T._train_plan(cands)
    cands[0] = (*cands[0][:3], T.STEPS + 1)
    with pytest.raises(ValueError):
        T.CS.select("C1", cands, plan)


def test_arms_extraction_walltime_matches_effective_qos():
    """Usare uno SHA reale per leggere l'header, non lo SHA fittizio dei test del planner."""
    sha = os.environ.get("WEARUSFM_REVIEW_SHA", "c21ca409298ef2374d6a5fea927410fc070302fc")
    campaign = {**A.CAMP, "expected_sha": sha}
    result = A.AP.plan(A.REPORT, 3e-4, campaign)
    nodes = A.EC.validate(result["eval_plans"]["objective"])
    incompatible = []
    for node in nodes:
        if node["kind"] != "extract":
            continue
        opts = A.EC.sbatch_options(node["sbatch"])
        hours, minutes, seconds = map(int, opts["--time"].split(":"))
        if "boost_qos_dbg" in A.EC.node_labels(node) and hours * 3600 + minutes * 60 + seconds > 1800:
            incompatible.append((node["id"], opts["--time"], sorted(A.EC.node_labels(node))))
    assert not incompatible, incompatible
