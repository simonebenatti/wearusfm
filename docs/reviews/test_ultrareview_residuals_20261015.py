"""Regressioni della review GPT di d5f2a1f / 5d7ced6 (solo CPU, cluster simulato).

Archiviate in Git il 21/10/2026: file storico gia' letto da Claude il 15/10,
rimasto non tracciato. Le aspettative qui sotto riguardano il candidato citato,
non attestano difetti ancora aperti sulla versione corrente.

WEARUSFM_REVIEW_SOURCE indica il checkout/snapshot di Claude da verificare.
Sul candidato 5d7ced6 questi sette casi DEVONO fallire: riproducono i residui,
non un fallimento della suite ordinaria. Nessun training o accesso a SLURM.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ["WEARUSFM_REVIEW_SOURCE"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "cpu"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("review_evaluation_helpers", ROOT / "tests/cpu/test_evaluation_chain.py")
W = _load("review_wsd_helpers", ROOT / "tests/cpu/test_wsd_chain.py")


@pytest.mark.parametrize("extra", [
    "--preset 300m", "--out-dir /R/different", "--batch 64",
    "--decay-frac 0.1", "--branch-fro /R/other/checkpoint.pt",
])
def test_free_args_cannot_change_planned_training(extra):
    plan = W._plan()
    plan["runs"] = plan["runs"][:1]
    plan["runs"][0]["args"] = extra
    with pytest.raises(ValueError):
        W.W.plan_jobs(plan)


def test_queue_cap_counts_both_partition_and_qos(tmp_path):
    serial = E._node("serial-probe", "probe", queue="lrd_all_serial")
    gpu = E._node("gpu-extract", "extract", queue="normal")
    serial["sbatch"] = ["--partition=lrd_all_serial", "--qos=normal"]
    gpu["sbatch"] = ["--partition=boost_usr_prod", "--qos=normal"]
    plan = {"schema": "eval-plan-v0", "campaign": "repro", "agent": "claude",
            "limits": {"normal": 1}, "nodes": [serial, gpu]}
    ssh, state = E._cluster(tmp_path)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    assert E.EC.main([str(p), "--submit", "--registro", str(tmp_path / "jobs"), "--ssh", ssh]) == 0
    calls = E._sbatch_calls(state)
    assert all("--qos=normal" in c for c in calls)
    assert len(calls) <= 1, f"cap normal=1, ma sottomessi {len(calls)} job normal"


def test_sbatch_cannot_override_generated_exports():
    node = E._node("extract", "extract")
    node["sbatch"] = ["--export=ALL,CODE_DIR=/OTHER,EXPECTED_SHA=" + "b" * 40]
    plan = {"schema": "eval-plan-v0", "campaign": "repro", "agent": "claude", "nodes": [node]}
    with pytest.raises(ValueError):
        E.EC.validate(plan)
