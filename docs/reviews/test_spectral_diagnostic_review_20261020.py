"""Regressioni indipendenti GPT sulla consegna 6086df2.

Eseguire con WEARUSFM_REVIEW_SOURCE al checkout da verificare.
Le asserzioni descrivono il comportamento atteso, non i difetti osservati.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = Path(os.environ["WEARUSFM_REVIEW_SOURCE"])
spec = importlib.util.spec_from_file_location("spectral_diagnostic_reviewed", SOURCE / "scripts/spectral_diagnostic.py")
SD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SD)


def test_constant_branch_does_not_abort_diagnostic_report(tmp_path, monkeypatch):
    """Una diagnostica accessoria non deve far fallire le sonde gia' scritte quando i rami collassano."""
    rng = np.random.default_rng(123)
    subjects = [f"s{i:02d}" for i in range(12)]
    s = np.repeat(subjects, 8)
    data = {"epn612": {"raw": rng.normal(size=(len(s), 5)), "init": rng.normal(size=(len(s), 3)),
                        "trained": np.zeros((len(s), 3)), "s": s, "subjects": subjects}}
    monkeypatch.setattr(SD, "load", lambda *args: data)
    args = SimpleNamespace(task="p2", arm="B_seed0", sets=[], redundancy=True, seed=0,
                           eval_root=tmp_path / "cache", downstream_root=tmp_path / "raw", out_dir=tmp_path / "out")
    assert SD.probe(args) == 0
    report = json.loads((args.out_dir / "p2/B_seed0/redundancy.json").read_text())
    assert report["effective_rank"]["trained"]["epn612"]["effective_rank"] == 0
    # Nessun R2 numerico fittizio: null o un oggetto con stato esplicito sono accettabili.
    r2 = report["datasets"]["epn612"]["r2_PW_from_R"]
    assert r2 is None or isinstance(r2, dict)


def test_distinct_campaigns_do_not_reuse_global_job_ids(tmp_path):
    """Cambiare campagna/cartella non deve ereditare lo stato sottomesso della campagna precedente."""
    plans = []
    for campaign in ("rami-first", "rami-second"):
        path = tmp_path / f"{campaign}.json"
        SD.main(["plan", "--code-dir", "/campaign-code", "--expected-sha", "a" * 40,
                 "--out-dir", f"/scratch/{campaign}", "--campaign", campaign, "--out", str(path)])
        plans.append(json.loads(path.read_text()))
    submitted = {n["id"] for n in plans[0]["nodes"]}
    read_state = lambda i: {"stato": "sottomesso", "job_slurm": "123"} if i in submitted else None
    rows = SD._script("evaluation_chain").schedule(plans[1], read_state, {"lrd_all_serial": 0})
    assert not any(r["status"] == "gia' sottomesso" for r in rows)
    assert sum(r["status"] == "sottomesso ora" for r in rows) == 2
