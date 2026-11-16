"""Test aggiuntivi del verificatore avversario (16/11/2026). Solo dati sintetici.

1) La scala RELATIVA delle repliche di E_dyn: nessun test consegnato ne' dei 53 aggiuntivi del revisore la protegge
   (nei test del revisore il braccio di controllo vale 1,0, dove differenza assoluta x100 e riduzione relativa coincidono).
2) Scala relativa anche nel verso inverso e trasformazione monotona dei limiti.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_compare as C


def pose_inputs():
    rng = np.random.default_rng(5)
    users = [f"u{i:02d}" for i in range(15)]
    vary = np.exp(rng.normal(5, .6, 15))
    ratio = .84 + rng.normal(0, .05, 15)
    po = {}
    for k, arm in enumerate(C.ARMS):
        r = ratio * (1 - .03 * k) * np.exp(rng.normal(0, .01, 15))
        rows = {u: {"n_windows": 3, "n_window_joints": 60, "n_valid_elements": 600,
                    "sums": {"vary": float(vary[i]), "centered": float(vary[i] * r[i])}} for i, u in enumerate(users)}
        po[arm] = {"complete": True, "arm": arm, "checkpoint_step": 19618, "which": "teacher",
                   "checkpoint_sha256": hashlib.sha256(arm.encode()).hexdigest(), "fit": {"selection": {"boundary_unresolved": False}},
                   "analysis": {"support_sha256": "s" * 64, "targets_sha256": "t" * 64, "dynamic_threshold": .05,
                                "groups": {"dynamic": {"n_selected_windows": 45, "centered": {"per_user": rows}}}}}
    return po, users, vary


def test_edyn_replicates_are_relative_reductions_not_absolute_differences():
    po, users, vary = pose_inputs()
    point, values, _ = C.pose_statistics(po, 19618, 4000, 3)
    assert .7 < point[0] < .95 and np.ptp(values[:, 0]) > 0          # controllo lontano da 1 e varianza non nulla
    zeros = {t: np.zeros(3) for t in C.TASKS}
    zs = {t: np.zeros((4000, 3)) for t in C.TASKS}
    pair = C.summarize_pair("CN", "C0", {"E_dyn": point, **zeros}, {"E_dyn": values, **zs})["E_dyn"]
    relative = 100 * (1 - values[:, 1] / values[:, 0])
    absolute = 100 * (values[:, 0] - values[:, 1])
    q = .05 / 18
    assert pair["benefit"] == pytest.approx(100 * (1 - point[1] / point[0]), abs=1e-12)
    assert pair["ci_simultaneous"] == pytest.approx(np.quantile(relative, [q, 1 - q]).tolist(), abs=1e-12)
    assert pair["ci95"] == pytest.approx(np.quantile(relative, [.025, .975]).tolist(), abs=1e-12)
    # la prova distingue davvero le due scale
    assert abs(np.quantile(absolute, q) - np.quantile(relative, q)) > 1e-3


def test_edyn_reverse_reading_is_monotone_transform_of_same_draws():
    po, users, vary = pose_inputs()
    point, values, _ = C.pose_statistics(po, 19618, 4000, 3)
    zeros = {t: np.zeros(3) for t in C.TASKS}
    zs = {t: np.zeros((4000, 3)) for t in C.TASKS}
    fwd = C.summarize_pair("CN", "CR", {"E_dyn": point, **zeros}, {"E_dyn": values, **zs})["E_dyn"]
    rev = C.summarize_pair("CR", "CN", {"E_dyn": point, **zeros}, {"E_dyn": values, **zs})["E_dyn"]
    t = lambda x: 100 * (1 - 1 / (1 - x / 100))
    assert rev["benefit"] == pytest.approx(t(fwd["benefit"]), abs=1e-10)
    # limiti scambiati e trasformati; l'interpolazione del quantile lascia uno scarto di ordine 1e-6
    assert rev["ci_simultaneous"][0] == pytest.approx(t(fwd["ci_simultaneous"][1]), abs=1e-4)
    assert rev["ci_simultaneous"][1] == pytest.approx(t(fwd["ci_simultaneous"][0]), abs=1e-4)
