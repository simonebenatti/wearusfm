"""Difetto dimostrato (review di Claude, 16/11/2026): tradeoff simmetrico letto come vittoria di CN per ordine del codice.

CN meglio di CR di 0,6 pp su P1; CR meglio di CN di 0,6 pp su P2; entrambi -5% di E_dyn contro C0. La regola del
paragrafo 6 e' soddisfatta nei due versi (contro C0 e nel confronto diretto): il protocollo non prevede un vincitore.
xfail(strict): quando il comparatore distinguera' il caso, il test passera' e l'xfail andra' tolto.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_compare as C


def build(p1, p2):
    cl, po = {}, {}
    y = np.tile(np.repeat(np.array([0, 1]), 1000), 4)
    for a in C.ARMS:
        cl[a] = {}
        for task, dss in C.TASKS.items():
            k = (p1 if task == "p1" else p2)[a]
            pred = np.where(np.tile(np.arange(1000), 8) < k, y, 1 - y)
            for ds in dss:
                cl[a][ds] = {"row": np.arange(8000), "y": y.copy(), "pred": pred.copy(),
                             "s": np.repeat(np.array(["a", "b", "c", "d"]), 2000), "fold": np.repeat(np.arange(4), 2000)}
        rows = {f"u{i}": {"n_windows": 2, "n_window_joints": 40, "n_valid_elements": 400,
                          "sums": {"vary": 10., "centered": 8. if a == "C0" else 7.6}} for i in range(4)}
        po[a] = {"complete": True, "arm": a, "checkpoint_step": 19618, "which": "teacher",
                 "checkpoint_sha256": hashlib.sha256(a.encode()).hexdigest(), "fit": {"selection": {"boundary_unresolved": False}},
                 "analysis": {"support_sha256": "s" * 64, "targets_sha256": "t" * 64, "dynamic_threshold": .05,
                              "groups": {"dynamic": {"n_selected_windows": 8, "centered": {"per_user": rows}}}}}
    return cl, po


def audits():
    return [{"arm": a, "task": t, "dataset": ds, "fold": f, "fit_phase": ph, "C": c, "n_iter": 5, "max_iter": 12000, "exception": None,
             "checkpoint_sha256": hashlib.sha256(a.encode()).hexdigest(), "convergence_warnings": 0, "converged_by_criterion": True}
            for a in C.ARMS for t, dss in C.TASKS.items() for ds in dss for f in range(4)
            for ph, grid in (("candidate", C.C_GRID), ("refit", (1.,))) for c in grid]


@pytest.mark.xfail(strict=True, reason="tradeoff simmetrico etichettato CN_supported_for_confirmation per ordine dei rami if/elif")
@pytest.mark.parametrize("swap", [False, True])
def test_symmetric_tradeoff_has_no_winner(swap):
    better_p1, better_p2 = ("CR", "CN") if swap else ("CN", "CR")
    p1 = {a: 706 if a == better_p1 else 700 for a in C.ARMS}
    p2 = {a: 706 if a == better_p2 else 700 for a in C.ARMS}
    r = C.compare(*build(p1, p2), audit_records=audits())
    assert r["contrasts"]["CN_minus_CR"]["useful_without_material_loss"] and r["reverse_head_to_head"]["useful_without_material_loss"]
    assert r["reading"] not in ("CN_supported_for_confirmation", "CR_supported_for_confirmation")
