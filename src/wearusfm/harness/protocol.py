"""Protocollo di valutazione: collega split per soggetto, normalizzazione train-only e
baseline (v10 §8). Regime "fine-tuning completo / da zero" - il regime "encoder
congelato + probe" richiede un encoder pre-addestrato (passo 6+, non ancora costruito) e
si aggiungera' come una seconda funzione con la stessa struttura, non qui.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearusfm.harness.baselines import HudginsLDABaseline
from wearusfm.harness.normalization import fit_train_only
from wearusfm.harness.splits import SubjectSplit, masks_from_split, split_subjects


@dataclass
class ProtocolResult:
    split: SubjectSplit
    train_balanced_accuracy: float
    val_balanced_accuracy: float
    test_balanced_accuracy: float
    n_train_windows: int
    n_val_windows: int
    n_test_windows: int


def run_hudgins_lda_protocol(
    windows: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    split: SubjectSplit | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
    normalize: bool = True,
) -> ProtocolResult:
    """windows: (N, T, C). labels: (N,). subject_ids: (N,), un subject_id per finestra.

    Split per soggetto (mai per finestra, v10 §8); se `split` non e' dato, se ne genera
    uno col protocollo 7:1:2. Normalizzazione stimata SOLO sul train (v10 §8) prima di
    estrarre le feature di Hudgins.
    """
    if split is None:
        split = split_subjects(subject_ids, ratios=ratios, seed=seed)
    masks = masks_from_split(subject_ids, split)

    train_w, train_y = windows[masks["train"]], labels[masks["train"]]
    val_w, val_y = windows[masks["val"]], labels[masks["val"]]
    test_w, test_y = windows[masks["test"]], labels[masks["test"]]

    if len(train_w) == 0:
        raise ValueError("split train vuoto: nessuna finestra assegnata")

    if normalize:
        stats = fit_train_only(train_w)
        train_w = stats.apply(train_w)
        val_w = stats.apply(val_w) if len(val_w) else val_w
        test_w = stats.apply(test_w) if len(test_w) else test_w

    model = HudginsLDABaseline().fit(train_w, train_y)

    train_acc = model.evaluate(train_w, train_y)
    val_acc = model.evaluate(val_w, val_y) if len(val_w) else float("nan")
    test_acc = model.evaluate(test_w, test_y) if len(test_w) else float("nan")

    return ProtocolResult(
        split=split,
        train_balanced_accuracy=train_acc,
        val_balanced_accuracy=val_acc,
        test_balanced_accuracy=test_acc,
        n_train_windows=len(train_w),
        n_val_windows=len(val_w),
        n_test_windows=len(test_w),
    )
