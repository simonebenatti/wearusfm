"""Split per soggetto (v10 §8): niente split casuale sulle finestre, sempre per soggetto,
altrimenti finestre della stessa sessione finiscono sia in train che in test.

Split ufficiali dei dataset hanno sempre la precedenza su uno split generato qui (v10 §8,
"Baseline non negoziabili": split per soggetto 7:1:2 e' quello usato da NeuroRVQ per
la ri-valutazione, non necessariamente lo split ufficiale del dataset originale).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        all_ids = [*self.train, *self.val, *self.test]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("un soggetto compare in piu' di uno split: leakage cross-split")


def split_subjects(
    subject_ids: Sequence[str],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> SubjectSplit:
    """Split deterministico per soggetto. `ratios` = (train, val, test), somma 1.0.

    Non e' lo split ufficiale di un dataset: usare questo solo quando un dataset non ne
    ha uno pubblicato, o per riprodurre esplicitamente il protocollo 7:1:2 di NeuroRVQ
    (v10 §8, §11) da confrontare - mai per generare il manifest finale a caso.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"le ratio devono sommare a 1.0, non {sum(ratios)}")
    unique_ids = sorted(set(subject_ids))
    if len(unique_ids) < 3:
        raise ValueError(f"servono almeno 3 soggetti distinti per uno split 3-vie, ne ho {len(unique_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1]))
    n_train = min(n_train, n - 2)  # lascia almeno 1 per val e 1 per test
    n_val = min(n_val, n - n_train - 1)

    train = tuple(shuffled[:n_train])
    val = tuple(shuffled[n_train : n_train + n_val])
    test = tuple(shuffled[n_train + n_val :])
    return SubjectSplit(train=train, val=val, test=test)


def masks_from_split(sample_subject_ids: Sequence[str], split: SubjectSplit) -> dict[str, np.ndarray]:
    """Maschere booleane per assegnare campioni (finestre) al loro split, dato il
    subject_id di ciascun campione."""
    ids = np.asarray(sample_subject_ids)
    train_set, val_set, test_set = set(split.train), set(split.val), set(split.test)
    return {
        "train": np.isin(ids, list(train_set)),
        "val": np.isin(ids, list(val_set)),
        "test": np.isin(ids, list(test_set)),
    }


def check_no_cross_dataset_subject_overlap(
    subject_ids_by_dataset: dict[str, Sequence[str]],
) -> list[tuple[str, str, str]]:
    """v10 §2.1: "la sovrapposizione di soggetti fra i DB NinaPro va controllata
    all'ingest". Ritorna la lista (dataset_a, dataset_b, subject_id) per ogni
    sovrapposizione trovata - non solleva eccezioni, la sovrapposizione puo' essere
    legittima (stesso soggetto reale in DB diversi) ma va SEMPRE dichiarata, mai
    scoperta dopo aver congelato gli split.
    """
    names = list(subject_ids_by_dataset)
    overlaps: list[tuple[str, str, str]] = []
    for i, a in enumerate(names):
        set_a = set(subject_ids_by_dataset[a])
        for b in names[i + 1 :]:
            set_b = set(subject_ids_by_dataset[b])
            for shared in sorted(set_a & set_b):
                overlaps.append((a, b, shared))
    return overlaps
