"""Baseline non negoziabili del protocollo (v10 §8): LDA su feature di Hudgins e' la
prima, quella con cui ogni FM va confrontato prima ancora di guardare i transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from wearusfm.harness.features_hudgins import extract_hudgins_features_batch


@dataclass
class HudginsLDABaseline:
    """LDA su feature di Hudgins. Lo scaler delle feature (non la normalizzazione del
    segnale grezzo, v10 §8 - quella e' un passo separato, vedi normalization.py) si
    adatta anch'esso solo sul train, per lo stesso motivo."""

    zc_threshold: float = 0.0
    ssc_threshold: float = 0.0
    _scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    _clf: LinearDiscriminantAnalysis = field(default_factory=LinearDiscriminantAnalysis, repr=False)
    _fitted: bool = False

    def _features(self, windows: np.ndarray) -> np.ndarray:
        return extract_hudgins_features_batch(windows, zc_threshold=self.zc_threshold, ssc_threshold=self.ssc_threshold)

    def fit(self, train_windows: np.ndarray, train_labels: np.ndarray) -> "HudginsLDABaseline":
        features = self._features(train_windows)
        features = self._scaler.fit_transform(features)
        self._clf.fit(features, train_labels)
        self._fitted = True
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() non ancora chiamato")
        features = self._features(windows)
        features = self._scaler.transform(features)
        return self._clf.predict(features)

    def evaluate(self, windows: np.ndarray, labels: np.ndarray) -> float:
        """Balanced accuracy (v10 §11 riporta BAcc per i gesti discreti)."""
        preds = self.predict(windows)
        return float(balanced_accuracy_score(labels, preds))
