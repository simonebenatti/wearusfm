"""Finite, train-only selection rules for the frozen pose readouts.

This module selects on pooled per-window MSE only. Dynamic error diagnoses
the schedule and the MSE/dynamics tradeoff; it never selects another model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

INITIAL_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
LOWER_LR = 3e-5
UPPER_LR = 1e-2
INITIAL_EPOCHS = 150
MAX_EPOCHS = 300
LOWER_EDGE_RELATIVE_TOLERANCE = 0.005


class NonDiagnosticError(ValueError):
    """A score has no supported observations or no dynamic denominator."""


@dataclass(frozen=True)
class FoldScore:
    mse_sum: float
    window_count: int
    dynamic_sse: float
    dynamic_target_ss: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FoldScore":
        count = value["window_count"]
        if isinstance(count, bool) or int(count) != count:
            raise ValueError("window_count must be an integer")
        return cls(float(value["mse_sum"]), int(count),
                   float(value["dynamic_sse"]), float(value["dynamic_target_ss"]))

    def validate(self) -> None:
        if isinstance(self.window_count, bool) or int(self.window_count) != self.window_count:
            raise ValueError("window_count must be an integer")
        if self.window_count <= 0:
            raise NonDiagnosticError("no supported windows in an inner held fold")
        for value in (self.mse_sum, self.dynamic_sse, self.dynamic_target_ss):
            if not math.isfinite(value) or value < 0:
                raise ValueError("score accumulators must be finite and nonnegative")


@dataclass(frozen=True)
class CurvePoint:
    exposure: float
    mse: float
    edyn: float
    fold_steps: tuple[int, ...] = ()

    def validate(self) -> None:
        if not math.isfinite(self.exposure) or self.exposure < 0:
            raise ValueError("invalid exposure")
        if not all(math.isfinite(x) and x >= 0 for x in (self.mse, self.edyn)):
            raise ValueError("nonfinite/negative selection metric")
        if any(isinstance(s, bool) or int(s) != s or s < 0 for s in self.fold_steps):
            raise ValueError("invalid optimizer-step count")


def pool_scores(scores: Sequence[FoldScore | Mapping[str, object]], exposure: float,
                fold_steps: Sequence[int] = ()) -> CurvePoint:
    """Pool sums, not fold ratios; every supported window has equal MSE weight."""
    parsed = [s if isinstance(s, FoldScore) else FoldScore.from_mapping(s) for s in scores]
    if not parsed:
        raise NonDiagnosticError("no fold scores")
    for score in parsed:
        score.validate()
    denominator = math.fsum(s.dynamic_target_ss for s in parsed)
    if denominator <= 0:
        raise NonDiagnosticError("zero pooled dynamic target sum of squares")
    point = CurvePoint(float(exposure),
                       math.fsum(s.mse_sum for s in parsed) / sum(s.window_count for s in parsed),
                       math.fsum(s.dynamic_sse for s in parsed) / denominator,
                       tuple(int(x) for x in fold_steps))
    point.validate()
    return point


@dataclass(frozen=True)
class CandidateCurve:
    lr: float
    horizon: int
    points: tuple[CurvePoint, ...]

    def validate(self) -> None:
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("invalid learning rate")
        if self.horizon <= 0 or int(self.horizon) != self.horizon:
            raise ValueError("invalid horizon")
        if not self.points or self.points[0].exposure != 0:
            raise ValueError("curve must retain the diagnostic step-zero point")
        previous = -1.0
        for point in self.points:
            point.validate()
            if point.exposure <= previous or point.exposure > self.horizon:
                raise ValueError("curve exposures must be strictly ordered within horizon")
            previous = point.exposure
        if self.points[-1].exposure != self.horizon:
            raise ValueError("incomplete candidate curve")
        if not any(p.exposure > 0 for p in self.points):
            raise ValueError("curve has no selectable checkpoint")

    def best(self, metric: str = "mse") -> CurvePoint:
        self.validate()
        if metric not in ("mse", "edyn"):
            raise ValueError("unsupported metric")
        return min((p for p in self.points if p.exposure > 0),
                   key=lambda p: (getattr(p, metric), p.exposure))


@dataclass(frozen=True)
class Selected:
    lr: float
    exposure: float
    mse: float
    edyn: float


def choose(candidates: Sequence[CandidateCurve]) -> Selected:
    if not candidates or len({c.lr for c in candidates}) != len(candidates):
        raise ValueError("candidate learning rates must be nonempty and unique")
    options = [(c, c.best("mse")) for c in candidates]
    candidate, point = min(options, key=lambda cp: (cp[1].mse, cp[1].exposure, cp[0].lr))
    return Selected(candidate.lr, point.exposure, point.mse, point.edyn)


def needs_epoch_extension(curve: CandidateCurve, initial_epochs: int = INITIAL_EPOCHS,
                          last_epochs: int = 5) -> bool:
    curve.validate()
    if curve.horizon != initial_epochs:
        raise ValueError("extension can be decided only at the initial horizon")
    boundary = max(1, initial_epochs - last_epochs + 1)
    return any(curve.best(metric).exposure >= boundary for metric in ("mse", "edyn"))


def next_learning_rate(candidates: Sequence[CandidateCurve]) -> float | None:
    """One addition at each initial-grid edge; never search beyond the two pins."""
    active = {c.lr for c in candidates}
    legal = set(INITIAL_LRS) | {LOWER_LR, UPPER_LR}
    if not set(INITIAL_LRS) <= active or not active <= legal:
        raise ValueError("all four initial learning rates must be complete")
    selected = choose(candidates)
    if selected.lr == min(active) and LOWER_LR not in active:
        return LOWER_LR
    if selected.lr == max(active) and UPPER_LR not in active:
        return UPPER_LR
    return None


def assessment(candidates: Sequence[CandidateCurve], *, maximum_epochs: int = MAX_EPOCHS,
               last_epochs: int = 5) -> dict[str, object]:
    """Report optimizer alarms and S-2 separately; neither changes the winner."""
    selected = choose(candidates)
    chosen = next(c for c in candidates if c.lr == selected.lr)
    best_dynamic = chosen.best("edyn")
    alarms: list[str] = []
    minima = {metric: chosen.best(metric).exposure for metric in ("mse", "edyn")}
    if chosen.horizon == maximum_epochs:
        boundary = max(1, maximum_epochs - last_epochs + 1)
        if any(epoch >= boundary for epoch in minima.values()):
            alarms.append("minimum_in_last_five_maximum_epochs")
    lower_gain = None
    if selected.lr == LOWER_LR:
        neighbor = min((c for c in candidates if c.lr > LOWER_LR), key=lambda c: c.lr)
        neighbor_mse = neighbor.best().mse
        lower_gain = 0.0 if neighbor_mse == 0 else (neighbor_mse - selected.mse) / neighbor_mse
        if lower_gain > LOWER_EDGE_RELATIVE_TOLERANCE:
            alarms.append("lower_lr_edge_material_gain")
    if selected.lr == UPPER_LR:
        alarms.append("upper_lr_edge_after_extension")
    if selected.exposure == 0.25:
        alarms.append("selected_first_quarter_epoch")
    distance = abs(selected.exposure - best_dynamic.exposure)
    infinite = best_dynamic.edyn == 0 and selected.edyn > 0
    regret = (0.0 if selected.edyn == 0 else None) if best_dynamic.edyn == 0 else (
        selected.edyn - best_dynamic.edyn) / best_dynamic.edyn
    tradeoff = distance > 5 and (infinite or (regret is not None and regret >= 0.02))
    return {
        "selected": asdict(selected), "optimizer_alarms": alarms,
        "optimization_unresolved": bool(alarms), "minimum_exposures": minima,
        "lower_edge_relative_mse_gain": lower_gain,
        "objective_tradeoff": {
            "flag": tradeoff, "selected": asdict(selected),
            "dynamic_minimum": asdict(best_dynamic), "exposure_distance": distance,
            "dynamic_relative_regret": regret, "regret_infinite": infinite,
            "is_optimizer_alarm": False,
        },
    }


def checkpoint_schedule(steps_per_fold: Sequence[int], end_epoch: int,
                        *, after_exposure: float = -1) -> list[tuple[float, tuple[int, ...]]]:
    """Synchronize by exposure, deduplicating identical pairs of optimizer steps.

    When a very small fixture maps several fractions to the same step pair,
    retain its greatest exposure (especially epoch one). This cannot create an
    artificial first-quarter alarm for a state that is already a complete epoch.
    """
    if end_epoch < 1 or any(int(s) != s or s < 1 for s in steps_per_fold) or not steps_per_fold:
        raise ValueError("invalid checkpoint schedule")
    exposures = [0.0, 0.25, 0.5, 0.75] + [float(e) for e in range(1, end_epoch + 1)]
    by_steps: dict[tuple[int, ...], float] = {}
    for exposure in exposures:
        steps = tuple(math.ceil(exposure * s) for s in steps_per_fold)
        by_steps[steps] = exposure
    return [(exposure, steps) for steps, exposure in by_steps.items() if exposure > after_exposure]


def curve_from_dict(value: Mapping[str, object]) -> CandidateCurve:
    if isinstance(value["horizon"], bool) or int(value["horizon"]) != value["horizon"]:
        raise ValueError("invalid horizon")
    points = []
    for record in value["points"]:
        item = dict(record)
        item["fold_steps"] = tuple(item.get("fold_steps", ()))
        points.append(CurvePoint(**item))
    curve = CandidateCurve(float(value["lr"]), int(value["horizon"]), tuple(points))
    curve.validate()
    return curve
