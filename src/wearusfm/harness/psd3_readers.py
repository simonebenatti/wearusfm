"""Experimental PSD3 readers, independent of the historical probe implementation.

The caller owns subject-disjoint folds and held-out scoring. This module never
accepts held-out labels. Preprocessing is fitted once on train and shared by the
readers; neither reader rescales its inputs. Fixed MLP epochs are a compute budget,
not a convergence criterion. No global NumPy or torch RNG state is consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral, Real
from time import perf_counter
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from wearusfm.harness.downstream_ab import fingerprint


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _real(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or
            not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"invalid {name}")


def _grid(values, name, *, positive=False):
    if not isinstance(values, tuple) or not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be a nonempty tuple of distinct numbers")
    for value in values:
        _real(value, name, positive=positive)


@dataclass(frozen=True)
class FeatureConfig:
    k: int = 128
    seed: int = 20261029
    n_iter: int = 7
    n_oversamples: int = 10

    def __post_init__(self):
        for name in ("k", "n_oversamples"):
            _integer(getattr(self, name), name)
        for name in ("seed", "n_iter"):
            _integer(getattr(self, name), name, minimum=0)
        for name in ("k", "seed", "n_iter", "n_oversamples"):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True)
class LogisticConfig:
    feature_dim: int = 128
    c_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    max_iter: int = 12000
    tol: float = 1e-4
    seed: int = 0

    def __post_init__(self):
        _integer(self.feature_dim, "feature_dim")
        _integer(self.max_iter, "max_iter")
        _integer(self.seed, "seed", minimum=0)
        _real(self.tol, "tol", positive=True)
        _grid(self.c_grid, "c_grid", positive=True)
        for name in ("feature_dim", "max_iter", "seed"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "tol", float(self.tol))
        object.__setattr__(self, "c_grid", tuple(float(c) for c in self.c_grid))


@dataclass(frozen=True)
class MLPConfig:
    feature_dim: int = 128
    hidden_dim: int = 32
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    lambdas: tuple[float, ...] = (0.0, 1e-3, 1e-2)
    seeds: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        for name in ("feature_dim", "hidden_dim", "epochs", "batch_size"):
            _integer(getattr(self, name), name)
        _real(self.lr, "lr", positive=True)
        _grid(self.lambdas, "lambdas")
        if (not isinstance(self.seeds, tuple) or not self.seeds or
                len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be a nonempty tuple of distinct integers")
        for seed in self.seeds:
            _integer(seed, "seed", minimum=0)
        for name in ("feature_dim", "hidden_dim", "epochs", "batch_size"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "lambdas", tuple(float(value) for value in self.lambdas))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))


@dataclass(frozen=True)
class PreparedFeatures:
    train: np.ndarray
    val: np.ndarray
    held: np.ndarray
    metadata: dict


def _features(*arrays, feature_dim=None):
    result = []
    for role, value in zip(("train", "val", "held"), arrays):
        x = np.asarray(value)
        if x.ndim != 2 or not all(x.shape) or x.dtype.kind not in "biuf":
            raise ValueError(f"{role} features must be a nonempty real matrix")
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite {role} features")
        result.append(x)
    if len({x.shape[1] for x in result}) != 1:
        raise ValueError("feature widths differ between roles")
    if feature_dim is not None and result[0].shape[1] != feature_dim:
        raise ValueError(f"expected exactly {feature_dim} input features")
    return result


def prepare_features(x_train, x_val, x_held, *, config=FeatureConfig()):
    """StandardScaler and randomized PCA, both train-only, with no dimension cap.

    PCA arithmetic is float64; its coordinates are converted once to float32 for
    both readers. The transform fingerprint depends only on fitted parameters.
    """
    arrays = _features(x_train, x_val, x_held)
    if len(arrays[0]) <= config.k or arrays[0].shape[1] < config.k:
        raise ValueError("PCA requires n_train > k and input width >= k; no automatic cap")
    scaler = StandardScaler().fit(np.asarray(arrays[0], dtype=np.float64))
    scaled = [scaler.transform(np.asarray(x, dtype=np.float64)) for x in arrays]
    pca = PCA(n_components=config.k, svd_solver="randomized", whiten=False,
              iterated_power=config.n_iter, n_oversamples=config.n_oversamples,
              random_state=config.seed).fit(scaled[0])
    # Use transform for every role (including train), not a distinct fit_transform path.
    transformed = [np.asarray(pca.transform(x), dtype=np.float32) for x in scaled]
    if not all(np.isfinite(x).all() for x in transformed):
        raise FloatingPointError("nonfinite PCA coordinates")
    ratio = pca.explained_variance_ratio_
    if not np.isfinite(ratio).all():
        raise ValueError("PCA explained variance is undefined")
    metadata = {
        "config": asdict(config), "fit_role": "train", "whiten": False,
        "input_shapes": [list(x.shape) for x in arrays],
        "input_dtypes": [str(x.dtype) for x in arrays],
        "fit_dtype": "float64", "output_dtype": "float32", "second_scaler": False,
        "train_input_fingerprint": fingerprint(arrays[0]),
        "scaler_fingerprint": fingerprint(scaler.mean_, scaler.scale_, scaler.var_),
        "pca_fingerprint": fingerprint(pca.components_, pca.mean_, pca.explained_variance_),
        "transform_fingerprint": fingerprint(scaler.mean_, scaler.scale_, scaler.var_,
                                             pca.components_, pca.mean_, pca.explained_variance_),
        "variance_explained": float(ratio.sum()), "explained_variance_ratio": ratio.tolist(),
    }
    return PreparedFeatures(*transformed, metadata)


def _labels(value, n, role):
    y = np.asarray(value)
    if y.ndim != 1 or len(y) != n or y.dtype.kind not in "biufUS":
        raise ValueError(f"invalid {role} classification labels")
    if y.dtype.kind in "US":
        y = np.char.decode(y, "utf-8") if y.dtype.kind == "S" else y
        if np.any(np.char.str_len(np.char.strip(y)) == 0):
            raise ValueError(f"blank {role} labels")
    elif not np.isfinite(y).all():
        raise ValueError(f"nonfinite {role} labels")
    return y


def _reader_inputs(x_train, y_train, x_val, y_val, x_held, feature_dim):
    xt, xv, xh = _features(x_train, x_val, x_held, feature_dim=feature_dim)
    yt, yv = _labels(y_train, len(xt), "train"), _labels(y_val, len(xv), "val")
    if (yt.dtype.kind == "U") != (yv.dtype.kind == "U"):
        raise ValueError("incompatible train/val label representations")
    classes, train_codes = np.unique(yt, return_inverse=True)
    if len(classes) < 2 or not np.isin(yv, classes).all():
        raise ValueError("need two train classes and val labels drawn from train classes")
    return xt, train_codes, xv, np.searchsorted(classes, yv), xh, classes


def _warnings(records):
    return [{"category": w.category.__name__, "message": str(w.message)} for w in records]


def _score(y, pred, records, role):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = float(balanced_accuracy_score(y, pred))
    records.extend({"role": role, **w} for w in _warnings(caught))
    if not np.isfinite(score):
        raise FloatingPointError(f"nonfinite {role} balanced accuracy")
    return score


def _failed(exc, row, audit, started, caught):
    row.update(status="failed", duration_s=perf_counter() - started,
               warnings=_warnings(caught), exception=f"{type(exc).__name__}: {exc}")
    audit.append(row)
    # Preserve partial diagnostics without disguising the exception as completion.
    exc.psd3_audit = audit


def fit_logistic(x_train, y_train, x_val, y_val, x_held, *, config=LogisticConfig()):
    """Six validation candidates, then a fresh fit on train only (seven by default)."""
    xt, yt, xv, yv, xh, classes = _reader_inputs(
        x_train, y_train, x_val, y_val, x_held, config.feature_dim)
    audit, metric_warnings, candidates = [], [], []
    train_hash = fingerprint(xt, yt, classes)

    def fit(c, phase):
        model = LogisticRegression(C=float(c), max_iter=config.max_iter, tol=config.tol,
                                   solver="lbfgs", penalty="l2", class_weight="balanced",
                                   random_state=config.seed, warm_start=False)
        row = {"phase": phase, "C": float(c), "seed": config.seed,
               "train_fingerprint": train_hash, "n_train": len(xt),
               "n_features": xt.shape[1], "classes": classes.tolist(),
               "parameters": model.get_params()}
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.fit(xt, yt)
                if not (np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()):
                    raise FloatingPointError("nonfinite fitted logistic parameters")
            except Exception as exc:
                _failed(exc, row, audit, started, caught)
                raise
        row.update(status="completed", duration_s=perf_counter() - started,
                   warnings=_warnings(caught), n_iter=np.asarray(model.n_iter_).tolist(),
                   reached_iteration_limit=bool(np.any(model.n_iter_ >= config.max_iter)))
        audit.append(row)
        return model

    for c in config.c_grid:
        model = fit(c, "candidate")
        score = _score(yv, model.predict(xv), metric_warnings, "val")
        audit[-1]["val_bacc"] = score
        candidates.append({"C": float(c), "val_bacc": score})
    best = max(candidates, key=lambda r: (r["val_bacc"], -r["C"]))
    model = fit(best["C"], "refit")
    train_score = _score(yt, model.predict(xt), metric_warnings, "train")
    val_score = _score(yv, model.predict(xv), metric_warnings, "val_refit")
    pred = classes[model.predict(xh)]
    return {
        "reader": "logistic", "status": "completed", "config": asdict(config),
        "classes": classes.tolist(), "selected_hyperparameter": best["C"],
        "selection_metric": "val_balanced_accuracy", "tie_break": "smaller_C",
        "selected_val_score": best["val_bacc"], "candidates": candidates,
        "scores": {str(config.seed): {"train": train_score, "val": val_score}},
        "predictions": {str(config.seed): pred.tolist()}, "audit": audit,
        "metric_warnings": metric_warnings, "expected_fits": len(config.c_grid) + 1,
    }


class _MLP(nn.Module):
    """Linear/ReLU/linear with local-generator PyTorch Linear initialization."""

    def __init__(self, in_dim, hidden_dim, n_classes, seed):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight1 = nn.Parameter(torch.empty(hidden_dim, in_dim, dtype=torch.float32, device="cpu"))
        self.bias1 = nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32, device="cpu"))
        self.weight2 = nn.Parameter(torch.empty(n_classes, hidden_dim, dtype=torch.float32, device="cpu"))
        self.bias2 = nn.Parameter(torch.empty(n_classes, dtype=torch.float32, device="cpu"))
        for weight, bias, fan_in in ((self.weight1, self.bias1, in_dim),
                                     (self.weight2, self.bias2, hidden_dim)):
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(weight, -bound, bound, generator=generator)
            nn.init.uniform_(bias, -bound, bound, generator=generator)

    def forward(self, x):
        return F.linear(F.relu(F.linear(x, self.weight1, self.bias1)), self.weight2, self.bias2)


def _class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("class weights require positive train support for every class")
    return len(y) / (n_classes * counts.astype(np.float64))


def _objective(model, logits, targets, weights, regularization):
    ce = (F.cross_entropy(logits, targets, reduction="none") * weights[targets]).mean()
    penalty = 0.5 * regularization * (model.weight1.square().sum() + model.weight2.square().sum())
    return ce + penalty, ce, penalty


def _predict_mlp(model, x, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite MLP prediction logits")
            predictions.append(logits.argmax(dim=1).numpy())
    return np.concatenate(predictions)


def fit_mlp(x_train, y_train, x_val, y_val, x_held, *, config=MLPConfig()):
    """Select lambda by mean val BAcc over seeds; retain its models, with no refit.

    Predictions are separate per seed, never an ensemble. Every candidate starts
    with the same initialization and epoch permutations for its seed. Adam has
    weight_decay=0: the explicit objective alone supplies L2 on the two weights.
    """
    xt, yt, xv, yv, xh, classes = _reader_inputs(
        x_train, y_train, x_val, y_val, x_held, config.feature_dim)
    tensor_arrays = [torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in (xt, xv, xh)]
    if not all(torch.isfinite(x).all() for x in tensor_arrays):
        raise ValueError("features exceed the finite float32 MLP range")
    tx, vx, hx = tensor_arrays
    targets = torch.from_numpy(yt.astype(np.int64))
    class_weights = _class_weights(yt, len(classes))
    weights = torch.tensor(class_weights, dtype=torch.float32, device="cpu")
    train_hash = fingerprint(xt, yt, classes)
    candidates, audit, metric_warnings = [], [], []
    best_models, best_key = None, None
    for regularization in config.lambdas:
        models, scores = {}, {}
        for seed in config.seeds:
            started = perf_counter()
            row = {"phase": "candidate", "lambda": float(regularization), "seed": seed,
                   "train_fingerprint": train_hash, "n_train": len(xt),
                   "n_features": xt.shape[1], "classes": classes.tolist(),
                   "class_weights": class_weights.tolist(), "epochs": [],
                   "parameters": {**asdict(config), "weight_decay": 0.0,
                                  "optimizer": "Adam", "device": "cpu", "dtype": "float32"}}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model = _MLP(config.feature_dim, config.hidden_dim, len(classes), seed)
                    row["parameter_count"] = sum(p.numel() for p in model.parameters())
                    row["initialization_fingerprint"] = fingerprint(
                        *(p.detach().numpy() for p in model.parameters()))
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
                    row["optimizer_defaults"] = dict(optimizer.defaults)
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    order_hash = hashlib.sha256()
                    for epoch in range(config.epochs):
                        epoch_started = perf_counter()
                        model.train()
                        permutation = torch.randperm(len(tx), generator=generator, device="cpu")
                        order_hash.update(permutation.numpy().tobytes())
                        totals = np.zeros(3, dtype=np.float64)
                        for start in range(0, len(tx), config.batch_size):
                            indices = permutation[start:start + config.batch_size]
                            optimizer.zero_grad(set_to_none=True)
                            losses = _objective(model, model(tx[indices]), targets[indices], weights, regularization)
                            if not all(torch.isfinite(loss).item() for loss in losses):
                                raise FloatingPointError("nonfinite MLP training loss")
                            losses[0].backward()
                            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise FloatingPointError("nonfinite MLP gradient")
                            optimizer.step()
                            if not all(torch.isfinite(p).all() for p in model.parameters()):
                                raise FloatingPointError("nonfinite MLP parameters")
                            totals += np.array([loss.detach().item() for loss in losses]) * len(indices)
                        row["epochs"].append({
                            "epoch": epoch + 1, "train_loss": float(totals[0] / len(tx)),
                            "weighted_ce": float(totals[1] / len(tx)),
                            "l2_penalty": float(totals[2] / len(tx)),
                            "duration_s": perf_counter() - epoch_started,
                        })
                    row["permutation_fingerprint"] = order_hash.hexdigest()
                except Exception as exc:
                    _failed(exc, row, audit, started, caught)
                    raise
            row.update(status="completed", duration_s=perf_counter() - started,
                       warnings=_warnings(caught), n_iter=config.epochs,
                       stopping_reason="fixed_epoch_budget", convergence_claim=False,
                       optimizer_steps=config.epochs * math.ceil(len(tx) / config.batch_size))
            audit.append(row)
            scores[str(seed)] = {
                "train": _score(yt, _predict_mlp(model, tx, config.batch_size), metric_warnings, "train"),
                "val": _score(yv, _predict_mlp(model, vx, config.batch_size), metric_warnings, "val"),
            }
            row["scores"] = scores[str(seed)]
            models[seed] = model
        val_score = float(np.mean([s["val"] for s in scores.values()]))
        candidates.append({"lambda": float(regularization), "val_bacc": val_score, "scores": scores})
        key = (val_score, regularization)
        if best_key is None or key > best_key:
            best_key, best_models = key, models
    best = max(candidates, key=lambda c: (c["val_bacc"], c["lambda"]))
    # First and only held-out inference happens after all candidates are scored.
    predictions = {str(seed): classes[_predict_mlp(best_models[seed], hx, config.batch_size)].tolist()
                   for seed in config.seeds}
    return {
        "reader": "mlp", "status": "completed", "config": asdict(config),
        "classes": classes.tolist(), "selected_hyperparameter": best["lambda"],
        "selection_metric": "mean_seed_val_balanced_accuracy", "tie_break": "larger_lambda",
        "selected_val_score": best["val_bacc"], "candidates": candidates,
        "scores": best["scores"], "predictions": predictions, "audit": audit,
        "metric_warnings": metric_warnings, "expected_fits": len(config.lambdas) * len(config.seeds),
        "prediction_policy": "separate_seeds_no_ensemble_no_refit",
        "stopping_reason": "fixed_epoch_budget_not_convergence",
    }
