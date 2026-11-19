"""Bounded readout training with paired fold checkpoints and isolated RNG.

Only the public ``run_search`` / ``run_refits`` entry points are production
entry points. They use the fixed recipe. Small private recipes exist solely
for CPU fixtures and remain marked in every state they produce.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from . import heads, metrics, selection


@dataclass(frozen=True)
class Recipe:
    initial_epochs: int = 150
    maximum_epochs: int = 300
    last_epochs: int = 5
    batch_size: int = 32
    clip_norm: float = 1.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    test_only: bool = False


@dataclass(frozen=True)
class _TestRecipe(Recipe):
    initial_epochs: int = 3
    maximum_epochs: int = 6
    last_epochs: int = 1
    batch_size: int = 4
    test_only: bool = True


def _check_recipe(recipe: Recipe, *, production: bool = False) -> None:
    if production and (type(recipe) is not Recipe or recipe != Recipe()):
        raise ValueError("production entry point requires the exact frozen recipe")
    if not recipe.test_only and recipe != Recipe():
        raise ValueError("nonstandard production recipe")
    if recipe.test_only and not isinstance(recipe, _TestRecipe):
        raise ValueError("test recipe must be explicitly private")
    if not (1 <= recipe.initial_epochs <= recipe.maximum_epochs <= 300):
        raise ValueError("invalid bounded horizon")
    if recipe.batch_size < 1 or not 1 <= recipe.last_epochs <= recipe.initial_epochs:
        raise ValueError("invalid training recipe")


def configure_numerics() -> dict[str, object]:
    """Configure the frozen FP32 profile before any CUDA work."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG conflicts with the frozen recipe")
    if torch.cuda.is_initialized() and workspace != ":4096:8":
        raise RuntimeError("configure the CUBLAS workspace before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) not in (None, "8"):
            raise ValueError(f"{name} must be 8 for production benchmark and fits")
        os.environ[name] = "8"
    torch.set_num_threads(8)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return numerical_profile()


def numerical_profile() -> dict[str, object]:
    return {
        "torch": str(torch.__version__), "precision": "float32_no_autocast",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _hash_value(value: object, digest: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
        # Byte views also support dtypes such as bfloat16 without NumPy conversion.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
            _hash_value(key, digest)
            _hash_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _hash_value(item, digest)
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")


def payload_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _hash_value(value, digest)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    """Normalize tuple/list and integer mapping keys before locking JSON manifests."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _cpu_copy(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


@dataclass
class TensorTrainingSet:
    features: torch.Tensor
    target: torch.Tensor
    support: torch.Tensor
    window_ids: Sequence[str]
    crop: int = 6
    _rows: torch.Tensor = field(init=False, repr=False)
    _digest: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.target.ndim != 3 or self.support.ndim != 3:
            raise ValueError("training arrays must be N,T,D tensors")
        if self.crop < 0 or any(t.device.type != "cpu" for t in (self.features, self.target, self.support)):
            raise ValueError("training storage must be on CPU with a nonnegative crop")
        n, time, _ = self.features.shape
        if len(self.target) != n or len(self.support) != n or len(self.window_ids) != n:
            raise ValueError("window population mismatch")
        self.window_ids = tuple(self.window_ids)
        if any(not isinstance(x, str) for x in self.window_ids) or len(set(self.window_ids)) != n:
            raise ValueError("window IDs must be unique canonical strings")
        centers = time - 2 * self.crop
        if centers <= 0 or self.support.shape[1] != centers:
            raise ValueError("support must describe exactly the central crop")
        if self.target.shape[1] not in (time, centers) or self.target.shape[2] != self.support.shape[2]:
            raise ValueError("target does not align with support")
        if self.support.dtype != torch.bool:
            raise ValueError("support must be boolean")
        supported = self.support.flatten(1).any(1).tolist()
        ordered = sorted((self.window_ids[i], i) for i in range(n) if supported[i])
        if not ordered:
            raise ValueError("no supported training windows")
        self._rows = torch.tensor([i for _, i in ordered], dtype=torch.long)

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def excluded_windows(self) -> int:
        return len(self.window_ids) - self.size

    @property
    def identity(self) -> str:
        if self._digest is None:
            digest = hashlib.sha256()
            _hash_value({"window_ids": self.window_ids, "crop": self.crop}, digest)
            for tensor in (self.features, self.target, self.support):
                _hash_value([str(tensor.dtype), list(tensor.shape)], digest)
                for chunk in tensor.split(64):
                    _hash_value(chunk, digest)
            self._digest = digest.hexdigest()
        return self._digest

    def batch(self, ordered_positions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
        rows = self._rows[ordered_positions]
        x = self.features[rows].to(device=device, dtype=torch.float32)
        y = self.target[rows].to(device=device, dtype=torch.float32)
        if y.shape[1] == self.features.shape[1]:
            y = y[:, self.crop:self.features.shape[1] - self.crop]
        mask = self.support[rows].to(device=device)
        if not torch.isfinite(x).all() or not torch.isfinite(y[mask]).all():
            raise FloatingPointError("nonfinite input or supported target")
        return x, y, mask

    def validate_production_shape(self) -> None:
        if self.crop != 6 or self.features.shape[1:] != (160, 384) or self.support.shape[1:] != (148, 20):
            raise ValueError("production readout requires N,160,384 and crop6/20 joints")


Factory = Callable[[int, int], nn.Module]
Evaluator = Callable[[nn.Module, int], selection.FoldScore | Mapping[str, object]]


class Trajectory:
    """One optimizer, one shuffle stream and one dropout stream, independently resumable."""

    def __init__(self, factory: Factory, data: TensorTrainingSet, *, lr: float, seed: int,
                 fold: int, device: str | torch.device = "cpu", _recipe: Recipe = Recipe()):
        _check_recipe(_recipe)
        if fold not in (0, 1, 2) or seed not in (0, 1, 2):
            raise ValueError("fold and readout seed must be frozen values")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("invalid learning rate")
        if not _recipe.test_only:
            data.validate_production_shape()
        self.recipe, self.data = _recipe, data
        self.lr, self.seed, self.fold = lr, seed, fold
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            self.model = factory(seed, fold).to(device=self.device, dtype=torch.float32)
        self.initial_sha256 = heads.state_sha256(self.model)
        self.optimizer = torch.optim.AdamW(heads.optimizer_groups(self.model, _recipe.weight_decay),
                                           lr=lr, betas=_recipe.betas, eps=_recipe.eps)
        seeds = heads.trajectory_seeds(seed, fold)
        self.shuffle = torch.Generator(device="cpu").manual_seed(seeds["shuffle"])
        self.dropout_cpu = torch.Generator(device="cpu").manual_seed(seeds["dropout"]).get_state()
        self.dropout_cuda = None
        if self.device.type == "cuda":
            self.dropout_cuda = torch.Generator(device=self.device).manual_seed(seeds["dropout"]).get_state()
        self.updates = 0
        self.permutation: torch.Tensor | None = None
        self.position = 0
        self.epoch_order_sha256: list[str] = []
        self.epoch_losses: list[dict[str, float | int]] = []
        self.loss_sum, self.loss_windows = 0.0, 0

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.data.size / self.recipe.batch_size)

    @contextmanager
    def _rng(self):
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        torch.set_rng_state(self.dropout_cpu)
        if self.dropout_cuda is not None:
            torch.cuda.set_rng_state(self.dropout_cuda, self.device)
        try:
            yield
        finally:
            self.dropout_cpu = torch.get_rng_state().clone()
            if self.device.type == "cuda":
                self.dropout_cuda = torch.cuda.get_rng_state(self.device).clone()
                torch.cuda.set_rng_state(cuda, self.device)
            torch.set_rng_state(cpu)

    def train_to_step(self, target_step: int) -> None:
        if int(target_step) != target_step or target_step < self.updates:
            raise ValueError("training can only continue forward")
        if target_step > self.recipe.maximum_epochs * self.steps_per_epoch:
            raise ValueError("requested training exceeds the frozen horizon")
        self.model.train()
        with self._rng(), torch.autocast(device_type=self.device.type, enabled=False):
            while self.updates < target_step:
                if self.permutation is None:
                    self.permutation = torch.randperm(self.data.size, generator=self.shuffle)
                    self.position = 0
                    self.epoch_order_sha256.append(payload_sha256(self.permutation))
                positions = self.permutation[self.position:self.position + self.recipe.batch_size]
                x, y, support = self.data.batch(positions, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(x)
                prediction = prediction[:, self.data.crop:x.shape[1] - self.data.crop]
                if prediction.shape != y.shape:
                    raise ValueError("head output does not align with the central targets")
                loss = metrics.per_window_mse(prediction, y, support)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip_norm,
                                                     error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient")
                self.optimizer.step()
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise FloatingPointError("nonfinite updated parameter")
                self.loss_sum += float(loss.detach().cpu()) * len(positions)
                self.loss_windows += len(positions)
                self.updates += 1
                self.position += len(positions)
                if self.position == self.data.size:
                    self.epoch_losses.append({"epoch": len(self.epoch_losses) + 1,
                                              "loss_sum": self.loss_sum, "window_count": self.loss_windows})
                    self.loss_sum, self.loss_windows = 0.0, 0
                    self.permutation, self.position = None, 0

    def evaluate(self, evaluator: Evaluator) -> selection.FoldScore:
        mode = self.model.training
        self.model.eval()
        try:
            with self._rng(), torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                result = evaluator(self.model, self.fold)
            score = result if isinstance(result, selection.FoldScore) else selection.FoldScore.from_mapping(result)
            score.validate()
            return score
        finally:
            self.model.train(mode)

    def state(self) -> dict[str, object]:
        value = {
            "schema": "pose-head-trajectory-v1", "recipe": asdict(self.recipe),
            "lr": self.lr, "seed": self.seed, "fold": self.fold, "device_type": self.device.type,
            "dataset_sha256": self.data.identity, "initial_sha256": self.initial_sha256,
            "numeric": numerical_profile(), "steps_per_epoch": self.steps_per_epoch,
            "model": _cpu_copy(self.model.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "updates": self.updates, "permutation": _cpu_copy(self.permutation), "position": self.position,
            "shuffle_rng": self.shuffle.get_state().clone(), "dropout_cpu": self.dropout_cpu.clone(),
            "dropout_cuda": _cpu_copy(self.dropout_cuda), "epoch_order_sha256": list(self.epoch_order_sha256),
            "epoch_losses": copy.deepcopy(self.epoch_losses), "loss_sum": self.loss_sum,
            "loss_windows": self.loss_windows,
        }
        value["payload_sha256"] = payload_sha256(value)
        return value

    def restore(self, state: Mapping[str, object]) -> None:
        value = dict(state)
        digest = value.pop("payload_sha256", None)
        if payload_sha256(value) != digest:
            raise ValueError("trajectory payload checksum mismatch")
        expected = {"schema": "pose-head-trajectory-v1", "recipe": asdict(self.recipe), "lr": self.lr,
                    "seed": self.seed, "fold": self.fold, "device_type": self.device.type,
                    "dataset_sha256": self.data.identity, "initial_sha256": self.initial_sha256,
                    "numeric": numerical_profile(), "steps_per_epoch": self.steps_per_epoch}
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("trajectory provenance/recipe mismatch")
        updates, position = value["updates"], value["position"]
        if not isinstance(updates, int) or not 0 <= updates <= self.recipe.maximum_epochs * self.steps_per_epoch:
            raise ValueError("invalid checkpoint progress")
        permutation = value["permutation"]
        in_epoch = updates % self.steps_per_epoch
        if in_epoch == 0:
            if permutation is not None or position != 0:
                raise ValueError("completed epoch has a partial permutation")
        else:
            if (not isinstance(permutation, torch.Tensor) or permutation.dtype != torch.long or
                    permutation.shape != (self.data.size,) or
                    not torch.equal(permutation.sort().values, torch.arange(self.data.size)) or
                    position != in_epoch * self.recipe.batch_size):
                raise ValueError("invalid partial-epoch shuffle state")
        if len(value["epoch_losses"]) != updates // self.steps_per_epoch:
            raise ValueError("epoch-loss history disagrees with progress")
        expected_orders = math.ceil(updates / self.steps_per_epoch)
        if len(value["epoch_order_sha256"]) != expected_orders:
            raise ValueError("shuffle history disagrees with progress")
        if (value["loss_windows"] != position or not math.isfinite(value["loss_sum"]) or value["loss_sum"] < 0 or
                any(record["epoch"] != i + 1 or record["window_count"] != self.data.size or
                    not math.isfinite(record["loss_sum"]) or record["loss_sum"] < 0
                    for i, record in enumerate(value["epoch_losses"]))):
            raise ValueError("training-loss history disagrees with progress")
        if permutation is not None and value["epoch_order_sha256"][-1] != payload_sha256(permutation):
            raise ValueError("current permutation disagrees with its recorded hash")
        expected_groups = [{k: v for k, v in group.items() if k != "params"}
                           for group in self.optimizer.state_dict()["param_groups"]]
        received_groups = [{k: v for k, v in group.items() if k != "params"}
                           for group in value["optimizer"]["param_groups"]]
        if expected_groups != received_groups:
            raise ValueError("optimizer recipe changed in checkpoint")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("nonfinite model in checkpoint")
        if any(not torch.isfinite(t).all() for states in self.optimizer.state.values()
               for t in states.values() if isinstance(t, torch.Tensor)):
            raise ValueError("nonfinite optimizer in checkpoint")
        self.updates, self.position = updates, position
        self.permutation = _cpu_copy(permutation)
        self.shuffle.set_state(value["shuffle_rng"])
        self.dropout_cpu = value["dropout_cpu"].clone()
        self.dropout_cuda = _cpu_copy(value["dropout_cuda"])
        self.epoch_order_sha256 = list(value["epoch_order_sha256"])
        self.epoch_losses = copy.deepcopy(value["epoch_losses"])
        self.loss_sum, self.loss_windows = value["loss_sum"], value["loss_windows"]


class PairedCandidate:
    """Advance and select a single LR using the same exposure in both folds."""

    def __init__(self, trajectories: Sequence[Trajectory], evaluator: Evaluator,
                 evaluation_identity: str | None = None):
        if len(trajectories) != 2 or [t.fold for t in trajectories] != [0, 1]:
            raise ValueError("paired selection requires folds 0 and 1 in order")
        if any(t.seed != 0 or t.lr != trajectories[0].lr or t.recipe != trajectories[0].recipe for t in trajectories):
            raise ValueError("paired trajectories must share CV seed, LR and recipe")
        self.trajectories = list(trajectories)
        self.evaluator = evaluator
        self.recipe, self.lr = trajectories[0].recipe, trajectories[0].lr
        if len({t.initial_sha256 for t in trajectories}) != 1:
            raise ValueError("initial weights differ between paired folds")
        if evaluation_identity is None and self.recipe.test_only:
            evaluation_identity = "test-only-evaluator"
        if not self.recipe.test_only and (not isinstance(evaluation_identity, str) or
                                          re.fullmatch(r"[0-9a-f]{64}", evaluation_identity) is None):
            raise ValueError("production selection requires a pinned inner evaluation identity")
        self.evaluation_identity = evaluation_identity
        self.points: list[selection.CurvePoint] = []
        self.best_states: list[dict[str, object]] | None = None
        self.best_exposure: float | None = None
        self.approved_horizon = self.recipe.initial_epochs

    def advance(self, horizon: int, on_checkpoint: Callable[["PairedCandidate"], None] | None = None) -> None:
        if horizon not in (self.recipe.initial_epochs, self.recipe.maximum_epochs):
            raise ValueError("horizon is outside the finite policy")
        if horizon == self.recipe.maximum_epochs and self.approved_horizon != horizon:
            if not self.points or self.points[-1].exposure != self.recipe.initial_epochs:
                raise ValueError("epoch extension requires the completed initial pair")
            if not selection.needs_epoch_extension(self.curve(), self.recipe.initial_epochs, self.recipe.last_epochs):
                raise ValueError("epoch extension is not triggered")
            self.approved_horizon = horizon
        if horizon != self.approved_horizon:
            raise ValueError("cannot shorten an already approved continuation")
        after = self.points[-1].exposure if self.points else -1
        steps = [t.steps_per_epoch for t in self.trajectories]
        for exposure, fold_steps in selection.checkpoint_schedule(steps, horizon, after_exposure=after):
            for trajectory, step in zip(self.trajectories, fold_steps):
                trajectory.train_to_step(step)
            point = selection.pool_scores([t.evaluate(self.evaluator) for t in self.trajectories], exposure, fold_steps)
            previous_best = min((p for p in self.points if p.exposure > 0),
                                key=lambda p: (p.mse, p.exposure), default=None)
            self.points.append(point)
            if exposure > 0 and (previous_best is None or point.mse < previous_best.mse):
                self.best_states = [t.state() for t in self.trajectories]
                self.best_exposure = exposure
            if on_checkpoint is not None:
                on_checkpoint(self)

    def curve(self) -> selection.CandidateCurve:
        curve = selection.CandidateCurve(self.lr, self.approved_horizon, tuple(self.points))
        curve.validate()
        return curve

    def state(self) -> dict[str, object]:
        value = {"schema": "pose-head-paired-v1", "lr": self.lr, "recipe": asdict(self.recipe),
                 "evaluation_identity": self.evaluation_identity,
                 "approved_horizon": self.approved_horizon, "points": [asdict(p) for p in self.points],
                 "best_exposure": self.best_exposure, "best_states": _cpu_copy(self.best_states),
                 "current_states": [t.state() for t in self.trajectories]}
        value["payload_sha256"] = payload_sha256(value)
        return value

    def restore(self, state: Mapping[str, object]) -> None:
        value = dict(state)
        digest = value.pop("payload_sha256", None)
        if payload_sha256(value) != digest:
            raise ValueError("paired checkpoint checksum mismatch")
        if (value.get("schema") != "pose-head-paired-v1" or value.get("lr") != self.lr or
                value.get("recipe") != asdict(self.recipe) or value.get("evaluation_identity") != self.evaluation_identity):
            raise ValueError("paired checkpoint provenance mismatch")
        horizon = value["approved_horizon"]
        if horizon not in (self.recipe.initial_epochs, self.recipe.maximum_epochs):
            raise ValueError("invalid approved horizon")
        points = [selection.CurvePoint(**{**p, "fold_steps": tuple(p["fold_steps"])}) for p in value["points"]]
        schedule = selection.checkpoint_schedule([t.steps_per_epoch for t in self.trajectories], horizon)
        if len(points) > len(schedule) or any((p.exposure, p.fold_steps) != expected for p, expected in zip(points, schedule)):
            raise ValueError("paired curve is not a prefix of the frozen schedule")
        for point in points:
            point.validate()
        if horizon == self.recipe.maximum_epochs:
            initial = tuple(p for p in points if p.exposure <= self.recipe.initial_epochs)
            initial_curve = selection.CandidateCurve(self.lr, self.recipe.initial_epochs, initial)
            if not selection.needs_epoch_extension(initial_curve, self.recipe.initial_epochs, self.recipe.last_epochs):
                raise ValueError("checkpoint contains an untriggered extension")
        positive = [p for p in points if p.exposure > 0]
        best = min(positive, key=lambda p: (p.mse, p.exposure), default=None)
        if value["best_exposure"] != (best.exposure if best else None):
            raise ValueError("best pair does not match the pooled MSE curve")
        if best is None and value["best_states"] is not None:
            raise ValueError("step zero cannot be a selected pair")
        if len(value["current_states"]) != 2 or (best is not None and len(value["best_states"]) != 2):
            raise ValueError("incomplete paired states")
        expected_current_steps = points[-1].fold_steps if points else (0, 0)
        for i, trajectory in enumerate(self.trajectories):
            if best is not None:
                trajectory.restore(value["best_states"][i])
                if trajectory.updates != best.fold_steps[i]:
                    raise ValueError("selected pair exposure mismatch")
            trajectory.restore(value["current_states"][i])
            if trajectory.updates != expected_current_steps[i]:
                raise ValueError("current pair exposure mismatch")
        self.points, self.approved_horizon = points, horizon
        self.best_states, self.best_exposure = _cpu_copy(value["best_states"]), value["best_exposure"]


def save_checkpoint(path: str | Path, state: Mapping[str, object]) -> str:
    """Atomically write a caller-owned checkpoint; return the file hash for its manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(dict(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: str | Path, *, expected_sha256: str) -> dict[str, object]:
    path = Path(path)
    if len(expected_sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("checkpoint file hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("checkpoint is not a state mapping")
    return value


def _run_search(factory: Factory, fitting: Sequence[TensorTrainingSet], evaluator: Evaluator,
                *, device: str | torch.device, recipe: Recipe,
                on_checkpoint: Callable[[PairedCandidate], None] | None = None,
                evaluation_identity: str | None = None,
                fitting_evaluator: Evaluator | None = None) -> dict[str, object]:
    _check_recipe(recipe)
    if len(fitting) != 2:
        raise ValueError("search requires two fitting folds")
    candidates: list[selection.CandidateCurve] = []
    states: dict[float, dict[str, object]] = {}
    initial_hashes = None
    queued = list(selection.INITIAL_LRS)
    while queued:
        lr = queued.pop(0)
        trajectories = [Trajectory(factory, data, lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
                        for fold, data in enumerate(fitting)]
        hashes = [t.initial_sha256 for t in trajectories]
        if initial_hashes is not None and hashes != initial_hashes:
            raise ValueError("initial weights differ between learning rates")
        initial_hashes = hashes
        pair = PairedCandidate(trajectories, evaluator, evaluation_identity)
        pair.advance(recipe.initial_epochs, on_checkpoint)
        if selection.needs_epoch_extension(pair.curve(), recipe.initial_epochs, recipe.last_epochs):
            pair.advance(recipe.maximum_epochs, on_checkpoint)
        candidates.append(pair.curve())
        states[lr] = pair.state()
        if not queued:
            added = selection.next_learning_rate(candidates)
            if added is not None:
                queued.append(added)
    report = selection.assessment(candidates, maximum_epochs=recipe.maximum_epochs, last_epochs=recipe.last_epochs)
    report.update({"recipe": asdict(recipe), "curves": [asdict(c) for c in candidates],
                   "initial_sha256_per_fold": initial_hashes, "selection_locked": True,
                   "evaluation_identity": pair.evaluation_identity})
    if fitting_evaluator is not None:
        selected_lr = report["selected"]["lr"]
        diagnostics = {}
        for fold, data in enumerate(fitting):
            trajectory = Trajectory(factory, data, lr=selected_lr, seed=0, fold=fold, device=device, _recipe=recipe)
            trajectory.restore(states[selected_lr]["best_states"][fold])
            diagnostics[fold] = asdict(trajectory.evaluate(fitting_evaluator))
        report["selected_fitting_diagnostics"] = diagnostics
    report = _json_ready(report)
    report["selection_sha256"] = payload_sha256(report)
    return {"selection": report, "candidate_states": states}


def run_search(factory: Factory, fitting: Sequence[TensorTrainingSet], evaluator: Evaluator,
               *, device: str | torch.device, recipe: Recipe = Recipe(),
               on_checkpoint: Callable[[PairedCandidate], None] | None = None,
               evaluation_identity: str | None = None,
               fitting_evaluator: Evaluator | None = None) -> dict[str, object]:
    """Production CV entry point: no horizon/batch override and no development data."""
    _check_recipe(recipe, production=True)
    if not isinstance(evaluation_identity, str) or re.fullmatch(r"[0-9a-f]{64}", evaluation_identity) is None:
        raise ValueError("production selection requires a pinned inner evaluation identity")
    configure_numerics()
    return _run_search(factory, fitting, evaluator, device=device, recipe=recipe, on_checkpoint=on_checkpoint,
                       evaluation_identity=evaluation_identity, fitting_evaluator=fitting_evaluator)


def _run_refits(factory: Factory, full_train: TensorTrainingSet, selected_report: Mapping[str, object],
                *, device: str | torch.device, recipe: Recipe,
                fitting_evaluator: Evaluator | None = None) -> dict[str, object]:
    report = dict(selected_report)
    digest = report.pop("selection_sha256", None)
    if (payload_sha256(report) != digest or report.get("selection_locked") is not True or
            report.get("recipe") != _json_ready(asdict(recipe))):
        raise ValueError("selection must be locked, intact and from the same recipe before refit")
    selected = report["selected"]
    curves = [selection.curve_from_dict(c) for c in report["curves"]]
    if asdict(selection.choose(curves)) != selected:
        raise ValueError("selection disagrees with the frozen MSE rule")
    states, diagnostics = {}, {}
    for seed in (0, 1, 2):
        trajectory = Trajectory(factory, full_train, lr=selected["lr"], seed=seed, fold=2,
                                device=device, _recipe=recipe)
        if seed == 0 and trajectory.initial_sha256 not in report["initial_sha256_per_fold"]:
            raise ValueError("refit seed-zero initialization differs from CV")
        print(json.dumps({"event": "refit_initialization", "seed": seed, "fold": 2,
                          "initial_sha256": trajectory.initial_sha256, "updates": trajectory.updates}), flush=True)
        trajectory.train_to_step(math.ceil(selected["exposure"] * trajectory.steps_per_epoch))
        states[seed] = trajectory.state()
        if fitting_evaluator is not None:
            diagnostics[seed] = asdict(trajectory.evaluate(fitting_evaluator))
    return {"selection_sha256": digest, "refit_states": states, "fitting_diagnostics": diagnostics,
            "refits_locked": True, "refit_sha256": payload_sha256(states), "recipe": asdict(recipe)}


def run_refits(factory: Factory, full_train: TensorTrainingSet, selected_report: Mapping[str, object],
               *, device: str | torch.device, recipe: Recipe = Recipe(),
               fitting_evaluator: Evaluator | None = None) -> dict[str, object]:
    _check_recipe(recipe, production=True)
    configure_numerics()
    return _run_refits(factory, full_train, selected_report, device=device, recipe=recipe,
                       fitting_evaluator=fitting_evaluator)
