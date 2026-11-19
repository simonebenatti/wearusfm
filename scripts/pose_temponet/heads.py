"""The five prespecified readers, independent of the FM and raw TEMPONet."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

HEADS = ("L0", "L13", "MLP", "TCN_post", "TCN_pre")
PARAMETERS = {"L0": 8468, "L13": 100628, "MLP": 73000,
              "TCN_post": 72904, "TCN_pre": 72904}
EPS = 1e-5


def trajectory_seeds(seed: int, fold: int) -> dict[str, int]:
    if type(seed) is not int or seed < 0 or type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("expected nonnegative seed and fold 0, 1 or full-refit 2")
    return {"initialization": 100000 + seed,
            "dropout": 200000 + 100 * fold + seed,
            "shuffle": 300000 + 100 * fold + seed}


class PoseReadout(nn.Module):
    """FP32 (batch,time,384) -> (batch,time,20), with time preserved.

    Dropout uses Torch's device RNG. The trajectory controller owns its separate
    saved RNG state; this class neither reseeds it nor consumes a shuffle RNG.
    Use make_head for the frozen initialization and global-RNG isolation.
    """
    def __init__(self, name: str):
        super().__init__()
        if name not in HEADS:
            raise ValueError("unknown pose readout")
        self.name = name
        self.norm = nn.LayerNorm(384, eps=EPS, elementwise_affine=True)
        if name == "L0":
            self.out = nn.Linear(384, 20)
        elif name == "L13":
            self.out = nn.Conv1d(384, 20, kernel_size=13, padding=6)
        else:
            if name == "MLP":
                dimensions = (384, 96, 96, 96, 64, 32, 20)
                transforms = [nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:])]
            else:
                transforms = [nn.Conv1d(384, 64, 1),
                              nn.Conv1d(64, 64, 3, dilation=1, padding=1),
                              nn.Conv1d(64, 64, 3, dilation=2, padding=2),
                              nn.Conv1d(64, 64, 3, dilation=3, padding=3),
                              nn.Conv1d(64, 32, 1), nn.Conv1d(32, 20, 1)]
            layers = []
            for i, layer in enumerate(transforms):
                layers.append(layer)
                if i < 5:
                    layers.append(nn.ReLU())
                    if i in (3, 4):
                        layers.append(nn.Dropout(p=0.1))
            self.deep = nn.Sequential(*layers)
            # Its default initialization must not alter the deep-path RNG.
            with torch.random.fork_rng(devices=[]):
                self.affine = nn.Linear(384, 20)
        self._initialize()

    def _initialize(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        if self.name in ("L0", "L13"):
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            return
        transforms = [m for m in self.deep if isinstance(m, (nn.Linear, nn.Conv1d))]
        for layer in transforms[:-1]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(transforms[-1].weight)
        nn.init.zeros_(transforms[-1].bias)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 384 or not features.is_floating_point():
            raise ValueError("expected floating (batch,time,384) features")
        if self.norm.weight.dtype != torch.float32:
            raise ValueError("pose readout parameters must remain FP32")
        with torch.autocast(device_type=features.device.type, enabled=False):
            h = self.norm(features.float())
            if self.name == "L0":
                return self.out(h)
            if self.name == "L13":
                return self.out(h.transpose(1, 2)).transpose(1, 2)
            deep = self.deep(h if self.name == "MLP" else h.transpose(1, 2))
            if self.name != "MLP":
                deep = deep.transpose(1, 2)
            return self.affine(h) + deep


def make_head(name: str, seed: int = 0) -> PoseReadout:
    """Fresh CPU model, same initial state across folds/LRs, no global RNG effect."""
    init_seed = trajectory_seeds(seed, 0)["initialization"]
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA and would escape the CPU fork.
        torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(init_seed).get_state())
        model = PoseReadout(name)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[name]:
        raise RuntimeError("pose readout parameter count differs from specification")
    return model


def optimizer_groups(model: nn.Module, weight_decay: float = 0.01) -> list[dict]:
    """Decay only matrix/kernel weights, never biases or LayerNorm parameters."""
    if weight_decay < 0 or not torch.isfinite(torch.tensor(weight_decay)):
        raise ValueError("invalid weight decay")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and name.endswith("weight") else no_decay).append(parameter)
    return [{"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0}]


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        metadata = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
