"""Supervised EPN development: frozen / adapted / random encoder, same head.

No pretraining losses, EMA, decoder or pose targets are used downstream. Data
preparation matches fm_features (including two crops and subject calibration).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import numpy as np
import torch
from torch import nn

from wearusfm.harness import fm_features as FF
from wearusfm.model import channel_codes as CC
from wearusfm.model.channel_identity import codes_to_tensors
from wearusfm.model.local_encoder import sets_to_tensors
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM
from wearusfm.model.spectral_step1 import pool_p1p2

REGIMES = ("frozen", "finetune", "scratch")
LR_GRID = {"frozen": (1e-3, 1e-2), "finetune": (1e-5, 1e-4), "scratch": (1e-4, 1e-3)}
# The encoder grid differs by regime; the head grid is exactly (1e-3, 1e-2)
# in every regime. Keep these multipliers in the persisted training recipe.
HEAD_LR_MULTIPLIER = {"frozen": 1, "finetune": 100, "scratch": 10}
ENCODER_MODULES = ("tokenizer", "identity", "local", "pool", "backbone")


class NumericalDivergence(RuntimeError):
    """A candidate's numerical outcome, distinct from an implementation failure.

    ``step`` counts completed updates; ``epoch`` is one-based. History contains
    only fully evaluated epochs. Context identifies a failed train update or
    prediction batch. OOM, invalid configuration and other runtime errors retain
    their original exception types and must not be classified as divergence.
    """
    def __init__(self, kind, *, step=None, epoch=None, history=None, context=None):
        self.kind, self.step, self.epoch = kind, step, epoch
        self.history = deepcopy(history if history is not None else [])
        self.context = deepcopy(context if context is not None else {})
        super().__init__(f"nonfinite {kind} (epoch={epoch}, completed_updates={step})")

    def as_dict(self):
        return {"kind": self.kind, "step": self.step, "epoch": self.epoch,
                "history": deepcopy(self.history), "context": deepcopy(self.context)}


class Classifier(nn.Module):
    def __init__(self, model, classes, frozen, head_seed=0):
        super().__init__()
        self.encoder = model
        self.frozen = frozen
        model.requires_grad_(False)
        if not frozen:
            for name in ENCODER_MODULES:
                getattr(model, name).requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(head_seed)
            self.head = nn.Sequential(nn.LayerNorm(2 * model.cfg.dim), nn.Linear(2 * model.cfg.dim, classes))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, crops):
        features = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            for inp in crops:
                enc = self.encoder.encode(inp, None)
                features.append(pool_p1p2(enc, inp.counts, inp.qc_valid))
        return self.head(torch.stack(features).mean(dim=0).float())


def build_classifier(state, regime, classes, seed=0):
    if regime not in REGIMES:
        raise ValueError("unregistered regime")
    mcfg = state["config"]["model"]
    if set(mcfg) - {f.name for f in fields(FMConfig)}:
        raise ValueError("unknown model config")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = WearUsFM(FMConfig(**mcfg))
    if regime != "scratch":
        model.load_state_dict(state["teacher"], strict=True)
    return Classifier(model, classes, regime == "frozen", head_seed=seed)


def optimizer_for(model, regime, lr):
    if regime not in REGIMES or lr not in LR_GRID[regime]:
        raise ValueError("unregistered learning rate")
    head_lr = HEAD_LR_MULTIPLIER[regime] * lr
    groups = [{"params": list(model.head.parameters()), "lr": head_lr}]
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder:
        groups.append({"params": encoder, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=.01)


def encode_labels(y, train_indices):
    """Fit vocabulary and balancing weights on train only; reject unseen labels."""
    y = np.asarray(y).astype(str)
    classes, counts = np.unique(y[train_indices], return_counts=True)
    if len(classes) < 2 or not set(y).issubset(classes):
        raise ValueError("class absent from train; do not silently drop held samples")
    labels = np.searchsorted(classes, y)
    weights = len(train_indices) / (len(classes) * counts.astype(np.float64))
    return labels, classes, torch.tensor(weights, dtype=torch.float32)


class PreparedWindows:
    """CPU preprocessing only, no learned scaler across subjects or label access.

    First ten windows of each subject supply its unlabeled calibration, exactly
    as in frozen probes. This is a declared calibrated-subject protocol, not a
    zero-calibration new-user claim. Filtering is per crop as in FF.
    """
    def __init__(self, windows, subjects, fs, montage):
        self.fs, self.montage = fs, montage
        self.layout, self.code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
        self.sets = CC.attention_sets(self.layout, 8)
        scales = FF.subject_scales(windows, subjects, fs, (20., 450.), 10)
        self.scales = scales
        self.crops = []
        for a, b in FF._crops(windows.shape[1], fs, 4.):
            self.crops.append([FF._filtered(w[a:b], fs, (20., 450.)) / np.float32(scales[s])
                               for w, s in zip(windows, subjects)])

    def batch(self, indices, device):
        n, c = len(indices), self.layout.n_channels
        move = lambda d: {k: v.to(device) for k, v in d.items()}
        codes = move(codes_to_tensors(CC.pack_codes([self.code] * n)))
        sets = move(sets_to_tensors(CC.pack_attention_sets([self.sets] * n)))
        qc = torch.as_tensor(self.layout.qc_valid).repeat(n).to(device)
        return [ModelInputs([torch.from_numpy(x[i]).to(device) for i in indices], [self.fs] * n,
                            codes, sets, [c] * n, qc) for x in self.crops]


def role_indices(subjects, roles):
    if set(roles) != {"train", "val", "held"}:
        raise ValueError("three subject-disjoint roles required")
    sets = [set(roles[k]) for k in ("train", "val", "held")]
    if any(not s for s in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("empty or overlapping subject roles")
    if set.union(*sets) != set(subjects):
        raise ValueError("role coverage differs from dataset")
    return {k: np.flatnonzero(np.isin(subjects, v)) for k, v in roles.items()}


def balanced_accuracy(y, pred, classes):
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes if np.any(y == c)]))


@torch.no_grad()
def predict(model, data, indices, device, batch=8, *, step=None, epoch=None, history=None):
    model.eval()
    out = []
    for start in range(0, len(indices), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits = model(data.batch(indices[start:start + batch], device))
        if not torch.isfinite(logits).all():
            raise NumericalDivergence("prediction", step=step, epoch=epoch, history=history,
                                      context={"phase": "evaluation", "batch_start": start})
        out.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(out)


def fit_candidate(model, data, labels, roles, weights, regime, lr, device, *, seed=0,
                  batch=8, epochs=5, max_updates=None, callback=None):
    """Fixed train budget, validation-only checkpoint selection; never reads held.

    Candidate selection over the two LRs is performed outside this function.
    Report a truncated budget as such; reaching it is not proof of convergence.
    """
    if epochs < 1 or (max_updates is not None and max_updates < 1):
        raise ValueError("positive training budget required")
    optimizer = optimizer_for(model, regime, lr)
    initial_lrs = [g["lr"] for g in optimizer.param_groups]
    total = epochs * int(np.ceil(len(roles["train"]) / batch))
    if max_updates is not None:
        total = min(total, max_updates)
    weights = weights.to(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, history, step = -np.inf, None, None, [], 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(roles["train"])
        losses = []
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            warm = min(1., (step + 1) / max(1, min(100, total // 10)))
            decay = .5 * (1 + np.cos(np.pi * step / total))
            for g, base_lr in zip(optimizer.param_groups, initial_lrs):
                g["lr"] = base_lr * warm * decay
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(labels[idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                logits = model(data.batch(idx, device))
                if not torch.isfinite(logits).all():
                    raise NumericalDivergence("prediction", step=step, epoch=epoch + 1, history=history,
                                              context={"phase": "train", "attempted_update": step + 1})
                loss = (nn.functional.cross_entropy(logits.float(), target, reduction="none") * weights[target]).mean()
            if not torch.isfinite(loss):
                raise NumericalDivergence("loss", step=step, epoch=epoch + 1, history=history,
                                          context={"phase": "train", "attempted_update": step + 1})
            loss.backward()
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=False)
            if not torch.isfinite(norm):
                raise NumericalDivergence("gradient", step=step, epoch=epoch + 1, history=history,
                                          context={"phase": "train", "attempted_update": step + 1})
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
            if step == total:
                break
        pred = predict(model, data, roles["val"], device, batch, step=step, epoch=epoch + 1, history=history)
        score = balanced_accuracy(labels[roles["val"]], pred, np.arange(len(weights)))
        history.append({"epoch": epoch + 1, "step": step, "loss": float(np.mean(losses)),
                        "last_gradient_norm": float(norm), "val_bacc": score})
        if score > best:  # deterministic tie: first epoch
            best = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if callback:
            callback(history[-1])
        if step == total:
            break
    model.load_state_dict(best_state)
    return {"val_bacc": best, "best_epoch": best_epoch, "history": history, "updates": step, "planned_updates": total,
            "stopped": "fixed_budget", "convergence_claim": False}
