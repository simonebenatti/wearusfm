"""Matched C0/CN/CR experiment: global candidates, shared forwards, explicit state.

Opt-in only. RnC is a replicated GLOBAL scalar (one logical replica per rank),
not a sum of independently constructed local pair losses. Differentiable gather
sums candidate gradients; DDP averages parameter gradients. Together these give
the full-batch gradient with each rank evaluating the same scalar.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
import copy
import math
import torch
import torch.nn.functional as F

from wearusfm.model.contrastive import ProjectionHead, window_readout, normalize_projection, info_nce
from wearusfm.data.pose_targets import N_JOINTS
from wearusfm.model.rank_contrast import pose_rank_contrast, anchor_support
from wearusfm.training.contrastive_memory import (
    CandidateQueue, identities_from_batch, admissible_mask, gather_variable, gather_objects, candidate_diagnostics,
)


@dataclass(frozen=True)
class ContrastiveConfig:
    nce_weight: float
    rnc_weight: float
    arm: str = "C0"
    nce_temperature: float = 0.2
    rnc_temperature: float = 0.1
    rnc_stride: int = 8
    rnc_resolution: float = 0.0
    capacity_per_dataset: int = 1024
    max_age: int = 256
    min_negatives: int = 32
    gap_seconds: float = 4.0
    fill_steps: int = 256
    ramp_steps: int = 256
    grad_every: int = 500
    grad_steps: tuple[int, ...] = (256, 512, 1962, 5000, 19618)
    grad_dense_start: int = 0  # inclusive, 1-based; 0/0 disables the dense panel
    grad_dense_end: int = 0
    calibration_only: bool = False
    coverage_block: int = 1024
    enforce_coverage: bool = True
    coverage_min_queries: int = 32
    coverage_dataset_fraction: float = .8
    coverage_nce_fraction: float = .9
    coverage_rnc_fraction: float = .9
    coverage_quotas: dict[str, float] | None = None
    coverage_global_batch_size: int = 32
    coverage_exempt_datasets: tuple[str, ...] = ()

    def __post_init__(self):
        if self.arm not in ("C0", "CN", "CR"):
            raise ValueError("contrastive arm must be C0, CN or CR")
        for k in ("nce_weight", "rnc_weight", "nce_temperature", "rnc_temperature", "gap_seconds"):
            v = getattr(self, k)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"invalid contrastive {k}")
        if not all(1e-4 <= getattr(self, k) <= 1. for k in ("nce_weight", "rnc_weight")):
            raise ValueError("contrastive weights must be explicit and in [1e-4, 1]")
        for k in ("rnc_stride", "capacity_per_dataset", "max_age", "min_negatives", "ramp_steps", "coverage_block",
                  "coverage_min_queries", "coverage_global_batch_size"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(f"invalid contrastive {k}")
        for k in ("fill_steps", "grad_every", "grad_dense_start", "grad_dense_end"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(f"invalid contrastive {k}")
        if not isinstance(self.grad_steps, (tuple, list)) or any(type(v) is not int or v < 1 for v in self.grad_steps):
            raise ValueError("grad_steps must contain positive integer update numbers")
        object.__setattr__(self, "grad_steps", tuple(self.grad_steps))
        if not (self.grad_dense_start == self.grad_dense_end == 0 or 1 <= self.grad_dense_start <= self.grad_dense_end):
            raise ValueError("invalid dense gradient diagnostic interval")
        if type(self.rnc_resolution) not in (int, float) or not math.isfinite(self.rnc_resolution) or self.rnc_resolution < 0:
            raise ValueError("invalid contrastive resolution")
        if type(self.calibration_only) is not bool or type(self.enforce_coverage) is not bool:
            raise ValueError("calibration_only/enforce_coverage must be boolean")
        for name in ("coverage_dataset_fraction", "coverage_nce_fraction", "coverage_rnc_fraction"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid " + name)
        if not isinstance(self.coverage_exempt_datasets, (tuple, list)) or any(
                not isinstance(d, str) or not d for d in self.coverage_exempt_datasets):
            raise ValueError("invalid coverage_exempt_datasets")
        object.__setattr__(self, "coverage_exempt_datasets", tuple(self.coverage_exempt_datasets))
        if self.coverage_quotas is None:
            if self.coverage_exempt_datasets:
                raise ValueError("coverage exemptions require pinned sampling quotas")
        else:
            quotas = self.coverage_quotas
            if not isinstance(quotas, dict) or not quotas or any(
                    not isinstance(d, str) or not d or type(q) not in (float, int) or not math.isfinite(q) or not 0 < q <= 1
                    for d, q in quotas.items()) or not math.isclose(sum(quotas.values()), 1., abs_tol=1e-9):
                raise ValueError("invalid normalized coverage_quotas")
            exempt = tuple(sorted(d for d, q in quotas.items()
                                  if q * self.coverage_global_batch_size * self.max_age < self.min_negatives + 1))
            if self.coverage_exempt_datasets != exempt:
                raise ValueError("coverage exemptions differ from frozen quota rule")

    def weights(self, step):
        factor = min(1., max(0., (step + 1 - self.fill_steps) / self.ramp_steps))
        return (self.nce_weight * factor if self.arm == "CN" else 0.,
                self.rnc_weight * factor if self.arm == "CR" else 0.)

    def wants_gradients(self, step):
        update = step + 1
        return ((self.calibration_only and step >= self.fill_steps) or
                (self.grad_every > 0 and update % self.grad_every == 0) or update in self.grad_steps or
                (0 < self.grad_dense_start <= update <= self.grad_dense_end))


def attach_contrastive_head(model):
    """Attach the training-only projector without changing the base model schema/RNG."""
    if hasattr(model, "contrastive_head"):
        raise ValueError("contrastive projector already attached")
    # The tokenizer owns float64 filter parameters even in the float32 model.
    reference = next(model.backbone.parameters())
    model.contrastive_head = ProjectionHead(model.cfg.dim).to(device=reference.device, dtype=reference.dtype)
    return model


def _collective_error(error):
    """All ranks learn of input errors before entering tensor collectives."""
    errors = gather_objects([error])
    bad = [e for e in errors if e is not None]
    if bad:
        raise ValueError("contrastive: " + " | ".join(map(str, bad)))


def _pad_time(tensor, width):
    if tensor.shape[1] == width:
        return tensor
    shape = list(tensor.shape)
    shape[1] = width - tensor.shape[1]
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)


def _global_width(tensor):
    n = torch.tensor(tensor.shape[1], device=tensor.device, dtype=torch.int64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.MAX)
    return int(n)


def physical_pair_mask(identities, sample_indices, patch_indices, patch_ms, device):
    """Symmetric pair exclusion, not removal/reweighting of duplicate anchors."""
    points = []
    for s, p in zip(sample_indices.tolist(), patch_indices.tolist()):
        ident = identities[s]
        sample = ident.start + int(math.ceil((p + 0.5) * patch_ms * ident.fs / 1000. - 1e-9))
        points.append((ident.dataset, ident.subject, ident.session, ident.trial, sample, ident.fs))
    mask = torch.tensor([[a != b for b in points] for a in points], device=device, dtype=torch.bool)
    return mask.reshape(len(points), len(points)), points


class ContrastiveRuntime:
    def __init__(self, config: dict):
        self.cfg = ContrastiveConfig(**config)
        self.queue = CandidateQueue(self.cfg.capacity_per_dataset, self.cfg.max_age)
        self.pending = None
        self.ids = None
        self.step = None
        self.expected_datasets = ()
        self.coverage = self._empty_coverage()
        self.coverage_reports = []
        self.failure_reason = None
        self._next_coverage = None
        self.failed_coverage_report = None

    @staticmethod
    def _empty_coverage():
        return {"updates": 0, "rnc_active": 0, "datasets": {}}

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("cannot checkpoint contrastive state before successful-step commit")
        return {"schema": 1, "config": asdict(self.cfg), "queue": self.queue.state_dict(),
                "expected_datasets": list(self.expected_datasets), "coverage": self.coverage,
                "coverage_reports": self.coverage_reports}

    def load_state_dict(self, state, completed_steps):
        if state.get("schema") != 1 or state.get("config") != asdict(self.cfg):
            raise ValueError("changed contrastive recipe/state")
        self.queue.load_state_dict(state["queue"])
        if self.queue.last_step != completed_steps - 1:
            raise ValueError("queue age disagrees with checkpoint step")
        if tuple(state.get("expected_datasets", ())) != self.expected_datasets:
            raise ValueError("contrastive coverage dataset set changed")
        self.coverage = state["coverage"]
        self.coverage_reports = state["coverage_reports"]
        if not 0 <= self.coverage["updates"] < self.cfg.coverage_block:
            raise ValueError("invalid contrastive coverage checkpoint")

    def prepare(self, batch, step):
        self.failure_reason = None
        self._next_coverage = None
        error = None
        try:
            ids = identities_from_batch(batch)
            if self.pending is not None or step != self.queue.last_step + 1:
                raise ValueError("contrastive commit/step out of order")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            ids = []
        _collective_error(error)
        self.ids, self.step = ids, step
        self.global_ids = gather_objects(list(ids))

    def counts(self, device):
        return {"contrast_nce": torch.tensor(len(self.ids), device=device, dtype=torch.int64),
                "contrast_rnc": torch.ones((), device=device, dtype=torch.int64)}

    def _numerical_stop(self, terms, device, error):
        """All ranks return matching nonfinite terms to the trainer's STOP path."""
        errors = gather_objects([error])
        if not any(errors):
            return False
        self.failure_reason = "contrastive numerical failure: " + " | ".join(str(e) for e in errors if e)
        terms.num.update({key: torch.full((), float("nan"), device=device) for key in self.counts(device)})
        terms.den.update(self.counts(device))
        wn, wr = self.cfg.weights(self.step)
        terms.weight.update(contrast_nce=wn, contrast_rnc=wr)
        return True

    def attach(self, terms, student, teacher, micro, enc, enc_t):
        """Called by trainer after base loss, inside the DDP objective forward."""
        from wearusfm.training.jepa import pose_time_valid

        tv = pose_time_valid(enc.patch_valid, micro.inp.counts)
        offsets = [0]
        for c in micro.inp.counts:
            offsets.append(offsets[-1] + c)
        physical = torch.stack([(enc.patch_valid[a:b] & micro.inp.qc_valid[a:b, None]).any(0)
                                for a, b in zip(offsets[:-1], offsets[1:])])
        q_read, q_ok = window_readout(enc.z, physical, enc.key_time_valid)
        k_read, k_ok = window_readout(enc_t.z, physical)
        error = None
        try:
            if not bool(torch.isfinite(enc.z).all()) or not bool(torch.isfinite(enc_t.z).all()):
                raise ValueError("nonfinite student/teacher representation")
            if micro.pose is not None and not bool(torch.isfinite(micro.pose[0][micro.pose[1]]).all()):
                raise ValueError("nonfinite valid pose target")
            q_raw = student.contrastive_head(q_read)
            with torch.no_grad():
                k_raw = teacher.contrastive_head(k_read)
            # Invalid windows carry finite differentiable zeros, not NaN norms.
            q = normalize_projection(q_raw, valid=q_ok)
            k = normalize_projection(k_raw, valid=k_ok)
        except ValueError as exc:
            error = f"{type(exc).__name__}: {exc}"
        if self._numerical_stop(terms, enc.z.device, error):
            return terms
        all_k = gather_variable(k.detach())
        all_valid = gather_variable(k_ok)
        snapshot = self.queue.snapshot(self.step, all_k, self.global_ids, all_valid)
        neg = admissible_mask(self.ids, snapshot.identities, gap_seconds=self.cfg.gap_seconds, device=q.device)
        error = None
        try:
            per_query, eligible, _ = info_nce(q, k.detach(), snapshot.keys, neg,
                                              temperature=self.cfg.nce_temperature, min_negatives=self.cfg.min_negatives)
        except ValueError as exc:
            error = str(exc)
        if self._numerical_stop(terms, enc.z.device, error):
            return terms
        eligible = eligible & q_ok & k_ok
        nce_num = torch.where(eligible, per_query, torch.zeros_like(per_query)).sum()
        # All branches update the identical queue policy once, even when loss weight is zero.
        self.pending = (all_k.detach(), self.global_ids, all_valid.detach())

        zm = enc.z.float().mean(2) if enc.z.dtype != torch.float64 else enc.z.mean(2)
        width = _global_width(zm)
        z_global = gather_variable(_pad_time(zm, width), with_grad=True)
        if micro.pose is None:
            y = zm.new_zeros((*zm.shape[:2], N_JOINTS))
            valid = torch.zeros_like(y, dtype=torch.bool)
        else:
            y, valid = micro.pose[:2]
        target_global = gather_variable(_pad_time(y.detach().to(zm.dtype), width))
        valid_global = gather_variable(_pad_time(valid, width))
        tv_global = gather_variable(_pad_time(tv, width))
        # Exclude duplicated physical instants, regardless of montage, in BOTH roles.
        si, pi, *_ = anchor_support(valid_global & tv_global[..., None], self.cfg.rnc_stride, "all")
        allowed, point_ids = physical_pair_mask(self.global_ids, si, pi, student.cfg.patch_ms, zm.device)
        error = None
        try:
            rnc, _, windows, rlog = pose_rank_contrast(z_global, (target_global, valid_global), tv_global,
                temperature=self.cfg.rnc_temperature, stride=self.cfg.rnc_stride, positives="all", per_window=True,
                resolution=self.cfg.rnc_resolution, pair_allowed=allowed)
        except ValueError as exc:
            error = str(exc)
        if self._numerical_stop(terms, enc.z.device, error):
            return terms
        wn, wr = self.cfg.weights(self.step)
        counts = self.counts(zm.device)
        terms.num.update(contrast_nce=nce_num, contrast_rnc=rnc)
        terms.den.update(counts)
        terms.weight.update(contrast_nce=wn, contrast_rnc=wr)
        terms.log_sum["contrast_nce_eligible"] = eligible.sum().detach()
        terms.log_sum["contrast_windows"] = counts["contrast_nce"]
        diagnostics = candidate_diagnostics(self.ids, snapshot, neg, self.step,
                                            datasets={i.dataset for i in self.global_ids})
        def record_diagnostic(key, value):
            # Snapshot-level deduplication is the same on every rank.
            if "duplicate_current_discarded" in key or "previous_keys_replaced" in key:
                terms.log_num[key] = zm.new_tensor(float(value), dtype=torch.float64)
                terms.log_den[key] = zm.new_ones((), dtype=torch.float64)
            else:
                terms.log_sum[key] = counts["contrast_nce"].new_tensor(value)
        for key, value in diagnostics.items():
            if "__" not in key:
                record_diagnostic(key, value)
        for suffix in ("", *("__" + d for d in sorted({i.dataset for i in self.global_ids}))):
            denominator = diagnostics["contrast_nce_negative_count" + suffix]
            for numerator, metric in (("negative_age_sum", "negative_age_mean"), ("negative_current", "negative_current_fraction"),
                                      ("negative_same_subject", "negative_same_subject_fraction"),
                                      ("negative_same_recording", "negative_same_recording_fraction")):
                key = "contrast_nce_" + metric + suffix
                terms.log_num[key] = zm.new_tensor(float(diagnostics["contrast_nce_" + numerator + suffix]), dtype=torch.float64)
                terms.log_den[key] = zm.new_tensor(float(denominator), dtype=torch.float64)
        self.pending_stats = {"rnc_active": bool(windows > 0), "datasets": {}}
        for dataset in sorted({i.dataset for i in self.global_ids}):
            idx = torch.tensor([i.dataset == dataset for i in self.ids], device=zm.device)
            queries, good = int(idx.sum()), int((idx & eligible).sum())
            self.pending_stats["datasets"][dataset] = [queries, good]
            terms.log_sum[f"contrast_nce_queries__{dataset}"] = counts["contrast_nce"].new_tensor(queries)
            terms.log_sum[f"contrast_nce_eligible__{dataset}"] = counts["contrast_nce"].new_tensor(good)
            loss_sum = torch.where(idx & eligible, per_query.detach(), 0.).sum()
            terms.log_sum[f"contrast_nce_loss_sum__{dataset}"] = loss_sum
            terms.log_num[f"contrast_nce_loss__{dataset}"] = loss_sum
            terms.log_den[f"contrast_nce_loss__{dataset}"] = zm.new_tensor(float(queries))
            for key, value in diagnostics.items():
                if key.endswith("__" + dataset):
                    record_diagnostic(key, value)
        for key, value in (("contrast_rnc_windows", windows), ("contrast_rnc_anchors", rlog["anchors"]),
                           ("contrast_rnc_candidate_anchors", len(point_ids)), ("contrast_rnc_pairs", rlog["pairs"]),
                           ("contrast_rnc_duplicate_instants", len(point_ids) - len(set(point_ids))),
                           ("contrast_rnc_max_instant_multiplicity", max(Counter(point_ids).values(), default=0)),
                           ("contrast_rnc_pairs_excluded_same_instant", int((~allowed).sum()) - len(point_ids)),
                           ("contrast_rnc_recordings", len({self.global_ids[s].recording for s in si.tolist()})),
                           ("contrast_rnc_subjects", len({(self.global_ids[s].dataset, self.global_ids[s].subject) for s in si.tolist()})),
                           ("contrast_nce_weight", wn), ("contrast_rnc_weight", wr)):
            dtype = torch.float64 if key in ("contrast_nce_weight", "contrast_rnc_weight") else zm.dtype
            terms.log_num[key] = zm.new_tensor(float(value), dtype=dtype)
            terms.log_den[key] = zm.new_ones((), dtype=dtype)
        if self.cfg.wants_gradients(self.step):
            from wearusfm.training.contrastive_diagnostics import cross_window_rank_alignment
            alignment = cross_window_rank_alignment(z_global, target_global, valid_global, tv_global,
                                                     self.cfg.rnc_stride, allowed)
            for key, value in alignment.items():
                terms.log_num[key] = zm.new_tensor(value, dtype=torch.float64)
                terms.log_den[key] = zm.new_ones((), dtype=torch.float64)
        return terms

    def before_update(self):
        """Evaluate prospective coverage without mutating queue or completed-update state."""
        if self.pending is None:
            raise RuntimeError("missing contrastive pending keys")
        if self._next_coverage is not None:
            return None
        records = gather_objects([self.pending_stats])
        coverage = copy.deepcopy(self.coverage)
        if self.step >= self.cfg.fill_steps:
            coverage["updates"] += 1
            coverage["rnc_active"] += int(records[0]["rnc_active"])
            for record in records:
                for dataset, (total, eligible) in record["datasets"].items():
                    row = coverage["datasets"].setdefault(dataset, [0, 0])
                    row[0] += total
                    row[1] += eligible
            if coverage["updates"] == self.cfg.coverage_block:
                report = self.coverage_report(coverage)
                if self.cfg.enforce_coverage and not report["passed"]:
                    self.failed_coverage_report = report
                    return "contrastive coverage gate failed: " + str(report)
        self._next_coverage = coverage
        return None

    def commit(self):
        failure = self.before_update()
        if failure is not None:
            raise RuntimeError(failure)
        keys, ids, valid = self.pending
        self.queue.commit(keys, ids, valid, step=self.step)
        self.pending = None
        self.coverage = self._next_coverage
        self._next_coverage = None
        if self.coverage["updates"] == self.cfg.coverage_block:
            self.coverage_reports.append(self.coverage_report())
            self.coverage = self._empty_coverage()

    def coverage_report(self, coverage=None):
        through_step = self.queue.last_step if coverage is None else self.step
        coverage = self.coverage if coverage is None else coverage
        total = sum(r[0] for r in coverage["datasets"].values())
        good = sum(r[1] for r in coverage["datasets"].values())
        datasets = {d: {"queries": r[0], "eligible": r[1], "fraction": r[1] / max(r[0], 1)}
                    for d, r in coverage["datasets"].items()}
        for d in self.expected_datasets:
            datasets.setdefault(d, {"queries": 0, "eligible": 0, "fraction": 0.})
        nce = good / max(total, 1)
        rnc = coverage["rnc_active"] / max(coverage["updates"], 1)
        for d, row in datasets.items():
            row["exempt"] = d in self.cfg.coverage_exempt_datasets
        passed = (total > 0 and nce >= self.cfg.coverage_nce_fraction and rnc >= self.cfg.coverage_rnc_fraction and
                  all(r["exempt"] or (r["queries"] >= self.cfg.coverage_min_queries and
                                      r["fraction"] >= self.cfg.coverage_dataset_fraction) for r in datasets.values()))
        return {"through_step": through_step, "updates": coverage["updates"], "nce_fraction": nce,
                "rnc_active_fraction": rnc, "datasets": datasets, "passed": passed,
                "exempt_datasets": list(self.cfg.coverage_exempt_datasets),
                "policy": {k: getattr(self.cfg, k) for k in ("coverage_min_queries", "coverage_dataset_fraction",
                           "coverage_nce_fraction", "coverage_rnc_fraction", "coverage_global_batch_size", "coverage_quotas")}}

    def global_gradients(self, terms, global_den, shared):
        """Train-only norms of complete global gradients, before clipping.

        Each local objective uses the same DDP scaling as the eventual backward;
        autograd.grad bypasses parameter reduction, so we explicitly average it.
        All ranks call every component, including ranks with no pose/query.
        """
        ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        scaled = {k: n * ws / global_den[k].clamp(min=1) for k, n in terms.num.items()}
        base = sum(terms.weight[k] * value for k, value in scaled.items() if not k.startswith("contrast_"))
        values = {"base": base, "nce": scaled["contrast_nce"], "rnc": scaled["contrast_rnc"]}
        flat = {}
        for name, value in values.items():
            grads = torch.autograd.grad(value, shared, retain_graph=True, allow_unused=True)
            v = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1) for p, g in zip(shared, grads)])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(v)
                v.div_(ws)
            flat[name] = v
        norms = {k: torch.linalg.vector_norm(v, dtype=torch.float64) for k, v in flat.items()}
        out = {f"contrast_global_grad_{k}": float(n) for k, n in norms.items()}
        for k in ("nce", "rnc"):
            den = float(norms["base"] * norms[k])
            out[f"contrast_global_cos_{k}_base"] = float(torch.dot(flat[k].double(), flat["base"].double())) / den if den else 0.
        return out
