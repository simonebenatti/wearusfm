"""Physical identities and deterministic training-only candidate memory.

No collective or mutation happens on construction or import. ``snapshot`` is read-only;
``commit`` is called once AFTER a successful optimizer update, using pre-update teacher
keys. Steps are zero based: the first completed update is committed with ``step=0``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class WindowIdentity:
    dataset: str
    subject: str
    session: str
    trial: int | None
    start: int
    stop: int
    fs: float

    def __post_init__(self):
        for name in ("dataset", "subject", "session"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"contrastive identity requires nonempty {name}")
        if self.trial is not None and (not isinstance(self.trial, Integral) or isinstance(self.trial, bool)):
            raise ValueError("trial must be an integer or explicit None for a continuous recording")
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (self.start, self.stop)) or not 0 <= self.start < self.stop:
            raise ValueError("window requires integer 0 <= start < stop")
        if not isinstance(self.fs, Real) or isinstance(self.fs, bool) or not math.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("window sampling frequency must be finite and positive")
        object.__setattr__(self, "trial", None if self.trial is None else int(self.trial))
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "stop", int(self.stop))
        object.__setattr__(self, "fs", float(self.fs))

    @property
    def recording(self):
        return self.dataset, self.subject, self.session

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("incomplete or unknown physical identity fields")
        return cls(**value)


def identities_from_batch(batch) -> tuple[WindowIdentity, ...]:
    """Read the physical origin before signal/device conversion; montage is not identity."""
    if not (len(batch.rows) == len(batch.windows) == len(batch.fs)):
        raise ValueError("contrastive origin lengths differ")
    out = []
    for row, window, fs in zip(batch.rows, batch.windows, batch.fs):
        if any(name not in row or row[name] is None for name in ("dataset", "subject", "session")):
            raise ValueError("contrastive physical origin missing dataset/subject/session")
        if not all(hasattr(window, name) for name in ("trial", "start", "stop")):
            raise ValueError("contrastive origin requires explicit trial/start/stop")
        out.append(WindowIdentity(row["dataset"], str(row["subject"]), str(row["session"]),
                                  window.trial, window.start, window.stop, fs))
    return tuple(out)


def admissible_mask(queries, candidates, *, gap_seconds: float = 4.0, device=None) -> torch.Tensor:
    """NCE negative eligibility, with no labels or montage-based identities.

    Same subject alone is NOT excluded. Different trials within a recording lack a
    common coordinate axis and are conservatively excluded; explicit trial=None
    denotes a valid continuous coordinate axis. A gap exactly equal to the margin
    is admissible. Equal sampling rates use integer sample gaps, avoiding loss of
    precision when subtracting large absolute timestamps. Different rates use seconds.
    """
    if not isinstance(gap_seconds, Real) or isinstance(gap_seconds, bool) or not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("negative exclusion margin must be finite and nonnegative")
    if any(not isinstance(x, WindowIdentity) for x in (*queries, *candidates)):
        raise ValueError("negative eligibility requires validated physical identities")
    rows = []
    for q in queries:
        row = []
        for c in candidates:
            keep = q.dataset == c.dataset and q != c
            if keep and q.recording == c.recording:
                if q.trial != c.trial:
                    keep = False
                else:
                    if q.fs == c.fs:
                        gap_samples = max(q.start, c.start) - min(q.stop, c.stop)
                        keep = gap_samples >= gap_seconds * q.fs
                    else:
                        gap = max(q.start / q.fs, c.start / c.fs) - min(q.stop / q.fs, c.stop / c.fs)
                        keep = gap >= gap_seconds
            row.append(keep)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device).reshape(len(queries), len(candidates))


@dataclass(frozen=True)
class CandidateSnapshot:
    keys: torch.Tensor
    identities: tuple[WindowIdentity, ...]
    steps: tuple[int, ...]
    from_current: tuple[bool, ...]
    global_indices: tuple[int, ...]
    duplicate_current: tuple[str, ...] = ()
    replaced_previous: tuple[str, ...] = ()


def candidate_diagnostics(queries, snapshot, negative_mask, step, *, datasets=()):
    """Additive counts by query dataset; no labels, tensor gradients or mutation.

    Age/current/subject/recording counts refer to admissible negative pairs, even
    for a query below min_negatives. Snapshot duplicate counts are GLOBAL and must
    be averaged (not summed) over ranks by callers.
    """
    names = ("negative_count", "negative_age_sum", "negative_current", "negative_same_subject",
             "negative_same_recording", "negative_age_0", "negative_age_1_32", "negative_age_33_128",
             "negative_age_129_256", "negative_age_257_plus", "excluded_overlap")
    datasets = sorted(set(datasets) | {i.dataset for i in (*queries, *snapshot.identities)})
    totals = {name: 0 for name in names}
    by_dataset = {d: dict(totals) for d in datasets}
    mask = negative_mask.detach().cpu().tolist()
    for query, row in zip(queries, mask):
        values = by_dataset[query.dataset]
        for keep, candidate, written, current in zip(row, snapshot.identities, snapshot.steps, snapshot.from_current):
            if not keep:
                if query.recording == candidate.recording and query.trial == candidate.trial:
                    overlap = (max(query.start, candidate.start) < min(query.stop, candidate.stop) if query.fs == candidate.fs
                               else max(query.start / query.fs, candidate.start / candidate.fs) < min(query.stop / query.fs, candidate.stop / candidate.fs))
                    values["excluded_overlap"] += int(overlap)
                continue
            age = step - written
            values["negative_count"] += 1
            values["negative_age_sum"] += age
            values["negative_current"] += int(current)
            values["negative_same_subject"] += int(query.subject == candidate.subject)
            values["negative_same_recording"] += int(query.recording == candidate.recording)
            bucket = "0" if age == 0 else "1_32" if age <= 32 else "33_128" if age <= 128 else "129_256" if age <= 256 else "257_plus"
            values["negative_age_" + bucket] += 1
    result = {}
    for dataset, values in by_dataset.items():
        for name, value in values.items():
            totals[name] += value
            result[f"contrast_nce_{name}__{dataset}"] = value
    result.update({f"contrast_nce_{name}": value for name, value in totals.items()})
    for name, duplicate_datasets in (("duplicate_current_discarded", snapshot.duplicate_current),
                                      ("previous_keys_replaced", snapshot.replaced_previous)):
        result[f"contrast_nce_{name}"] = len(duplicate_datasets)
        for dataset in datasets:
            result[f"contrast_nce_{name}__{dataset}"] = duplicate_datasets.count(dataset)
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class CandidateQueue:
    """Per-dataset FIFO, newest physical key wins, deterministic rank/index ties.

    All ranks receive identical globally gathered inputs. Stored keys are detached
    CPU clones, so state serialization has no CUDA-device dependency. Snapshot
    moves only retained keys to the current key tensor's device and dtype.
    """

    VERSION = 1

    def __init__(self, capacity_per_dataset: int = 1024, max_age: int = 256):
        _integer(capacity_per_dataset, "capacity_per_dataset", 1)
        _integer(max_age, "max_age", 1)
        self.capacity_per_dataset = capacity_per_dataset
        self.max_age = max_age
        self.last_step = -1
        self.feature_dim = None
        self.dtype = None
        self._entries = []  # (identity, key CPU clone, step, global index)

    def __len__(self):
        return len(self._entries)

    def _inputs(self, keys, identities, valid, step):
        _integer(step, "step")
        if step != self.last_step + 1:
            raise ValueError(f"queue expects update {self.last_step + 1}, got {step}")
        identities = tuple(identities)
        if not isinstance(keys, torch.Tensor) or keys.ndim != 2 or keys.shape[1] < 1 or not keys.is_floating_point() or len(identities) != len(keys):
            raise ValueError("keys must be floating (N, D) aligned with physical identities")
        if self.feature_dim is not None and (keys.shape[1] != self.feature_dim or keys.dtype != self.dtype):
            raise ValueError("candidate key dimension or dtype changed")
        if any(not isinstance(m, WindowIdentity) for m in identities):
            raise ValueError("candidate memory requires validated physical identities")
        if valid is None:
            valid = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
        if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != (len(keys),):
            raise ValueError("candidate validity must be boolean (N,)")
        valid = valid.to(keys.device)
        if not bool(torch.isfinite(keys[valid]).all()):
            raise ValueError("nonfinite valid candidate key")
        # Lowest global rank/index wins exact current-batch ties. Invalid copies
        # cannot erase an older valid key or a later valid current-batch copy.
        seen, current = set(), []
        for i, keep in enumerate(valid.detach().cpu().tolist()):
            ident = identities[i]
            if keep and ident not in seen:
                current.append((ident, keys[i].detach().cpu().clone(), step, i))
                seen.add(ident)
        return current

    def snapshot(self, step, current_keys, current_ids, current_valid=None) -> CandidateSnapshot:
        current = self._inputs(current_keys, current_ids, current_valid, step)
        current_ids_set = {entry[0] for entry in current}
        previous = [e for e in self._entries if step - e[2] <= self.max_age and e[0] not in current_ids_set]
        entries = previous + current
        keys = torch.stack([e[1] for e in entries]).to(current_keys) if entries else current_keys.detach().new_empty((0, current_keys.shape[1]))
        seen, duplicates = set(), []
        valid_rows = [True] * len(current_ids) if current_valid is None else current_valid.detach().cpu().tolist()
        for ident, keep in zip(current_ids, valid_rows):
            if keep and ident in seen:
                duplicates.append(ident.dataset)
            if keep:
                seen.add(ident)
        replaced = tuple(e[0].dataset for e in self._entries if step - e[2] <= self.max_age and e[0] in current_ids_set)
        return CandidateSnapshot(keys.detach(), tuple(e[0] for e in entries), tuple(e[2] for e in entries),
                                 tuple([False] * len(previous) + [True] * len(current)), tuple(e[3] for e in entries),
                                 tuple(duplicates), replaced)

    def commit(self, keys, identities, valid=None, step=None) -> None:
        if step is None:
            raise ValueError("commit requires an explicit successful update index")
        current = self._inputs(keys, identities, valid, step)
        current_ids = {e[0] for e in current}
        entries = [e for e in self._entries if step - e[2] <= self.max_age and e[0] not in current_ids] + current
        counts, kept = {}, []
        for entry in reversed(entries):
            dataset = entry[0].dataset
            if counts.get(dataset, 0) < self.capacity_per_dataset:
                kept.append(entry)
                counts[dataset] = counts.get(dataset, 0) + 1
        self._entries = list(reversed(kept))
        self.last_step, self.feature_dim, self.dtype = step, keys.shape[1], keys.dtype

    def state_dict(self):
        return {"version": self.VERSION, "capacity_per_dataset": self.capacity_per_dataset, "max_age": self.max_age,
                "last_step": self.last_step, "feature_dim": self.feature_dim,
                "dtype": None if self.dtype is None else str(self.dtype),
                "entries": [{"identity": e[0].to_dict(), "key": e[1].clone(), "step": e[2], "global_index": e[3]}
                            for e in self._entries]}

    def load_state_dict(self, state):
        required = {"version", "capacity_per_dataset", "max_age", "last_step", "feature_dim", "dtype", "entries"}
        if not isinstance(state, dict) or set(state) != required or type(state["version"]) is not int or state["version"] != self.VERSION:
            raise ValueError("invalid contrastive queue state schema/version")
        _integer(state["capacity_per_dataset"], "capacity_per_dataset", 1)
        _integer(state["max_age"], "max_age", 1)
        if state["capacity_per_dataset"] != self.capacity_per_dataset or state["max_age"] != self.max_age:
            raise ValueError("contrastive memory policy changed on resume")
        last = state["last_step"]
        _integer(last, "last_step", -1)
        if not isinstance(state["entries"], list):
            raise ValueError("queue entries must be a list")
        dim, dtype = state["feature_dim"], state["dtype"]
        if last == -1:
            if dim is not None or dtype is not None or state["entries"]:
                raise ValueError("uninitialized queue has stored state")
            new_dtype = None
        else:
            _integer(dim, "feature_dim", 1)
            types = {str(t): t for t in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
            if not isinstance(dtype, str) or dtype not in types:
                raise ValueError("invalid stored key dtype")
            new_dtype = types[dtype]
        entries, seen, counts, previous_order = [], set(), {}, None
        for entry in state["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"identity", "key", "step", "global_index"}:
                raise ValueError("invalid queue entry schema")
            ident = WindowIdentity.from_dict(entry["identity"])
            _integer(entry["step"], "entry step")
            _integer(entry["global_index"], "global index")
            order = entry["step"], entry["global_index"]
            if entry["step"] > last or last - entry["step"] > self.max_age or (previous_order is not None and order <= previous_order):
                raise ValueError("queue order or age is inconsistent")
            if ident in seen:
                raise ValueError("duplicate physical key in queue checkpoint")
            key = entry["key"]
            if not isinstance(key, torch.Tensor) or key.shape != (dim,) or key.dtype != new_dtype or not bool(torch.isfinite(key).all()):
                raise ValueError("invalid stored candidate key")
            counts[ident.dataset] = counts.get(ident.dataset, 0) + 1
            if counts[ident.dataset] > self.capacity_per_dataset:
                raise ValueError("stored memory exceeds per-dataset capacity")
            entries.append((ident, key.detach().cpu().clone(), entry["step"], entry["global_index"]))
            seen.add(ident)
            previous_order = order
        # No partial mutation if any field above fails validation.
        self.last_step, self.feature_dim, self.dtype, self._entries = last, dim, new_dtype, entries


def _distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def gather_objects(local_items):
    """Flatten sequence objects in deterministic rank/index order."""
    if not isinstance(local_items, (list, tuple)):
        raise ValueError("gather_objects takes a sequence, not one opaque object")
    if not _distributed():
        return list(local_items)
    parts = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(parts, list(local_items))
    return [item for part in parts for item in part]


def _gather_layout(value, with_grad=False):
    if value.ndim < 1:
        raise ValueError("variable gather needs a leading candidate dimension")
    if not _distributed():
        return [value.shape[0]]
    meta = [(value.shape[0], tuple(value.shape[1:]), str(value.dtype), value.device.type,
             bool(with_grad), bool(value.requires_grad) if with_grad else False)]
    rows = gather_objects(meta)
    if any(row[1:] != rows[0][1:] for row in rows):
        raise ValueError("distributed candidate shapes/dtypes/devices/gradient participation differ")
    return [row[0] for row in rows]


def _gather_tensor(value, sizes):
    max_size = max(1, max(sizes))
    padded = value.new_zeros((max_size, *value.shape[1:]))
    padded[:len(value)] = value
    chunks = [torch.empty_like(padded) for _ in sizes]
    torch.distributed.all_gather(chunks, padded.contiguous())
    return torch.cat([part[:n] for part, n in zip(chunks, sizes)], dim=0)


class _GatherVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, sizes):
        ctx.sizes = sizes
        ctx.rank = torch.distributed.get_rank()
        return _gather_tensor(value, sizes)

    @staticmethod
    def backward(ctx, grad_output):
        # Every rank may use every candidate. SUM the contributions from all ranks,
        # including empty local ranks, before returning this rank's input gradient.
        grad = grad_output.contiguous().clone()
        if grad.numel():
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
        begin = sum(ctx.sizes[:ctx.rank])
        return grad[begin:begin + ctx.sizes[ctx.rank]], None


def gather_variable(value: torch.Tensor, *, with_grad: bool = False) -> torch.Tensor:
    """Gather variable candidate counts, including zero on a rank.

    With gradients each rank must participate in backward, even if its local loss
    contribution is zero. Backward SUMs all uses of every candidate. If every rank
    evaluates the same FULL global loss, DDP's subsequent mean cancels this sum;
    do not multiply that loss by world_size a second time. Detached keys are the
    default for NCE. This helper does not mutate queue state.
    """
    sizes = _gather_layout(value, with_grad)
    if not _distributed():
        return value if with_grad else value.detach()
    if with_grad:
        if not value.is_floating_point():
            raise ValueError("differentiable gather needs floating candidates")
        return _GatherVariable.apply(value, sizes)
    with torch.no_grad():
        return _gather_tensor(value.detach(), sizes)
