"""Observe one production backward at the DDP communication boundary.

GradBucket members are the actual parameter objects. Their membership, launch
order, and offsets come from the hook in this iteration, not DDP's delayed log.
The registered reference allreduce hook remains installed for the entire leg.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook


SCHEMA = "cn-restart-gradients-v1"


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_hash(tensor):
    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def _norm(tensor):
    norm = torch.linalg.vector_norm(tensor, dtype=torch.float64).item()
    if not math.isfinite(norm):
        raise ValueError("nonfinite observed gradient")
    return norm


def _average_observed(state, bucket):
    if state.capturing:
        state._snapshot(bucket)
    # With a user hook DDP does not pre-divide the bucket. PyTorch's reference
    # hook divides by group size, then SUMs in the original dtype, asynchronously.
    return allreduce_hook(state.group, bucket)


class ReductionObservation:
    def __init__(self, ddp, student, *, expected_parameters=270):
        self.group = ddp.process_group
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        self.parameters = [(name, param) for name, param in student.named_parameters() if param.requires_grad]
        if len(self.parameters) != expected_parameters:
            raise ValueError("unexpected number of trainable parameters")
        self.indices = {id(p): i for i, (_, p) in enumerate(self.parameters)}
        self.capturing = False
        self.started = False
        self.buckets = []
        ddp.register_comm_hook(self, _average_observed)

    def start(self):
        if self.started:
            raise ValueError("step 545 may only be observed once")
        self.started = True
        self.capturing = True

    def _snapshot(self, bucket):
        buffer = bucket.buffer().detach()
        if buffer.ndim != 1 or not buffer.is_contiguous() or not buffer.is_floating_point():
            raise ValueError("unsupported DDP gradient buffer")
        members = []
        for param, view in zip(bucket.parameters(), bucket.gradients(), strict=True):
            if (view.dtype != buffer.dtype or view.dtype != param.dtype or view.shape != param.shape
                    or view.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()):
                raise ValueError("DDP bucket gradient view does not identify the parameter")
            index = self.indices[id(param)]
            members.append((index, tuple(view.shape), tuple(view.stride()),
                            view.storage_offset() - buffer.storage_offset()))
        # One async device-to-device copy per bucket, before allreduce_hook mutates
        # the native unscaled gradient. All host IO is deferred until backward ends.
        self.buckets.append((bucket.index(), members, buffer.clone(), buffer))

    def finish(self, directory, rank):
        if not self.capturing or rank != self.rank:
            raise ValueError("missing observation or incorrect process-group rank")
        self.capturing = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensor_path = directory / f"vectors.rank{rank}.pt"
        receipt_path = directory / f"observation.rank{rank}.json"
        if tensor_path.exists() or receipt_path.exists():
            raise FileExistsError("step-545 evidence already exists")

        nparams = len(self.parameters)
        metadata = [None] * nparams
        vectors = {stage: [None] * nparams for stage in ("local", "reduced")}
        layout = []
        for index, members, local, reduced in self.buckets:
            # The reference allreduce uses the original buffer in-place. This is
            # called after the real backward and before clipping or optimizer.step.
            buffers = {"local": local.cpu(), "reduced": reduced.to(device="cpu", copy=True)}
            layout.append({"index": index, "nbytes": reduced.numel() * reduced.element_size(),
                           "dtype": str(reduced.dtype), "parameter_ids": [m[0] for m in members]})
            for position, (pid, shape, stride, offset) in enumerate(members):
                name, param = self.parameters[pid]
                if metadata[pid] is not None or param.grad is None:
                    raise ValueError("duplicated bucket member or missing reduced gradient")
                row = {"id": pid, "name": name, "shape": list(shape), "dtype": str(param.dtype),
                       "bucket_index": index, "bucket_position": position, "bucket_element_offset": offset}
                for stage, source in buffers.items():
                    view = source.as_strided(shape, stride, offset)
                    vectors[stage][pid] = view
                    row[stage + "_sha256"] = _tensor_hash(view)
                    row[stage + "_norm"] = _norm(view)
                grad = param.grad.detach()
                if (grad.shape != param.shape or grad.dtype != param.dtype
                        or _tensor_hash(grad.cpu()) != row["reduced_sha256"]):
                    raise ValueError("reduced bucket differs from the actual preclip parameter gradient")
                metadata[pid] = row
        self.buckets.clear()
        if any(row is None for row in metadata) or len({b["index"] for b in layout}) != len(layout):
            raise ValueError("incomplete step-545 bucket coverage")

        with tensor_path.open("xb") as stream:
            torch.save(vectors, stream)
        receipt = {"schema": SCHEMA, "update": 545, "rank": rank, "world_size": self.world_size,
                   "buckets": layout, "parameters": metadata, "vectors": {
                       "file": tensor_path.name, "bytes": tensor_path.stat().st_size, "sha256": _file_hash(tensor_path)}}
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, allow_nan=False, indent=2)
            stream.write("\n")
        return receipt


def load_observation(directory, rank, expected_parameters=270):
    """Read back and verify the rank's inventory and original, native-dtype vectors."""
    directory = Path(directory)
    tensor_path = directory / f"vectors.rank{rank}.pt"
    record = json.loads((directory / f"observation.rank{rank}.json").read_text())
    pin = record["vectors"]
    if (record["schema"] != SCHEMA or record["update"] != 545 or record["rank"] != rank
            or not 0 <= rank < record["world_size"] or tensor_path.is_symlink()
            or pin["file"] != tensor_path.name or pin["bytes"] != tensor_path.stat().st_size
            or pin["sha256"] != _file_hash(tensor_path)):
        raise ValueError("invalid observation receipt or vector pin")
    vectors = torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)
    rows, buckets = record["parameters"], record["buckets"]
    if (len(rows) != expected_parameters or len({r["name"] for r in rows}) != len(rows)
            or set(vectors) != {"local", "reduced"} or any(len(v) != len(rows) for v in vectors.values())
            or sorted(b["index"] for b in buckets) != list(range(len(buckets)))
            or sorted(pid for b in buckets for pid in b["parameter_ids"]) != list(range(len(rows)))):
        raise ValueError("invalid parameter or bucket inventory")
    for bucket in buckets:
        occupied = []
        for pos, pid in enumerate(bucket["parameter_ids"]):
            row = rows[pid]
            if ((row["id"], row["bucket_index"], row["bucket_position"]) != (pid, bucket["index"], pos)
                    or row["dtype"] != bucket["dtype"]):
                raise ValueError("bucket/parameter mapping differs")
            for stage in ("local", "reduced"):
                tensor = vectors[stage][pid]
                if (not tensor.is_floating_point() or list(tensor.shape) != row["shape"]
                        or str(tensor.dtype) != row["dtype"] or _tensor_hash(tensor) != row[stage + "_sha256"]
                        or tensor.storage_offset() != row["bucket_element_offset"]
                        or not math.isclose(row[stage + "_norm"], _norm(tensor), rel_tol=1e-12)):
                    raise ValueError("observed gradient differs from its receipt")
            occupied.append((row["bucket_element_offset"] * tensor.element_size(), tensor.numel() * tensor.element_size()))
        # Model parameters are dense, even if their strides are non-contiguous.
        end = 0
        for offset, length in sorted(occupied):
            if offset < end or offset + length > bucket["nbytes"]:
                raise ValueError("parameter span outside its bucket or overlapped")
            end = offset + length
    return record, vectors
