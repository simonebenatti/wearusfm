"""Replay one zero-weight distributed backward with matched full state, batch and RNG."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from contrastive_cuda_check import build_batches
from contrastive_prepare import config_from_dict, source_digest
from wearusfm.data.pretraining_loader import split_batch
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as trainer
from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
from wearusfm.training.jepa import make_teacher
from wearusfm.training.pairing import batch_sha256, state_sha256

CAMPAIGN = ROOT / "results/infra/contrastive_comparison_20261116"
RELEASE = CAMPAIGN / "releases/bf16_20261118/cuda.job.json"
COHORT = CAMPAIGN / "calibrated_cohort_20261118_bf16/cohort.json"
# Two independent fresh backwards for each arm in every profile, counterbalanced.
CASES = ("C0a", "CNa", "CRa", "CRb", "CNb", "C0b")
PROFILES = ("bf16_auto", "bf16_math", "fp32_math", "bf16_math_deterministic")
MATCH_FIELDS = ("initial_student", "initial_teacher", "queue", "coverage", "batch", "micro", "rng")
COMPARE = (
    ("C0a", "C0b"),
    ("CNa", "CNb"),
    ("CRa", "CRb"),
    ("C0a", "CNa"),
    ("C0a", "CRa"),
    ("C0b", "CNb"),
    ("C0b", "CRb"),
)


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def verified_file(spec, root=ROOT):
    path = root / spec["path"]
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != spec["sha256"] or ("bytes" in spec and path.stat().st_size != spec["bytes"]):
        raise ValueError(f"Input differs from pin: {path}")
    return path


def controls(precision):
    strict = precision == "bf16_math_deterministic"
    torch.use_deterministic_algorithms(strict)
    if precision != "bf16_auto":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "autocast_bf16": precision != "fp32_math",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def gradient_delta(a, b):
    """Relative vector error (symmetric norm scale), cosine and FP64 norms per block."""
    if a.keys() != b.keys() or not a:
        raise ValueError("Gradient parameter schema differs")
    terms = {}
    missing_changed = []
    for name in a:
        x, y = a[name], b[name]
        if x is None and y is None:
            continue
        if x is None or y is None:
            missing_changed.append(name)
            x = torch.zeros_like(y) if x is None else x
            y = torch.zeros_like(x) if y is None else y
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Gradient schema differs: {name}")
        x, y = x.double().flatten(), y.double().flatten()
        values = (
            float(torch.dot(x, x)),
            float(torch.dot(y, y)),
            float(torch.linalg.vector_norm(x - y)) ** 2,
            float(torch.dot(x, y)),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite gradient: {name}")
        group = name.split(".")[0]
        if name.startswith("backbone.blocks."):
            group = ".".join(name.split(".")[:3])
        for key in (group, "ALL"):
            acc = terms.setdefault(key, [0.0] * 4)
            for i, value in enumerate(values):
                acc[i] += value
    if "ALL" not in terms:
        raise ValueError("All parameter gradients missing")
    out = {}
    for key, (u, v, e, dot) in terms.items():
        nu, nv = math.sqrt(u), math.sqrt(v)
        out[key] = {
            "left_norm": nu,
            "right_norm": nv,
            "difference_norm": math.sqrt(e),
            "relative_vector_difference": math.sqrt(e) / max(nu, nv) if max(nu, nv) else 0.0,
            "cosine": dot / (nu * nv) if nu and nv else None,
        }
    return {"groups": out, "grad_presence_changed": missing_changed}


def check_match(left, right):
    differing = [key for key in MATCH_FIELDS if left[key] != right[key]]
    if differing:
        raise ValueError("Common-state evidence differs: " + ", ".join(differing))


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"], device)
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])


def warm_queue(saved):
    """Translate calibration queue timestamps to immediately before update 255.

    The calibration model stayed fixed. Keep the stored keys, IDs and ordering.
    Drop entries whose translated insertion time would precede update zero.
    Step numbers in the runtime are zero-based.
    """
    if saved["last_step"] != 511:
        raise ValueError("Unexpected calibration queue step")
    shift = 511 - 253
    entries = [
        {**entry, "step": entry["step"] - shift} for entry in saved["entries"] if entry["step"] >= shift
    ]
    if not entries:
        raise ValueError("No calibration queue keys in retained interval")
    return {**saved, "last_step": 253, "entries": entries}


def snapshot(name, cfg, initial, queue, local_batch, saved_rng, ctx, device, bf16):
    arm = name[:2]
    cfg = replace(cfg, contrastive={**cfg.contrastive, "arm": arm})
    student = attach_contrastive_head(WearUsFM(cfg.model)).to(device)
    student.load_state_dict(initial, strict=True)
    teacher = make_teacher(student)
    model = trainer._ddp_wrap(student, cfg, ctx, device) if ctx.active else None
    student.train()
    runtime = ContrastiveRuntime(cfg.contrastive)
    runtime.queue.load_state_dict(queue)
    micro = trainer._prepare(local_batch, device)
    restore_rng(saved_rng, device)
    fingerprint = {
        "initial_student": state_sha256(student.state_dict()),
        "initial_teacher": state_sha256(teacher.state_dict()),
        "queue": state_sha256(runtime.queue.state_dict()),
        "coverage": state_sha256(runtime.coverage),
        "batch": batch_sha256(local_batch),
        "micro": trainer._micro_digest([micro]),
        "rng": state_sha256(rng_state(device)),
    }
    runtime.prepare(local_batch, 254)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    failure, metrics = trainer._accumulate_terms(
        model, student, teacher, [micro], cfg, ctx, device, amp, contrastive=runtime
    )
    if failure or (metrics["contrast_nce_weight"], metrics["contrast_rnc_weight"]) != (0.0, 0.0):
        raise RuntimeError(f"Invalid zero-weight measurement: {failure}, arm {arm}")
    gradients = {
        n: p.grad.detach().to("cpu", copy=True) if p.grad is not None else None
        for n, p in student.named_parameters()
    }
    fingerprint.update(
        nce_queries=int(metrics["contrast_windows"]),
        nce_eligible=int(metrics["contrast_nce_eligible"]),
        rnc_windows=float(metrics["contrast_rnc_windows"]),
        rnc_pairs=float(metrics["contrast_rnc_pairs"]),
        nce_weight=metrics["contrast_nce_weight"],
        rnc_weight=metrics["contrast_rnc_weight"],
        after_rng=state_sha256(rng_state(device)),
        gradient_sha256=state_sha256(gradients),
        loss=float(metrics["total"]),
        trainer_norm=float(torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)),
    )
    if not all(math.isfinite(fingerprint[k]) for k in ("trainer_norm", "loss")):
        raise ValueError(f"Nonfinite measurement for {name}")
    return fingerprint, gradients


def read_inputs():
    release, cohort = load_json(RELEASE), load_json(COHORT)
    digest = source_digest()
    if digest != cohort["source_sha256"] or digest != release["source_sha256"]:
        raise ValueError("Frozen source digest differs")
    paths = {k: verified_file(v) for k, v in release["inputs"].items()}
    checkpoint = verified_file(release["calibration_checkpoint"])
    cfg = config_from_dict(load_json(verified_file(cohort["configs"]["C0"], COHORT.parent)))
    if (Path(cfg.loader.pose_root) / "pose_stats.json").resolve() != paths["pose_stats"].resolve():
        raise ValueError("Pose statistics path mismatch")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    if (
        state["step"] != 512
        or state["optimizer"]["state"]
        or state_sha256(state["student"]) != cohort["initial_state_sha256"]
        or state_sha256(state["teacher"]) != cohort["initial_state_sha256"]
    ):
        raise ValueError("Invalid fixed calibration state")
    return cfg, state["student"], warm_queue(state["contrastive_state"]["queue"]), paths, release, cohort


def one_profile(profile, output, scratch):
    ctx = trainer.dist_from_env()
    if ctx.world_size != 4 or not torch.cuda.is_available():
        raise ValueError("A four-rank CUDA allocation is required")
    torch.cuda.set_device(ctx.local_rank)
    device = torch.device("cuda", ctx.local_rank)
    cfg, initial, queue, paths, release, cohort = read_inputs()
    fixture = replace(cfg, datasets=("emg2pose",), batch_size=8)
    batch = build_batches(
        fixture,
        paths["manifest"],
        release["roots"],
        load_json(paths["scales"])["scales"],
        load_json(paths["window_seconds"])["window_s"],
    )[0]
    local_batch = split_batch(batch, 4)[ctx.rank]
    # Setup is common across profiles before switching compute backend.
    flags = controls(profile)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        seed = cfg.seed * 1_000_003 + ctx.rank + 1
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        saved_rng = rng_state(device)
        identities, pairs = {}, {}
        reference_gradients = {}
        for name in CASES:
            item, grad = snapshot(
                name, fixture, initial, queue, local_batch, saved_rng, ctx, device, flags["autocast_bf16"]
            )
            if (
                item["initial_student"] != cohort["initial_state_sha256"]
                or item["initial_teacher"] != cohort["initial_state_sha256"]
            ):
                raise ValueError("Model instance differs from frozen initialization")
            if identities:
                check_match(identities["C0a"], item)
            identities[name] = item
            # Store vectors transiently only on rank zero. All-rank hashes test DDP agreement.
            if ctx.is_main:
                reference_gradients[name] = grad
        ranks = [None] * ctx.world_size
        torch.distributed.all_gather_object(
            ranks,
            {
                "rank": ctx.rank,
                "global_batch_sha256": batch_sha256(batch),
                "cases": identities,
                "flags": flags,
            },
        )
        if ctx.is_main:
            if len({r["global_batch_sha256"] for r in ranks}) != 1:
                raise ValueError("Pre-split physical batch hashes differ between ranks")
            ranks_rng_after = {
                name: [r["cases"][name]["after_rng"] == r["cases"]["C0a"]["after_rng"] for r in ranks]
                for name in CASES
            }
            agreement = {
                name: len({r["cases"][name]["gradient_sha256"] for r in ranks}) == 1 for name in CASES
            }
            for a, b in COMPARE:
                pairs[a + "_" + b] = gradient_delta(reference_gradients[a], reference_gradients[b])
            # Parent compares precision/backend using the exact same case and matched inputs.
            torch.save(reference_gradients["C0a"], scratch / (profile + ".pt"))
            write_json(
                output / (profile + ".json"),
                {
                    "profile": profile,
                    "ranks": ranks,
                    "pairs": pairs,
                    "post_backward_rng_matches_C0a_by_rank": ranks_rng_after,
                    "reduced_gradient_agreement_by_case": agreement,
                    "initial": cohort["initial_state_sha256"],
                    "inputs": release["inputs"],
                    "calibration": release["calibration_checkpoint"],
                    "source_sha256": cohort["source_sha256"],
                    "device": torch.cuda.get_device_name(device),
                    "hostname": socket.gethostname(),
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "nccl": list(torch.cuda.nccl.version()),
                    "cudnn": torch.backends.cudnn.version(),
                },
            )
    finally:
        torch.distributed.destroy_process_group()


def unsupported(error):
    message = str(error).lower()
    return "deterministic" in message and any(
        x in message
        for x in (
            "does not have a deterministic implementation",
            "does not have a deterministic",
            "no deterministic implementation",
            "is not deterministic",
            "does not support deterministic",
        )
    )


def aggregate(profiles, scratch):
    """Check rank-wise fixture invariants, then compare gradient vectors across profiles."""
    reference = profiles["bf16_auto"]
    comparisons = {}
    for profile in profiles.values():
        if [r["rank"] for r in profile["ranks"]] != list(range(4)):
            raise ValueError("Incomplete four-rank profile")
        for a, b in zip(reference["ranks"], profile["ranks"]):
            if a["global_batch_sha256"] != b["global_batch_sha256"]:
                raise ValueError("Global batch changed between profiles")
            check_match(a["cases"]["C0a"], b["cases"]["C0a"])
    for left, right in (
        ("fp32_math", "bf16_math"),
        ("bf16_auto", "bf16_math"),
        ("bf16_math", "bf16_math_deterministic"),
    ):
        if left in profiles and right in profiles:
            a, b = [
                torch.load(str(scratch / (key + ".pt")), mmap=True, weights_only=True)
                for key in (left, right)
            ]
            result = gradient_delta(a, b)
            result["post_backward_rng_equal_by_rank"] = [
                lhs["cases"]["C0a"]["after_rng"] == rhs["cases"]["C0a"]["after_rng"]
                for lhs, rhs in zip(profiles[left]["ranks"], profiles[right]["ranks"])
            ]
            comparisons[left + "_" + right] = result
    return comparisons


def orchestrate(output):
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
    profiles, failures = {}, {}
    report = {
        "protocol": "zero-weight-fixed-state-20261118",
        "training_approved": False,
        "fixture": {
            "global_batch": 8,
            "dataset": "emg2pose",
            "update": 255,
            "queue": "calibration keys shifted by 258 steps, negative insertion times removed",
            "optimizer_updates": 0,
            "muscle_dropout": 0.4,
            "removed_pose_rank": 0,
            "rnc_and_nce_weights": 0,
            "mixed_dataset_batch32_replayed": False,
        },
        "cases": CASES,
    }
    with tempfile.TemporaryDirectory(prefix="contrastive-gradient-") as temporary:
        scratch = Path(temporary)
        for profile in PROFILES:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                str(Path(__file__).resolve()),
                "--out-dir",
                str(output),
                "--worker",
                profile,
                "--scratch",
                str(scratch),
            ]
            log = output / (profile + ".log")
            with log.open("x") as handle:
                proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            if proc.returncode:
                markers = [load_json(p) for p in output.glob(profile + ".error.rank*.json")]
                category = (
                    "unsupported_deterministic"
                    if profile == "bf16_math_deterministic"
                    and any(m["unsupported"] for m in markers)
                    and all(m["unsupported"] for m in markers)
                    else "failed"
                )
                failures[profile] = {
                    "category": category,
                    "errors": markers,
                    "exit": proc.returncode,
                    "log": log.name,
                }
                if category == "failed":
                    break
            else:
                profiles[profile] = load_json(output / (profile + ".json"))
        report.update(completed_profiles=list(profiles), failures=failures)
        if "bf16_auto" in profiles:
            try:
                report["cross_profile"] = aggregate(profiles, scratch)
            except (RuntimeError, ValueError, OSError) as exc:
                failures["aggregation"] = {"category": "failed", "message": str(exc)}
        unsupported_losses = {
            k: [
                case
                for case in CASES
                if any(
                    r["cases"][case]["nce_eligible"] <= 0 or r["cases"][case]["rnc_windows"] <= 0
                    for r in p["ranks"]
                )
            ]
            for k, p in profiles.items()
        }
        report["cases_without_loss_support"] = unsupported_losses
        rng_mismatches = {
            key: [
                name
                for name, ranks in value["post_backward_rng_matches_C0a_by_rank"].items()
                if not all(ranks)
            ]
            for key, value in profiles.items()
        }
        cross_rng_mismatches = [
            key
            for key, value in report.get("cross_profile", {}).items()
            if not all(value["post_backward_rng_equal_by_rank"])
        ]
        report["post_backward_rng_disagreements"] = rng_mismatches
        report["cross_profile_rng_disagreements"] = cross_rng_mismatches
        rank_mismatches = {
            k: [n for n, ok in p["reduced_gradient_agreement_by_case"].items() if not ok]
            for k, p in profiles.items()
        }
        report["reduced_gradient_disagreements"] = rank_mismatches
        if any(v["category"] == "failed" for v in failures.values()) or any(rank_mismatches.values()):
            report["status"] = "FAILED_REQUIRES_REVIEW"
        elif any(rng_mismatches.values()) or cross_rng_mismatches:
            report["status"] = "INCONCLUSIVE_RNG_CONSUMPTION"
        elif any(unsupported_losses.values()):
            report["status"] = "INCONCLUSIVE_MISSING_SUPPORT"
        elif failures:
            report["status"] = "INCONCLUSIVE_UNSUPPORTED_DETERMINISM"
        else:
            report["status"] = "MEASURED_FOR_REVIEW"
        write_json(output / "report.json", report)
    return 0 if report["status"] == "MEASURED_FOR_REVIEW" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--worker", choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            one_profile(args.worker, args.out_dir, args.scratch)
        except Exception as error:
            rank = int(os.environ.get("RANK", "-1"))
            write_json(
                args.out_dir / f"{args.worker}.error.rank{rank}.json",
                {"unsupported": unsupported(error), "type": type(error).__name__, "message": str(error)},
            )
            raise
        return 0
    return orchestrate(args.out_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
