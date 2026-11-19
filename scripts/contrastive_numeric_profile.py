"""Explicit prospective CUDA profile for the RMS100/fast125 cohort.

This is a process-scoped backend contract, not a change of loss or precision in
archived recipes. Callers still use the trainer's bf16 autocast. Actual autocast
is recorded inside the parity backward, not inferred from this outer context.
"""
from __future__ import annotations
from contextlib import contextmanager
import os

import torch

PROFILE = "bf16_math_tf32_off"


class NumericProfileUnavailable(RuntimeError):
    """Required execution profile cannot be provided; comparisons are inconclusive."""


def _cuda_autocast_flags():
    """Read actual CUDA autocast, including PyTorch 2.2's device-less APIs."""
    enabled = getattr(torch, "is_autocast_enabled", None)
    if not callable(enabled):
        raise NumericProfileUnavailable("CUDA autocast enabled-state API unavailable")
    try:
        active = enabled("cuda")
    except TypeError:
        # Before the device argument was introduced this API specifically read
        # CUDA, not CPU autocast. Do not substitute a planned precision flag.
        active = enabled()

    dtype = getattr(torch, "get_autocast_dtype", None)
    if callable(dtype):
        try:
            return active, str(dtype("cuda"))
        except TypeError:
            pass
    legacy_dtype = getattr(torch, "get_autocast_gpu_dtype", None)
    if not callable(legacy_dtype):
        raise NumericProfileUnavailable("CUDA autocast dtype API unavailable")
    return active, str(legacy_dtype())


def _numeric_flags():
    cuda = torch.backends.cuda
    autocast_active, autocast_dtype = _cuda_autocast_flags()
    return {
        "profile": PROFILE,
        "matmul_tf32": bool(cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "math_sdp_enabled": bool(cuda.math_sdp_enabled()),
        "flash_sdp_enabled": bool(cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(cuda.mem_efficient_sdp_enabled()),
        "cudnn_sdp_enabled": bool(cuda.cudnn_sdp_enabled()) if hasattr(cuda, "cudnn_sdp_enabled") else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction": bool(cuda.matmul.allow_bf16_reduced_precision_reduction),
        "fp16_reduced_precision_reduction": bool(cuda.matmul.allow_fp16_reduced_precision_reduction),
        "autocast_cuda_enabled": autocast_active,
        "autocast_cuda_dtype": autocast_dtype,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def numeric_flags():
    try:
        return _numeric_flags()
    except (AttributeError, TypeError, NotImplementedError) as exc:
        raise NumericProfileUnavailable("required numeric-profile API unavailable: " + str(exc)) from exc


def assert_numeric_flags(flags, *, require_autocast=False):
    wanted = {
        "profile": PROFILE, "matmul_tf32": False, "cudnn_tf32": False,
        "math_sdp_enabled": True, "flash_sdp_enabled": False,
        "mem_efficient_sdp_enabled": False, "deterministic_algorithms": False,
        "cudnn_deterministic": False, "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
    }
    if flags.get("cudnn_sdp_enabled") not in (False, None):
        raise NumericProfileUnavailable("required profile has cuDNN SDPA enabled")
    differing = [key for key, value in wanted.items() if flags.get(key) != value]
    if require_autocast and (flags.get("autocast_cuda_enabled") is not True or
                             flags.get("autocast_cuda_dtype") != "torch.bfloat16"):
        differing.append("actual bf16 CUDA autocast")
    if differing:
        raise NumericProfileUnavailable("required numeric profile differs: " + ", ".join(differing))
    return flags


def assert_numeric_profile(*, require_cuda=True, require_autocast=False):
    if require_cuda and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise NumericProfileUnavailable("required CUDA bf16 profile is unavailable")
    return assert_numeric_flags(numeric_flags(), require_autocast=require_autocast)


@contextmanager
def numeric_profile(*, require_cuda=True):
    """Select ordinary bf16 math, TF32 off; restore every altered flag on exit.

    Deterministic algorithms are deliberately not enabled: that is a different
    profile. Exact gradient equality is measured, not guaranteed by a flag.
    """
    before = numeric_flags()
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        yield assert_numeric_profile(require_cuda=require_cuda)
    finally:
        torch.use_deterministic_algorithms(before["deterministic_algorithms"],
                                           warn_only=before["deterministic_warn_only"])
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.backends.cuda.enable_math_sdp(before["math_sdp_enabled"])
        torch.backends.cuda.enable_flash_sdp(before["flash_sdp_enabled"])
        torch.backends.cuda.enable_mem_efficient_sdp(before["mem_efficient_sdp_enabled"])
        if before["cudnn_sdp_enabled"] is not None:
            torch.backends.cuda.enable_cudnn_sdp(before["cudnn_sdp_enabled"])
