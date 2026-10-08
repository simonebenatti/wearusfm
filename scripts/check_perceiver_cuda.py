#!/usr/bin/env python3
"""CUDA regression for Perceiver safe_scores, including checkpoint recomputation."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts


def run_case(dtype: torch.dtype) -> dict:
    torch.manual_seed(0)
    device = torch.device("cuda")
    counts, patches, dim, heads, latents = [16, 8], 40, 384, 6, 64
    offsets = offsets_from_counts(counts).to(device)
    visible = torch.ones(sum(counts), patches, dtype=torch.bool, device=device)
    visible[: counts[0], 7] = False
    visible[counts[0] :] = False

    plain = PerceiverPooling(dim, heads, latents).to(device).train()
    recomputed = PerceiverPooling(dim, heads, latents).to(device).train()
    recomputed.load_state_dict(plain.state_dict())
    x_plain = torch.randn(sum(counts), patches, dim, device=device, requires_grad=True)
    x_recomputed = x_plain.detach().clone().requires_grad_()
    amp = dtype == torch.bfloat16

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out_plain = plain(x_plain, offsets, visible)
        out_recomputed = checkpoint(
            lambda value: recomputed(value, offsets, visible),
            x_recomputed,
            use_reentrant=False,
        )
        loss_plain = out_plain.float().square().mean()
        loss_recomputed = out_recomputed.float().square().mean()

    assert torch.isfinite(out_plain).all() and torch.isfinite(out_recomputed).all()
    assert torch.isfinite(loss_plain) and torch.isfinite(loss_recomputed)
    atol, rtol = ((2e-2, 2e-2) if amp else (2e-5, 2e-5))
    torch.testing.assert_close(out_recomputed, out_plain, atol=atol, rtol=rtol)
    loss_plain.backward()
    loss_recomputed.backward()

    assert x_plain.grad is not None and x_recomputed.grad is not None
    assert torch.isfinite(x_plain.grad).all() and torch.isfinite(x_recomputed.grad).all()
    assert torch.count_nonzero(x_plain.grad[: counts[0], 7]) == 0
    assert torch.count_nonzero(x_plain.grad[counts[0] :]) == 0
    assert torch.count_nonzero(x_plain.grad[: counts[0], [0, 1, 2]]) > 0
    torch.testing.assert_close(x_recomputed.grad, x_plain.grad, atol=atol, rtol=rtol)

    max_parameter_grad_diff = 0.0
    for (name_a, parameter_a), (name_b, parameter_b) in zip(plain.named_parameters(), recomputed.named_parameters()):
        assert name_a == name_b and parameter_a.grad is not None and parameter_b.grad is not None
        assert torch.isfinite(parameter_a.grad).all() and torch.isfinite(parameter_b.grad).all()
        torch.testing.assert_close(parameter_b.grad, parameter_a.grad, atol=atol, rtol=rtol)
        max_parameter_grad_diff = max(
            max_parameter_grad_diff,
            float((parameter_b.grad.float() - parameter_a.grad.float()).abs().max()),
        )

    return {
        "dtype": str(dtype).removeprefix("torch."),
        "output_finite": True,
        "loss_finite": True,
        "all_gradients_finite": True,
        "fully_hidden_input_gradient_zero": True,
        "checkpoint_forward_matches": True,
        "checkpoint_backward_matches": True,
        "max_output_abs_diff": float((out_recomputed.float() - out_plain.float()).abs().max()),
        "max_input_grad_abs_diff": float((x_recomputed.grad.float() - x_plain.grad.float()).abs().max()),
        "max_parameter_grad_abs_diff": max_parameter_grad_diff,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    results = [run_case(torch.float32), run_case(torch.bfloat16)]
    print(json.dumps({
        "perceiver_safe_scores_cuda": "passed",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": results,
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
