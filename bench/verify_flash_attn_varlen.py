#!/usr/bin/env python3
"""Prova funzionale di flash_attn_varlen_func in bf16 su A100.

Precondizione del braccio L2 (packing), non ancora verificata (dataloader_bench_spec.md
§6: "l'import non e' una verifica"; fatto da verificare #9 in fatti_da_verificare.md).
Richiede CUDA: non gira sul Mac.

Simula esattamente l'uso previsto (spec §6): cross-attention del Perceiver, query = K
latenti per campione (fissi), chiavi/valori = C token di canale per campione (variabile
per campione, spec §2 - il campione e' il montaggio intero). cu_seqlens_q e cu_seqlens_k
distinti, come richiesto dalla spec.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/step0/flash_attn_varlen_check.json")
    p.add_argument("--k-latents", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    # C variabile per campione, come nella distribuzione sintetica (spec §3): dal caso
    # sparso (8) alla griglia HD piena (256)
    p.add_argument("--c-per-sample", type=int, nargs="+", default=[8, 12, 16, 32, 64, 128, 256])
    args = p.parse_args()

    result: dict = {"c_per_sample": args.c_per_sample, "k_latents": args.k_latents}
    try:
        import torch
        from flash_attn import flash_attn_varlen_func

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            result["status"] = "FAIL"
            result["error"] = "CUDA non disponibile: la prova richiede una GPU"
            _write(args.out, result)
            return 1

        import flash_attn

        result["flash_attn_version"] = flash_attn.__version__
        device = torch.device("cuda")

        b = len(args.c_per_sample)
        k = args.k_latents
        h = args.n_heads
        d = args.head_dim

        cu_seqlens_q = torch.arange(0, (b + 1) * k, k, dtype=torch.int32, device=device)
        c_tensor = torch.tensor(args.c_per_sample, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(c_tensor, dim=0).to(torch.int32)]
        )

        total_q = b * k
        total_k = int(sum(args.c_per_sample))
        max_seqlen_q = k
        max_seqlen_k = max(args.c_per_sample)

        q = torch.randn(total_q, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        kk = torch.randn(total_k, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(total_k, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)

        out = flash_attn_varlen_func(
            q, kk, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )
        result["forward_output_shape"] = list(out.shape)
        result["forward_output_dtype"] = str(out.dtype)
        assert out.shape == (total_q, h, d), f"forma inattesa: {out.shape}"
        assert torch.isfinite(out).all(), "output non finito dopo il forward"

        loss = out.float().pow(2).mean()
        loss.backward()

        grads_ok = (
            q.grad is not None and torch.isfinite(q.grad).all()
            and kk.grad is not None and torch.isfinite(kk.grad).all()
            and v.grad is not None and torch.isfinite(v.grad).all()
        )
        result["backward_grads_finite"] = bool(grads_ok)
        assert grads_ok, "gradienti assenti o non finiti dopo il backward"

        result["status"] = "OK"
        result["gpu_name"] = torch.cuda.get_device_name(0)

    except Exception as e:  # noqa: BLE001 - vogliamo il traceback completo nel JSON
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        _write(args.out, result)
        print(json.dumps(result, indent=2))
        return 1

    _write(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


def _write(out: str, result: dict) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
