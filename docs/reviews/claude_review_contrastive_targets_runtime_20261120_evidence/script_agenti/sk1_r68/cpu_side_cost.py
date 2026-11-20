"""Order of magnitude of the CPU-side work inside one parity case for the production T100-G 30M model.

Not a GPU timing: on Leonardo the same host work (model construction, state copies,
sha256 of student/teacher/gradients, float64 norm) adds to the GPU backward.
"""
import math
import time

import torch

import contrastive_target_parity as P
from contrastive_prepare import BASELINE, _read, config_from_dict
from wearusfm.training.pairing import state_sha256

torch.set_num_threads(1)
cfg = config_from_dict(_read(BASELINE))


def timed(fn, reps=3):
    out = []
    for _ in range(reps):
        t = time.perf_counter(); value = fn(); out.append(time.perf_counter() - t)
    return value, min(out)


model, t_build = timed(lambda: P.attach_contrastive_head(P.WearUsFM(cfg.model)), reps=2)
n = sum(p.numel() for p in model.parameters())
state = P._cpu_state(model)
mb = sum(v.numel() * v.element_size() for v in state.values()) / 2**20
_, t_copy = timed(lambda: P._cpu_state(model))
_, t_hash = timed(lambda: state_sha256(model.state_dict()))
for p in model.parameters():
    p.grad = torch.randn_like(p)
grads, t_grad_copy = timed(lambda: P._gradient_state(model))
_, t_grad_hash = timed(lambda: state_sha256(grads))
_, t_norm = timed(lambda: math.sqrt(sum(float(torch.linalg.vector_norm(v.double())) ** 2 for v in grads.values() if v is not None)))
_, t_finite = timed(lambda: any(v is not None and not bool(torch.isfinite(v).all()) for v in grads.values()))
per_case = t_build + 2 * t_hash + t_grad_copy + t_grad_hash + t_norm + t_finite  # host work of _measure_case, no GPU/DDP
per_point = 6 * per_case + 2 * t_copy + 4 * t_hash  # plus live copies and before/after fingerprints
print(f"parameters {n:,}; state {mb:.1f} MiB")
print(f"build {t_build:.3f}s copy {t_copy:.3f}s state_sha256 {t_hash:.3f}s grad_copy {t_grad_copy:.3f}s grad_sha256 {t_grad_hash:.3f}s norm64 {t_norm:.3f}s isfinite {t_finite:.3f}s")
print(f"host-only per case ~{per_case:.2f}s; per point (6 cases + live) ~{per_point:.2f}s; two points ~{2*per_point:.1f}s")
