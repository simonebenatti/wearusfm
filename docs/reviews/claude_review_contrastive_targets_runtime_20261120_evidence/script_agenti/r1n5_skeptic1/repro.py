"""R1-N5 skeptic 1: reproduce semantics of per-case memory and gate (read-only, no CUDA)."""
import copy
from unittest.mock import patch, MagicMock

import torch

import contrastive_target_parity as P
import test_contrastive_target_parity as T
from wearusfm.training.contrastive_objectives import ContrastiveConfig

# 1) _memory only reads process-wide counters, never resets them.
state = {"alloc": 3 << 30, "peak": 9 << 30}   # e.g. peak 9 GiB set earlier (update-100 diagnostic)
reset = MagicMock(name="reset_peak_memory_stats")
props = MagicMock(); props.total_memory = 64 << 30
with patch.object(torch.cuda, "memory_allocated", lambda d=None: state["alloc"]), \
     patch.object(torch.cuda, "memory_reserved", lambda d=None: state["alloc"]), \
     patch.object(torch.cuda, "max_memory_allocated", lambda d=None: state["peak"]), \
     patch.object(torch.cuda, "max_memory_reserved", lambda d=None: state["peak"]), \
     patch.object(torch.cuda, "get_device_properties", lambda d=None: props), \
     patch.object(torch.cuda, "reset_peak_memory_stats", reset):
    rows = []
    for name, case_peak in zip(P.CASES, (5, 5, 6, 6, 5, 5)):   # each case peaks lower than earlier global peak
        state["peak"] = max(state["peak"], case_peak << 30)
        rows.append((name, P._memory("cuda:0")["peak_allocated_bytes"] >> 30))
    print("1) per-case recorded peak GiB:", rows, "| reset called:", reset.called)
    print("   peak_scope:", P._memory("cuda:0")["peak_scope"])

# 2) Gate requires only presence/positivity of the per-case memory.
def report_with(mem):
    pts = [T.point(128), T.point(256)]
    for p in pts:
        for r in p["ranks"]:
            for c in r["cases"].values():
                c["memory"] = copy.deepcopy(mem)
    return {"schema": 1, "profile": P.PROFILE, "points": pts}
for label, mem in [("1 byte", {"cuda_measured": True, "peak_allocated_bytes": 1}),
                   ("1 EiB", {"cuda_measured": True, "peak_allocated_bytes": 1 << 60}),
                   ("zero", {"cuda_measured": True, "peak_allocated_bytes": 0}),
                   ("cpu", {"cuda_measured": False})]:
    print(f"2) gate with per-case memory {label}:", P.validate_parity_report(report_with(mem))["status"])

# 3) Diagnostic updates before the two probe points in the cohort configuration.
import json, subprocess
raw = json.loads(subprocess.check_output(["git", "-C", T.Path(P.__file__).resolve().parents[1].as_posix(), "show",
      "fbfde797:results/infra/contrastive_targets_20261118/review_configs/C0.config.json"]))["contrastive"]
cc = ContrastiveConfig(**{**raw, "nce_weight": 1., "rnc_weight": 1.})
print("3) diagnostic updates 1..128:", [u for u in range(1, 129) if cc.wants_gradients(u - 1)],
      "| 129..256:", [u for u in range(129, 257) if cc.wants_gradients(u - 1)],
      "| dense 513..576 count:", sum(cc.wants_gradients(u - 1) for u in range(513, 577)))
