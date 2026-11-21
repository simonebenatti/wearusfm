"""Simulation (CPU): endpoint payloads shaped like the real 576 checkpoints read on Leonardo.
Only cuda_rng[1..3] differ between continuous (fresh start: manual_seed(seed*1000003+rank+1) on all
devices) and any reload (manual_seed(seed) on all devices, then set_rng_state only on own device)."""
import sys, torch
sys.path[:0] = ["scripts/diagnostics", "scripts", "src"]
import contrastive_cn_restart_compare as C
def payload(other_seed):
    g = [torch.cuda.default_generators] if False else None
    own = torch.arange(16, dtype=torch.uint8)
    others = [torch.tensor(list(other_seed.to_bytes(8, "little")) + [0]*8, dtype=torch.uint8) for _ in range(3)]
    return {"student": {"w": torch.ones(3)}, "step": 576, "cuda_rng": [own, *others],
            "rank_rng": [{"torch_rng": torch.zeros(4, dtype=torch.uint8), "cuda_rng": own}]}
seed = 20261118
continuous = payload(seed * 1_000_003 + 0 + 1)
warm_reload_identical_trajectory = payload(seed)
d = C.checkpoint_differences(continuous, warm_reload_identical_trajectory)
print(d)
try:
    C.require(not d["metadata_or_schema_different"], "warm endpoint metadata/schema or nonfloat state differs")
except ValueError as e:
    print("compare() would stop here ->", type(e).__name__, e, "-> report status INCONCLUSIVE")
print("classify with endpoint_exact = exact_payload:", C.classify(False, True, False, d["exact_payload"], True))
