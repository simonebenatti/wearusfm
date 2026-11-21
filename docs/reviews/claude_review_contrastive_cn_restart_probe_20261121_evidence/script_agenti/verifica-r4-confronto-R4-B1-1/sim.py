# SIMULAZIONE: stato CUDA Philox come 16 byte uint8 (seed u64 + offset u64), come torch 2.2 su CUDA.
import sys, struct, torch
sys.path.insert(0, sys.argv[1] + "/scripts/diagnostics"); sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
from contrastive_cn_restart_compare import checkpoint_differences
def cuda_state(seed, offset): return torch.tensor(list(struct.pack("<QQ", seed, offset)), dtype=torch.uint8)
torch.manual_seed(123); cpu = torch.get_rng_state()
w = torch.randn(4, 3)
dev0 = cuda_state(1, 4096)  # dispositivo usato: identico se la traiettoria coincide
rank_rng = [{"torch_rng": cpu.clone(), "cuda_rng": dev0.clone()} for _ in range(4)]
def payload(seed_unused):
    return {"student": {"w": w.clone()}, "step": 576, "torch_rng": cpu.clone(),
            "cuda_rng": [dev0.clone()] + [cuda_state(seed_unused, 0) for _ in range(3)],
            "rank_rng": [{k: v.clone() for k, v in r.items()} for r in rank_rng], "world_size": 4}
continuous = payload(1)   # run.py:1463 manual_seed(0*1_000_003+0+1) -> manual_seed_all: dispositivi 1..3 seme 1
warm = payload(0)         # run.py:1383 manual_seed(0); 1450 ripristina solo il dispositivo 0
r = checkpoint_differences(continuous, warm)
print(r)
try:
    assert not r["metadata_or_schema_different"], "warm endpoint metadata/schema or nonfloat state differs"
    print("riga 248: passa")
except AssertionError as e:
    print("riga 248 solleverebbe:", e)
# Ordinaria: due riprese, stessi semi -> nessuna differenza
print(checkpoint_differences(payload(0), payload(0))["exact_payload"])
