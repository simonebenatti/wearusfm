# SIMULAZIONE CPU (nessuna GPU): stato Philox CUDA = 8 byte seed + 8 byte offset (torch 2.2).
import struct, torch, contrastive_cn_restart_compare as compare
def philox(seed, offset=0):
    return torch.frombuffer(bytearray(struct.pack("<Qq", seed, offset)), dtype=torch.uint8).clone()
seed = 1234  # valore qualunque di cfg.seed
own = philox(999, 4096)  # stato avanzato del device 0, identico nelle due traiettorie (rank_rng)
common = {"student": {"w": torch.ones(3)}, "rank_rng": [{"cuda_rng": own}], "torch_rng": torch.zeros(5, dtype=torch.uint8)}
continuous = {**common, "cuda_rng": [own] + [philox(seed * 1_000_003 + 1)] * 3}
warm_recovered = {**common, "cuda_rng": [own] + [philox(seed)] * 3}
d = compare.checkpoint_differences(continuous, warm_recovered)
print(d)
try:
    compare.require(not d["metadata_or_schema_different"], "warm endpoint metadata/schema or nonfloat state differs")
except ValueError as e:
    print("report -> INCONCLUSIVE:", e)
