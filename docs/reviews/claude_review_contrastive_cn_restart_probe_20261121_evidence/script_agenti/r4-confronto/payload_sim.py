"""SIMULATION: CUDA RNG states encoded like torch 2.2 Philox (seed u64 + offset u64); no CUDA on the Mac."""
import copy, struct, torch
import contrastive_cn_restart_compare as C
def cuda_state(seed, offset=0): return torch.tensor(list(struct.pack("<QQ", seed, offset)), dtype=torch.uint8)
seed = 0  # cfg.seed irrelevant: any seed s with s*1_000_003+1 != s
torch.manual_seed(0)
p = torch.nn.Parameter(torch.randn(3)); opt = torch.optim.AdamW([p]); p.grad = torch.randn(3); opt.step()
def payload(top_cuda):
    return {"student": {"w": p.detach().clone(), "num": torch.tensor(3)}, "teacher": {"w": p.detach().clone()},
            "optimizer": copy.deepcopy(opt.state_dict()), "step": 576, "alarm_counts": (0, 0),
            "config": {"seed": seed}, "torch_rng": torch.get_rng_state(), "cuda_rng": top_cuda,
            "stream": {"num_workers": 4, "segment": 64, "world_size": 4}, "data_signature": {"a": [1, 2.5]},
            "pose_stats_sha256": None,
            "contrastive_state": {"schema": 1, "queue": {"last_step": 575, "entries": [{"identity": {"fs": 2000.0, "trial": None}, "key": torch.ones(4, dtype=torch.bfloat16), "step": 575, "global_index": 1}]},
                                  "coverage": {"updates": 3, "rnc_active": 1, "datasets": {"d": [5, 4]}}, "coverage_reports": [{"fraction": 0.8}]},
            "contrastive_initial_sha256": "x" * 64, "contrastive_provenance": {"k": "v"},
            "rank_rng": [{"torch_rng": torch.get_rng_state(), "cuda_rng": cuda_state(7, 4096 * r)} for r in range(4)], "world_size": 4}
# continuous rank-0 process: devices 1..3 last seeded by manual_seed(seed*1_000_003 + rank + 1), never used
continuous = payload([cuda_state(7, 0)] + [cuda_state(seed * 1_000_003 + 1)] * 3)
# reload/warm rank-0 process: manual_seed(seed) at run.py:1383; only device 0 restored at run.py:1450
warm = payload([cuda_state(7, 0)] + [cuda_state(seed)] * 3)
r = C.checkpoint_differences(continuous, warm)
print(r)
print("compare() line 248 would raise:", bool(r["metadata_or_schema_different"]))
