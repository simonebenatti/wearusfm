# SIMULAZIONE: stato CUDA Philox come uint8[16] = seed(u64) + offset(u64), come in torch 2.2 (CUDAGeneratorImpl::get_state)
import struct, torch, contrastive_cn_restart_compare as c
def st(seed, off=0): return torch.tensor(list(struct.pack("<QQ", seed, off)), dtype=torch.uint8)
w = torch.randn(4, 3)
def payload(dev123_seed):
    return {"student": {"w": w.clone()}, "step": 576, "torch_rng": torch.get_rng_state(),
            "cuda_rng": [st(0, 9999)] + [st(dev123_seed)] * 3,
            "rank_rng": [{"torch_rng": torch.get_rng_state(), "cuda_rng": st(r, 9999)} for r in range(4)]}
cont = payload(1)   # continuo: run.py:1463 manual_seed(0*1_000_003+0+1)=1 su tutti i dispositivi del rank 0
warm = payload(0)   # ripresa: ultimo seed_all = run.py:1383 manual_seed(0); 1450 ripristina solo cuda:0
print(c.checkpoint_differences(cont, warm))
print(c.checkpoint_differences(cont, payload(1)))
