# SIMULAZIONE CPU: payload a 576 come li salva run.py:1482 (rank 0, get_rng_state_all su 4 dispositivi).
# continuo: manual_seed(seed*1_000_003+1) su tutti i device (run.py:1463); ripresa (ordinary o warm): manual_seed(seed) (run.py:1383)
# e set_rng_state del solo device 0 (run.py:1450). Caso peggiore per il rilievo: warm recupera TUTTI i float del continuo.
import torch, contrastive_cn_restart_compare as compare
def cuda_state(seed):
    g = torch.Generator(); g.manual_seed(seed); return g.get_state()
seed = 7
own = cuda_state(123)
base = {"student": {"w": torch.ones(3)}, "rank_rng": [{"cuda_rng": own, "torch_rng": cuda_state(5)}], "torch_rng": cuda_state(5)}
cont = {**base, "cuda_rng": [own] + [cuda_state(seed*1_000_003+1)]*3}
warm = {**base, "cuda_rng": [own] + [cuda_state(seed)]*3}
d = compare.checkpoint_differences(cont, warm); print(d)
print("gate 248 passa:", not d["metadata_or_schema_different"])
