# Simulazione CPU: schema RNG del payload del continuo (fresco) contro una ripresa (rank 0, 4 GPU visibili)
import sys, torch
from contrastive_cn_restart_compare import checkpoint_differences, classify
seed=7
def gen(s): g=torch.Generator(); g.manual_seed(s); return g.get_state()
# continuo: manual_seed(seed*1000003+1) semina tutte le GPU; rank 0 avanza solo il dispositivo 0
cont={"cuda_rng":[gen(99), gen(seed*1_000_003+1), gen(seed*1_000_003+1), gen(seed*1_000_003+1)], "w":torch.ones(3)}
# ripresa: manual_seed(seed) su tutte, set_rng_state solo del dispositivo 0 (traiettoria identica)
warm={"cuda_rng":[gen(99), gen(seed), gen(seed), gen(seed)], "w":torch.ones(3)}
d=checkpoint_differences(cont,warm); print(d)
print("line248 require passes:", not d["metadata_or_schema_different"])
