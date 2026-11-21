# Simulazione CPU: generatori Philox CUDA sostituiti da stati CPU seminati come nel trainer.
import torch, contrastive_cn_restart_compare as C
seed=1234
def st(s):
    g=torch.Generator(); g.manual_seed(s); return g.get_state()
w=torch.randn(4)
# continuo: rank0 risemina tutte le GPU con seed*1000003+1; GPU0 poi avanza (qui: stato "traiettoria" T)
T=st(99)
cont={"student":{"w":w.clone()},"cuda_rng":[T, *[st(seed*1_000_003+1)]*3],"rank_rng":[{"cuda_rng":T}],"step":576}
# ripresa (CN_A o warm con traiettoria identica): manual_seed(cfg.seed) su tutte, set_rng_state solo GPU0
resume={"student":{"w":w.clone()},"cuda_rng":[T, *[st(seed)]*3],"rank_rng":[{"cuda_rng":T}],"step":576}
d=C.checkpoint_differences(cont,resume); print(d)
try:
    C.require(not d["metadata_or_schema_different"],"warm endpoint metadata/schema or nonfloat state differs"); print("passa")
except ValueError as e: print("ValueError ->",e,"-> INCONCLUSIVE")
print(C.classify(False,True,False,d["exact_payload"],True))
