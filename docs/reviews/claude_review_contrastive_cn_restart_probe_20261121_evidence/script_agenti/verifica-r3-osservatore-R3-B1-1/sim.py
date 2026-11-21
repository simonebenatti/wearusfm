import torch, copy, sys
sys.path[:0]=["clone/src","clone/scripts","clone/scripts/diagnostics"]
import contrastive_cn_restart_compare as c
# Simulation: CPU generator states stand in for per-device CUDA states of the rank-0 process.
def st(seed):
    g=torch.Generator(); g.manual_seed(seed); return g.get_state()
seed=0
cont={"student":{"w":torch.ones(3)},"rank_rng":[{"torch_rng":st(5),"cuda_rng":st(6)}],
      "cuda_rng":[st(9)]+[st(seed*1_000_003+1)]*3,"step":576}
warm=copy.deepcopy(cont); warm["cuda_rng"]=[st(9)]+[st(seed)]*3
d=c.checkpoint_differences(cont,warm); print(d)
try:
    c.require(not d["metadata_or_schema_different"],"warm endpoint metadata/schema or nonfloat state differs"); print("passes")
except ValueError as e: print("ValueError ->",e,"-> main() writes INCONCLUSIVE")
print(c.classify(False,True,False,d["exact_payload"],True))
