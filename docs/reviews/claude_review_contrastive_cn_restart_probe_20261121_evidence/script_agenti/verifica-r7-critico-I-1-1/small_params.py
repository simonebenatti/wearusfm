import sys, json, torch
sys.path[:0]=["src","scripts"]
from contrastive_target_resume import WearUsFM, attach_contrastive_head, config_from_dict
cfg=json.load(open("results/infra/contrastive_targets_20261118/reloads_execution_20261121/CN_A/config.json"))
with torch.random.fork_rng(devices=[]):
    m=attach_contrastive_head(WearUsFM(config_from_dict(cfg).model))
ps=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
print(len(ps), sum(p.numel() for _,p in ps))
for i,(n,p) in enumerate(ps):
    if p.numel()<=64: print(i,n,tuple(p.shape),p.dtype)
