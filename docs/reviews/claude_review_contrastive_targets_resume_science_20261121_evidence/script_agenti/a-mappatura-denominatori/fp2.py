import json, itertools
from wearusfm.training.pairing import state_sha256
from contrastive_prepare import config_from_dict
from wearusfm.training import run as trainer
C='/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_resume/clone/'
raw=json.load(open(C+'results/infra/contrastive_targets_20261118/benchmark_execution_20261121/CN/config.json'))
cfg=config_from_dict(raw); lr=trainer._lr_at(544,cfg)
T='864a86decd9d15929e4bf5a0400908c0697ae08ded112e95b5db8140570e9837'
# torch 2.2 AdamW defaults: lr betas eps weight_decay amsgrad foreach maximize capturable differentiable fused
base=dict(lr=lr,betas=(0.9,0.999),eps=1e-08,weight_decay=0.05,amsgrad=False,foreach=None,maximize=False,capturable=False,differentiable=False,fused=None)
for n in range(265,275):
  for extra in ({},{'decoupled_weight_decay':True}):
    g=[{**base,**extra,'params':list(range(n))}]
    h=state_sha256(g)
    if h==T: print('MATCH n',n,extra)
print('done')
g=[{**base,'params':list(range(270))}]; print(state_sha256(g))
