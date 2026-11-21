import json,sys,torch
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head
c=json.load(open(sys.argv[1]))['model']
import dataclasses
fields={f.name for f in dataclasses.fields(FMConfig)}
cfg=FMConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in c.items() if k in fields})
m=WearUsFM(cfg); attach_contrastive_head(m)
ps=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
print('n params', len(ps), 'numel', sum(p.numel() for _,p in ps))
for i,(n,p) in enumerate(ps):
    if i>=262 or i<2: print(i,n,tuple(p.shape),p.dtype,p.numel())
print('float64 params:',[ (i,n) for i,(n,p) in enumerate(ps) if p.dtype==torch.float64])
print('contrastive_head numel', sum(p.numel() for n,p in ps if n.startswith('contrastive_head')))
