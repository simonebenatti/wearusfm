import json,sys,dataclasses,typing,torch
from wearusfm.training import run as R
from wearusfm.model.contrastive import ProjectionHead
d=json.load(open(sys.argv[1]))
def build(cls,v):
    if not dataclasses.is_dataclass(cls) or not isinstance(v,dict): return v
    hints=typing.get_type_hints(cls); kw={}
    for f in dataclasses.fields(cls):
        if f.name in v:
            t=hints[f.name]; kw[f.name]=build(t,v[f.name]) if dataclasses.is_dataclass(t) else (tuple(v[f.name]) if isinstance(v[f.name],list) else v[f.name])
    return cls(**kw)
cfg=build(R.RunConfig,d)
m=R.WearUsFM(cfg.model)
ps=[(n,tuple(p.shape),p.dtype,p.numel()) for n,p in m.named_parameters() if p.requires_grad]
print(len(ps))
for n,s,dt,k in ps:
    if k<=16: print(n,s,dt,k)
