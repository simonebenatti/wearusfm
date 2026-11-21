import json, torch
import contrastive_cn_restart_compare as C
from contrastive_target_resume import WearUsFM, attach_contrastive_head, config_from_dict, read, rows
P = C.PACKAGE
CN = P/"benchmark_execution_20261121/CN"; A = P/"reloads_execution_20261121/CN_A"; B = P/"reloads_execution_20261121/CN_B"
print("B phase", rows(B/"trace-0.jsonl", 544, 576)[0]["phase"])
cfg = read(C.ARCHIVE["ordinary"]/"config.json")
with torch.random.fork_rng(devices=[]):
    model = attach_contrastive_head(WearUsFM(config_from_dict(cfg).model))
named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
allp = list(model.named_parameters())
print("trainable", len(named), "all", len(allp), "dtypes", {str(p.dtype) for _,p in named})
ident = [(i,n,tuple(p.shape),str(p.dtype)) for i,(n,p) in enumerate(named)]
logged = rows(CN/"trace-0.ddp.jsonl", 544, 576)[0]
sizes = [int(s) for s in logged["rebuilt_bucket_sizes"].split(",")]
ids = [tuple(map(int,g.split())) for g in logged["rebuilt_per_bucket_param_indices"].split(",")]
# consistency of logged sizes with model numel*element_size
ok = all(sum(named[i][1].numel()*named[i][1].element_size() for i in g)==s for s,g in zip(sizes,ids))
print("logged sizes match model numel*esize:", ok, "dtype per bucket", [{str(named[i][1].dtype) for i in g} for g in ids])
params = [{"id":i,"name":n,"shape":list(s),"dtype":d} for i,n,s,d in ident]
by = {}
for i,_,_,d in ident: by.setdefault(d,[]).append(i)
def nb(g): return sum(named[i][1].numel()*named[i][1].element_size() for i in g)
# DDP initial: reversed bucket list
cold = [{"index":k,"nbytes":nb(g),"dtype":d,"parameter_ids":g} for k,(d,g) in enumerate(reversed(list(by.items())))]
hot = [{"index":k,"nbytes":s,"dtype":str(named[g[0]][1].dtype),"parameter_ids":list(g)} for k,(s,g) in enumerate(zip(sizes,ids))]
print("cold bytes", [b["nbytes"] for b in cold], "logged initial", logged["bucket_sizes"])
try:
    C.layouts({"buckets":cold,"parameters":params},{"buckets":hot,"parameters":params},logged); print("layouts OK")
except Exception as e: print("layouts EXC", e)
# mutation: swap two ids in hot
hot2 = json.loads(json.dumps(hot)); hot2[0]["parameter_ids"][0], hot2[0]["parameter_ids"][1] = hot2[0]["parameter_ids"][1], hot2[0]["parameter_ids"][0]
try: C.layouts({"buckets":cold,"parameters":params},{"buckets":hot2,"parameters":params},logged); print("swap NOT detected")
except Exception as e: print("swap detected:", e)
