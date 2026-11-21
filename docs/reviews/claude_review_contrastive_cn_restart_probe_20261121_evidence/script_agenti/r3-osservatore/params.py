import json, dataclasses, torch
from wearusfm.model.fm import WearUsFM, FMConfig
from wearusfm.training.contrastive_objectives import attach_contrastive_head
d = json.load(open('results/infra/contrastive_targets_20261118/benchmark_execution_20261121/CN/config.json'))['model']
names = {f.name for f in dataclasses.fields(FMConfig)}
cfg = FMConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})
torch.manual_seed(0)
m = WearUsFM(cfg); attach_contrastive_head(m)
ps = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
print("n", len(ps), "bytes", {str(dt): sum(p.numel()*p.element_size() for _, p in ps if p.dtype == dt) for dt in {p.dtype for _, p in ps}})
print("noncontig", [n for n, p in ps if not p.is_contiguous()])
print("f64", [(n, p.numel()) for n, p in ps if p.dtype == torch.float64])
print("zero-numel", [n for n, p in ps if p.numel() == 0])
