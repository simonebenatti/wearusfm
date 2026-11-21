import sys, collections
sys.path[:0]=["src","scripts","scripts/diagnostics","tests/cpu"]
from pathlib import Path
from test_contrastive_runtime import _config, _pose_setup
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head
tmp=Path(sys.argv[1]); tmp.mkdir(exist_ok=True)
root, manifest, pose = _pose_setup(tmp)
cfg=_config(pose, arm="CN")
m=attach_contrastive_head(WearUsFM(cfg.model))
ps=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
print(len(ps), collections.Counter(str(p.dtype) for _,p in ps), sum(p.numel()*p.element_size() for _,p in ps))
print([n for n,p in ps if p.dtype!=__import__('torch').float32])
print(cfg.batch_size, cfg.contrastive)
