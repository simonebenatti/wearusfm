import json, sys
from pathlib import Path
C = Path(sys.argv[1])
sys.path[:0] = [str(C/'src'), str(C/'scripts')]
import torch
from contrastive_prepare import build_review_configs, BASELINE, _read, config_from_dict
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head, ContrastiveConfig
cfgs = build_review_configs(_read(BASELINE))
print(sorted(cfgs))
for name, raw in cfgs.items():
    print(name, raw.get('max_steps'), raw.get('schedule'), raw.get('batch_size'), 'eval_every', raw.get('eval_every'), 'ckpt_every', raw.get('ckpt_every'),
          'fixed_eval_every', raw.get('fixed_eval_every'), 'pose_panel_every', raw.get('pose_panel_every'), 'log_batch_origin', raw.get('log_batch_origin'),
          'prefetch', raw.get('prefetch'), 'stream_segment', raw.get('stream_segment'))
raw = cfgs[sorted(cfgs)[0]]
cfg = config_from_dict(raw)
torch.manual_seed(0)
m = attach_contrastive_head(WearUsFM(cfg.model))
n = sum(p.numel() for p in m.parameters())
b = sum(p.numel()*p.element_size() for p in m.parameters())
bb = sum(t.numel()*t.element_size() for t in m.state_dict().values())
print('params', n, 'param bytes', b, 'state_dict bytes', bb)
cc = ContrastiveConfig(**raw['contrastive'])
print('contrastive', {k: getattr(cc, k) for k in ('capacity_per_dataset','max_age','fill_steps','ramp_steps','coverage_block','grad_every','grad_steps','grad_dense_start','grad_dense_end')})
print('datasets', raw.get('datasets'))
est = 2*bb + 2*b
print('estimated checkpoint bytes (student+teacher state + 2 Adam moments, no queue):', est, round(est/2**30,3), 'GiB')
