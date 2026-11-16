import json
import torch
from rank_contrast import pose_rank_contrast

g = torch.Generator().manual_seed(107)
z = torch.randn(4, 6, 12, generator=g, dtype=torch.float64, requires_grad=True)
y = torch.randn(4, 6, 3, generator=g, dtype=torch.float64)
mask = torch.ones_like(y, dtype=torch.bool)
tv = torch.ones(4, 6, dtype=torch.bool)
kw = dict(temperature=0.2, stride=2, positives='cross', per_window=True)
whole, num, den, _ = pose_rank_contrast(z, (y, mask), tv, **kw)
gwhole, = torch.autograd.grad(whole, z, retain_graph=True)
rows = {'whole': {'loss': float(whole.detach()), 'count': int(den), 'grad_norm': float(gwhole.norm())}}
for size in (1, 2):
    pieces = [pose_rank_contrast(z[i:i+size], (y[i:i+size], mask[i:i+size]), tv[i:i+size], **kw) for i in range(0,4,size)]
    den = sum(p[2] for p in pieces)
    reduced = sum(p[1] for p in pieces)/den.clamp(min=1)
    grad, = torch.autograd.grad(reduced, z, retain_graph=True)
    rows[f'split_{size}'] = {'loss': float(reduced.detach()), 'count': int(den), 'grad_norm': float(grad.norm()), 'max_grad_delta': float((grad-gwhole).abs().max())}
assert rows['split_1']['loss'] == 0 and rows['whole']['loss'] > 0
assert rows['split_2']['max_grad_delta'] > 1e-6
print(json.dumps(rows, indent=2))
