"""Logical autograd/Adam checks only: no optimizer steps or training."""
import math
import sys

sys.path.insert(0, sys.argv[1] + "/src")
import torch
from wearusfm.training.jepa import LossTerms, pose_terms
from wearusfm.model.pose_head import PoseHead

p = torch.tensor(0.3, requires_grad=True)
y = torch.arange(1, 17, dtype=torch.float32).reshape(16, 1, 1) / 10
mask = torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool).reshape(16, 1, 1)
tv = torch.ones(16, 1, dtype=torch.bool)
loss, _, den, _ = pose_terms(p.expand_as(y), (y, mask), tv, True)
reference = torch.autograd.grad(loss, p)[0].item()
reduced = 0.0
counts = []
for rank in range(4):
    for micro in range(2):
        sl = slice(4 * rank + 2 * micro, 4 * rank + 2 * micro + 2)
        _, num, local_den, _ = pose_terms(p.expand(2, 1, 1), (y[sl], mask[sl]), tv[sl], True)
        terms = LossTerms(num={"pose": num}, den={"pose": local_den}, weight={"pose": 1.0})
        reduced += torch.autograd.grad(terms.backward_loss({"pose": den}, world_size=4), p)[0].item() / 4
        counts.append(int(local_den))
assert reference == reduced
print({"global_grad": reference, "reconstructed_ddp_grad": reduced, "counts": counts})

torch.manual_seed(0)
head = PoseHead(8)
x = torch.randn(2, 3, 8, requires_grad=True)
head(x).sub(1).square().mean().backward()
assert x.grad.abs().max().item() == 0
assert head.norm.weight.grad.abs().max().item() == 0
assert head.out.weight.grad.norm().item() > 0
assert head.out.bias.grad.norm().item() > 0
print({"input_grad_max": x.grad.abs().max().item(), "norm_grad_max": head.norm.weight.grad.abs().max().item(),
       "output_weight_grad_norm": head.out.weight.grad.norm().item(), "output_bias_grad_norm": head.out.bias.grad.norm().item()})

def second_adam_update(a, b):
    m = 0.9 * (0.1 * a) + 0.1 * b
    v = 0.999 * (0.001 * a * a) + 0.001 * b * b
    return (m / (1 - 0.9**2)) / math.sqrt(v / (1 - 0.999**2))

print({"adam_second_unclipped": second_adam_update(1, 100), "adam_second_clipped": second_adam_update(1, 1)})
