"""Counterexample on logits, not a training benchmark."""
import json
import torch
import conftest
from wearusfm.model.rank_contrast import rank_contrast_pairs

n = 4
groups = torch.tensor([0, 0, 1, 1])
cand = ~torch.eye(n, dtype=torch.bool)
cross = cand & (groups[:, None] != groups[None, :])
# All four poses identical: every off-diagonal target distance equals zero.
dist = torch.zeros(n, n, dtype=torch.float64)
logits = torch.zeros(n, n, dtype=torch.float64, requires_grad=True)
num, chance, _ = rank_contrast_pairs(logits, dist, cand, cross)
row_loss = num[0] / chance[0]
grad = torch.autograd.grad(row_loss, logits)[0]
all_num, all_chance, _ = rank_contrast_pairs(logits, dist, cand, cand)
all_grad = torch.autograd.grad(all_num[0] / all_chance[0], logits)[0]
assert grad[0, 1] > 0  # descent decreases same-window similarity despite equal pose
assert grad[0, 2] < 0 and grad[0, 3] < 0
assert torch.allclose(all_grad[0], torch.zeros(n, dtype=torch.float64), atol=1e-14)

# Excluding same-window pairs from BOTH roles removes this forced competition.
both_num, both_chance, _ = rank_contrast_pairs(logits, dist, cross, cross)
both_grad = torch.autograd.grad(both_num[0] / both_chance[0], logits)[0]
assert torch.allclose(both_grad[0], torch.zeros(n, dtype=torch.float64), atol=1e-14)

print(json.dumps({
    "setup": "four identical poses, two windows, uniform logits, anchor 0",
    "cross_normalized_loss": row_loss.item(),
    "cross_dloss_dlogits_row0": grad[0].tolist(),
    "all_dloss_dlogits_row0": all_grad[0].tolist(),
    "exclude_same_window_both_roles_dloss_dlogits_row0": both_grad[0].tolist(),
    "iid_example_p_at_most_one_pose_window_local_batch8_p023": .77**8+8*.23*.77**7,
    "iid_example_only_not_loader_measurement": True,
}, indent=2))
