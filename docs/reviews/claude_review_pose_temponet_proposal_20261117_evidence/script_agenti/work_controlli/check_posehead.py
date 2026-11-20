"""Controllo ESEGUITO: la PoseHead dello snapshot c15aac9 e' LN affine (eps 1e-5) -> Linear(384, 20) a zero: 8.468 parametri."""
import sys
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
import torch
from wearusfm.model.pose_head import PoseHead
h = PoseHead(384)
print("parametri:", sum(p.numel() for p in h.parameters()), "| LN affine:", h.norm.elementwise_affine, "| eps:", h.norm.eps,
      "| pesi e bias a zero:", bool((h.out.weight == 0).all() and (h.out.bias == 0).all()),
      "| gamma=1, beta=0:", bool((h.norm.weight == 1).all() and (h.norm.bias == 0).all()))
x = torch.randn(2, 160, 384)
print("uscita iniziale tutta zero:", bool((h(x) == 0).all()), "| forma", tuple(h(x).shape))
