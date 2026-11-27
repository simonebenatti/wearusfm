"""Real ReductionObservation._snapshot/finish with an all-zero and a -0.0 gradient, then load_observation under torch 2.2 shim."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import torch, torch.distributed as dist
import contrastive_cn_restart_observe as ob
out = Path(sys.argv[1])
dist.get_rank = lambda g: 0; dist.get_world_size = lambda g: 4
model = torch.nn.Linear(2, 1)
obs = ob.ReductionObservation(SimpleNamespace(process_group=None, register_comm_hook=lambda s, h: None), model, expected_parameters=2)
obs.start()
flat = torch.tensor([0., -0., -0.])
members = list(model.parameters()); views = [flat[:2].view_as(model.weight), flat[2:].view_as(model.bias)]
obs._snapshot(SimpleNamespace(buffer=lambda: flat, parameters=lambda: members, gradients=lambda: views, index=lambda: 0))
for p, v in zip(model.parameters(), views): p.grad = v.clone()
obs.finish(out, 0)
native = torch.load
def shim(f, *a, **k):
    if k.get("mmap") and not isinstance(f, str): raise ValueError("mmap needs str")
    return native(f, *a, **k)
torch.load = shim
r, v = ob.load_observation(out, 0, expected_parameters=2)
print("norms", [(p["local_norm"], type(p["local_norm"]).__name__) for p in r["parameters"]], "loaded", v["reduced"][1])
# int norm in receipt (hand-edited) is still accepted; bool rejected
rec = json.loads((out / "observation.rank0.json").read_text()); rec["parameters"][0]["local_norm"] = 0
(out / "observation.rank0.json").write_text(json.dumps(rec)); ob.load_observation(out, 0, expected_parameters=2); print("int norm accepted")
