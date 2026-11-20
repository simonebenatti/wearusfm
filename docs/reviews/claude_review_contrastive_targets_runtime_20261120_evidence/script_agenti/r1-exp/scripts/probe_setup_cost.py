"""Rough CPU proxy of per-probe setup cost with the production 30M model and a full queue."""
import copy
import json
import sys
import time
from pathlib import Path

import torch

from contrastive_prepare import config_from_dict
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
from wearusfm.training.contrastive_memory import WindowIdentity
from wearusfm.training.jepa import make_teacher
from wearusfm.training.pairing import state_sha256

torch.set_num_threads(1)
raw = json.loads(Path(sys.argv[1]).read_text())
raw["contrastive"]["nce_weight"] = raw["contrastive"]["rnc_weight"] = .01
cfg = config_from_dict(raw)
t = time.perf_counter()
live = attach_contrastive_head(WearUsFM(cfg.model))
teacher_live = make_teacher(live)
print("params", sum(p.numel() for p in live.parameters()), "build+teacher", round(time.perf_counter() - t, 2))
student_state = {k: v.detach().clone() for k, v in live.state_dict().items()}
teacher_state = {k: v.detach().clone() for k, v in teacher_live.state_dict().items()}
runtime = ContrastiveRuntime(cfg.contrastive)
datasets = sorted(cfg.contrastive["coverage_quotas"])
runtime.expected_datasets = tuple(datasets)
# Fill a realistic queue: 255 updates x 32 windows, capacity 1024/dataset, quotas as configured.
quotas = cfg.contrastive["coverage_quotas"]
import random
rnd = random.Random(0)
for step in range(255):
    ids = []
    for i in range(32):
        d = rnd.choices(datasets, weights=[quotas[x] for x in datasets])[0]
        ids.append(WindowIdentity(d, "s%d" % rnd.randrange(50), "r%d" % rnd.randrange(5), None, step * 1000 + i * 10, step * 1000 + i * 10 + 5, 2000.))
    keys = torch.nn.functional.normalize(torch.randn(32, 128), dim=-1)
    runtime.queue.commit(keys, ids, step=step)
print("queue entries", len(runtime.queue))
t = time.perf_counter()
saved = copy.deepcopy(runtime.state_dict())
print("deepcopy runtime state", round(time.perf_counter() - t, 3))
t = time.perf_counter()
h = state_sha256(saved["queue"])
print("hash queue", round(time.perf_counter() - t, 3))
t = time.perf_counter()
state_sha256(student_state); state_sha256(teacher_state)
print("hash student+teacher", round(time.perf_counter() - t, 3))
for i in range(2):
    t = time.perf_counter()
    s = attach_contrastive_head(WearUsFM(cfg.model))
    t_build = time.perf_counter() - t
    s.load_state_dict(student_state, strict=True)
    te = make_teacher(s)
    te.load_state_dict(teacher_state, strict=True)
    t_model = time.perf_counter() - t
    rt = ContrastiveRuntime(cfg.contrastive)
    rt.expected_datasets = tuple(datasets)
    cl = copy.deepcopy(saved)
    rt.load_state_dict(cl, completed_steps=cl["queue"]["last_step"] + 1)
    t_runtime = time.perf_counter() - t - t_model
    state_sha256(s.state_dict()); state_sha256(te.state_dict()); state_sha256(rt.queue.state_dict())
    t_total = time.perf_counter() - t
    print("probe setup", i, "build", round(t_build, 2), "model+teacher", round(t_model, 2), "runtime", round(t_runtime, 2), "with fingerprints", round(t_total, 2))
