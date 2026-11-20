"""Independent reproduction for R1-N6 (zero-weight arms differ only by label).

Run from the clone root with PYTHONPATH=src:tests/cpu:scripts. Writes only under WORK.
"""
import json
import math
import random
import sys
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import torch

WORK = Path(sys.argv[1])
REPO = Path.cwd()

import contrastive_target_parity as P
from contrastive_numeric_profile import numeric_profile
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader
from wearusfm.training.contrastive_memory import WindowIdentity
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime
from wearusfm.training import jepa as J
from wearusfm.training.pairing import state_sha256

out = {}

# 1) review configs: recursive diff
def flat(d, p=""):
    r = {}
    if isinstance(d, dict):
        for k, v in d.items():
            r.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            r.update(flat(v, f"{p}[{i}]"))
    else:
        r[p] = d
    return r

base = REPO / "results/infra/contrastive_targets_20261118/review_configs"
F = {a: flat(json.loads((base / f"{a}.config.json").read_text())) for a in ("C0", "CN", "CR")}
out["config_diffs"] = {f"{a}/{b}": sorted(k for k in set(F[a]) | set(F[b]) if F[a].get(k, "<absent>") != F[b].get(k, "<absent>"))
                       for a, b in (("C0", "CN"), ("C0", "CR"), ("CN", "CR"))}

# 2) weights at the two prescribed points, production fill/ramp, arbitrary admissible lambdas (sign of zero too)
prod = json.loads((base / "C0.config.json").read_text())["contrastive"]
rng = random.Random(7)
weights = {}
for trial in range(200):
    ln, lr = 10 ** rng.uniform(-4, 0), 10 ** rng.uniform(-4, 0)
    for arm in ("C0", "CN", "CR"):
        cc = ContrastiveConfig(**{**prod, "arm": arm, "nce_weight": ln, "rnc_weight": lr})
        for step in (127, 255):
            w = cc.weights(step)
            key = f"{arm}@update{step + 1}"
            weights.setdefault(key, set()).add(tuple((x, math.copysign(1., x)) for x in w))
        if trial == 0:
            weights.setdefault(f"{arm}@update257", set()).add(tuple(cc.weights(256)))
out["weights"] = {k: sorted(map(str, v)) for k, v in weights.items()}
out["wants_gradients_255"] = {arm: ContrastiveConfig(**{**prod, "arm": arm, "nce_weight": .1, "rnc_weight": .1}).wants_gradients(255)
                              for arm in ("C0", "CN", "CR")}

# 3) six real CPU probes per point, with instrumentation (commit/before_update calls, autograd graph signature)
root, manifest, pose = _pose_setup(WORK)
cfg = _config(pose)
cfg = replace(cfg, model=replace(cfg.model, muscle_dropout=.4), contrastive={**cfg.contrastive, "fill_steps": 256})
loader = PretrainLoader(P.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
batch = loader.batch(4, np.random.default_rng(42))
micros = [P.trainer._prepare(batch, "cpu")]
torch.manual_seed(0)
student = P.attach_contrastive_head(P.WearUsFM(cfg.model))
teacher = P.make_teacher(student)
with torch.no_grad():
    next(teacher.parameters()).add_(.001)

calls = {"commit": 0, "before_update": 0}
orig_commit, orig_before = ContrastiveRuntime.commit, ContrastiveRuntime.before_update
def counting(name, fn):
    def wrapped(self, *a, **k):
        calls[name] += 1
        return fn(self, *a, **k)
    return wrapped
ContrastiveRuntime.commit = counting("commit", orig_commit)
ContrastiveRuntime.before_update = counting("before_update", orig_before)

graphs = []
orig_bl = J.LossTerms.backward_loss
def graph_signature(t):
    seen, order, stack = set(), [], [t.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        order.append(type(node).__name__)
        stack.extend(n for n, _ in node.next_functions)
    return len(order), state_sha256(sorted(order))
def bl(self, *a, **k):
    loss = orig_bl(self, *a, **k)
    graphs.append({"weights": {k2: self.weight[k2] for k2 in ("contrast_nce", "contrast_rnc")},
                   "signature": graph_signature(loss)})
    return loss
J.LossTerms.backward_loss = bl

def run_point(last_step, weights_patch=None):
    runtime = ContrastiveRuntime(cfg.contrastive)
    runtime.expected_datasets = ("emg2pose",)
    keydim = student.contrastive_head.layers[-1].out_features
    key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
    ident = WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)
    runtime.queue.commit(key, [ident], step=0)
    saved = runtime.state_dict()
    saved["queue"]["last_step"] = last_step
    saved["queue"]["entries"][0]["step"] = last_step
    initial = P.rng_state(torch.device("cpu"))
    graphs.clear()
    rows = {}
    before_calls = dict(calls)
    orig_w = ContrastiveConfig.weights
    if weights_patch is not None:
        ContrastiveConfig.weights = weights_patch
    try:
        with numeric_profile(require_cuda=False):
            for name in P.CASES:
                m = P._measure_case(name, cfg, student.state_dict(), teacher.state_dict(), saved, batch, micros,
                                    initial, P.trainer.DistContext(), torch.device("cpu"),
                                    P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, require_cuda=False)
                rows[name] = {"grad": m["gradient_sha256"][:16], "w": (m["nce_weight"], m["rnc_weight"]),
                              "nce_eligible": m["nce_eligible"], "rnc_pairs": m["rnc_pairs"]}
    finally:
        ContrastiveConfig.weights = orig_w
    sigs = [g["signature"] for g in graphs]
    return {"cases": rows, "unique_grad": sorted({r["grad"] for r in rows.values()}),
            "graph_signatures_unique": sorted(map(str, set(sigs))), "n_graphs": len(sigs),
            "loss_weights_seen": sorted({str(g["weights"]) for g in graphs}),
            "commit_calls": calls["commit"] - before_calls["commit"],
            "before_update_calls": calls["before_update"] - before_calls["before_update"]}

out["update128"] = run_point(126)
out["update256"] = run_point(254)

# 4) positive control: label made non-inert (tiny CN weight at zero-weight steps) -> within-arm still identical, between-arm differs
def leaky(self, step):
    return (1e-3 if self.arm == "CN" else 0., 0.)
pc = run_point(126, leaky)
c = pc["cases"]
pc["within_equal"] = {a: c[a + "a"]["grad"] == c[a + "b"]["grad"] for a in ("C0", "CN", "CR")}
pc["between_equal"] = {f"{x}/{y}": c[x]["grad"] == c[y]["grad"] for x, y in P.PAIRS[3:]}
out["positive_control_leaky_CN"] = pc

print(json.dumps(out, indent=1, default=str))
