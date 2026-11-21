"""Warm the real Gloo reducer without advancing the model, optimizer, input stream or RNG."""
import copy
from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import contrastive_cn_restart_leg as leg


class TestModelFixture(torch.nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(135))
        self.register_buffer("calls", torch.tensor(0), persistent=False)

    def forward(self, x):
        self.calls.add_(1)
        value = torch.nn.functional.dropout(x, p=.1, training=True)
        for layer in self.layers:
            value = value + .01 * layer(value).tanh()
        return value.square().sum()


class FixtureRuntime:
    def __init__(self):
        self.step = 544
        self.pending = None
        self.ids = ["staged first batch"]
        self.coverage = {"updates": 9}
        self.queue = {"last": 543}

    def state_dict(self):
        assert self.pending is None
        return {"coverage": self.coverage, "queue": self.queue}


def fake_accumulate(ddp, student, teacher, micros, cfg, ctx, device, amp, *, contrastive):
    # All three process RNG domains advance, and the runtime develops pending state.
    value = micros[0] + np.random.uniform() + random.random()
    ddp(value).backward()
    contrastive.pending = torch.ones(1)
    contrastive.coverage["updates"] += 1
    contrastive.queue["last"] = 544
    return None, {"ok": 1}


def run_gloo(rank, world, destination, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=50))
    try:
        root = Path(destination)
        torch.set_num_threads(1)
        for variant in ("ordinary", "warm"):
            directory = root / variant
            directory.mkdir(exist_ok=True)
            torch.manual_seed(252)
            student = TestModelFixture()
            teacher = copy.deepcopy(student)
            ddp = DistributedDataParallel(student, bucket_cap_mb=.001)
            opt = torch.optim.AdamW(student.parameters(), lr=.001)
            random.seed(100 + rank)
            np.random.seed(100 + rank)
            torch.manual_seed(100 + rank)
            batch = [torch.randn(3,4) + rank]
            batch[0].anchor_targets = [torch.tensor(2.)]
            runtime = FixtureRuntime()
            ctx = SimpleNamespace(rank=rank, world_size=world)
            def cpu_identity(_):
                return {"host": "gloo", "device_index": rank, "device_name": "cpu", "gpu_uuid": "test"}
            with (patch.object(leg.trainer, "_accumulate_terms", fake_accumulate),
                  patch.object(leg.trainer, "_micro_digest", leg.state_sha256),
                  patch.object(leg, "device_identity", cpu_identity),
                  leg.instrument_leg(variant, directory)):
                opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                leg.trainer._accumulate_terms(ddp, student, teacher, batch, None, ctx, "cpu", None,
                                              contrastive=runtime)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                opt.step()
            assert runtime.coverage == {"updates": 10}
            assert student.calls.item() == 1
    finally:
        dist.destroy_process_group()


def test_real_gloo_observes_cold_and_rebuilt_reductions_and_neutral_warmup(tmp_path):
    world = 4
    mp.spawn(run_gloo, args=(world, str(tmp_path), (tmp_path / "store").as_uri()), nprocs=world, join=True)
    for rank in range(world):
        by_variant = {}
        boundaries = {}
        for variant in ("ordinary", "warm"):
            directory = tmp_path / variant
            boundaries[variant] = json.loads((directory / f"boundary.rank{rank}.json").read_text())
            snap = boundaries[variant]
            assert snap["null_warmups"] == int(variant == "warm")
            assert snap["before"] == snap["after_warmup"]
            assert snap["before"]["grad_none"] is True
            by_variant[variant] = json.loads((directory / f"observation.rank{rank}.json").read_text())
            post = json.loads((directory / f"post_update.rank{rank}.json").read_text())
            assert [row["id"] for row in post["rows"]] == list(range(270))
            assert all(len(row["exp_avg_sha256"]) == 64 for row in post["rows"])
        assert boundaries["ordinary"]["before"] == boundaries["warm"]["before"]
        assert len(by_variant["ordinary"]["buckets"]) < len(by_variant["warm"]["buckets"])
        for a,b in zip(by_variant["ordinary"]["parameters"], by_variant["warm"]["parameters"], strict=True):
            assert a["local_sha256"] == b["local_sha256"]
            assert len(a["reduced_sha256"]) == len(b["reduced_sha256"]) == 64


def test_null_backward_restores_rng_and_gradients_after_failure():
    student = torch.nn.Linear(4, 4)
    teacher = copy.deepcopy(student)
    opt = torch.optim.AdamW(student.parameters())
    runtime = FixtureRuntime()
    initial = leg.state_sha256(leg.rng_states("cpu"))
    def fail(surrogate):
        assert surrogate is not runtime
        surrogate.coverage.clear()
        student(torch.randn(4)).sum().backward()
        random.random()
        np.random.uniform()
        return "test failure", {}
    with pytest.raises(RuntimeError, match="test failure"):
        leg.null_backward(fail, student, teacher, opt, runtime, [torch.zeros(1)], "cpu")
    assert leg.state_sha256(leg.rng_states("cpu")) == initial
    assert runtime.coverage == {"updates": 9}
    assert all(p.grad is None for p in student.parameters())
    assert not opt.state


def test_diagnostic_leg_rejects_swapped_release_and_output(tmp_path):
    job = json.loads(leg.ARCHIVED_JOB.read_text())
    directory = tmp_path / "ordinary"
    plan = {"output": str(tmp_path)}
    args = ["--variant", "ordinary", "--out-dir", str(directory), "--stage", "benchmark", "--time-limit-s", "1770"]
    for key, flag in (("config", "--config"), ("release", "--release"), ("audit", "--audit-report"),
                      ("cohort", "--cohort"), ("calibration_report", "--calibration-report")):
        args.extend([flag, str(leg.ROOT / job[key]["path"])])
    args.extend(["--calibration-run-dir", str(Path(job["calibration_summary"]["path"]).parent)])
    for k in ("manifest", "scales", "window_seconds"):
        args.extend(["--" + k.replace("_", "-"), job["inputs"][k]["path"]])
    for directory_name in job["roots"]:
        args.extend(["--root", directory_name])
    assert leg.validate_argv(args, plan)[:2] == ("ordinary", directory)
    changed = args.copy()
    changed[changed.index("--release") + 1] = "foreign.json"
    with pytest.raises(ValueError, match="input paths"):
        leg.validate_argv(changed, plan)
    directory.mkdir()
    with pytest.raises(ValueError, match="output must be fresh"):
        leg.validate_argv(args, plan)


def test_null_shadow_uses_the_prepared_real_infonce_runtime_and_anchor_inputs(tmp_path):
    from test_contrastive_runtime import _config, _pose_setup
    from wearusfm.data.pretraining_loader import PretrainLoader
    from wearusfm.model.fm import WearUsFM
    from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
    from wearusfm.training.contrastive_memory import WindowIdentity
    from wearusfm.training.jepa import make_teacher

    root, manifest, pose = _pose_setup(tmp_path)
    cfg = _config(pose, arm="CN")
    loader = PretrainLoader(leg.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
    batch = loader.batch(4, np.random.default_rng(42))
    micros = [leg.trainer._prepare(batch, "cpu")]
    student = attach_contrastive_head(WearUsFM(cfg.model))
    teacher = make_teacher(student)
    optimizer = torch.optim.AdamW(student.parameters())
    runtime = ContrastiveRuntime(cfg.contrastive)
    runtime.expected_datasets = ("emg2pose",)
    keydim = student.contrastive_head.layers[-1].out_features
    key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
    ident = WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)
    runtime.queue.commit(key, [ident], step=0)
    state = runtime.state_dict()
    state["queue"]["last_step"] = state["queue"]["entries"][0]["step"] = 543
    runtime.load_state_dict(state, completed_steps=544)
    runtime.prepare(batch, 544)
    before = leg.snapshot(student, teacher, optimizer, runtime, micros, "cpu")
    def backward(shadow):
        assert shadow is not runtime and shadow.step == 544 and len(shadow.global_ids) == 4
        result = leg.trainer._accumulate_terms(None, student, teacher, micros, cfg,
            leg.trainer.DistContext(), "cpu", nullcontext(), contrastive=shadow)
        assert shadow.pending is not None and runtime.pending is None
        assert any(param.grad is not None for param in student.parameters())
        return result
    leg.null_backward(backward, student, teacher, optimizer, runtime, micros, "cpu")
    assert leg.snapshot(student, teacher, optimizer, runtime, micros, "cpu") == before
    assert runtime.queue.last_step == 543
