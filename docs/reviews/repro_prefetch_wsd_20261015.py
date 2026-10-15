"""Additional CPU review checks: prefetch with WSD branching and exact DDP resume.

Run explicitly with PYTHONPATH=src:tests/cpu; no cluster resources required.
The existing WSD scenarios are reused with prefetch enabled, and checkpoint
comparisons additionally include optimizer state and saved random generators.
"""

from dataclasses import replace

import pytest
import torch

import test_training_ddp as D
import test_training_wsd as W


def _same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for item, other in zip(a, b, strict=True):
            _same(item, other)
    else:
        assert a == b


def _same_checkpoint(a, b):
    sa, sb = (torch.load(p, weights_only=False) for p in (a, b))
    for key in ("student", "teacher", "optimizer", "step", "torch_rng", "cuda_rng", "rank_rng"):
        assert (key in sa) == (key in sb), key
        if key in sa:
            _same(sa[key], sb[key])


def _enable_prefetch(monkeypatch):
    train = W.R.train

    def prefetched(cfg, *args, **kwargs):
        return train(replace(cfg, prefetch=True, prefetch_audit=True), *args, **kwargs)

    monkeypatch.setattr(W.R, "train", prefetched)
    monkeypatch.setattr(W, "_same_weights", _same_checkpoint)


@pytest.mark.parametrize("stop", [7, 8, 9])
def test_prefetch_wsd_resume_keeps_optimizer_rng_and_named_checkpoints(tmp_path, monkeypatch, stop):
    _enable_prefetch(monkeypatch)
    W.test_resume_across_the_decay_boundary(tmp_path, stop)


def test_prefetch_wsd_stable_decay_and_continuation_branches(tmp_path, monkeypatch):
    _enable_prefetch(monkeypatch)
    W.test_branches_equal_runs_from_scratch(tmp_path)


@pytest.mark.parametrize("accum", [1, 2])
def test_prefetch_ddp_resume_keeps_optimizer_and_every_rank_rng(tmp_path, accum):
    plain = tmp_path / "plain"
    chained = tmp_path / "chained"
    D._run(tmp_path, "continuous", plain, max_steps=4, accum=accum, prefetch=True)
    D._run(tmp_path, "interrupted", chained, max_steps=2, accum=accum, prefetch=True)
    D._run(tmp_path, "resumed", chained, max_steps=4, accum=accum, prefetch=True)
    _same_checkpoint(plain / "checkpoint.pt", chained / "checkpoint.pt")
    # Ending a segment also evaluates collapse diagnostics: compare the actual
    # loss/count/update records, allowing only timings and extra diagnostics.
    for a, b in zip(D._lines(plain), D._lines(chained), strict=True):
        keys = (a.keys() & b.keys()) - {"t_data_s", "t_data_s_mean", "t_step_s"}
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
