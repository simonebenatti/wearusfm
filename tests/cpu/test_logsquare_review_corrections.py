"""Regression checks for Claude's preparatory-collaudo review, without CUDA jobs."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_tokenizer_preflight import miniature
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "logsquare_review_corrections", REPO / "scripts/logsquare_preflight.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class Encoder:
    """Keep a non-RNG call counter to emulate nondeterministic CUDA output."""
    def __init__(self, *, drift=0., shift=0., nonfinite=False):
        self.calls = 0
        self.drift, self.shift, self.nonfinite = drift, shift, nonfinite

    def eval(self):
        return self

    def encode(self, inp, visible):
        self.calls += 1
        # This draw must be replayed by the gate, unlike the emulated drift.
        value = torch.rand(2, 3) + (self.calls - 1) * self.drift + self.shift
        if self.nonfinite:
            value[0, 0] = float("nan")
        return SimpleNamespace(local=value, z=value * 2)


def paired_encoders(control=None, candidate=None):
    rng = L.P.rng_state("cpu")
    return {name: {"student": encoder, "cfg": SimpleNamespace(amp_bf16=True), "rng": rng}
            for name, encoder in zip(L.ARMS, (control or Encoder(), candidate or Encoder()))}


def test_initial_gate_replays_rng_checks_control_twice_and_persists_success(tmp_path):
    arms = paired_encoders()
    path = tmp_path / "initial.json"
    report = L.check_initial_function(arms, SimpleNamespace(inp=None, visible=None), "cpu", path)
    assert report == json.loads(path.read_text())
    assert report["status"] == "passed"
    assert arms["T100-L"]["student"].calls == 2
    assert arms["T100-LQ"]["student"].calls == 1
    for comparison in (report["control_self_comparison"], report["between_arms"]):
        for values in comparison.values():
            assert values == {"exact": True, "finite": True, "max_abs_difference": 0.}


@pytest.mark.parametrize("drift", [.25, 1.])
def test_nondeterministic_control_fails_before_candidate_comparison(tmp_path, drift):
    arms = paired_encoders(control=Encoder(drift=drift))
    path = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="nondeterministic_control_forward"):
        L.check_initial_function(arms, SimpleNamespace(inp=None, visible=None), "cpu", path)
    report = json.loads(path.read_text())
    assert report["status"] == "nondeterministic_control_forward"
    assert report["between_arms"] is None
    assert arms["T100-LQ"]["student"].calls == 0
    assert report["control_self_comparison"]["local"]["max_abs_difference"] == pytest.approx(drift)
    assert report["control_self_comparison"]["z"]["max_abs_difference"] == pytest.approx(2 * drift)


def test_deterministic_control_distinguishes_true_candidate_mismatch(tmp_path):
    arms = paired_encoders(candidate=Encoder(shift=.5))
    path = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="initial_function_mismatch"):
        L.check_initial_function(arms, SimpleNamespace(inp=None, visible=None), "cpu", path)
    report = json.loads(path.read_text())
    assert report["status"] == "initial_function_mismatch"
    assert all(v["exact"] for v in report["control_self_comparison"].values())
    assert report["between_arms"]["local"]["max_abs_difference"] == pytest.approx(.5)
    assert report["between_arms"]["z"]["max_abs_difference"] == pytest.approx(1.)


def test_nonfinite_control_diagnostic_is_valid_json_and_stops_before_candidate(tmp_path):
    arms = paired_encoders(control=Encoder(nonfinite=True))
    path = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match="nonfinite_control_forward"):
        L.check_initial_function(arms, SimpleNamespace(inp=None, visible=None), "cpu", path)
    report = json.loads(path.read_text())
    assert report["between_arms"] is None
    assert arms["T100-LQ"]["student"].calls == 0
    for values in report["control_self_comparison"].values():
        assert not values["finite"] and not values["exact"]
        assert values["max_abs_difference"] is None


def record(step, norm, log_norm=None):
    return {"step": step, "grad_norm_before_clip": norm,
            "gradient_norms_before_clip": {} if log_norm is None else {"logsquare": log_norm}}


def clipping_part(rank=0):
    # At initialization [3, 4] shared gradients plus log-gradient [12] give
    # exactly 5, 12, 13. Later records need not retain this decomposition.
    return {"rank": rank, "arms": {
        "T100-L": {"records": [record(0, 5.), record(1, 0.), record(2, .25)]},
        "T100-LQ": {"records": [record(0, 13., 12.), record(1, 0., 0.), record(2, .5, .4)]}}}


def torch_clip_factor(vector, max_norm):
    p = torch.nn.Parameter(torch.zeros(len(vector), dtype=torch.float64))
    p.grad = torch.tensor(vector, dtype=torch.float64)
    original = p.grad.clone()
    torch.nn.utils.clip_grad_norm_([p], max_norm)
    active = original != 0
    return float((p.grad[active] / original[active])[0]) if active.any() else 1.


@pytest.mark.parametrize("max_norm", [.1, 1., 20.])
def test_clip_factors_match_torch_oracle_and_preserve_ranks(max_norm):
    configs = {name: SimpleNamespace(grad_clip=max_norm) for name in L.ARMS}
    result = L.clipping_summary([clipping_part(0), clipping_part(3)], configs)
    expected = {
        "T100-L": [torch_clip_factor(g, max_norm) for g in ([3., 4.], [0.], [.25])],
        "T100-LQ": [torch_clip_factor(g, max_norm) for g in ([3., 4., 12.], [0.], [.5])]}
    ratio = np.array(expected["T100-LQ"]) / np.array(expected["T100-L"])
    for name in L.ARMS:
        assert set(result[name]["per_rank"]) == {"0", "3"}
        for rank in result[name]["per_rank"].values():
            assert rank["steps"] == [0, 1, 2]
            np.testing.assert_allclose(rank["clip_factor_per_step"], expected[name], rtol=1e-14)
            np.testing.assert_allclose(rank["clip_factor_LQ_over_L_per_step"], ratio, rtol=1e-14)
            assert rank["fraction_steps_clipped"] == pytest.approx(sum(f < 1 for f in expected[name]) / 3)
    q = result["T100-LQ"]["per_rank"]["0"]
    assert q["logsquare_gradient_squared_share_per_step"] == pytest.approx([144 / 169, None, .64])
    assert q["first_step_norm_decomposition_relative_residual"] == 0.
    assert result["T100-L"]["per_rank"]["0"]["logsquare_gradient_squared_share_per_step"] is None


def test_all_zero_gradient_norms_remain_json_safe_and_unclipped():
    part = {"rank": 2, "arms": {name: {"records": [record(0, 0., 0.)]} for name in L.ARMS}}
    result = L.clipping_summary([part], {name: SimpleNamespace(grad_clip=1.) for name in L.ARMS})
    json.dumps(result, allow_nan=False)
    for name in L.ARMS:
        rank = result[name]["per_rank"]["2"]
        assert rank["clip_factor_per_step"] == [1.]
        assert rank["clip_factor_LQ_over_L_per_step"] == [1.]
        assert rank["fraction_steps_clipped"] == 0.
    q = result["T100-LQ"]["per_rank"]["2"]
    assert q["logsquare_gradient_squared_share_per_step"] == [None]
    assert q["first_step_norm_decomposition_relative_residual"] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.])
def test_clip_summary_rejects_invalid_norms(invalid):
    part = clipping_part()
    part["arms"]["T100-LQ"]["records"][0]["grad_norm_before_clip"] = invalid
    with pytest.raises(ValueError, match="invalid gradient norm"):
        L.clipping_summary([part], {name: SimpleNamespace(grad_clip=1.) for name in L.ARMS})


def test_clip_summary_rejects_unpaired_steps():
    part = clipping_part()
    part["arms"]["T100-LQ"]["records"][1]["step"] = 7
    with pytest.raises(ValueError, match="unpaired"):
        L.clipping_summary([part], {name: SimpleNamespace(grad_clip=1.) for name in L.ARMS})


def test_jsonl_journal_retains_completed_steps_and_never_replaces_existing_run(tmp_path):
    path = tmp_path / "rank0.jsonl"
    records = {name: {"grad_norm_before_clip": .25} for name in L.ARMS}
    L.journal_step(path, 0, "first-batch", records)
    first = path.read_bytes()
    assert json.loads(first)["step"] == 1
    L.journal_step(path, 1, "second-batch", records)
    assert path.read_bytes().startswith(first)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"step": i + 1, "batch_sha256": batch, "arms": records}
                     for i, batch in enumerate(("first-batch", "second-batch"))]
    complete = path.read_bytes()
    with pytest.raises(FileExistsError):
        L.journal_step(path, 0, "replacement-batch", records)
    assert path.read_bytes() == complete


def test_explicit_nondefault_epsilon_controls_moments_and_arm_collectors():
    epsilon = .01
    x = torch.tensor([0., .05, .2, 1.], dtype=torch.float32)[:, None].expand(-1, 64)
    moments = L.LogMoments(epsilon)
    moments.add("native200", x, "identity")
    result = moments.summary()["native200"]["logsquare"]
    z = x.numpy()
    expected = np.log(z * z + np.float32(epsilon)).astype(np.float64)
    derivative = (2 * z / (np.float32(epsilon) + z * z)).astype(np.float64)
    np.testing.assert_allclose(result["mean_per_kernel"], expected.mean(0), atol=2e-7)
    np.testing.assert_allclose(result["derivative_rms_per_kernel"], np.sqrt((derivative ** 2).mean(0)), atol=2e-7)
    assert result["below_floor_fraction_per_kernel"] == [.5] * 64
    assert result["analytic_max_abs_derivative"] == 10.
    assert result["epsilon"] == epsilon

    configs, _, _ = miniature()
    baseline = configs["T100-L"]
    configs = {"T100-L": baseline, "T100-LQ": replace(
        baseline, model=replace(baseline.model, tokenizer_logsquare=True, tokenizer_logsquare_eps=epsilon))}
    torch.manual_seed(baseline.seed)
    historical = L.P.state_sha256(WearUsFM(baseline.model).state_dict())
    arms = L.create_arms(configs, R.DistContext(), "cpu", historical, epsilon)
    assert all(arm["features"].epsilon == epsilon for arm in arms.values())
    with pytest.raises(ValueError, match="diagnostic/model epsilon mismatch"):
        L.create_arms(configs, R.DistContext(), "cpu", historical, .001)
