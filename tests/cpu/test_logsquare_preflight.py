"""Independent moment oracle and paired real trainer updates."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_tokenizer_preflight import miniature
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("logsquare_preflight_test", REPO / "scripts/logsquare_preflight.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def test_log_moments_match_independent_numpy_oracle_and_merge():
    x = torch.randn(91, 64, generator=torch.Generator().manual_seed(91))
    x[:10] = 0
    a, b = L.LogMoments(), L.LogMoments()
    a.add("g", x[:40], "identity")
    b.add("g", x[40:], "identity")
    a.merge(b.serializable())
    got = a.summary()["g"]["logsquare"]
    z = x.numpy()
    y = np.log(z * z + np.float32(.001)).astype(np.float64)
    np.testing.assert_allclose(got["mean_per_kernel"], y.mean(0), atol=2e-7)
    np.testing.assert_allclose(got["std_per_kernel"], y.std(0), atol=2e-7)
    np.testing.assert_allclose(got["below_floor_fraction_per_kernel"], (z*z < .001).mean(0), atol=0)
    d = (2*z / (.001+z*z)).astype(np.float64)
    np.testing.assert_allclose(got["derivative_rms_per_kernel"], np.sqrt((d*d).mean(0)), atol=2e-7)
    assert got["analytic_max_abs_derivative"] == pytest.approx(1/np.sqrt(.001))
    a.add("g", x, "identity")  # repeated accumulation, including custom fields
    assert a.summary()["g"]["n_tokens"] == 182
    with pytest.raises(ValueError, match="nonfinite"):
        a.add("bad", torch.full((1,64), float("inf")), "identity")


def test_paired_real_training_has_identical_support_rng_and_learns_new_branch():
    configs, micro, rows = miniature()
    cfg = configs["T100-L"]
    configs = {"T100-L": cfg, "T100-LQ": replace(cfg, model=replace(cfg.model, tokenizer_logsquare=True))}
    torch.manual_seed(cfg.seed)
    expected_hash = L.P.state_sha256(WearUsFM(cfg.model).state_dict())
    arms = L.create_arms(configs, R.DistContext(), "cpu", expected_hash)
    L.check_initial_function(arms, micro, "cpu")
    for step in range(2):
        records = {name: L.P.update_arm(arm, micro, rows, step, R.DistContext(), "cpu")
                   for name, arm in arms.items()}
        L.verify_pair(records)
        assert records["T100-LQ"]["gradient_norms_before_clip"]["logsquare"] > 0
        assert records["T100-LQ"]["update_norms"]["logsquare"] > 0
    assert arms["T100-LQ"]["student"].tokenizer.logsquare_weight.count_nonzero() > 0
    broken = copy.deepcopy(records)
    broken["T100-LQ"]["support"]["pose_support_sha256"] = "wrong"
    with pytest.raises(ValueError, match="support"):
        L.verify_pair(broken)
    broken = copy.deepcopy(records)
    broken["T100-LQ"]["rng_after_sha256"] = "wrong"
    with pytest.raises(ValueError, match="RNG"):
        L.verify_pair(broken)
    with pytest.raises(ValueError, match="historical"):
        L.create_arms(configs, R.DistContext(), "cpu", "wrong")


def test_cli_preserves_defaults_and_rejects_unregistered_combination():
    spec = importlib.util.spec_from_file_location("logsquare_cli_test", REPO / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]
    def config(extra):
        return cli.build_config(cli.build_parser().parse_args(base + extra))
    assert config([]).model.tokenizer_logsquare is False
    assert config(["--tokenizer-logsquare", "--window-ms", "100"]).model.tokenizer_logsquare
    for extra in (["--tokenizer-logsquare-eps", ".001"],
                  ["--tokenizer-logsquare", "--tokenizer-activation", "gelu"],
                  ["--tokenizer-logsquare", "--tokenizer-logsquare-eps", "nan"]):
        with pytest.raises(ValueError):
            config(extra)


def test_registered_plan_pins_exact_candidate_and_rejects_epsilon_change(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("logsquare_plan_cli", REPO / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    entries = json.loads((REPO / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_text())
    argv = entries["arm_configurations"]["T100-L"]["validated_trainer_argv"][1:]
    baseline = cli.build_config(cli.build_parser().parse_args(argv))
    plan_path = REPO / "results/infra/logsquare_review_20261104/preflight_plan.json"
    plan = json.loads(plan_path.read_text())
    # Remote data are not read on the Mac; the historical loader's remote
    # pin checks are tested separately. Exercise the new plan's real local pins.
    monkeypatch.setattr(L.P, "load_plan", lambda p: ({"batches":64},
                        {"T100-L":baseline}, {"plan":L.P.file_identity(p)}))
    _, configs, states = L.load_plan(plan_path)
    expected = R.config_to_dict(baseline)
    expected["model"]["tokenizer_logsquare"] = True
    assert R.config_to_dict(configs["T100-LQ"]) == expected
    assert states["candidate_config"]["sha256"] == plan["candidate_config"]["sha256"]
    plan["epsilon"] = .01
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="epsilon"):
        L.load_plan(changed)
    plan["epsilon"] = .001
    plan["candidate_config"]["sha256"] = "wrong"
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="SHA mismatch"):
        L.load_plan(changed)
