"""Independent update oracle and guards for the new scaling technical screen."""
import copy
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_tokenizer_preflight import miniature
from wearusfm.training import run as R
from wearusfm.training import jepa as J

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scaling_preflight_test", ROOT / "scripts/scaling_preflight.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def actual_config():
    spec = importlib.util.spec_from_file_location("scaling_train_cli_test", ROOT / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    comparison = json.loads((ROOT / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_text())
    argv = comparison["arm_configurations"]["T100-G"]["validated_trainer_argv"]
    return cli.build_config(cli.build_parser().parse_args(argv[1:]))


def test_only_ladder_dimensions_change_and_absolute_rank_alarm_is_preserved():
    cfg = actual_config()
    configs = S.configurations(cfg)
    a, b = (R.config_to_dict(configs[name]) for name in S.ARMS)
    assert b["model"]["dim"] == 896 and b["model"]["n_heads"] == 14
    assert b["model"]["backbone_layers"] == 18 and b["model"]["local_layers"] == 3
    assert b["alarm"] == a["alarm"] and b["alarm"]["erank_min"] == 38.4
    for key in R.LADDER["300m"]:
        b["model"][key] = a["model"][key]
    assert a == b


@pytest.mark.parametrize("field,value", [("lr", 3e-4), ("batch_size", 16), ("grad_clip", 2), ("seed", 1)])
def test_production_optimizer_recipe_cannot_change_silently(field, value):
    with pytest.raises(ValueError, match="recipe"):
        S.configurations(replace(actual_config(), **{field: value}))


@pytest.mark.parametrize("step,phase", [(0, "native_warmup"), (64, "peak_lr_stress")])
def test_native_and_stress_updates_match_independent_production_update(step, phase):
    configs, micro, rows = miniature()
    cfg = configs["T100-G"]
    arm = S.create_arm(cfg, R.DistContext(), "cpu")
    initial_rng = copy.deepcopy(arm["rng"])
    plain = R.WearUsFM(cfg.model)
    plain.load_state_dict(arm["student"].state_dict())
    teacher = J.make_teacher(plain)
    optimizer = torch.optim.AdamW(plain.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    expected_cfg = cfg if phase == "native_warmup" else replace(cfg, warmup_steps=0)
    optimizer.param_groups[0]["lr"] = R._lr_at(step, expected_cfg)
    S.P.restore_rng(initial_rng, "cpu")
    failure, _ = R._accumulate_terms(None, plain, teacher, [micro], expected_cfg,
                                    R.DistContext(), "cpu", nullcontext())
    assert failure is None
    grad = torch.nn.utils.clip_grad_norm_(plain.parameters(), cfg.grad_clip)
    optimizer.step()
    J.ema_update(teacher, plain, cfg.jepa.ema_momentum)
    got = S.update_arm(arm, micro, rows, step, 64, R.DistContext(), "cpu")
    assert arm["cfg"] is cfg and got["scientific_update"] is False and got["phase"] == phase
    assert got["lr"] == R._lr_at(step, expected_cfg)
    assert got["clip_factor"] == pytest.approx(min(1, cfg.grad_clip / (float(grad) + 1e-6)))
    for name, value in plain.state_dict().items():
        assert torch.equal(value, arm["student"].state_dict()[name]), name
        assert torch.equal(teacher.state_dict()[name], arm["teacher"].state_dict()[name]), name


def test_different_dimensions_keep_same_frontend_but_not_same_model_rng():
    configs, _, _ = miniature()
    cfg = configs["T100-G"]
    a = S.create_arm(cfg, R.DistContext(), "cpu")
    b = S.create_arm(replace(cfg, model=replace(cfg.model, dim=cfg.model.dim * 2)), R.DistContext(), "cpu")
    assert a["initial"]["frontend_and_gain_sha256"] == b["initial"]["frontend_and_gain_sha256"]
    assert a["initial"]["state_sha256"] != b["initial"]["state_sha256"]
    assert not torch.equal(a["rng"]["cpu"], b["rng"]["cpu"])
    assert "backbone_first" not in a["groups"]
    for label in ("first", "last"):
        assert next(iter(a["groups"][f"backbone_{label}_qkv"])).endswith("time_attn.qkv.weight")
        assert next(iter(a["groups"][f"backbone_{label}_mlp"])).endswith("mlp.0.weight")


def test_native_to_stress_transition_preserves_adamw_moments_and_matches_oracle():
    configs, micro, rows = miniature()
    cfg = configs["T100-G"]
    arm = S.create_arm(cfg, R.DistContext(), "cpu")
    plain = R.WearUsFM(cfg.model)
    plain.load_state_dict(arm["student"].state_dict())
    teacher = J.make_teacher(plain)
    opt = torch.optim.AdamW(plain.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for step in (63, 64):
        actual_cfg = cfg if step == 63 else replace(cfg, warmup_steps=0)
        S.P.restore_rng(copy.deepcopy(arm["rng"]), "cpu")
        opt.zero_grad(set_to_none=True)
        opt.param_groups[0]["lr"] = R._lr_at(step, actual_cfg)
        failure, _ = R._accumulate_terms(None, plain, teacher, [micro], actual_cfg,
                                        R.DistContext(), "cpu", nullcontext())
        assert failure is None
        torch.nn.utils.clip_grad_norm_(plain.parameters(), cfg.grad_clip)
        opt.step()
        J.ema_update(teacher, plain, cfg.jepa.ema_momentum)
        S.update_arm(arm, micro, rows, step, 64, R.DistContext(), "cpu")
        for name, value in plain.state_dict().items():
            assert torch.equal(value, arm["student"].state_dict()[name]), name
            assert torch.equal(teacher.state_dict()[name], arm["teacher"].state_dict()[name]), name
        actual = arm["optimizer"].state_dict()
        expected = opt.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        for key, moments in expected["state"].items():
            for field, value in moments.items():
                assert torch.equal(value, actual["state"][key][field]), (key, field)
    assert all(float(state["step"]) == 2 for state in arm["optimizer"].state.values())


def paired_fixture():
    result = {"initial": {"frontend_and_gain_sha256": "a" * 64}, "delivered": [{"batch_sha256": "b" * 64}],
              "records": [{"step": 1, "support": {"target": "same"}, "lr": 1e-4, "phase": "peak_lr_stress", "rng_after_sha256": "one"}]}
    return {name: copy.deepcopy(result) for name in S.ARMS}


def test_pairing_checks_data_targets_and_frontend_but_allows_different_dropout_rng():
    parts = paired_fixture()
    parts[S.ARMS[1]]["records"][0]["rng_after_sha256"] = "different"
    S.verify_pairing(parts, 1, 0)
    for key in ("delivered", "initial", "records"):
        bad = copy.deepcopy(parts)
        if key == "delivered":
            bad[S.ARMS[1]][key][0]["batch_sha256"] = "wrong"
        elif key == "initial":
            bad[S.ARMS[1]][key]["frontend_and_gain_sha256"] = "wrong"
        else:
            bad[S.ARMS[1]][key][0]["support"]["target"] = "wrong"
        with pytest.raises(ValueError):
            S.verify_pairing(bad, 1, 0)


@pytest.mark.parametrize("change", ["missing", "step", "phase"])
def test_missing_or_wrong_updates_cannot_receive_complete_pairing(change):
    parts = paired_fixture()
    rows = parts[S.ARMS[1]]["records"]
    if change == "missing":
        rows.clear()
    elif change == "step":
        rows[0]["step"] = 2
    else:
        rows[0]["phase"] = "native_warmup"
    with pytest.raises(ValueError):
        S.verify_pairing(parts, 1, 0)


def test_memory_gate_uses_single_arm_rank_max_and_timing_uses_component_maxima():
    parts = []
    for rank in range(2):
        rows = [{"forward_backward_s": 2. if rank == 0 else 1., "clip_optimizer_ema_s": 1. if rank == 0 else 2.,
                 "memory_peak_allocated_bytes": 50 + 10 * rank, "memory_peak_reserved_bytes": 70,
                 "grad_norm_before_clip": 2., "clip_factor": .5, "relative_update_norms": {"projection": .001}}
                 for _ in range(20)]
        parts.append({"total_device_memory_bytes": 100, "arms": {"large": {"initial": {}, "records": rows}}})
    got = S.summarize(parts, "large", 10, .9)
    assert got["allocated_fraction"] == .6 and got["peak_allocated_bytes"] == 60
    assert got["phases"]["peak_lr_stress"]["normal_update_mean_s"] == 4.
    parts[1]["arms"]["large"]["records"][3]["memory_peak_allocated_bytes"] = 91
    with pytest.raises(ValueError, match="memory gate"):
        S.summarize(parts, "large", 10, .9)


def test_frozen_plan_cannot_expand_dose_or_world_before_loading_external_inputs(tmp_path):
    plan = json.loads((ROOT / "results/infra/scaling_300m_20261104/preflight_plan.json").read_text())
    for key, value in (("native_warmup_batches", 65), ("peak_stress_batches", 128), ("world_size", 8), ("max_memory_fraction", 1.)):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps({**plan, key: value}))
        with pytest.raises(ValueError):
            S.load_plan(path)
