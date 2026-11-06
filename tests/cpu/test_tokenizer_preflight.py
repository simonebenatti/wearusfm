"""Real trainer updates and independent audit oracles for the tokenizer preflight."""
import copy
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_tokenizer_comparison import CFG, ARMS, scene
from wearusfm.training import jepa as J
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tokenizer_preflight_test", REPO / "scripts/tokenizer_preflight.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def miniature():
    base = R.small_config()
    configs = {name: replace(base, model=replace(CFG, window_ms=window, tokenizer_activation=activation, muscle_dropout=.4),
               jepa=replace(base.jepa, jepa_weight=0., pose_weight=1., loss_per_window=True, masked_loss_radius=0))
               for name, (window, activation) in zip(P.ARMS, ARMS)}
    inp, visible, targets, pose = scene()
    micro = R._Micro(inp, visible, (~visible).long(), targets, None, None, pose=pose)
    rows = [{"dataset": "emg2pose", "split": "pretraining"}, {"dataset": "camargo", "split": "pretraining"}]
    return configs, micro, rows


def test_feature_moments_merge_and_nonlinearity_match_direct_numpy_oracle():
    x = torch.randn(129, 64, generator=torch.Generator().manual_seed(24), dtype=torch.float64)
    x[:, 24:48] *= .2
    x[:, 48:] *= .03
    a, b = P.FeatureMoments(), P.FeatureMoments()
    a.add("g", x[:70], "gelu")
    b.add("g", x[70:], "gelu")
    a.merge(b.serializable())
    got = a.summary()["g"]
    y = torch.nn.functional.gelu(x).numpy()
    z = x.numpy()
    assert got["n_tokens"] == 129
    np.testing.assert_allclose(got["scaled_mean"], z.mean(0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got["scaled_std"], z.std(0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got["activated_mean"], y.mean(0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got["activated_std"], y.std(0), rtol=1e-12, atol=1e-15)
    _, singular, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
    np.testing.assert_allclose(got["scaled_centered_singular_values"], singular / np.sqrt(len(z)), rtol=1e-10, atol=1e-14)
    for family, s in (("fourier", slice(0, 24)), ("spline", slice(24, 48)), ("mlp", slice(48, 64))):
        expected = np.sqrt(np.square(y[:, s] - .5 * z[:, s]).sum() / np.square(z[:, s]).sum())
        assert got["families"][family]["gelu_minus_half_x_rms_over_x_rms"] == pytest.approx(expected, rel=1e-12)
    zeros = P.FeatureMoments()
    zeros.add("zero", torch.zeros(3, 64), "identity")
    assert zeros.summary()["zero"]["rank_relative_1e3"] == 0
    assert zeros.summary()["zero"]["families"]["mlp"]["gelu_minus_half_x_rms_over_x_rms"] is None
    with pytest.raises(ValueError, match="nonfinite"):
        zeros.add("bad", torch.full((1, 64), float("nan")), "gelu")


def test_support_oracle_checks_qc_padding_and_keeps_pose_outside_hidden_queries():
    configs, micro, rows = miniature()
    audit, expected = P.support_audit(micro, configs["T100-G"], rows)
    assert expected["ch"].numel() == 388
    assert audit["windows"][0]["queries"] == 300 and audit["windows"][1]["queries"] == 88
    assert [r["pose_valid_elements"] for r in audit["windows"]] == [620, 230]
    assert audit["loss_counts"]["pose"] == 2
    assert not expected["visible"][3].any() and not expected["patch_valid"][16:, 12:].any()
    assert set(expected["time"][expected["ch"] < 16].tolist()) == set(range(4, 24))
    assert set(expected["time"][expected["ch"] >= 16].tolist()) == set(range(4, 12))


def test_three_real_updates_match_plain_trainer_and_preserve_pairing_and_rng():
    configs, micro, rows = miniature()
    arms = P.create_arms(configs, R.DistContext(), "cpu")
    initial_rng = copy.deepcopy(arms["T25-L"]["rng"])
    # A separate, uninstrumented trainer update is the oracle for gradients,
    # clipping, AdamW, EMA and the actual first-step LR.
    cfg = configs["T25-L"]
    plain = R.WearUsFM(cfg.model)
    plain.load_state_dict(arms["T25-L"]["student"].state_dict())
    teacher = J.make_teacher(plain)
    optimizer = torch.optim.AdamW(plain.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.param_groups[0]["lr"] = R._lr_at(0, cfg)
    P.restore_rng(initial_rng, "cpu")
    failure, _ = R._accumulate_terms(None, plain, teacher, [micro], cfg, R.DistContext(), "cpu", nullcontext())
    assert failure is None
    torch.nn.utils.clip_grad_norm_(plain.parameters(), cfg.grad_clip)
    optimizer.step()
    J.ema_update(teacher, plain, cfg.jepa.ema_momentum)
    expected_rng = state_sha256(P.rng_state("cpu"))
    before_inputs = R._micro_digest([micro])
    records = {name: P.update_arm(arm, micro, rows, 0, R.DistContext(), "cpu") for name, arm in arms.items()}
    P.verify_paired(records)
    assert before_inputs == R._micro_digest([micro])
    assert records["T25-L"]["rng_after_sha256"] == expected_rng
    assert records["T25-L"]["lr"] == cfg.lr / cfg.warmup_steps
    for name, value in plain.state_dict().items():
        assert torch.equal(value, arms["T25-L"]["student"].state_dict()[name]), name
        assert torch.equal(teacher.state_dict()[name], arms["T25-L"]["teacher"].state_dict()[name]), name
    for record in records.values():
        assert all(v > 0 for v in record["gradient_norms_before_clip"].values())
        assert all(v > 0 for v in record["update_norms"].values())
        assert record["normal_update_s"] > 0 and record["diagnostic_s"] > 0
    assert set(arms["T100-G"]["features"].summary()) == {
        f"{phase}/{dataset}/{fs}/{mode}" for phase in ("initial", "trajectory_preupdate")
        for dataset, fs in (("emg2pose", "2000"), ("camargo", "1000"))
        for mode in ("masked_visible", "unmasked_same_visible")}
    broken = copy.deepcopy(records)
    broken["T100-G"]["support"]["pose_support_sha256"] = "wrong"
    with pytest.raises(ValueError, match="parity"):
        P.verify_paired(broken)


def test_masked_feature_support_is_identical_and_ignores_hidden_input_perturbations():
    configs, micro, rows = miniature()
    arm = P.create_arms(configs, R.DistContext(), "cpu")["T100-G"]
    first, changed = P.FeatureMoments(), P.FeatureMoments()
    before_rng = state_sha256(P.rng_state("cpu"))
    baseline = P.collect_features(arm["student"], micro, rows, 0, first)
    altered = copy.deepcopy(micro)
    offset = 0
    for s, (fs, count) in enumerate(zip(micro.inp.fs, micro.inp.counts)):
        hidden = ~micro.visible[offset:offset + count, :micro.inp.signals[s].shape[1] // int(fs / 40)]
        altered.inp.signals[s][hidden.repeat_interleave(int(fs / 40), 1)] += 100.
        offset += count
    assert isinstance(baseline, str) and len(baseline) == 64
    P.collect_features(arm["student"], altered, rows, 0, changed)
    # The raw byte hash may distinguish +0/-0 after multiplying hidden samples
    # by zero; the actual features must remain identical.
    assert state_sha256(P.rng_state("cpu")) == before_rng
    aa, bb = first.summary(), changed.summary()
    for key in aa:
        if key.endswith("masked_visible"):
            assert aa[key] == bb[key]
        else:
            assert aa[key]["scaled_mean"] != bb[key]["scaled_mean"]
        other_mode = key.replace("masked_visible", "unmasked_same_visible")
        assert aa[key]["n_tokens"] == aa[other_mode]["n_tokens"]


def test_file_pins_reject_corruption_and_changes_after_validation(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"value":1}')
    identity = P.file_identity(path)
    assert P.pinned_file({"path": path.name, "sha256": identity["sha256"]}, tmp_path) == identity
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="SHA mismatch"):
        P.pinned_file({"path": path.name, "sha256": identity["sha256"]}, tmp_path)
    with pytest.raises(ValueError, match="changed"):
        P.check_unchanged({"input": identity})


def test_production_plan_has_exact_allocation_and_stream_constraints(tmp_path):
    plan = json.loads((REPO / "results/infra/tokenizer_execution_20261103/preflight_plan.json").read_text())
    for key, value in (("batches", 63), ("batches", 65), ("world_size", 1), ("global_batch_size", 8), ("num_workers", 0)):
        changed = {**plan, key: value}
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="64 batches|historical"):
            P.load_plan(path)


def test_one_real_loader_batch_runs_complete_audit_and_writes_no_checkpoint(tmp_path, monkeypatch):
    from test_pretraining_loader import _tree, _cfg
    from test_pose_targets import _pose_tree, _stats

    root, manifest = _tree(tmp_path)
    pose_root = tmp_path / "pose"
    _pose_tree(pose_root, 24000)
    _stats(pose_root, mean=[.5] * 20, std=[.25] * 20)
    configs, _, _ = miniature()
    configs = {name: replace(cfg, loader=_cfg(pose_root=str(pose_root), multiscale_anchor=True),
                            datasets=("emg2pose",), batch_size=2, nested=None) for name, cfg in configs.items()}
    files = {"manifest": manifest, "pose_stats": pose_root / "pose_stats.json"}
    for name, data in (("scales", {"scales": {}}), ("windows", {"window_s": {}})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        files[name] = path
    states = {name: P.file_identity(path) for name, path in files.items()}
    states["plan"] = states["scales"]  # the core run is exercised without weakening production CLI constraints
    plan = {"roots": [str(root)], "batches": 1, "num_workers": 0}
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(P, "code_identity", lambda code: code)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: pytest.fail("preflight must not save a checkpoint"))
    P.run(plan, configs, states, out, R.DistContext(), "cpu", "a" * 40)
    report = json.loads((out / "report.json").read_text())
    rank = json.loads((out / "rank0.json").read_text())
    event = json.loads((out / "rank0.jsonl").read_text())
    assert report["complete"] and report["windows_exposed_each_arm"] == 2
    assert report["exposure_windows"] == {"emg2pose/2000": 2}
    assert rank["sources_unchanged_after_first_seen"]
    assert {Path(p).name for p in rank["sources_stat_first_seen"]} >= {"metadata.json", "data_int16.npy", "pose_200hz_f16.npy", "pose_valid.npy"}
    assert len(event["fingerprints"]["origin_sha256"]) == len(event["fingerprints"]["batch_sha256"]) == 64
    assert set(event["arms"]) == set(P.ARMS)
    assert sorted(p.name for p in out.iterdir()) == ["rank0.json", "rank0.jsonl", "report.json"]


@pytest.mark.parametrize("missing_old_field", [None, "tokenizer_activation", "keep_readout"])
def test_production_config_binding_reconstructs_all_three_full_cli_configs(tmp_path, missing_old_field):
    spec = importlib.util.spec_from_file_location("preflight_cli_fixture", REPO / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    comparison = json.loads((REPO / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_text())
    plan = {"protocol": P.PROTOCOL, "batches": 64, "global_batch_size": 32, "num_workers": 6, "world_size": 4,
            "roots": [str(tmp_path / "processed")], "inputs": {}}
    (tmp_path / "pose").mkdir()
    for name in ("manifest", "scales", "windows", "pose_stats"):
        path = tmp_path / ("pose/pose_stats.json" if name == "pose_stats" else name + ".json")
        path.write_text("{}")
        plan["inputs"][name] = {"path": str(path), "sha256": R.sha256_file(path)}
    mapping = {"--manifest": plan["inputs"]["manifest"]["path"], "--scales": plan["inputs"]["scales"]["path"],
               "--window-seconds": plan["inputs"]["windows"]["path"], "--pose-root": str(tmp_path / "pose")}
    for name, entry in comparison["arm_configurations"].items():
        argv, rewritten, i = entry["validated_trainer_argv"], [], 0
        while i < len(argv):
            arg = argv[i]
            if arg == "--root":
                i += 2
                continue
            rewritten.append(arg)
            if arg in mapping:
                rewritten.append(mapping[arg])
                i += 2
            else:
                i += 1
        rewritten += ["--root", plan["roots"][0]]
        entry["validated_trainer_argv"] = rewritten
        cfg = cli.build_config(cli.build_parser().parse_args(rewritten[1:]))
        path = tmp_path / f"{name}.json"
        stored = R.config_to_dict(cfg)
        # Legacy configs may omit only the two fields introduced by LQ.
        stored["model"].pop("tokenizer_logsquare")
        stored["model"].pop("tokenizer_logsquare_eps")
        if missing_old_field is not None and name == "T100-L":
            stored["model"].pop(missing_old_field)
        path.write_text(json.dumps(stored))
        entry.update(config=str(path), sha256=R.sha256_file(path))
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison))
    plan["comparison"] = {"path": str(path), "sha256": R.sha256_file(path)}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    if missing_old_field is not None:
        with pytest.raises(ValueError, match="CLI/config mismatch"):
            P.load_plan(plan_path)
        return
    _, cfgs, states = P.load_plan(plan_path)
    assert set(cfgs) == set(P.ARMS) and cfgs["T100-G"].model.tokenizer_activation == "gelu"
    assert cfgs["T100-G"].jepa.masked_loss_radius == 0
    P.check_unchanged(states)
    # Pinning only the comparison is insufficient: a config must match its own
    # SHA and the CLI-derived recipe too.
    bad = json.loads((tmp_path / "T100-G.json").read_text())
    bad["lr"] *= 2
    (tmp_path / "T100-G.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="SHA mismatch"):
        P.load_plan(plan_path)
