"""Scaling cache regressions and real, small teacher checkpoint restoration.

The large fixed architectures are validated without allocating a 292M model.
Only external I/O/training is replaced in controller tests; cache comparisons
and file/provenance guards use real NPZ/torch files.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import scaling_evaluation as S
import tokenizer_evaluation as E
import test_tokenizer_evaluation as OLD
from wearusfm.harness import pose_eval as PE
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training import run as R

CODE, STATS, CKPT_SHA = OLD.CODE, OLD.STATS, OLD.CKPT_SHA
pin, save_json = OLD.pin, OLD.save_json


def model_config(arm):
    cfg = json.loads((S.ROOT / "results/infra/tokenizer_review_20261103/config_T100-G.json").read_text())["model"]
    cfg.update(R.LADDER["30m" if arm == S.ARMS[0] else "300m"])
    return cfg


@pytest.mark.parametrize("arm", S.ARMS)
def test_registered_architectures_are_exact_and_no_small_or_mixed_preset_passes(arm):
    cfg = model_config(arm)
    S.model_recipe(cfg, arm)
    assert cfg["dim"] == S.DIMENSIONS[arm]
    for key, value in (("dim", 16), ("n_heads", 4), ("local_layers", 1), ("backbone_layers", 1),
                       ("decoder_layers", 1), ("k_latents", 4), ("pose_head", False),
                       ("window_ms", 25.), ("tokenizer_activation", "identity")):
        bad = {**cfg, key: value}
        with pytest.raises(ValueError, match="recipe"):
            S.model_recipe(bad, arm)


@pytest.mark.parametrize("arm", S.ARMS)
def test_public_checkpoint_gate_checks_fixed_dose_and_architecture_before_io(arm, monkeypatch):
    cfg = {"model": model_config(arm), "max_steps": 5000}
    observed = []
    def loader(pin, config, got_arm, stats, signature, source):
        observed.append((config, got_arm, stats, signature, source))
        return {"real_loader_tested_separately": True}
    monkeypatch.setattr(S, "load_teacher_endpoint", loader)
    result = S.validate_checkpoint({}, cfg, arm, 5000, STATS, {"data": True}, {})
    assert result == {"real_loader_tested_separately": True} and len(observed) == 1
    for step in (19618, 4999, True, 5000.):
        with pytest.raises(ValueError, match="dose"):
            S.validate_checkpoint({}, cfg, arm, step, STATS, {}, {})
    cfg["model"]["dim"] = 16
    with pytest.raises(ValueError, match="recipe"):
        S.validate_checkpoint({}, cfg, arm, 5000, STATS, {}, {})
    assert len(observed) == 1


def endpoint(tmp_path, arm):
    cfg = replace(R.sanity_config(), max_steps=5000, model=FMConfig(dim=16, n_heads=4, k_latents=4,
        local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0., pose_head=True,
        window_ms=100., tokenizer_activation="gelu"))
    model = WearUsFM(cfg.model)
    model.tokenizer.fe_gain.fill_(3.125)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.tokenizer.proj.weight.grad = torch.ones_like(model.tokenizer.proj.weight)
    optimizer.step()
    optimizer.state[model.tokenizer.proj.weight]["step"].fill_(5000)
    config = R.config_to_dict(cfg)
    signature = {"manifest_sha256": "d" * 64, "scales_sha256": "e" * 64, "loader": config["loader"]}
    state = {"step": 5000, "config": config, "teacher": copy.deepcopy(model.state_dict()),
             "student": model.state_dict(), "pose_stats_sha256": STATS, "data_signature": signature,
             "world_size": 4, "stream": {"num_workers": 6, "segment": cfg.stream_segment, "world_size": 4},
             "alarm_counts": (0, 0), "optimizer": optimizer.state_dict(),
             "rank_rng": [{"torch_rng": torch.Generator().manual_seed(i).get_state(),
                           "cuda_rng": torch.zeros(16, dtype=torch.uint8)} for i in range(4)]}
    cp, cf, rp, sp = [tmp_path / name for name in ("checkpoint_step0005000.pt", "phase_config.json",
                                                    "training_receipt.json", "phase_summary.json")]
    torch.save(state, cp)
    save_json(cf, config)
    source = {"sha256": "d" * 64, "scientific_source_sha256": {"src/training.py": "e" * 64}}
    old = arm == S.ARMS[0]
    receipt = {"protocol": S.T.PROTOCOL if old else S.TRAIN_PROTOCOL, "complete": True,
        "arm": "T100-G" if old else arm, "checkpoint_step": 5000,
        "checkpoint": str(cp), "checkpoint_sha256": R.sha256_file(cp), "config_path": str(cf),
        "config_file_sha256": R.sha256_file(cf), "code_sha": S.REFERENCE_CODE if old else CODE,
        "training_source_contract_sha256": source["sha256"], "scientific_source_sha256": source["scientific_source_sha256"]}
    save_json(rp, receipt)
    summary = {"stopped": "max_steps", "steps": 5000}
    save_json(sp, summary)
    p = {**pin(cp), **{k + "_path": str(f) for k, f in (("config", cf), ("receipt", rp), ("summary", sp))},
         **{k + "_sha256": R.sha256_file(f) for k, f in (("config", cf), ("receipt", rp), ("summary", sp))}}
    return p, config, signature, state, receipt, summary, source


def load(scene, arm):
    p, config, signature, _, _, _, source = scene
    return S.load_teacher_endpoint(p, config, arm, STATS, signature, source)


@pytest.mark.parametrize("arm", S.ARMS)
def test_real_teacher_restores_gain_and_uses_separate_training_receipt_protocols(tmp_path, arm):
    scene = endpoint(tmp_path, arm)
    before = S.P.file_identity(scene[0]["path"])
    result = load(scene, arm)
    assert result["postload_recipe_passed"] and result["fe_gain"] == 3.125
    assert result["training_code_sha"] == (S.REFERENCE_CODE if arm == S.ARMS[0] else CODE)
    assert S.P.file_identity(scene[0]["path"]) == before


@pytest.mark.parametrize("corruption", ["step", "config", "signature", "stats", "teacher_missing", "teacher_nan",
    "student_nan", "student_layout", "optimizer", "rng", "world", "stream", "alarm", "receipt_arm",
    "receipt_protocol", "incomplete", "summary", "source", "source_hash", "gain", "STOP"])
def test_real_endpoint_rejects_partial_corrupt_or_misattributed_state(tmp_path, corruption):
    arm = S.ARMS[1]
    scene = endpoint(tmp_path, arm)
    p, config, signature, state, receipt, summary, source = scene
    if corruption == "step": state["step"] = 4999
    if corruption == "config": state["config"] = {**config, "max_steps": 19618}
    if corruption == "signature": state["data_signature"] = {"different": True}
    if corruption == "stats": state["pose_stats_sha256"] = "f" * 64
    if corruption == "teacher_missing": state["teacher"].pop("tokenizer.proj.weight")
    if corruption == "teacher_nan": state["teacher"]["tokenizer.proj.weight"][0, 0] = float("nan")
    if corruption == "student_nan": state["student"]["tokenizer.proj.weight"][0, 0] = float("nan")
    if corruption == "student_layout": state["student"]["tokenizer.proj.weight"] = torch.zeros(1)
    if corruption == "optimizer": state["optimizer"] = {}
    if corruption == "rng": state["rank_rng"] = []
    if corruption == "world": state["world_size"] = 8
    if corruption == "stream": state["stream"]["num_workers"] = 1
    if corruption == "alarm": state["alarm_counts"] = [-1, 0]
    if corruption == "receipt_arm": receipt["arm"] = "T100-G"
    if corruption == "receipt_protocol": receipt["protocol"] = S.T.PROTOCOL
    if corruption == "incomplete": receipt["complete"] = False
    if corruption == "summary": summary["stopped"] = "time_limit"
    if corruption == "source": receipt["scientific_source_sha256"] = {}
    if corruption == "source_hash": receipt["training_source_contract_sha256"] = "0" * 64
    if corruption == "gain": state["teacher"]["tokenizer.fe_gain"].fill_(0.)
    if corruption == "STOP": (tmp_path / "STOP").write_text("stop")
    torch.save(state, p["path"])
    p.update(pin(p["path"]))
    receipt["checkpoint_sha256"] = p["sha256"]
    save_json(Path(p["receipt_path"]), receipt)
    p["receipt_sha256"] = R.sha256_file(p["receipt_path"])
    save_json(Path(p["summary_path"]), summary)
    p["summary_sha256"] = R.sha256_file(p["summary_path"])
    with pytest.raises((ValueError, RuntimeError)):
        load(scene, arm)


def test_historical_control_does_not_accept_a_new_training_code_identity(tmp_path):
    scene = endpoint(tmp_path, S.ARMS[0])
    scene[4]["code_sha"] = CODE
    save_json(Path(scene[0]["receipt_path"]), scene[4])
    scene[0]["receipt_sha256"] = R.sha256_file(scene[0]["receipt_path"])
    with pytest.raises(ValueError, match="historical training code"):
        load(scene, S.ARMS[0])


def pose_scene(tmp_path, arm, split="train"):
    actual, reference, data = OLD.pose_scene(tmp_path, split)
    dim = S.dimension(arm)
    data["dim"] = dim
    data["lat"] = np.ones((2, 160, dim), np.float16)
    PE.save_npz(actual, data)
    return actual, reference, data


def pose_check(scene, arm):
    actual, reference, _ = scene
    return S.verify_pose_cache(actual, reference, "train", "/checkpoint.pt", CODE, STATS,
                               {"dim": S.dimension(arm)}, arm)


def test_d384_pose_validator_regresses_exactly_against_historical_guard(tmp_path):
    scene = pose_scene(tmp_path, S.ARMS[0])
    assert pose_check(scene, S.ARMS[0]) == OLD.pose_check(scene)


@pytest.mark.parametrize("arm", S.ARMS)
def test_both_pose_widths_preserve_targets_and_ignore_only_padding(tmp_path, arm):
    scene = pose_scene(tmp_path, arm)
    assert pose_check(scene, arm)["n_windows"] == 2
    scene[2]["lat"][1, 150:] = np.nan
    PE.save_npz(scene[0], scene[2])
    assert pose_check(scene, arm)["n_users"] == 2


@pytest.mark.parametrize("arm", S.ARMS)
@pytest.mark.parametrize("corruption", ["target", "mask", "meta", "width", "precision", "nonfinite_unlabelled", "pred", "teacher"])
def test_pose_both_widths_reject_geometry_support_and_real_time_errors(tmp_path, arm, corruption):
    scene = pose_scene(tmp_path, arm)
    path, _, data = scene
    if corruption == "target": data["target"][0, 0, 0] += 1
    if corruption == "mask": data["mask"][0, 0, 0] = False
    if corruption == "meta": data["meta"][0]["start"] += 10
    if corruption == "width": data["lat"] = data["lat"][..., :-1]
    if corruption == "precision": data["lat"] = data["lat"].astype(np.float32)
    if corruption == "nonfinite_unlabelled": data["lat"][0, 17, 0] = np.nan
    if corruption == "pred": data["pred"][0, 0, 0] = np.inf
    if corruption == "teacher": data["config"]["which"] = "student"
    PE.save_npz(path, data)
    with pytest.raises(ValueError):
        pose_check(scene, arm)


def classification_scene(tmp_path, arm, dataset="epn612"):
    actual, reference, data, splits, dataset = OLD.classification_scene(tmp_path, dataset)
    dim = S.dimension(arm)
    data["encoder_dim"] = np.asarray(dim)
    data["x_pool"] = np.full((len(data["y"]), 2 * dim), 3., np.float32)
    data["x_local"] = np.zeros((len(data["y"]), 8 * dim), np.float16)
    data["x_dec"] = np.zeros_like(data["x_local"])
    np.savez(actual, **data)
    return actual, reference, data, splits, dataset


def class_check(scene, arm):
    actual, ref, _, splits, dataset = scene
    return S.verify_classification_cache(actual, ref, dataset, "/checkpoint.pt", CKPT_SHA, CODE, splits, arm)


@pytest.mark.parametrize("dataset", ("epn612", "uci_emg", "ninapro_db2"))
def test_d384_classification_validator_regresses_exactly_against_historical_guard(tmp_path, dataset):
    scene = classification_scene(tmp_path, S.ARMS[0], dataset)
    assert class_check(scene, S.ARMS[0]) == OLD.class_check(scene)


@pytest.mark.parametrize("dataset", ("epn612", "uci_emg", "ninapro_db2"))
def test_d896_classification_preserves_exact_rows_and_fold_roles_of_d384(tmp_path, dataset):
    small, large = tmp_path / "small", tmp_path / "large"
    small.mkdir(); large.mkdir()
    a = class_check(classification_scene(small, S.ARMS[0], dataset), S.ARMS[0])
    b = class_check(classification_scene(large, S.ARMS[1], dataset), S.ARMS[1])
    assert b["pool_shape"][1] == 1792 and b["pool_dtype"] == "<f4"
    assert a["ordered_rows_sha256"] == b["ordered_rows_sha256"]
    assert a["fold_roles_sha256"] == b["fold_roles_sha256"]


@pytest.mark.parametrize("arm", S.ARMS)
@pytest.mark.parametrize("corruption", ["labels", "input", "nan", "width", "precision", "subject_order", "objects", "nan_labels", "code", "encoder_dim"])
def test_classification_rejects_corruption_without_isfinite_on_text(tmp_path, arm, corruption):
    scene = classification_scene(tmp_path, arm)
    path, _, data, _, _ = scene
    if corruption == "labels": data["y"] = data["y"][::-1]
    if corruption == "input": data["input_sha256"] = np.asarray("1" * 64)
    if corruption == "nan": data["x_pool"][0, 0] = np.nan
    if corruption == "width": data["x_pool"] = data["x_pool"][:, :-1]
    if corruption == "precision": data["x_pool"] = data["x_pool"].astype(np.float16)
    if corruption == "subject_order": data["s"] = data["s"][::-1]
    if corruption == "objects": data["y"] = data["y"].astype(object)
    if corruption == "nan_labels": data["y"] = np.full(len(data["y"]), np.nan)
    if corruption == "code": data["code_sha"] = np.asarray("0" * 40)
    if corruption == "encoder_dim": data["encoder_dim"] = np.asarray(123)
    np.savez(path, **data)
    with pytest.raises(ValueError):
        class_check(scene, arm)


def test_command_builder_is_unchanged_and_supports_both_endpoint_paths(tmp_path):
    plan = S.make_plan({}, {k: {"path": "/" + k} for k in S.INPUTS},
        {"pose_root": "/pose", "processed_root": "/processed", "roots": ["/root1", "/root2"],
         "epn_roots": ["/epn1", "/epn2"], "uci_root": "/uci"}, {}, {}, {})
    for arm in S.ARMS:
        ckpt = "/" + arm + "/checkpoint_step0005000.pt"
        for stage, task in (("extract-pose", None), ("extract-classification", "p1"), ("extract-classification", "p2")):
            new = S.extraction_commands(stage, plan, ckpt, tmp_path, task, "cuda")
            assert new == E.extraction_commands(stage, plan, ckpt, tmp_path, task, "cuda")
            assert all(ckpt in command for command in new.values())
    with pytest.raises(ValueError):
        S.extraction_commands("unknown", plan, ckpt, tmp_path, None, "cuda")


def test_extraction_primitive_bytes_match_historical_baseline_and_mutation_is_rejected(tmp_path):
    states = S.extraction_source_states()
    assert len(states) == 5
    for path, expected in S.EXTRACTION_SOURCE_SHA256.items():
        historical = S.subprocess.check_output(["git", "show", S.EXTRACTION_REFERENCE_CODE + ":" + path], cwd=S.ROOT)
        import hashlib
        assert hashlib.sha256(historical).hexdigest() == expected
        destination = tmp_path / path
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(S.ROOT / path, destination)
    S.extraction_source_states(tmp_path)
    (tmp_path / "scripts/pose_eval.py").write_text("# changed extractor\n")
    with pytest.raises(ValueError, match="SHA"):
        S.extraction_source_states(tmp_path)


@pytest.mark.parametrize("dim", (384, 896))
def test_real_pool_primitive_places_backbone_then_local_at_both_widths(dim):
    from wearusfm.model.fm import Encoded
    from wearusfm.model.spectral_step1 import pool_p1p2
    counts = [2, 1]
    valid = torch.tensor([[True, True, False], [True, True, False], [True, True, False]])
    tv = torch.tensor([[True, True, False], [True, True, False]])
    enc = Encoded(torch.full((3, 3, dim), 71.), torch.full((2, 3, 4, dim), 13.), tv,
                  torch.zeros(3, dim), valid, valid)
    result = pool_p1p2(enc, counts, torch.ones(3, dtype=torch.bool))
    assert result.shape == (2, 2 * dim) and result.dtype == torch.float32
    assert torch.all(result[:, :dim] == 13.) and torch.all(result[:, dim:] == 71.)


@pytest.fixture
def context(tmp_path, monkeypatch):
    """Local source-contract boundary + real reconstructed CLI/full configurations."""
    import scaling_train as ST
    import train_jepa
    frozen = json.loads((S.ROOT / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_text())
    argv = list(frozen["arm_configurations"]["T100-G"]["validated_trainer_argv"])
    inputs = {}
    for name in S.INPUTS:
        value = {"scales": {}} if name == "scales" else {"window_s": {}} if name == "windows" else {}
        inputs[name] = save_json(tmp_path / ("pose_stats.json" if name == "pose_stats" else name + ".json"), value)
    for flag, name in (("--manifest", "manifest"), ("--scales", "scales"), ("--window-seconds", "windows")):
        argv[argv.index(flag) + 1] = inputs[name]["path"]
    argv[argv.index("--pose-root") + 1] = str(tmp_path)
    cfg = train_jepa.build_config(train_jepa.build_parser().parse_args(argv[1:]))
    cf = save_json(tmp_path / "config.json", R.config_to_dict(cfg))
    comparison = save_json(tmp_path / "comparison.json", {"arm_configurations": {"T100-G": {
        "config": cf["path"], "sha256": cf["sha256"], "validated_trainer_argv": argv}}})
    refs = {split: {"path": str(tmp_path / (split + ".npz")), "sha256": "1" * 64, "bytes": 1, **selection}
            for split, selection in S.Q.SELECTION.items()}
    contract = {"comparison": comparison, "inputs": {k: inputs[k] for k in ("manifest", "scales", "windows", "pose_stats")},
                "reference_checkpoint_sha256": CKPT_SHA}
    contract_pin = save_json(tmp_path / "source_contract.json", contract)
    checks = {"contract": contract, "states": {}, "config_30m": cfg,
              "config_300m": S.S.configurations(cfg)[S.ARMS[1]]}
    plan = S.make_plan(comparison, inputs, {"pose_root": str(tmp_path), "processed_root": "/processed",
        "roots": ["/r1", "/r2"], "epn_roots": ["/e1", "/e2"], "uci_root": "/uci"}, refs,
        {ds: {"path": "/" + ds + ".npz", "sha256": "1" * 64} for ds in sum(S.DATASETS.values(), ())},
        contract_pin, {arm: {"path": "/" + arm + "/checkpoint_step0005000.pt", "sha256": CKPT_SHA} for arm in S.ARMS})
    path = tmp_path / "plan.json"
    save_json(path, plan)
    monkeypatch.setattr(ST, "check_source_contract", lambda *args: checks)
    monkeypatch.setattr(S.P, "code_identity", lambda *args: CODE)
    seen = []
    def guard(pin, config, arm, step, stats, signature, source):
        assert config == R.config_to_dict(checks["config_30m" if arm == S.ARMS[0] else "config_300m"])
        S.model_recipe(config["model"], arm)
        assert step == 5000 and source["sha256"] == contract_pin["sha256"]
        assert stats == inputs["pose_stats"]["sha256"]
        seen.append(arm)
        return {"states": {}}
    monkeypatch.setattr(S, "validate_checkpoint", guard)
    return {"plan": plan, "path": path, "checks": checks, "seen": seen, "tmp": tmp_path}


@pytest.mark.parametrize("arm", S.ARMS)
def test_context_binds_both_full_recipes_to_training_source_contract(context, arm):
    plan, states, _, code = S.load_context(context["path"], arm, 5000, CODE)
    assert context["seen"] == [arm] and code == CODE
    assert set(S.INPUTS) <= set(states)
    assert len([k for k in states if k.startswith("extraction_source_")]) == 5


@pytest.mark.parametrize("mutation", ("comparison", "input", "control", "base_cfg", "large_cfg", "dose", "missing_endpoint"))
def test_context_rejects_otherwise_valid_but_unpaired_comparison_or_control(context, mutation):
    plan = context["plan"]
    if mutation == "comparison":
        other = json.loads(Path(plan["comparison"]["path"]).read_text())
        other["different_comparison"] = True
        plan["comparison"] = save_json(context["tmp"] / "other_comparison.json", other)
    if mutation == "input":
        plan["inputs"]["manifest"] = save_json(context["tmp"] / "other_manifest.json", {"different": True})
    if mutation == "control": plan["checkpoints"][S.ARMS[0]]["sha256"] = "3" * 64
    if mutation == "base_cfg": context["checks"]["config_30m"] = replace(context["checks"]["config_30m"], lr=5e-5)
    if mutation == "large_cfg": context["checks"]["config_300m"] = replace(context["checks"]["config_300m"], lr=5e-5)
    if mutation == "dose": plan["step"] = 19618
    if mutation == "missing_endpoint": plan["checkpoints"].pop(S.ARMS[0])
    save_json(context["path"], plan)
    with pytest.raises(ValueError):
        S.load_context(context["path"], S.ARMS[1], 5000, CODE)
    assert not context["seen"]


@pytest.mark.parametrize("corrupt", (False, True))
def test_extraction_receipt_written_last_only_after_both_real_d896_caches_pass(tmp_path, monkeypatch, corrupt):
    import scaling_train as ST
    sources = tmp_path / "sources"
    sources.mkdir()
    arm = S.ARMS[1]
    scenes = {split: pose_scene(sources, arm, split) for split in ("train", "val")}
    input_file = tmp_path / "input.json"
    input_file.write_text('{}')
    checkpoint = tmp_path / "checkpoint_step0005000.pt"
    checkpoint.write_bytes(b"pinned checkpoint")
    plan = S.make_plan({}, {}, {}, {s: scene[1] for s, scene in scenes.items()}, {}, pin(input_file),
                       {arm: {"path": str(checkpoint)}})
    plan_path = tmp_path / "plan.json"
    save_json(plan_path, plan)
    states = {"plan": S.P.file_identity(plan_path), "pose_stats": S.P.file_identity(input_file),
              "checkpoint": S.P.file_identity(checkpoint), "training_source_contract": S.P.file_identity(input_file)}
    for split, (_, _, data) in scenes.items():
        data["config"].update(pose_stats_sha256=states["pose_stats"]["sha256"],
                              checkpoint_pose_stats_sha256=states["pose_stats"]["sha256"], checkpoint=str(checkpoint))
    guard = {"checkpoint_sha256": states["checkpoint"]["sha256"], "model_config": {"dim": 896},
             "postload_recipe_passed": True}
    monkeypatch.setattr(S, "load_context", lambda *args: (plan, states, guard, CODE))
    monkeypatch.setattr(S.P, "code_identity", lambda *args: CODE)
    monkeypatch.setattr(ST, "check_source_contract", lambda *args: {})
    raw = tmp_path / "data.npy"
    np.save(raw, np.zeros((100, 8), np.int16))
    monkeypatch.setattr(E, "pose_sources", lambda plan, split: ([{"i": i} for i in range(2)], E.source_states([raw])))
    monkeypatch.setattr(S, "extraction_commands", lambda *args: {s: [s] for s in ("train", "val")})
    out = tmp_path / "extraction"
    delivered = []
    def extractor(command, **kwargs):
        split = command[0]
        assert not (out / "receipt.json").exists()
        assert kwargs["env"]["WEARUSFM_CODE_SHA"] == CODE
        data = copy.deepcopy(scenes[split][2])
        if split == "val" and corrupt:
            data["target"][0, 0, 0] += 1
        PE.save_npz(out / ("train_subset.npz" if split == "train" else "val.npz"), data)
        delivered.append(split)
    monkeypatch.setattr(S.subprocess, "run", extractor)
    if corrupt:
        with pytest.raises(ValueError, match="target"):
            S.execute("extract-pose", plan_path, arm, 5000, CODE, out, device="cpu")
        assert not (out / "receipt.json").exists()
    else:
        outputs = S.execute("extract-pose", plan_path, arm, 5000, CODE, out, device="cpu")
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["complete"] and receipt["checkpoint_step"] == 5000
        assert set(receipt["outputs"]) == {"train", "val"}
        for name, item in outputs.items():
            prov = json.loads(Path(item["provenance_path"]).read_text())
            assert item["sha256"] == R.sha256_file(item["path"])
            assert prov["model_config"]["dim"] == 896 and prov["arm"] == arm and prov["split"] == name
            assert item["provenance_sha256"] == R.sha256_file(item["provenance_path"])
    assert delivered == ["train", "val"]
