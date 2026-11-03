"""Actual checkpoint loading, immutable provenance and paired extraction oracles."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import tokenizer_evaluation as E
from wearusfm.harness import pose_eval as PE
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training import run as R

CODE, CKPT_SHA, STATS = "a" * 40, "b" * 64, "c" * 64


def pin(path):
    return {k: E.P.file_identity(path)[k] for k in ("path", "sha256", "bytes")}


def save_json(path, data):
    path.write_text(json.dumps(data))
    return pin(path)


@pytest.fixture
def checkpoint(tmp_path):
    cfg = replace(R.sanity_config(), max_steps=5000, model=FMConfig(dim=16, n_heads=4, k_latents=4,
                  local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0., pose_head=True,
                  window_ms=100., tokenizer_activation="gelu"))
    model = WearUsFM(cfg.model)
    # A deliberately saved gain unlike constructor initialization proves restore.
    model.tokenizer.fe_gain.fill_(3.125)
    config = R.config_to_dict(cfg)
    signature = {"manifest_sha256": "d" * 64, "scales_sha256": "e" * 64, "loader": config["loader"]}
    state = {"step": 5000, "config": config, "teacher": model.state_dict(), "pose_stats_sha256": STATS, "data_signature": signature}
    cp = tmp_path / "checkpoint_step0005000.pt"
    torch.save(state, cp)
    cf = tmp_path / "phase_config.json"
    save_json(cf, config)
    rp = tmp_path / "training_receipt.json"
    receipt = {"protocol": "tokenizer-training-20261103-v1", "complete": True, "arm": "T100-G", "checkpoint_step": 5000,
               "checkpoint": str(cp), "checkpoint_sha256": E.R.sha256_file(cp), "config_path": str(cf),
               "config_file_sha256": E.R.sha256_file(cf), "code_sha": CODE}
    save_json(rp, receipt)
    p = {**pin(cp), "config_path": str(cf), "config_sha256": E.R.sha256_file(cf),
         "receipt_path": str(rp), "receipt_sha256": E.R.sha256_file(rp)}
    return p, config, signature, state, receipt


def guard(scene):
    p, config, signature, _, _ = scene
    return E.validate_checkpoint(p, config, "T100-G", 5000, STATS, signature)


def test_actual_gelu_checkpoint_load_restores_saved_gain_and_preserves_source(checkpoint):
    before = E.P.file_identity(checkpoint[0]["path"])
    got = guard(checkpoint)
    assert got["postload_recipe_passed"] is True and got["fe_gain"] == 3.125
    assert got["model_config"]["tokenizer_activation"] == "gelu"
    assert got["model_config"]["window_ms"] == 100.
    assert before == E.P.file_identity(checkpoint[0]["path"])


@pytest.mark.parametrize("corruption", ["step", "config", "signature", "stats", "missing_weight", "nan", "receipt", "sha", "STOP"])
def test_checkpoint_guards_reject_corruption(checkpoint, corruption):
    p, config, signature, state, receipt = checkpoint
    cp, rp = Path(p["path"]), Path(p["receipt_path"])
    if corruption == "step":
        state["step"] = 19618
    elif corruption == "config":
        state["config"] = copy.deepcopy(config)
        state["config"]["model"]["window_ms"] = None
    elif corruption == "signature":
        state["data_signature"] = {"other": True}
    elif corruption == "stats":
        state["pose_stats_sha256"] = "f" * 64
    elif corruption == "missing_weight":
        state["teacher"].pop("tokenizer.proj.weight")
    elif corruption == "nan":
        state["teacher"]["tokenizer.proj.weight"][0, 0] = float("nan")
    elif corruption == "receipt":
        receipt["arm"] = "T100-L"
    elif corruption == "sha":
        p["sha256"] = "0" * 64
    elif corruption == "STOP":
        (cp.parent / "STOP").write_text("stop")
    if corruption in ("step", "config", "signature", "stats", "missing_weight", "nan"):
        torch.save(state, cp)
        p.update(pin(cp))
        receipt["checkpoint_sha256"] = p["sha256"]
    save_json(rp, receipt)
    p["receipt_sha256"] = E.R.sha256_file(rp)
    with pytest.raises((ValueError, RuntimeError)):
        guard(checkpoint)


def test_old_loader_silent_activation_omission_is_rejected(checkpoint, monkeypatch):
    original = E.FF.load_model
    def old_load(*args, **kwargs):
        model = original(*args, **kwargs)
        model.tokenizer.tokenizer_activation = "identity"  # same tensor keys
        return model
    monkeypatch.setattr(E.FF, "load_model", old_load)
    with pytest.raises(ValueError, match="activation/window"):
        guard(checkpoint)


def pose_scene(tmp_path, split="train"):
    rng = np.random.default_rng(7)
    meta = [{"user": "u" + str(i), "session": "s", "stage": "test", "side": "left", "stem": "r" + str(i),
             "recording": i, "start": 0, "fs": 2000., "n_patches": 160 if i == 0 else 150} for i in range(2)]
    target = rng.normal(size=(2, 160, 20)).astype(np.float32)
    mask = np.ones_like(target, dtype=bool)
    mask[1, 150:] = False
    mask[0, 15:20] = False
    data = {"lat": np.zeros((2, 160, 384), np.float16), "pred": np.zeros_like(target), "target": target,
            "mask": mask, "std": np.ones((2, 20), np.float32), "pose_var": np.array([.2, .5], np.float32),
            "meta": meta, "has_head": True, "dim": 384, "n_recordings": 2,
            "config": {"n_windows": 10, "window_s": 4., "seed": 20261026, "pose_stats_sha256": STATS, "per_side": False,
                       "checkpoint_pose_stats_sha256": STATS, "checkpoint": "/checkpoint.pt", "code_sha": CODE,
                       "split": split, "which": "teacher"}}
    reference, actual = tmp_path / (split + "_ref.npz"), tmp_path / (split + "_actual.npz")
    PE.save_npz(reference, data)
    data["lat"][:] = 1  # Different encoder information is expressly permitted.
    PE.save_npz(actual, data)
    return actual, {**pin(reference), "n_windows": 2, "n_users": 2, "n_recordings": 2}, data


def pose_check(scene):
    actual, reference, _ = scene
    return E.verify_pose_cache(actual, reference, "train", "/checkpoint.pt", CODE, STATS, {"dim": 384})


def test_pose_parity_ignores_changed_features_but_preserves_exact_targets(tmp_path):
    scene = pose_scene(tmp_path)
    assert pose_check(scene)["n_windows"] == 2
    actual, _, data = scene
    data["lat"][1, 150:] = np.nan  # Padding is not real input time.
    PE.save_npz(actual, data)
    assert pose_check(scene)["n_users"] == 2


@pytest.mark.parametrize("corruption", ["target", "mask", "metadata", "precision", "lat_nonfinite_missing_target", "pred_nonfinite", "teacher"])
def test_pose_cache_fails_on_identity_or_real_time_corruption(tmp_path, corruption):
    scene = pose_scene(tmp_path)
    actual, _, data = scene
    if corruption == "target":
        data["target"][0, 0, 0] += 1
    elif corruption == "mask":
        data["mask"][0, 0, 0] = False
    elif corruption == "metadata":
        data["meta"][0]["start"] += 50
    elif corruption == "precision":
        data["lat"] = data["lat"].astype(np.float32)
    elif corruption == "lat_nonfinite_missing_target":
        data["lat"][0, 17, 0] = np.nan
    elif corruption == "pred_nonfinite":
        data["pred"][0, 0, 0] = np.inf
    elif corruption == "teacher":
        data["config"]["which"] = "student"
    PE.save_npz(actual, data)
    with pytest.raises(ValueError):
        pose_check(scene)


def classification_scene(tmp_path, dataset="epn612"):
    task = "p1" if dataset.startswith("ninapro") else "p2"
    all_subjects = np.array([f"s{i:02}" for i in range(30)])
    subjects = all_subjects[:10] if task == "p1" else np.asarray(E.C.CP.p2_internal_subjects(all_subjects))
    s = np.repeat(subjects, 2)
    y = np.resize(np.array(["noGesture", "fist"]) if dataset == "epn612" else np.array([1, 2]), len(s))
    splits_path = tmp_path / "splits.json"
    splits = save_json(splits_path, {"datasets": {dataset: {"pretraining": subjects.tolist(), "test": ["external"]}}})
    data = {"x_pool": np.zeros((len(s), 768), np.float32), "x_local": np.zeros((len(s), 8 * 384), np.float16),
            "x_dec": np.zeros((len(s), 8 * 384), np.float16), "y": y, "s": s, "subjects": subjects,
            "all_subjects": all_subjects, "checkpoint": np.asarray("/checkpoint.pt"), "checkpoint_sha256": np.asarray(CKPT_SHA),
            "input_sha256": np.asarray("f" * 64), "splits_sha256": np.asarray(splits["sha256"] if task == "p1" else E.C.CP.P2_SPLITS_SHA256),
            "extraction_recipe": np.asarray(json.dumps(E.C.recipe(task))), "encoder_dim": np.asarray(384), "channels": np.asarray(8),
            "protocol": np.asarray(E.C.PROTOCOL), "calibration_protocol": np.asarray(E.C.CP.PROTOCOL), "task": np.asarray(task),
            "feature_set": np.asarray(E.C.CR.FEATURE_SET), "code_sha": np.asarray(CODE), "montage_signature": np.asarray(json.dumps([[0, i] for i in range(8)]))}
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, **data)
    data["x_pool"][:] = 3
    np.savez(new, **data)
    return new, pin(old), data, splits, dataset


def class_check(scene):
    new, old, _, splits, dataset = scene
    return E.verify_classification_cache(new, old, dataset, "/checkpoint.pt", CKPT_SHA, CODE, splits)


@pytest.mark.parametrize("dataset", ["epn612", "uci_emg", "ninapro_db2"])
def test_classification_real_string_and_numeric_label_contracts(tmp_path, dataset):
    got = class_check(classification_scene(tmp_path, dataset))
    assert got["pool_shape"][1] == 768 and got["pool_dtype"] == "<f4"
    assert len(got["fold_roles_sha256"]) == 64


@pytest.mark.parametrize("corruption", ["labels", "input", "pool_nan", "pool_width", "subject_order", "object_labels", "nan_labels", "code"])
def test_classification_fail_closed_without_calling_isfinite_on_strings(tmp_path, corruption):
    scene = classification_scene(tmp_path)
    new, _, data, _, _ = scene
    if corruption == "labels":
        data["y"] = data["y"][::-1]
    elif corruption == "input":
        data["input_sha256"] = np.asarray("1" * 64)
    elif corruption == "pool_nan":
        data["x_pool"][0, 0] = np.nan
    elif corruption == "pool_width":
        data["x_pool"] = data["x_pool"][:, :-1]
    elif corruption == "subject_order":
        data["s"] = data["s"][::-1]
    elif corruption == "object_labels":
        data["y"] = data["y"].astype(object)
    elif corruption == "nan_labels":
        data["y"] = np.full(len(data["y"]), np.nan)
    elif corruption == "code":
        data["code_sha"] = np.asarray("0" * 40)
    np.savez(new, **data)
    with pytest.raises(ValueError):
        class_check(scene)


def test_command_contract_keeps_teacher_original_selection_and_no_crop(tmp_path):
    plan = E.make_plan({}, {k: {"path": "/" + k} for k in E.INPUTS},
        {"pose_root": "/pose", "processed_root": "/processed", "roots": ["/root1", "/root2"],
         "epn_roots": ["/epn1", "/epn2"], "uci_root": "/uci"}, {}, {})
    pose = E.extraction_commands("extract-pose", plan, "/checkpoint.pt", tmp_path, None, "cuda")
    for split, argv in pose.items():
        args = dict(zip(argv[2::2], argv[3::2]))
        assert args["--which"] == "teacher" and args["--split"] == split
        assert args["--seed"] == "20261026" and args["--window-s"] == "4"
        assert "--tokenizer-activation" not in argv and "--window-ms" not in argv
    assert "--n-users" in pose["train"] and "--n-users" not in pose["val"]
    for task in E.DATASETS:
        argv = E.extraction_commands("extract-classification", plan, "/checkpoint.pt", tmp_path, task, "cuda")[task]
        assert all(dataset in argv for dataset in E.DATASETS[task])
        assert "--crop" not in argv and "--which" not in argv  # Original channel extractor loads teacher itself.


def test_source_guard_detects_modified_data_and_new_sidecar(tmp_path):
    arr, sidecar = tmp_path / "data_int16.npy", tmp_path / "pose_meta.json"
    np.save(arr, np.zeros((100, 8), np.int16))
    before = E.source_states([arr, sidecar])
    assert before[str(sidecar)] is None
    sidecar.write_text('{}')
    assert E.source_states([arr, sidecar]) != before
    sidecar.unlink()
    np.save(arr, np.ones((100, 8), np.int16))
    assert E.source_states([arr, sidecar]) != before


@pytest.mark.parametrize("corrupt", [False, True])
def test_execute_receipt_is_last_and_failed_second_split_cannot_complete(tmp_path, monkeypatch, corrupt):
    sources = tmp_path / "sources"
    sources.mkdir()
    scenes = {split: pose_scene(sources, split) for split in ("train", "val")}
    input_file = tmp_path / "input.json"
    input_file.write_text('{}')
    fake_checkpoint = tmp_path / "checkpoint_step0005000.pt"
    fake_checkpoint.write_bytes(b'pinned checkpoint')
    plan = E.make_plan({}, {}, {}, {s: scene[1] for s, scene in scenes.items()}, {},
                       {"5000": {"T100-G": {"path": str(fake_checkpoint)}}})
    plan_path = tmp_path / "plan.json"
    save_json(plan_path, plan)
    states = {"plan": E.P.file_identity(plan_path), "pose_stats": {**E.P.file_identity(input_file), "sha256": STATS},
              "checkpoint": E.P.file_identity(fake_checkpoint)}
    # Actual immutable state checks are exercised, including source hashes;
    # give the fixture's stats pin its actual digest and adjust embedded cache.
    states["pose_stats"]["sha256"] = E.R.sha256_file(input_file)
    for split, (_, _, data) in scenes.items():
        data["config"]["pose_stats_sha256"] = states["pose_stats"]["sha256"]
        data["config"]["checkpoint_pose_stats_sha256"] = states["pose_stats"]["sha256"]
        data["config"]["checkpoint"] = str(fake_checkpoint)
    guard_info = {"checkpoint_sha256": states["checkpoint"]["sha256"], "model_config": {"dim": 384},
                  "postload_recipe_passed": True}
    monkeypatch.setattr(E, "load_context", lambda *args: (plan, states, guard_info, CODE))
    monkeypatch.setattr(E.P, "code_identity", lambda *args: CODE)
    raw_file = tmp_path / "data_int16.npy"
    np.save(raw_file, np.zeros((100, 8), np.int16))
    def pose_sources(plan, split):
        return [{"i": i} for i in range(2)], E.source_states([raw_file])
    monkeypatch.setattr(E, "pose_sources", pose_sources)
    def commands(stage, plan, checkpoint, out, task, device):
        return {s: [s, str(out / (s + '.npz'))] for s in ('train', 'val')}
    monkeypatch.setattr(E, "extraction_commands", commands)
    out = tmp_path / "extraction"
    delivered = []
    def extractor(command, **kwargs):
        split = command[0]
        assert not (out / 'receipt.json').exists()
        assert kwargs['env']['WEARUSFM_CODE_SHA'] == CODE
        data = copy.deepcopy(scenes[split][2])
        if split == 'val' and corrupt:
            data['target'][0, 0, 0] += 1
        destination = out / ('train_subset.npz' if split == 'train' else 'val.npz')
        PE.save_npz(destination, data)
        delivered.append(split)
    monkeypatch.setattr(E.subprocess, 'run', extractor)
    if corrupt:
        with pytest.raises(ValueError, match='target'):
            E.execute('extract-pose', plan_path, 'T100-G', 5000, CODE, out, device='cpu')
        assert not (out / 'receipt.json').exists()
    else:
        outputs = E.execute('extract-pose', plan_path, 'T100-G', 5000, CODE, out, device='cpu')
        receipt = json.loads((out / 'receipt.json').read_text())
        assert receipt['complete'] is True and receipt['checkpoint_unchanged'] is True
        assert set(receipt['outputs']) == {'train', 'val'}
        for split, item in outputs.items():
            assert item['sha256'] == E.R.sha256_file(item['path'])
            pp = Path(item['provenance_path'])
            assert item['provenance_sha256'] == E.R.sha256_file(pp)
            prov = json.loads(pp.read_text())
            assert prov['which'] == 'teacher' and prov['split'] == split
            assert prov['cache_bytes'] == Path(item['path']).stat().st_size
    assert delivered == ['train', 'val']


def test_load_context_rebuilds_actual_frozen_cli_and_data_signature(tmp_path, monkeypatch):
    import train_jepa
    recorded = json.loads((E.REPO / 'results/infra/tokenizer_review_20261103/comparison_configurations.json').read_text())
    argv = list(recorded['arm_configurations']['T100-G']['validated_trainer_argv'])
    inputs = {}
    for name in E.INPUTS:
        path = tmp_path / ('pose_stats.json' if name == 'pose_stats' else name + '.json')
        value = {'scales': {}} if name == 'scales' else {'window_s': {}} if name == 'windows' else {}
        inputs[name] = save_json(path, value)
    for flag, name in (('--manifest', 'manifest'), ('--scales', 'scales'), ('--window-seconds', 'windows')):
        argv[argv.index(flag) + 1] = inputs[name]['path']
    argv[argv.index('--pose-root') + 1] = str(tmp_path)
    cfg = train_jepa.build_config(train_jepa.build_parser().parse_args(argv[1:]))
    config_path = tmp_path / 'frozen_config.json'
    cp = save_json(config_path, R.config_to_dict(cfg))
    comparison = save_json(tmp_path / 'comparison.json', {'arm_configurations': {'T100-G': {
        'config': cp['path'], 'sha256': cp['sha256'], 'validated_trainer_argv': argv}}})
    refs = {split: {'path': str(tmp_path / (split + '.npz')), 'sha256': '0' * 64, 'bytes': 1, **selection}
            for split, selection in E.Q.SELECTION.items()}
    plan = E.make_plan(comparison, inputs, {'pose_root': str(tmp_path), 'processed_root': '/processed',
        'roots': ['/r1', '/r2'], 'epn_roots': ['/e1', '/e2'], 'uci_root': '/uci'}, refs, {},
        {'5000': {'T100-G': {'path': '/checkpoint_step0005000.pt'}}})
    pp = tmp_path / 'plan.json'
    save_json(pp, plan)
    seen = []
    def checkpoint_guard(pin, expected, arm, step, stats, signature):
        assert expected == R.config_to_dict(cfg) and expected['model']['tokenizer_activation'] == 'gelu'
        assert arm == 'T100-G' and step == 5000 and stats == inputs['pose_stats']['sha256']
        assert signature == E.canonical(R.calibration_signature(cfg, inputs['manifest']['path'], {}, {}))
        seen.append(True)
        return {'states': {}}
    monkeypatch.setattr(E, 'validate_checkpoint', checkpoint_guard)
    monkeypatch.setattr(E.P, 'code_identity', lambda *args: CODE)
    got, states, _, code = E.load_context(pp, 'T100-G', 5000, CODE)
    assert seen == [True] and code == CODE and got['inputs']['manifest']['path'] == inputs['manifest']['path']
    assert set(E.INPUTS) <= set(states)
    Path(inputs['manifest']['path']).write_text('{"changed": true}')
    with pytest.raises(ValueError, match='SHA'):
        E.load_context(pp, 'T100-G', 5000, CODE)
    assert seen == [True]
