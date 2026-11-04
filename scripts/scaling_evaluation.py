"""Attest and extract the frozen 30M/300M teacher endpoints at 5,000 updates.

This is a separate campaign contract, not an override of the tokenizer guards.
The dimensional extractors and pure selection/hash primitives are reused. No
training, probe/ridge fit, job submission, or post-backbone feature is added here.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import tokenizer_evaluation as E
import tokenizer_train as T
import scaling_preflight as S
from wearusfm.harness import fm_features as FF, psd3_inputs as I
from wearusfm.harness.downstream_ab import fingerprint
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "scaling-evaluation-20261104-v1"
TRAIN_PROTOCOL = "scaling-training-20261104-v1"
REFERENCE_CODE = "afb644ea10ecc2a716160b4d8a4904e805c366fb"
# The actual extraction commit of the historical T100-G 5k pose cache. These
# script bytes also equal b315dad, used for the final tokenizer classification.
# All src/wearusfm bytes are additionally pinned by the training source contract.
EXTRACTION_REFERENCE_CODE = "f2061b8bc91a6355213793a9719121135ad48fa2"
EXTRACTION_SOURCE_SHA256 = {
    "scripts/pose_eval.py": "044037f0c351db467775cb03209defb2fd4c4b3bd95c017f247f44933a5d49a9",
    "scripts/channel_readout.py": "746e2ce9d81da27bc089d1ec8b62f270d8b70c9f5c42a2420370f4b7f113c20a",
    "scripts/calibration_probe.py": "197d19d061aa188a5a6c87305e60087f25b346e1966ad949c1a6ab7f8522aac3",
    "scripts/tokenizer_evaluation.py": "00d5f42583bc3203df5041558486647dd05f66f59df8adf3ae7728cf3e6c127b",
    "scripts/tokenizer_pose_readout.py": "18a15436e48045447661d3aaab6a8770819527b4ff97a2837d9be3930bad7fb3",
}
ARMS = ("T100-G-30M", "T100-G-300M")
DIMENSIONS = dict(zip(ARMS, (384, 896)))
STEP = 5000
DATASETS, INPUTS = E.DATASETS, E.INPUTS
P, Q, C = E.P, E.Q, E.C
require, pin_file, canonical = P.require, E.pin_file, E.canonical


def dimension(arm):
    require(arm in ARMS, "unregistered scaling arm")
    return DIMENSIONS[arm]


def extraction_source_states(root=ROOT):
    return {"extraction_source_" + path: pin_file({"path": str(Path(root) / path), "sha256": sha})
            for path, sha in EXTRACTION_SOURCE_SHA256.items()}


def model_recipe(model, arm):
    """Architecture guard is independent of constructing a 292M CPU model."""
    dimension(arm)
    preset = R.LADDER["30m" if arm == ARMS[0] else "300m"]
    expected = {**preset, "k_latents": 64, "patch_ms": 25., "window_ms": 100.,
                "tokenizer_activation": "gelu", "pose_head": True, "grad_checkpoint": True}
    require(all(model.get(k) == v for k, v in expected.items()), "scaling model preset/recipe mismatch")


def make_plan(comparison, inputs, paths, pose_references, classification_references,
              source_contract, checkpoints=None):
    """No placeholder checkpoint is valid: actual immutable hashes come after training.

    Checkpoint pins include config, completion receipt and phase-summary file pins.
    In particular the historical summary is the immutable 5000 phase copy at
    $WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/summary.json,
    not the mutable scratch run summary (which now describes 19618).
    The historical 30M has its original tokenizer receipt; the new 300M has the
    scaling-training receipt, including the exact scientific source contract.
    """
    return copy.deepcopy({"protocol": PROTOCOL, "step": STEP, "arms": list(ARMS),
        "comparison": comparison, "inputs": inputs, "paths": paths,
        "pose_references": pose_references, "classification_references": classification_references,
        "training_source_contract": source_contract, "checkpoints": checkpoints or {}})


def validate_training_state(state, config, stats_sha256, signature):
    """Check endpoint state and structural optimizer/RNG integrity, as production.

    This does not certify a future 19618 resume: the historical optimizer check
    does not bind every moment to the full parameter list. A continuation needs
    its own resume validation; extraction loads and verifies the teacher exactly.
    """
    require(type(state.get("step")) is int and state["step"] == STEP, "checkpoint step mismatch")
    require(state.get("config") == config, "checkpoint full configuration mismatch")
    require(state.get("world_size") == 4 and state.get("stream") ==
            {"num_workers": 6, "segment": config["stream_segment"], "world_size": 4}, "checkpoint stream mismatch")
    require(state.get("pose_stats_sha256") == stats_sha256, "checkpoint pose statistics mismatch")
    require(bool(signature) and state.get("data_signature") == signature, "checkpoint data signature mismatch")
    T.validate_rng(state.get("rank_rng"))
    T.validate_optimizer(state.get("optimizer"), SimpleNamespace(**config), STEP)
    counts = state.get("alarm_counts")
    require(isinstance(counts, (tuple, list)) and len(counts) == 2 and
            all(type(v) is int and v >= 0 for v in counts), "invalid alarm counters")
    for name in ("teacher", "student"):
        weights = state.get(name)
        require(isinstance(weights, dict) and weights and all(isinstance(v, torch.Tensor) and
                bool(torch.isfinite(v).all()) for v in weights.values()), "missing/nonfinite " + name)
    teacher, student = state["teacher"], state["student"]
    require(teacher.keys() == student.keys() and all(teacher[k].shape == student[k].shape and
            teacher[k].dtype == student[k].dtype for k in teacher), "student/teacher layout mismatch")


def load_teacher_endpoint(pin, expected_config, arm, stats_sha256, signature, source_contract):
    """File/state primitive; public campaign validation also enforces the preset.

    The separated loader permits CPU tests with a small *real* checkpoint, while
    the public guard always requires D384 or D896 and the full frozen config.
    """
    dimension(arm)
    path = Path(pin["path"])
    require(path.is_absolute() and path.name == "checkpoint_step0005000.pt", "not an immutable named endpoint")
    require(not (path.parent / "STOP").exists(), "STOP present")
    states = {"checkpoint": pin_file(pin)}
    for name in ("config", "receipt", "summary"):
        states[name] = pin_file({"path": pin[name + "_path"], "sha256": pin[name + "_sha256"]})
    config, receipt, summary = (json.loads(Path(states[name]["path"]).read_text())
                                for name in ("config", "receipt", "summary"))
    require(config == expected_config, "source config differs from frozen full recipe")
    require(summary.get("stopped") == "max_steps" and type(summary.get("steps")) is int and
            summary["steps"] == STEP, "training phase incomplete")
    old = arm == ARMS[0]
    require(receipt.get("protocol") == (T.PROTOCOL if old else TRAIN_PROTOCOL) and
            receipt.get("complete") is True, "incomplete/wrong training receipt")
    for key, value in {"arm": "T100-G" if old else arm, "checkpoint_step": STEP,
                       "checkpoint": str(path), "checkpoint_sha256": pin["sha256"],
                       "config_path": states["config"]["path"],
                       "config_file_sha256": pin["config_sha256"]}.items():
        require(receipt.get(key) == value, "training receipt mismatch: " + key)
    require(re.fullmatch(r"[0-9a-f]{40}", receipt.get("code_sha", "")) is not None,
            "missing training code identity")
    if old:
        require(receipt["code_sha"] == REFERENCE_CODE, "historical training code mismatch")
    else:
        require(receipt.get("training_source_contract_sha256") == source_contract["sha256"] and
                receipt.get("scientific_source_sha256") == source_contract["scientific_source_sha256"],
                "scientific training source identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_training_state(state, config, stats_sha256, signature)
    teacher = state["teacher"]
    model = FF.load_model(str(path), device="cpu", which="teacher")  # strict state load
    require(canonical(asdict(model.cfg)) == config["model"], "loaded model configuration differs")
    require(model.cfg.tokenizer_activation == model.tokenizer.tokenizer_activation == "gelu" and
            model.cfg.window_ms == 100. and model.tokenizer.window_s == .1,
            "loaded tokenizer activation/window differs")
    loaded = model.state_dict()
    require(loaded.keys() == teacher.keys() and all(v.dtype == teacher[k].dtype and
            torch.equal(v, teacher[k]) for k, v in loaded.items()), "teacher state not restored exactly")
    require(torch.equal(model.tokenizer.fe_gain, teacher["tokenizer.fe_gain"]), "checkpoint gain not restored")
    gain = float(model.tokenizer.fe_gain)
    require(np.isfinite(gain) and gain > 0, "invalid checkpoint gain")
    model_config = canonical(asdict(model.cfg))
    del loaded, model, state, teacher
    gc.collect()
    P.check_unchanged(states)
    require(not (path.parent / "STOP").exists(), "STOP appeared during validation")
    return {"states": states, "model_config": model_config, "fe_gain": gain,
            "checkpoint_sha256": pin["sha256"], "checkpoint_step": STEP,
            "training_code_sha": receipt["code_sha"], "postload_recipe_passed": True,
            "gain_policy": "exact saved buffer; no evaluation-data calibration"}


def validate_checkpoint(pin, expected_config, arm, step, stats_sha256, data_signature, source_contract):
    require(type(step) is int and step == STEP, "unregistered scaling dose")
    model_recipe(expected_config["model"], arm)
    require(expected_config["max_steps"] == STEP, "wrong training endpoint configuration")
    return load_teacher_endpoint(pin, expected_config, arm, stats_sha256, data_signature, source_contract)


def verify_pose_cache(path, reference, split, checkpoint, code_sha, stats_sha256, model_config, arm):
    """Identical target/support checks to the old D384 guard; explicit scaling width."""
    dim = dimension(arm)
    require(model_config.get("dim") == dim, "pose model configuration dimension mismatch")
    require(split in ("train", "val"), "wrong pose split")
    ref_state = pin_file(reference)
    old, new = Q.read_targets(reference["path"]), Q.read_targets(path)
    Q.check_selection(old, reference)
    users = Q.check_selection(new, reference)
    require(Q.target_signature(new) == Q.target_signature(old), "pose metadata/target/mask/std/pose_var differs")
    info = I.inspect_cache(path)
    n, t, joints = new["target"].shape
    require((t, joints) == (160, 20), "pose geometry differs")
    for key, shape, dtype in (("lat", [n, t, dim], "<f2"), ("pred", [n, t, 20], "<f4")):
        require(info["arrays"][key]["shape"] == shape and info["arrays"][key]["dtype"] == dtype,
                "pose cache shape/precision differs: " + key)
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(str(z["config_json"]))
        require(int(z["dim"]) == dim and bool(z["has_head"]), "pose model dimension/head mismatch")
        for key, value in {"checkpoint": checkpoint, "which": "teacher", "split": split, "code_sha": code_sha,
                           "pose_stats_sha256": stats_sha256, "checkpoint_pose_stats_sha256": stats_sha256,
                           "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}.items():
            require(config.get(key) == value, "pose extraction recipe differs: " + key)
        real = np.arange(t)[None, :] < new["lengths"][:, None]
        for key in ("lat", "pred"):
            values = z[key]
            require(np.isfinite(values[real]).all(), "nonfinite real input-time pose " + key)
            del values
    require(P.file_identity(reference["path"]) == ref_state, "historical pose reference changed")
    return {"split": split, "config": config, "model_config": model_config, "stats_sha256": stats_sha256,
            "n_windows": n, "n_users": len(users), "n_recordings": new["n_recordings"],
            "targets": Q.target_signature(new), "reference_cache_sha256": reference["sha256"]}


def verify_classification_cache(path, reference, dataset, checkpoint, checkpoint_sha256, code_sha, splits, arm):
    """Preserve exact rows, text/numeric labels, subjects and deterministic folds."""
    dim = dimension(arm)
    ref_state = pin_file(reference)
    task = "p1" if dataset in DATASETS["p1"] else "p2"
    require(dataset in DATASETS[task], "unregistered dataset")
    info, old_info = I.inspect_cache(path), I.inspect_cache(reference["path"])
    meta, old_meta = info["metadata"], old_info["metadata"]
    for key in ("input_sha256", "splits_sha256", "extraction_recipe", "montage_signature", "channels", "task"):
        require(meta.get(key) == old_meta.get(key) and key in meta, "classification input/recipe mismatch: " + key)
    for key, value in {"checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256, "code_sha": code_sha,
                       "extraction_recipe": C.recipe(task), "encoder_dim": dim, "task": task,
                       "protocol": C.PROTOCOL, "feature_set": C.CR.FEATURE_SET, "calibration_protocol": C.CP.PROTOCOL}.items():
        require(meta.get(key) == value, "classification metadata mismatch: " + key)
    expected_splits = splits["sha256"] if task == "p1" else C.CP.P2_SPLITS_SHA256
    require(meta["splits_sha256"] == expected_splits, "classification split identity mismatch")
    with np.load(path, allow_pickle=False) as z, np.load(reference["path"], allow_pickle=False) as old:
        arrays = {k: z[k] for k in ("y", "s", "subjects")}
        for key, values in arrays.items():
            I._labels(values, key)
            require(values.dtype == old[key].dtype and np.array_equal(values, old[key]),
                    "classification rows/labels/subjects differ: " + key)
        y, s, subjects = (arrays[k] for k in ("y", "s", "subjects"))
        require(s.dtype.kind in "US" and subjects.dtype.kind in "US" and len(y) == len(s) and
                sorted(set(s.tolist())) == subjects.tolist(), "classification subject coverage/order mismatch")
        if task == "p1":
            require(subjects.tolist() == C.CP.pretraining_subjects(json.loads(Path(splits["path"]).read_text()), dataset),
                    "official test subjects leaked")
        else:
            all_subjects = z["all_subjects"]
            I._labels(all_subjects, "all_subjects")
            require(all_subjects.dtype == old["all_subjects"].dtype and np.array_equal(all_subjects, old["all_subjects"]) and
                    subjects.tolist() == C.CP.p2_internal_subjects(all_subjects.tolist()), "P2 internal subject selection differs")
        require(info["arrays"]["x_pool"]["shape"] == [len(y), 2 * dim] and
                info["arrays"]["x_pool"]["dtype"] == "<f4", "pool shape/precision differs")
        require(np.isfinite(z["x_pool"]).all(), "nonfinite pool features")
        roles = C.CP.fold_roles(subjects.tolist())
        I.validate_roles(roles, subjects)
        rows_sha = fingerprint(np.arange(len(y), dtype=np.int64), y, s)
    require(P.file_identity(reference["path"]) == ref_state, "historical classification reference changed")
    return {"dataset": dataset, "task": task, "n_windows": len(y), "n_users": len(subjects),
            **{k: meta[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")},
            "ordered_rows_sha256": rows_sha, "fold_roles_sha256": fingerprint(roles),
            "reference_cache_sha256": reference["sha256"], "pool_shape": [len(y), 2 * dim], "pool_dtype": "<f4"}


def extraction_commands(stage, plan, checkpoint, out, task, device):
    require(stage in ("extract-pose", "extract-classification"), "wrong extraction stage")
    # These primitive command builders have no arm/dose/width assumption. Calling
    # tokenizer_evaluation.execute or its campaign load_context would be wrong.
    return E.extraction_commands(stage, plan, checkpoint, Path(out), task, device)


def load_context(plan_path, arm, step, expected_code_sha):
    import scaling_train as ST

    code = P.code_identity(expected_code_sha)
    states = {"plan": P.file_identity(plan_path)}
    plan = json.loads(Path(plan_path).read_text())
    dimension(arm)
    require(plan.get("protocol") == PROTOCOL and plan.get("arms") == list(ARMS) and
            type(step) is int and step == plan.get("step") == STEP, "wrong scaling extraction protocol/dose")
    require(set(plan["checkpoints"]) == set(ARMS), "both actual endpoint pins required")
    require(set(plan["inputs"]) == INPUTS and set(plan["pose_references"]) == {"train", "val"}, "missing input/reference pins")
    require(set(plan["classification_references"]) == set(sum(DATASETS.values(), ())), "missing classification references")
    for split, selection in Q.SELECTION.items():
        require(all(plan["pose_references"][split].get(k) == v for k, v in selection.items()), "historical pose selection differs")
    for key in ("pose_root", "processed_root", "uci_root"):
        require(Path(plan["paths"][key]).is_absolute(), "relative source root: " + key)
    require(all(Path(p).is_absolute() for key in ("roots", "epn_roots") for p in plan["paths"][key]), "relative dataset root")
    states["comparison"] = pin_file(plan["comparison"])
    states.update({key: pin_file(pin) for key, pin in plan["inputs"].items()})
    for key in INPUTS:
        plan["inputs"][key]["path"] = states[key]["path"]
    states["training_source_contract"] = pin_file(plan["training_source_contract"])
    checked = ST.check_source_contract(states["training_source_contract"]["path"])
    contract = {**checked["contract"], "sha256": states["training_source_contract"]["sha256"]}
    states.update({"scientific_source_" + key: value for key, value in checked["states"].items()})
    states.update(extraction_source_states())
    require(states["comparison"]["sha256"] == contract["comparison"]["sha256"], "comparison differs from training source contract")
    for key in ("manifest", "scales", "windows", "pose_stats"):
        require(states[key]["sha256"] == contract["inputs"][key]["sha256"] and
                states[key]["path"] == contract["inputs"][key]["path"], "input differs from training source contract: " + key)
    require(plan["checkpoints"][ARMS[0]]["sha256"] == contract["reference_checkpoint_sha256"], "wrong historical control checkpoint")
    comparison = json.loads(Path(states["comparison"]["path"]).read_text())
    entry = comparison["arm_configurations"]["T100-G"]
    states["expected_config"] = pin_file({"path": entry["config"], "sha256": entry["sha256"]})
    spec = importlib.util.spec_from_file_location("scaling_eval_trainer_cli", ROOT / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    argv = entry["validated_trainer_argv"]
    require(argv[0] == "scripts/train_jepa.py", "wrong trainer executable")
    args = cli.build_parser().parse_args(argv[1:])
    base = cli.build_config(args)
    require(R.config_to_dict(base) == json.loads(Path(states["expected_config"]["path"]).read_text()), "frozen CLI/config mismatch")
    require(R.config_to_dict(base) == R.config_to_dict(checked["config_30m"]), "base configuration differs from training source contract")
    for arg, key in (("manifest", "manifest"), ("scales", "scales"), ("window_seconds", "windows")):
        require(str(Path(getattr(args, arg)).resolve()) == states[key]["path"], "training input differs: " + key)
    cfg = replace(S.configurations(base)[arm], max_steps=STEP)
    require(R.config_to_dict(cfg) == R.config_to_dict(checked["config_30m" if arm == ARMS[0] else "config_300m"]),
            "effective configuration differs from training source contract")
    actual_loader, historical = canonical(asdict(cfg.loader)), canonical(asdict(R.sanity_config().loader))
    require(all(actual_loader[k] == historical[k] for k in E.PREPROCESSING), "pose extractor preprocessing differs")
    require(str(Path(plan["paths"]["pose_root"]) / "pose_stats.json") == states["pose_stats"]["path"] ==
            str(Path(cfg.loader.pose_root) / "pose_stats.json"), "pose root/statistics mismatch")
    scales = json.loads(Path(states["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(states["windows"]["path"]).read_text())["window_s"]
    signature = canonical(R.calibration_signature(cfg, states["manifest"]["path"], scales, windows))
    guard = validate_checkpoint(plan["checkpoints"][arm], R.config_to_dict(cfg), arm, step,
                                states["pose_stats"]["sha256"], signature, contract)
    states.update(guard["states"])
    return plan, states, guard, code


def execute(stage, plan_path, arm, step, expected_code_sha, out, task=None, device="cuda"):
    started = time.perf_counter()
    require(stage in ("validate", "extract-pose", "extract-classification"), "wrong extraction stage")
    require(stage != "extract-classification" or task in DATASETS, "classification task required")
    plan, states, guard, code = load_context(plan_path, arm, step, expected_code_sha)
    checkpoint = plan["checkpoints"][arm]["path"]
    out = Path(out)
    require(out.is_absolute() and not out.exists() and not out.is_symlink() and out.resolve() == out, "output must be new and absolute")
    require(not any(out.is_relative_to(p) for p in (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"),
            Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))), "output inside code")
    common = {"protocol": PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": step,
              "checkpoint_sha256": guard["checkpoint_sha256"], "code_sha": code, "plan_sha256": states["plan"]["sha256"],
              "postload_recipe_passed": True, "model_config": guard["model_config"], "stats_sha256": states["pose_stats"]["sha256"],
              "training_source_contract_sha256": states["training_source_contract"]["sha256"]}
    out.mkdir(parents=True, exist_ok=False)
    outputs, commands, raw = {}, {}, {}
    if stage != "validate":
        commands = extraction_commands(stage, plan, checkpoint, out, task, device)
        for key, command in commands.items():
            if stage == "extract-pose":
                records, raw_before = E.pose_sources(plan, key)
                reference = plan["pose_references"][key]
                Q.validate_reference(reference)
                require(len(records) == reference["n_recordings"], "selected recording count differs")
            else:
                raw_before = E.classification_sources(plan, task)
                for dataset in DATASETS[task]:
                    states["reference_" + dataset] = pin_file(plan["classification_references"][dataset])
            with (out / (key + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env={**os.environ, "WEARUSFM_CODE_SHA": code},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            require(raw_before == E.source_states([Path(p) for p in raw_before]), "raw source changed during extraction")
            if stage == "extract-pose":
                records_after, _ = E.pose_sources(plan, key)
                require(records_after == records, "selected record list changed during extraction")
                paths = {key: out / ("train_subset.npz" if key == "train" else "val.npz")}
            else:
                require(E.classification_sources(plan, task) == raw_before, "classification source inventory changed")
                paths = {dataset: out / (dataset + ".npz") for dataset in DATASETS[task]}
            raw[key] = {"file_states": raw_before, "unchanged": True}
            for name, path in paths.items():
                before = P.file_identity(path)
                details = verify_pose_cache(path, plan["pose_references"][name], name, checkpoint, code,
                    states["pose_stats"]["sha256"], guard["model_config"], arm) if stage == "extract-pose" else \
                    verify_classification_cache(path, plan["classification_references"][name], name, checkpoint,
                    guard["checkpoint_sha256"], code, states["splits"], arm)
                require(P.file_identity(path) == before, "output cache changed during validation")
                provenance = {**common, **details, "cache": str(path), "cache_sha256": before["sha256"], "cache_bytes": before["bytes"]}
                pp = Path(str(path) + ".provenance.json")
                P.write_json(pp, provenance)
                outputs[name] = {"path": str(path), "sha256": before["sha256"], "bytes": before["bytes"],
                                 "provenance_path": str(pp), "provenance_sha256": R.sha256_file(pp)}
    P.check_unchanged(states)
    # Revalidate the scientific source bytes; the code identity covers wrappers.
    import scaling_train as ST
    ST.check_source_contract(states["training_source_contract"]["path"])
    require(P.code_identity(code) == code and not (Path(checkpoint).parent / "STOP").exists(), "code/STOP changed during extraction")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
               "device": device, "torch_threads": torch.get_num_threads(),
               "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    P.write_json(out / "receipt.json", {**common, "complete": True, "stage": stage, "checkpoint_unchanged": True,
        "outputs": outputs, "checkpoint_guard": guard, "source_states": states, "raw_inputs": raw,
        "commands": commands, "runtime": runtime, "elapsed_seconds": time.perf_counter() - started})
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "extract-pose", "extract-classification"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--step", type=int, choices=(STEP,), required=True)
    parser.add_argument("--task", choices=tuple(DATASETS))
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    execute(args.stage, args.plan, args.arm, args.step, args.expected_code_sha, args.out, args.task, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
