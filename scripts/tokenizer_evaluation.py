"""Attest tokenizer checkpoints and run unchanged teacher extraction commands.

No training, metric selection, GPU submission or calibration on evaluation data.
Receipts are written last; a partial output directory is never a valid artifact.
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
import subprocess
import sys
import time

import numpy as np
import torch

from wearusfm.data import pose_targets as PT
from wearusfm.harness import fm_features as FF, pose_eval as PE, psd3_inputs as I
from wearusfm.harness.downstream_ab import fingerprint
from wearusfm.training import run as R

import channel_readout as C
import tokenizer_pose_readout as Q
import tokenizer_preflight as P

PROTOCOL = "tokenizer-evaluation-20261103-v1"
ARMS = P.ARMS
REPO = Path(__file__).resolve().parents[1]
DATASETS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
INPUTS = {"csv", "manifest", "scales", "windows", "pose_stats", "splits"}
PREPROCESSING = ("patch_ms", "min_window_s", "max_window_s", "align_ms", "run_margin_s", "split_at_gaps",
                 "filter_band_hz", "filter_margin_s", "notch_hz", "scale_chunk_s", "scale_chunks", "k_neighbors")
require = P.require


def make_plan(comparison, inputs, paths, pose_references, classification_references, checkpoints=None):
    """Pure constructor: reference/input pins precede training; checkpoint pins follow it.

    ``checkpoints[str(step)][arm]`` contains path/sha256, config_path/config_sha256,
    receipt_path/receipt_sha256. References contain path/sha256/bytes and, for
    pose, n_windows/n_users/n_recordings. All runtime paths must be absolute.
    """
    return copy.deepcopy({"protocol": PROTOCOL, "comparison": comparison, "inputs": inputs, "paths": paths,
                          "pose_references": pose_references, "classification_references": classification_references,
                          "checkpoints": checkpoints or {"5000": {}, "19618": {}}})


def canonical(value):
    return json.loads(json.dumps(value))


def pin_file(pin):
    identity = P.pinned_file(pin)
    require("bytes" not in pin or identity["bytes"] == pin["bytes"], "pinned byte count mismatch")
    return identity


def validate_checkpoint(pin, expected_config, arm, step, stats_sha256, data_signature):
    """Strictly load the teacher on CPU, including actual GELU and stored gain.

    The normal constructor initializes a synthetic gain, then strict state loading
    replaces it with the saved buffer. No evaluation signal calibrates this gain.
    This function never mutates the checkpoint or adjusts its configuration.
    """
    require(arm in ARMS and type(step) is int and step in (5000, 19618), "unregistered arm/step")
    path = Path(pin["path"])
    require(path.is_absolute() and path.name == f"checkpoint_step{step:07d}.pt", "not an immutable named endpoint")
    require(not (path.parent / "STOP").exists(), "STOP present")
    states = {"checkpoint": pin_file(pin),
              "config": pin_file({"path": pin["config_path"], "sha256": pin["config_sha256"]}),
              "receipt": pin_file({"path": pin["receipt_path"], "sha256": pin["receipt_sha256"]})}
    config = json.loads(Path(states["config"]["path"]).read_text())
    receipt = json.loads(Path(states["receipt"]["path"]).read_text())
    require(config == expected_config, "source config differs from frozen full recipe")
    require(receipt.get("protocol") == "tokenizer-training-20261103-v1" and receipt.get("complete") is True,
            "incomplete training receipt")
    for key, value in {"arm": arm, "checkpoint_step": step, "checkpoint": str(path),
                       "checkpoint_sha256": pin["sha256"], "config_file_sha256": pin["config_sha256"],
                       "config_path": states["config"]["path"]}.items():
        require(receipt.get(key) == value, "training receipt mismatch: " + key)
    require(isinstance(receipt.get("code_sha"), str) and len(receipt["code_sha"]) == 40, "missing training code identity")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(type(state.get("step")) is int and state["step"] == step, "checkpoint step mismatch")
    require(state.get("config") == config, "checkpoint full configuration mismatch")
    require(state.get("pose_stats_sha256") == stats_sha256, "checkpoint pose stats mismatch")
    require(state.get("data_signature") == data_signature, "checkpoint source data signature mismatch")
    teacher = state.get("teacher")
    require(isinstance(teacher, dict) and teacher, "missing teacher")
    require(all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in teacher.values()), "nonfinite teacher state")
    # FF.load_model itself performs strict load_state_dict. Comparing actual
    # config is additional protection: GELU contributes no state_dict key.
    model = FF.load_model(str(path), device="cpu", which="teacher")
    require(canonical(asdict(model.cfg)) == config["model"], "loaded model configuration differs")
    activation = "gelu" if arm == "T100-G" else "identity"
    window = None if arm == "T25-L" else 100.
    require(model.cfg.tokenizer_activation == model.tokenizer.tokenizer_activation == activation and
            model.cfg.window_ms == window and model.tokenizer.window_s == (.025 if window is None else .1),
            "loaded tokenizer activation/window differs")
    require(model.state_dict().keys() == teacher.keys() and
            all(v.dtype == teacher[k].dtype and torch.equal(v, teacher[k]) for k, v in model.state_dict().items()),
            "teacher state not restored exactly")
    require(torch.equal(model.tokenizer.fe_gain, teacher["tokenizer.fe_gain"]), "checkpoint gain not restored")
    gain = float(model.tokenizer.fe_gain)
    require(np.isfinite(gain) and gain > 0, "invalid checkpoint gain")
    model_config = canonical(asdict(model.cfg))
    del model, state, teacher
    gc.collect()
    P.check_unchanged(states)
    require(not (path.parent / "STOP").exists(), "STOP appeared during validation")
    return {"states": states, "model_config": model_config, "fe_gain": gain,
            "checkpoint_sha256": pin["sha256"], "checkpoint_step": step,
            "training_code_sha": receipt["code_sha"], "postload_recipe_passed": True,
            "gain_policy": "standard constructor initialization overwritten by exact checkpoint buffer; no data calibration"}


def load_context(plan_path, arm, step, expected_code_sha):
    code = P.code_identity(expected_code_sha)
    states = {"plan": P.file_identity(plan_path)}
    plan = json.loads(Path(plan_path).read_text())
    require(plan.get("protocol") == PROTOCOL and arm in ARMS and step in (5000, 19618), "wrong extraction protocol/arm/dose")
    require(set(plan["inputs"]) == INPUTS, "missing input pins")
    require(set(plan["pose_references"]) == {"train", "val"}, "missing pose references")
    for split, selection in Q.SELECTION.items():
        require(all(plan["pose_references"][split].get(k) == v for k, v in selection.items()), "historical pose selection differs")
    for key in ("pose_root", "processed_root", "uci_root"):
        require(Path(plan["paths"][key]).is_absolute(), "relative source root: " + key)
    require(all(Path(p).is_absolute() for key in ("roots", "epn_roots") for p in plan["paths"][key]), "relative dataset root")
    states["comparison"] = pin_file(plan["comparison"])
    states.update({key: pin_file(pin) for key, pin in plan["inputs"].items()})
    # Repo-relative split pins are permitted, but subprocesses receive resolved
    # paths and cannot accidentally read a file from another working directory.
    for key in INPUTS:
        plan["inputs"][key]["path"] = states[key]["path"]
    comparison = json.loads(Path(states["comparison"]["path"]).read_text())
    entry = comparison["arm_configurations"][arm]
    states["expected_config"] = pin_file({"path": entry["config"], "sha256": entry["sha256"]})
    spec = importlib.util.spec_from_file_location("tokenizer_eval_trainer_cli", REPO / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    argv = entry["validated_trainer_argv"]
    require(argv[0] == "scripts/train_jepa.py", "wrong trainer executable")
    cfg = cli.build_config(cli.build_parser().parse_args(argv[1:]))
    require(R.config_to_dict(cfg) == json.loads(Path(states["expected_config"]["path"]).read_text()), "frozen CLI/config mismatch")
    cfg = replace(cfg, max_steps=step)
    require(cfg.model.dim == 384 and cfg.model.k_latents == 64 and cfg.model.patch_ms == 25 and cfg.model.pose_head and
            cfg.jepa.jepa_weight == 0 and cfg.jepa.masked_loss_radius == 0, "not the registered tokenizer campaign")
    actual_loader, historical = canonical(asdict(cfg.loader)), canonical(asdict(R.sanity_config().loader))
    require(all(actual_loader[k] == historical[k] for k in PREPROCESSING), "pose extractor preprocessing differs from training")
    require(str(Path(plan["paths"]["pose_root"]) / "pose_stats.json") == states["pose_stats"]["path"] ==
            str(Path(cfg.loader.pose_root) / "pose_stats.json"), "pose root/statistics mismatch")
    scales = json.loads(Path(states["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(states["windows"]["path"]).read_text())["window_s"]
    signature = canonical(R.calibration_signature(cfg, states["manifest"]["path"], scales, windows))
    pin = plan["checkpoints"][str(step)][arm]
    guard = validate_checkpoint(pin, R.config_to_dict(cfg), arm, step, states["pose_stats"]["sha256"], signature)
    states.update(guard["states"])
    return plan, states, guard, code


def verify_pose_cache(path, reference, split, checkpoint, code_sha, stats_sha256, model_config):
    """Metadata/targets exactly paired; features may differ but must be finite on real input time."""
    ref_state = pin_file(reference)
    old, new = Q.read_targets(reference["path"]), Q.read_targets(path)
    Q.check_selection(old, reference)
    users = Q.check_selection(new, reference)
    require(Q.target_signature(new) == Q.target_signature(old), "pose metadata/target/mask/std/pose_var differs")
    info = I.inspect_cache(path)
    n, t, joints = new["target"].shape
    require((t, joints) == (160, 20), "pose geometry differs")
    for key, shape, dtype in (("lat", [n, t, 384], "<f2"), ("pred", [n, t, 20], "<f4")):
        require(info["arrays"][key]["shape"] == shape and info["arrays"][key]["dtype"] == dtype, "pose cache shape/precision differs: " + key)
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(str(z["config_json"]))
        require(int(z["dim"]) == 384 and bool(z["has_head"]), "pose model dimension/head mismatch")
        for key, value in {"checkpoint": checkpoint, "which": "teacher", "split": split, "code_sha": code_sha,
                           "pose_stats_sha256": stats_sha256, "checkpoint_pose_stats_sha256": stats_sha256,
                           "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}.items():
            require(config.get(key) == value, "pose extraction recipe differs: " + key)
        real = np.arange(t)[None, :] < new["lengths"][:, None]
        # Read one array at a time, not both val latents and predictions together.
        for key in ("lat", "pred"):
            values = z[key]
            require(np.isfinite(values[real]).all(), "nonfinite real input-time pose " + key)
            del values
    require(P.file_identity(reference["path"]) == ref_state, "historical pose reference changed")
    return {"split": split, "config": config, "model_config": model_config, "stats_sha256": stats_sha256,
            "n_windows": n, "n_users": len(users), "n_recordings": new["n_recordings"],
            "targets": Q.target_signature(new), "reference_cache_sha256": reference["sha256"]}


def verify_classification_cache(path, reference, dataset, checkpoint, checkpoint_sha256, code_sha, splits):
    """Check original row order, labels (including text), subject roles and pool768."""
    ref_state = pin_file(reference)
    task = "p1" if dataset in DATASETS["p1"] else "p2"
    require(dataset in DATASETS[task], "unregistered dataset")
    info, old_info = I.inspect_cache(path), I.inspect_cache(reference["path"])
    meta, old_meta = info["metadata"], old_info["metadata"]
    for key in ("input_sha256", "splits_sha256", "extraction_recipe", "montage_signature", "channels", "task"):
        require(meta.get(key) == old_meta.get(key) and key in meta, "classification input/recipe mismatch: " + key)
    for key, value in {"checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256, "code_sha": code_sha,
                       "extraction_recipe": C.recipe(task), "encoder_dim": 384, "task": task,
                       "protocol": C.PROTOCOL, "feature_set": C.CR.FEATURE_SET, "calibration_protocol": C.CP.PROTOCOL}.items():
        require(meta.get(key) == value, "classification metadata mismatch: " + key)
    expected_splits = splits["sha256"] if task == "p1" else C.CP.P2_SPLITS_SHA256
    require(meta["splits_sha256"] == expected_splits, "classification split identity mismatch")
    with np.load(path, allow_pickle=False) as z, np.load(reference["path"], allow_pickle=False) as old:
        arrays = {k: z[k] for k in ("y", "s", "subjects")}
        for key, values in arrays.items():
            I._labels(values, key)
            require(values.dtype == old[key].dtype and np.array_equal(values, old[key]), "classification rows/labels/subjects differ: " + key)
        y, s, subjects = (arrays[k] for k in ("y", "s", "subjects"))
        require(s.dtype.kind in "US" and subjects.dtype.kind in "US" and len(y) == len(s) and
                sorted(set(s.tolist())) == subjects.tolist(), "classification subject coverage/order mismatch")
        if task == "p1":
            require(subjects.tolist() == C.CP.pretraining_subjects(json.loads(Path(splits["path"]).read_text()), dataset), "official test subjects leaked")
        else:
            all_subjects = z["all_subjects"]
            I._labels(all_subjects, "all_subjects")
            require(np.array_equal(all_subjects, old["all_subjects"]) and
                    subjects.tolist() == C.CP.p2_internal_subjects(all_subjects.tolist()), "P2 internal subject selection differs")
        require(info["arrays"]["x_pool"]["shape"] == [len(y), 768] and info["arrays"]["x_pool"]["dtype"] == "<f4", "pool shape/precision differs")
        x = z["x_pool"]
        require(np.isfinite(x).all(), "nonfinite pool features")
        roles = C.CP.fold_roles(subjects.tolist())
        I.validate_roles(roles, subjects)
        rows_sha = fingerprint(np.arange(len(y), dtype=np.int64), y, s)
    require(P.file_identity(reference["path"]) == ref_state, "historical classification reference changed")
    return {"dataset": dataset, "task": task, "n_windows": len(y), "n_users": len(subjects),
            **{k: meta[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")},
            "ordered_rows_sha256": rows_sha, "fold_roles_sha256": fingerprint(roles),
            "reference_cache_sha256": reference["sha256"], "pool_shape": [len(y), 768], "pool_dtype": "<f4"}


def pose_sources(plan, split):
    """Stat every selected source before extraction; missing sidecars remain missing."""
    recs = PE.recordings_for(Path(plan["inputs"]["csv"]["path"]), Path(plan["paths"]["processed_root"]), split,
                             n_users=40 if split == "train" else None,
                             n_recordings=200 if split == "train" else None, seed=20261026)
    paths = []
    for rec in recs:
        session = Path(rec["path"])
        pose = PT.pose_dir(Path(plan["paths"]["pose_root"]), session)
        required = (session / "metadata.json", session / "data_int16.npy", pose / PT.POSE_FILE, pose / PT.VALID_FILE)
        require(all(p.is_file() for p in required), "missing pose source file")
        paths.extend((*required, pose / PT.META_FILE))
    return recs, source_states(paths)


def source_states(paths):
    return {str(p): P.SourceGuard.stat(p) if p.exists() else None for p in paths}


def classification_sources(plan, task):
    paths = []
    if task == "p1":
        splits = json.loads(Path(plan["inputs"]["splits"]["path"]).read_text())
        for dataset in DATASETS[task]:
            for subject in C.CP.pretraining_subjects(splits, dataset):
                root = C._root_of([Path(p) for p in plan["paths"]["roots"]], dataset, subject)
                for session in C.P1.sessions_of(root, dataset, subject):
                    paths.extend(session / name for name in ("metadata.json", "data_int16.npy", "labels.npz"))
    else:
        for root in plan["paths"]["epn_roots"]:
            paths.extend(p / (p.name + ".json") for p in sorted(Path(root).glob("user*")) if (p / (p.name + ".json")).exists())
        for directory in Path(plan["paths"]["uci_root"]).rglob("*"):
            if directory.is_dir() and directory.name.isdigit():
                paths.extend(sorted(directory.glob("*.txt")))
    require(paths and all(p.is_file() for p in paths), "missing classification source files")
    return source_states(paths)


def extraction_commands(stage, plan, checkpoint, out, task, device):
    if stage == "extract-pose":
        commands = {}
        for split, name in (("train", "train_subset.npz"), ("val", "val.npz")):
            command = [sys.executable, str(REPO / "scripts/pose_eval.py"), "--stage", "extract", "--checkpoint", checkpoint,
                       "--which", "teacher", "--csv", plan["inputs"]["csv"]["path"], "--processed-root", plan["paths"]["processed_root"],
                       "--pose-root", plan["paths"]["pose_root"], "--split", split, "--n-windows", "10", "--window-s", "4",
                       "--seed", "20261026", "--batch", "32", "--device", device, "--out", str(out / name)]
            if split == "train":
                command += ["--n-users", "40", "--n-recordings", "200"]
            commands[split] = command
        return commands
    require(task in DATASETS, "classification task required")
    command = [sys.executable, str(REPO / "scripts/channel_readout.py"), "--stage", "extract", "--checkpoint", checkpoint,
               "--task", task, "--datasets", *DATASETS[task], "--features", str(out), "--batch", "64", "--device", device]
    if task == "p1":
        command += ["--splits", plan["inputs"]["splits"]["path"]]
        for root in plan["paths"]["roots"]:
            command += ["--root", root]
    else:
        command += ["--uci-root", plan["paths"]["uci_root"]]
        for root in plan["paths"]["epn_roots"]:
            command += ["--epn-root", root]
    return {task: command}


def execute(stage, plan_path, arm, step, expected_code_sha, out, task=None, device="cuda"):
    started = time.perf_counter()
    require(stage != "extract-classification" or step == 19618, "classification is registered only at 19618")
    plan, states, guard, code = load_context(plan_path, arm, step, expected_code_sha)
    checkpoint = plan["checkpoints"][str(step)][arm]["path"]
    out = Path(out)
    require(out.is_absolute() and not out.exists() and not out.is_symlink() and out.resolve() == out, "output must be new and absolute")
    require(not out.is_relative_to(REPO) and not any(out.is_relative_to(Path(p)) for p in
            ("/leonardo_work/IscrB_WearUsFM/wearusfm", "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns")), "output inside code")
    common = {"protocol": PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": step,
              "checkpoint_sha256": guard["checkpoint_sha256"], "code_sha": code, "plan_sha256": states["plan"]["sha256"],
              "postload_recipe_passed": True, "model_config": guard["model_config"], "stats_sha256": states["pose_stats"]["sha256"]}
    out.mkdir(parents=True, exist_ok=False)
    outputs, commands, raw = {}, {}, {}
    if stage != "validate":
        commands = extraction_commands(stage, plan, checkpoint, out, task, device)
        for key, command in commands.items():
            if stage == "extract-pose":
                records, raw_before = pose_sources(plan, key)
                reference = plan["pose_references"][key]
                # Hash/selection validation happens before expensive extraction, too.
                Q.validate_reference(reference)
                require(len(records) == reference["n_recordings"], "selected recording count differs")
            else:
                raw_before = classification_sources(plan, task)
                for dataset in DATASETS[task]:
                    states["reference_" + dataset] = pin_file(plan["classification_references"][dataset])
            with (out / (key + ".log")).open("x") as log:
                subprocess.run(command, cwd=REPO, env={**os.environ, "WEARUSFM_CODE_SHA": code}, stdout=log, stderr=subprocess.STDOUT, check=True)
            raw_after = source_states([Path(p) for p in raw_before])
            require(raw_before == raw_after, "raw source changed during extraction")
            if stage == "extract-pose":
                records_after, _ = pose_sources(plan, key)
                require(records_after == records, "selected record list changed during extraction")
                paths = {key: out / ("train_subset.npz" if key == "train" else "val.npz")}
            else:
                require(classification_sources(plan, task) == raw_before, "classification source inventory changed")
                paths = {dataset: out / (dataset + ".npz") for dataset in DATASETS[task]}
            raw[key] = {"file_states": raw_before, "unchanged": True}
            for name, path in paths.items():
                before = P.file_identity(path)
                details = verify_pose_cache(path, plan["pose_references"][name], name, checkpoint, code,
                                            states["pose_stats"]["sha256"], guard["model_config"]) if stage == "extract-pose" else \
                          verify_classification_cache(path, plan["classification_references"][name], name, checkpoint,
                                                      guard["checkpoint_sha256"], code, states["splits"])
                require(P.file_identity(path) == before, "output cache changed during validation")
                provenance = {**common, **details, "cache": str(path), "cache_sha256": before["sha256"], "cache_bytes": before["bytes"]}
                pp = Path(str(path) + ".provenance.json")
                P.write_json(pp, provenance)
                outputs[name] = {"path": str(path), "sha256": before["sha256"], "bytes": before["bytes"],
                                 "provenance_path": str(pp), "provenance_sha256": R.sha256_file(pp)}
    P.check_unchanged(states)
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
    parser.add_argument("--step", type=int, choices=(5000, 19618), required=True)
    parser.add_argument("--task", choices=tuple(DATASETS))
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    execute(args.stage, args.plan, args.arm, args.step, args.expected_code_sha, args.out, args.task, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
