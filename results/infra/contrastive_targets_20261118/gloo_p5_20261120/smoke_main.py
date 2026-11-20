"""Fixture-only four-process entry into the real benchmark main, with visible CPU adapters."""
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("scripts", "src", "tests/cpu")]
import torch
import contrastive_run as controller
import contrastive_numeric_profile as profile
import contrastive_target_parity as parity
from contrastive_prepare import source_digest
from test_pose_training import _pose_setup
from test_contrastive_runtime import _config
from wearusfm.training import run as trainer
from wearusfm.training.contrastive_objectives import attach_contrastive_head
from wearusfm.training.pairing import state_sha256

file_sha = controller.sha


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def fixture_initial(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return state_sha256(attach_contrastive_head(trainer.WearUsFM(cfg.model)).state_dict())


def worker(rank, store, argv, cfg, release, cohort, worker_patches):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4", TORCH_DIST_INIT_BARRIER="1")
    torch.set_num_threads(1)
    work = store.parent
    out = Path(argv[argv.index("--out-dir") + 1])
    events = []

    def cuda_admission():
        events.append("controller_cuda_admission_adapted")
        return True

    def select_device(local):
        assert local == rank
        events.append("controller_cuda_device_adapted")

    def init_group(backend):
        assert backend == "nccl"
        torch.distributed.init_process_group("gloo", rank=rank, world_size=4,
            init_method=f"file://{store}", timeout=timedelta(seconds=120))
        events.append("gloo_initialized")

    def fixture_recipe(raw, stage, calibration, calibration_dir):
        assert stage == "benchmark" and raw == trainer.config_to_dict(cfg)
        return controller.config_from_dict(raw)

    def fixture_release(actual, stage, config_path, inputs, *args, **kwargs):
        assert actual == release and actual["fixture_only"] is True
        assert stage == actual["stage"] and cfg.contrastive["arm"] == actual["arm"]
        assert file_sha(config_path) == actual["config_sha256"]
        assert {key: file_sha(path) for key, path in inputs.items()} == actual["inputs"]
        assert actual["source_sha256"] == source_digest()
        assert file_sha(kwargs["cohort_path"]) == actual["cohort_sha256"]
        assert json.loads(kwargs["cohort_path"].read_text()) == cohort
        return cohort

    real_train = trainer.train

    def train_on_cpu(*args, **kwargs):
        assert kwargs.pop("device") == f"cuda:{rank}" and kwargs.pop("num_workers") == 6
        assert torch.distributed.get_backend() == "gloo"
        events.append("real_trainer_entered")
        summary = real_train(*args, device="cpu", num_workers=0, **kwargs)
        events.append("real_trainer_completed")
        return summary

    cpu_profile = profile.numeric_profile
    def local_profile(*, require_cuda):
        assert require_cuda
        return cpu_profile(require_cuda=False)

    def local_profile_flags(*, require_cuda=True, require_autocast=False):
        assert not require_autocast
        return profile.assert_numeric_profile(require_cuda=False)

    # Restrict device substitution to main's imported torch namespace. Trainer,
    # DDP, RNG and the report see the real CPU device and its actual properties.
    torch_adapter = SimpleNamespace(cuda=SimpleNamespace(is_available=cuda_admission, set_device=select_device),
        distributed=SimpleNamespace(init_process_group=init_group, barrier=torch.distributed.barrier,
                                    destroy_process_group=torch.distributed.destroy_process_group))
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(controller, "torch", torch_adapter))
        stack.enter_context(patch.object(controller, "validate_recipe", fixture_recipe))
        stack.enter_context(patch.object(controller, "validate_release", fixture_release))
        stack.enter_context(patch.object(profile, "numeric_profile", local_profile))
        stack.enter_context(patch.object(trainer, "train", train_on_cpu))
        if release["arm"] == "C0":
            stack.enter_context(patch.object(parity, "UPDATES", (2, 3)))
            stack.enter_context(patch.object(parity, "assert_numeric_profile", local_profile_flags))
        if worker_patches:
            worker_patches(stack, cfg, release)
        controller.main()
    assert not torch.distributed.is_initialized()
    events.append("main_returned_group_destroyed")
    write(work / (out.name + f".rank{rank}.json"), events)


def launch(work, arm, phase, cfg, manifest, data, cohort, *, extra_release=None, worker_patches=None):
    work = Path(work).resolve()
    manifest, data = Path(manifest).resolve(), Path(data).resolve()
    assert not torch.cuda.is_available() and cfg.model.dim <= 32 and cfg.max_steps <= 576
    assert arm == cfg.contrastive["arm"] and arm in ("C0", "CN", "CR")
    assert work.is_dir() and work.is_relative_to(Path(tempfile.gettempdir()).resolve())
    out = work / f"{arm}_{phase}"
    packet = work / f"{arm}_{phase}_fixture"
    packet.mkdir(exist_ok=False)
    config = packet / "config.json"
    cohort_path = packet / "cohort.json"
    scales, window = packet / "scales.json", packet / "window.json"
    write(config, trainer.config_to_dict(cfg))
    write(cohort_path, cohort)
    write(scales, {"scales": {}})
    write(window, {"window_s": {}})
    inputs = {"manifest": manifest, "scales": scales, "window_seconds": window,
              "pose_stats": Path(cfg.loader.pose_root) / "pose_stats.json"}
    release = {"stage": "benchmark", "phase": phase, "arm": arm, "fixture_only": True,
               "source_sha256": source_digest(), "config_sha256": file_sha(config),
               "cohort_sha256": file_sha(cohort_path), "inputs": {k: file_sha(v) for k, v in inputs.items()}}
    for key, value in (extra_release or {}).items():
        assert key not in release
        release[key] = value
    release_path = packet / "release.json"
    write(release_path, release)
    argv = ["contrastive_run.py", "--config", str(config), "--stage", "benchmark", "--release", str(release_path),
            "--manifest", str(manifest), "--root", str(data), "--window-seconds", str(window), "--scales", str(scales),
            "--out-dir", str(out), "--cohort", str(cohort_path), "--time-limit-s", "300"]
    torch.multiprocessing.spawn(worker, args=(work / (out.name + ".store"), argv, cfg, release, cohort, worker_patches),
                                nprocs=4, join=True)
    return out


def run_control():
    import argparse
    import gzip
    import hashlib
    import platform
    import shutil

    args = argparse.ArgumentParser(description="Run C0 orchestration smoke on four local CPU processes")
    args.add_argument("--evidence", type=Path, required=True, help="fresh evidence directory")
    destination = args.parse_args().evidence.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix="wearusfm-p5-c0-") as tmp:
        work = Path(tmp)
        torch.set_num_threads(1)
        data, manifest, pose = _pose_setup(work)
        base = _config(pose, "C0")
        cfg = replace(base, max_steps=3, batch_size=32, amp_bf16=True,
                      model=replace(base.model, muscle_dropout=.4),
                      loader=replace(base.loader, rms_window_ms=100, ms_fast_window_ms=125),
                      contrastive={**base.contrastive, "fill_steps": 256})
        cohort = {"initial_state_sha256": fixture_initial(cfg), "calibration_sha256": None}
        out = launch(work, "C0", "full", cfg, manifest, data, cohort)
        summary = json.loads((out / "summary.json").read_text())
        report = json.loads((out / "zero_weight_parity.json").read_text())
        assert (summary["stopped"], summary["steps"], summary["world_size"]) == ("max_steps", 3, 4)
        assert report["gate"]["status"] == "INCONCLUSIVE" and all(not point["errors"] for point in report["gate"]["points"])
        assert report["training_approved"] is False and report["summary_sha256"] == file_sha(out / "summary.json")
        assert report["initial_state_sha256"] == cohort["initial_state_sha256"]
        assert report["provenance"] == summary["contrastive"]["provenance"]
        assert [point["update"] for point in report["points"]] == [2, 3]
        for point in report["points"]:
            assert [row["rank"] for row in point["ranks"]] == list(range(4))
            assert point["gate"]["within_and_between_byte_identical"]
            for row in point["ranks"]:
                assert row["live_state_restored"] and set(row["cases"]) == set(parity.CASES)
                for case in row["cases"].values():
                    assert case["nce_eligible"] > 0 and case["rnc_pairs"] > 0 and case["rnc_windows"] > 0
                    assert case["global_windows"] == 32 and case["nce_weight"] == case["rnc_weight"] == 0
                    assert case["memory"]["cuda_measured"] is False and not case["flags"]["autocast_cuda_enabled"]
        paths = [out / name for name in ("zero_weight_parity.json", "summary.json", "config.json", "metrics.jsonl")]
        for rank in range(4):
            receipt = out / f"numeric_profile.rank{rank}.json"
            records = out / f"zero_weight_parity.rank{rank}.json"
            local = json.loads(records.read_text())
            numeric = json.loads(receipt.read_text())
            assert numeric["rank"] == rank and numeric["world_size"] == 4 and numeric["phase"] == "full"
            assert numeric["source_sha256"] == report["source_sha256"]
            assert numeric["config_canonical_sha256"] == report["config_sha256"]
            assert not numeric["flags"]["autocast_cuda_enabled"]
            assert [entry["record"] for entry in local["points"]] == [point["ranks"][rank] for point in report["points"]]
            paths += [receipt, records, work / f"C0_full.rank{rank}.json"]
        destination.mkdir()
        digests = {}
        for path in paths:
            payload = path.read_bytes()
            (destination / (path.name + ".gz")).write_bytes(gzip.compress(payload, mtime=0))
            digests[path.name] = hashlib.sha256(payload).hexdigest()
        write(destination / "receipt.json",
              {"local_orchestration": "PASSED", "arm": "C0", "production_parity": report["gate"]["status"],
               "training_approved": False, "device": "cpu", "arithmetic": "float32", "backend": "gloo",
               "torch": str(torch.__version__), "python": platform.python_version(), "world_size": 4,
               "fixture": "synthetic emg2pose, global batch 32, 3 steps, parity at 2 and 3",
               "cases_per_rank_per_point": len(parity.CASES), "source_sha256": report["source_sha256"],
               "uncompressed_sha256": digests})
        print("C0 local orchestration passed; production gate INCONCLUSIVE:", destination, flush=True)


if __name__ == "__main__":
    run_control()
