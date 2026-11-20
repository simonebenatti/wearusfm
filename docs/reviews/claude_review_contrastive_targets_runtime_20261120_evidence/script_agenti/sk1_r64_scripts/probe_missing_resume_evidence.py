"""Skeptic probe R6-4: real CPU common-fork run, then remove one evidence file at a time.

Independent of the r6b probe: builds its own continuous + two fresh reloads (ranks=1,
fork 2, end 4) with the production trace_resume_updates / prepare_reload code, then
calls the real compare_three_paths with each file missing (restored afterwards).
"""
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

import contrastive_target_resume as C
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256


def train(folder, cfg, manifest, data, phase, initial_sha, provenance):
    with C.trace_resume_updates(folder / "trace-0.jsonl", phase, fork_step=2, end_step=4):
        summary = R.train(cfg, manifest, [data], folder, log=lambda _: None,
                          expected_contrastive_initial_sha256=initial_sha, contrastive_provenance=provenance)
    assert summary["steps"] == 4 and summary["stopped"] == "max_steps", summary


def fresh(rank, folder, cfg, manifest, data, phase, initial_sha, provenance):
    torch.set_num_threads(1)
    train(folder, cfg, manifest, data, phase, initial_sha, provenance)


def main(base):
    tmp = Path(tempfile.mkdtemp(dir=base))
    data, manifest, pose = _pose_setup(tmp)
    b = _config(pose, sys.argv[2] if len(sys.argv) > 2 else "CR")
    cfg = replace(b, max_steps=4, model=replace(b.model, muscle_dropout=.4), log_batch_origin=True,
                  schedule=replace(b.schedule, total_steps=20, keep_checkpoints=(2,)))
    calibration = replace(cfg, contrastive={**cfg.contrastive, "calibration_only": True})
    provenance = {"source_sha256": "probe", "input_sha256": {}}
    R.train(calibration, manifest, [data], tmp / "calibration", log=lambda _: None,
            contrastive_provenance=provenance)
    initial = C.checkpoint(tmp / "calibration/checkpoint.pt")["student"]
    initial_sha = state_sha256(initial)
    paths = {phase: tmp / phase for phase in C.PHASES}
    train(paths["continuous"], cfg, manifest, data, "continuous", initial_sha, provenance)
    fork = paths["continuous"] / "checkpoint_step0000002.pt"
    fork_sha = C.file_sha(fork)
    for phase in C.PHASES[1:]:
        C.prepare_reload(fork, fork_sha, paths[phase], cfg, provenance, initial_sha,
                         phase=phase, fork_step=2, end_step=4, ranks=1)
        torch.multiprocessing.spawn(fresh, args=(paths[phase], cfg, manifest, data, phase,
                                                 initial_sha, provenance), nprocs=1)
    call = lambda: C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha,
                                         ranks=1, fork_step=2, end_step=4)
    intact = call()
    print("arm", intact["arm"], "intact status", intact["status"], "passed", intact["passed"])
    victims = [paths["continuous"] / "trace-0.jsonl", paths["reload_a"] / "trace-0.jsonl",
               paths["reload_b"] / "trace-0.jsonl", paths["reload_a"] / "start.json",
               paths["reload_b"] / "start.json"]
    for victim in victims:
        saved = victim.read_bytes()
        victim.unlink()
        try:
            out = call()
            print("missing", victim.parent.name + "/" + victim.name, "-> returned status", out["status"])
        except Exception as exc:  # report the exact class and message
            print("missing", victim.parent.name + "/" + victim.name, "->", type(exc).__name__, "|",
                  str(exc).replace(str(tmp), "<tmp>"))
        finally:
            victim.write_bytes(saved)
    again = call()
    print("restored status", again["status"], "passed", again["passed"])
    # A truncated trace (one update missing) is the tested rejection path.
    victim = paths["reload_b"] / "trace-0.jsonl"
    saved = victim.read_bytes()
    victim.write_text(saved.decode().splitlines()[0] + "\n")
    try:
        call()
        print("truncated trace -> returned")
    except Exception as exc:
        print("truncated", victim.parent.name + "/" + victim.name, "->", type(exc).__name__, "|",
              str(exc).replace(str(tmp), "<tmp>"))
    finally:
        victim.write_bytes(saved)
    print("tmp", tmp)


if __name__ == "__main__":
    main(sys.argv[1])
