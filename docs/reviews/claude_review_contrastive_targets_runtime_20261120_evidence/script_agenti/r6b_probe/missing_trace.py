"""Real CPU compare_three_paths output: JSON round trip equality (authenticate_preflight uses actual != receipt)."""
import sys, json, tempfile
from dataclasses import replace
from pathlib import Path
import torch
import contrastive_target_resume as C
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256
import test_contrastive_target_resume as TR

def main(base):
    tmp = Path(tempfile.mkdtemp(dir=base))
    data, manifest, pose = _pose_setup(tmp)
    b = _config(pose, "CR")
    cfg = replace(b, max_steps=4, model=replace(b.model, muscle_dropout=.4), log_batch_origin=True,
                  schedule=replace(b.schedule, total_steps=20, keep_checkpoints=(2,)))
    calibration = replace(cfg, contrastive={**cfg.contrastive, "calibration_only": True})
    provenance = {"source_sha256": "test", "input_sha256": {}}
    R.train(calibration, manifest, [data], tmp / "calibration", log=lambda _: None, contrastive_provenance=provenance)
    initial = C.checkpoint(tmp / "calibration/checkpoint.pt")["student"]
    initial_sha = state_sha256(initial)
    paths = {phase: tmp / phase for phase in C.PHASES}
    TR._live_train(paths["continuous"], cfg, manifest, data, "continuous", initial_sha, provenance)
    fork = paths["continuous"] / "checkpoint_step0000002.pt"
    fork_sha = C.file_sha(fork)
    for phase in C.PHASES[1:]:
        C.prepare_reload(fork, fork_sha, paths[phase], cfg, provenance, initial_sha, phase=phase, fork_step=2, end_step=4, ranks=1)
        torch.multiprocessing.spawn(TR._fresh_reload, args=(paths[phase], cfg, manifest, data, phase, initial_sha, provenance), nprocs=1)
    import os
    for victim in (paths["reload_b"] / "trace-0.jsonl", paths["reload_a"] / "start.json"):
        saved = victim.read_bytes(); victim.unlink()
        try:
            C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
            print("missing", victim.name, "-> returned normally")
        except Exception as exc:
            print("missing", victim.name, "->", type(exc).__name__)
        victim.write_bytes(saved)
    result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    text = json.dumps(result, allow_nan=False)
    print("status", result["status"], "passed", result["passed"])
    print("json round-trip equal:", json.loads(text) == result)
    print("evidence keys per phase:", {k: sorted(v) for k, v in result["evidence_sha256"].items()})
    print("bytes", len(text))

if __name__ == "__main__":
    main(sys.argv[1])
