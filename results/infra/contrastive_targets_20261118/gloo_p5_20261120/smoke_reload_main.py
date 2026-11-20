"""Exercise the actual benchmark full and resume_{a,b} branches of main with four Gloo ranks."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests/cpu")]

import torch
import contrastive_run as entry
import contrastive_target_resume as resume
from contrastive_release import pin
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training.contrastive_objectives import attach_contrastive_head
from wearusfm.training.pairing import state_sha256
from wearusfm.training.run import WearUsFM


FORK, END = 2, 4


def fixture_variant(original, replacements):
    """Apply only exact CPU/short-dose predicates to a process-local copy of a validator."""
    source = inspect.getsource(original)
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"fixture adapter no longer matches {original.__qualname__}: {before}")
        source = source.replace(before, after)
    # Functions resolve peer calls via the production module, including the scoped
    # fixture validators installed by the caller. No repository source is edited.
    local = {}
    exec(compile(source, f"<Gloo fixture: {original.__qualname__}>", "exec"), original.__globals__, local)
    return local[original.__name__]


def install_fork_reader(stack):
    cpu_fork = fixture_variant(resume.validate_fork, [
        ("fork_step=FORK, end_step=END", "fork_step=2, end_step=4"),
        ('not isinstance(r.get("cuda_rng"), torch.Tensor)', 'r["cuda_rng"] is not None'),
    ])
    stack.enter_context(patch.object(resume, "validate_fork", cpu_fork))


def worker_patches(stack, cfg, release):
    """Leave prepare_reload (copy, exclusive start, receipt) and the actual trainer intact."""
    install_fork_reader(stack)
    cpu_target_fork = fixture_variant(entry.validate_target_fork, [
        ('summary.get("steps") != 576', 'summary.get("steps") != 4'),
        ('"checkpoint_step0000544.pt"', '"checkpoint_step0000002.pt"'),
    ])
    stack.enter_context(patch.object(entry, "validate_target_fork", cpu_target_fork))
    for name in ("prepare_reload", "trace_resume_updates"):
        actual = getattr(resume, name)
        stack.enter_context(patch.object(resume, name, partial(actual, fork_step=FORK, end_step=END)))


def summarize(work, cfg, paths, initial, initial_sha, fork, seconds, out):
    # The production comparator requires CUDA RNG fingerprints from every rank. On
    # Gloo it must instead see four genuine CPU streams and an explicitly absent
    # CUDA stream. Preserve the original comparator and its three pairwise gates.
    compare = fixture_variant(resume.compare_three_paths, [
        ('for domain in ("cpu", "cuda") for when in ("before", "after")',
         'for domain in ("cpu",) for when in ("before", "after")'),
    ])
    with ExitStack() as stack:
        install_fork_reader(stack)
        result = compare(paths, cfg, initial_sha, initial, resume.file_sha(fork), ranks=4,
                         fork_step=FORK, end_step=END)
    rng, evidence = {}, {}
    for phase, directory in paths.items():
        counts = []
        for rank in range(4):
            trace = resume.rows(directory / f"trace-{rank}.jsonl", FORK, END)
            ddp = resume.rows(directory / f"trace-{rank}.ddp.jsonl", FORK, END)
            assert all(row["rank"] == rank for row in ddp)
            assert all(row[f"rng_cpu_{when}_sha256"] and row[f"rng_cuda_{when}_sha256"] is None
                       for row in trace for when in ("before", "after"))
            counts.append(len(trace))
        state = resume.checkpoint(directory / "checkpoint.pt")
        assert len(state["rank_rng"]) == 4 and all(
            torch.is_tensor(r["torch_rng"]) and r["cuda_rng"] is None for r in state["rank_rng"])
        rng[phase] = counts
        evidence[phase] = sorted(p.name for p in directory.iterdir() if p.name in {
            "config.json", "summary.json", "metrics.jsonl", "start.json", "origin_sessions.json",
        } or p.name.startswith(("numeric_profile.rank", "trace-")))
    archive = out / "reload_gloo_receipts.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for phase in paths:
            for filename in evidence[phase]:
                zipped.write(paths[phase] / filename, f"{phase}/{filename}")
    result.update({
        "scope": "local synthetic CN; CPU FP32, four fresh Gloo workers per path; not a production CUDA gate",
        "entry_point": "scripts/contrastive_run.py:main",
        "cuda_tested": False,
        "training_approved": False,
        "fixture_changes": ["tiny real data/model and local artifact pins, 0 loader workers",
          "main uses real Gloo barrier; numeric profile is CPU; no CUDA allocation",
          "fork/update 2..4 represent 544..576; the binder checks the scaled step and real checkpoint name",
          "CPU RNG required and CUDA RNG required absent on all four ranks; no CUDA RNG synthesized"],
        "seconds_per_phase": seconds,
        "rank_trace_rows_per_phase": rng,
        "receipts_archive": archive.name,
        "receipts_archive_sha256": resume.file_sha(archive),
        "fixture_initial_sha256": initial_sha,
        "real_trainer": True,
        "cpu_rng_only": True,
        "source_pins": {name: resume.file_sha(ROOT / "scripts" / name)
                        for name in ("contrastive_run.py", "contrastive_target_resume.py")},
    })
    # Keep the receipts and full evidence pins, plus the maximum of the per-tensor
    # distances for readability. The production reader applied its rule to each tensor.
    for pair in result["pairs"].values():
        distances = pair.pop("floating_tensor_relative_to_left", {})
        pair["floating_tensors_compared"] = len(distances)
        pair["max_floating_tensor_relative_to_left"] = max(distances.values(), default=None)
    (out / "reload_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise RuntimeError(f"restart comparator returned {result['status']}; inspect reload_result.json")


def main():
    from smoke_main import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path, help="empty/fresh evidence output directory")
    parser.add_argument("--work", type=Path, help="optional new scratch directory for local fixture")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    if any((args.evidence / name).exists() for name in ("reload_result.json", "reload_gloo_receipts.zip")):
        raise ValueError("refuse to replace existing local evidence")
    work = args.work or Path(tempfile.mkdtemp(prefix="wearusfm-p5-reload-"))
    if args.work:
        work.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, manifest, pose = _pose_setup(work)
    base = _config(pose, "CN")
    cfg = replace(base, max_steps=END, batch_size=32, model=replace(base.model, muscle_dropout=.4),
                  loader=replace(base.loader, rms_window_ms=100, ms_fast_window_ms=125),
                  log_batch_origin=True, alarm=replace(base.alarm, patience=20),
                  schedule=replace(base.schedule, total_steps=20, keep_checkpoints=(FORK,)))
    torch.manual_seed(cfg.seed)
    initial = attach_contrastive_head(WearUsFM(cfg.model)).state_dict()
    initial_sha = state_sha256(initial)
    cohort = {"initial_state_sha256": initial_sha, "calibration_sha256": None}
    paths = {p: work / f"CN_{p}" for p in ("full", "resume_a", "resume_b")}
    fork = paths["full"] / f"checkpoint_step{FORK:07d}.pt"
    seconds = {}
    for phase in paths:
        extra = {} if phase == "full" else {
            "fork_checkpoint": pin(fork), "continuous_summary": pin(paths["full"] / "summary.json"),
        }
        start = time.monotonic()
        launch(work, "CN", phase, cfg, manifest, data, cohort, extra_release=extra,
               worker_patches=worker_patches)
        seconds[phase] = round(time.monotonic() - start, 3)
    logical_paths = dict(zip(resume.PHASES, paths.values()))
    summarize(work, cfg, logical_paths, initial, initial_sha, fork, seconds, args.evidence)


if __name__ == "__main__":
    main()
