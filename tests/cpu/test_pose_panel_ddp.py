"""Pannello della posa su due processi CPU: training invariato ed errori iniziali coordinati."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

from test_pose_panel import _base, _multi_tree  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.pairing import state_sha256  # noqa: E402


def _worker(rank, store, root, manifest, pose_root, out, results, panel, invalid):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        held = {}
        make_teacher, make_panel = R.make_teacher, R._pose_panel_set
        calls = {"panel": 0}

        def capture(student):
            held["student"] = student
            held["teacher"] = make_teacher(student)
            return held["teacher"]

        def count_panel(*args, **kwargs):
            calls["panel"] += 1
            assert rank == 0, "il pannello va costruito solo sul processo 0"
            return make_panel(*args, **kwargs)

        R.make_teacher, R._pose_panel_set = capture, count_panel
        cfg = replace(_base(Path(pose_root)), max_steps=2, batch_size=4, fixed_eval_every=0,
                      pose_panel_every=1 if panel else 0, pose_panel_users=4 if invalid else 2,
                      pose_panel_recordings=1, pose_panel_windows=2)
        try:
            summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), log=lambda _: None,
                              dist=R.DistContext(rank, 2, rank))
        except ValueError as exc:
            if not invalid:
                raise
            result = {"error_type": type(exc).__name__, "error": str(exc), "panel_calls": calls["panel"]}
        else:
            assert not invalid, "la selezione impossibile del pannello deve rifiutare il run"
            result = {"summary": summary, "panel_calls": calls["panel"],
                      "student": state_sha256(held["student"].state_dict()),
                      "teacher": state_sha256(held["teacher"].state_dict())}
        Path(results, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _run(tmp_path, data, tag, *, panel, invalid=False):
    root, manifest, pose_root = data
    results = tmp_path / f"results_{tag}"
    results.mkdir()
    out = tmp_path / tag
    mp.spawn(_worker, args=(str(tmp_path / f"store_{tag}"), str(root), str(manifest), str(pose_root),
                            str(out), str(results), panel, invalid), nprocs=2, join=True)
    return out, [json.loads((results / f"rank{rank}.json").read_text()) for rank in range(2)]


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pose_panel_preserves_two_rank_training_and_rng(tmp_path):
    data = _multi_tree(tmp_path)
    plain, plain_ranks = _run(tmp_path, data, "plain", panel=False)
    panel, panel_ranks = _run(tmp_path, data, "panel", panel=True)
    assert [r["panel_calls"] for r in plain_ranks] == [0, 0]
    assert [r["panel_calls"] for r in panel_ranks] == [1, 0]
    for key in ("student", "teacher"):
        assert len({r[key] for r in plain_ranks + panel_ranks}) == 1
    assert all(r["summary"]["steps"] == 2 and r["summary"]["stopped"] == "max_steps" for r in panel_ranks)

    a, b = (torch.load(p / "checkpoint.pt", weights_only=False) for p in (plain, panel))
    for key in ("student", "teacher", "optimizer", "rank_rng"):
        assert state_sha256(a[key]) == state_sha256(b[key]), key
    timing = {"t_data_s", "t_data_s_mean", "t_step_s"}
    assert [{k: v for k, v in r.items() if k not in timing} for r in _records(plain / "metrics.jsonl")] == \
        [{k: v for k, v in r.items() if k not in timing} for r in _records(panel / "metrics.jsonl")]
    assert not (plain / "pose_panel.jsonl").exists()
    logged = _records(panel / "pose_panel.jsonl")
    assert [r["step"] for r in logged] == [0, 1, 2]
    assert all(r["n_windows_valid"] == 4 and r["n_users"] == 2 for r in logged)


def test_pose_panel_construction_failure_reaches_both_ranks(tmp_path):
    data = _multi_tree(tmp_path)  # tre utenti: la richiesta di quattro e' impossibile solo nella costruzione sul rank 0
    out, results = _run(tmp_path, data, "invalid", panel=True, invalid=True)
    assert [r["panel_calls"] for r in results] == [1, 0]
    assert {r["error_type"] for r in results} == {"ValueError"}
    assert len({r["error"] for r in results}) == 1
    assert all("pannello della posa" in r["error"] for r in results)
    assert not (out / "checkpoint.pt").exists()
    assert not (out / "metrics.jsonl").exists()
