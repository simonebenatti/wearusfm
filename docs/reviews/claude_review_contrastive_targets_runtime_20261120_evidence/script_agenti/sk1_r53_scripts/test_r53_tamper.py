"""Scettico 1, R5-3: negative probes for the untested restart guards (outside the clone).

Each test tampers one artefact of the real CPU common-fork fixture so that ONLY the named
guard can detect it, then restores the bytes. On unmutated code all must pass (the guard
works); with the corresponding mutation applied the test must fail (the gap is closable).
"""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import torch

import contrastive_target_resume as C
from test_contrastive_target_resume import common_paths  # noqa: F401  (module fixture)


@contextmanager
def restored(*paths):
    saved = {p: p.read_bytes() for p in paths}
    try:
        yield
    finally:
        for p, data in saved.items():
            p.write_bytes(data)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _compare(common):
    paths, cfg, initial_sha, initial, fork_sha, _ = common
    return C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)


def test_boundary_state_only(common_paths):
    paths = common_paths[0]
    trace = paths["reload_a"] / "trace-0.jsonl"
    with restored(trace):
        rows = _rows(trace)
        assert "boundary" in rows[0] and "boundary" not in rows[-1]
        rows[0]["boundary"]["optimizer_moments"] = "changed"
        _write_rows(trace, rows)
        result = _compare(common_paths)
        assert result["status"] == "FAILED_REQUIRES_REVIEW" and not result["passed"]
        assert result["boundary_mismatches"] == [{"phase": "reload_a", "rank": 0, "field": "optimizer_moments"}]
        assert all(not p["input_rng_counter_order_differences"] for p in result["pairs"].values())


def test_boundary_rng_same_in_all_three_paths(common_paths):
    paths = common_paths[0]
    traces = [paths[p] / "trace-0.jsonl" for p in C.PHASES]
    with restored(*traces):
        for trace in traces:
            rows = _rows(trace)
            rows[0]["rng_cpu_before_sha256"] = "changed"   # identical in the three paths
            _write_rows(trace, rows)
        result = _compare(common_paths)
        assert result["status"] == "FAILED_REQUIRES_REVIEW" and not result["passed"]
        assert {m["field"] for m in result["boundary_mismatches"]} == {"rng_cpu_before_sha256"}
        assert len(result["boundary_mismatches"]) == 3
        assert all(not p["input_rng_counter_order_differences"] for p in result["pairs"].values())


def test_start_json_protocol(common_paths):
    paths = common_paths[0]
    start = paths["reload_b"] / "start.json"
    with restored(start):
        record = json.loads(start.read_text())
        record["protocol"] = "other"
        start.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="same immutable fork bytes"):
            _compare(common_paths)


def test_endpoint_stop_file(common_paths):
    paths = common_paths[0]
    stop = paths["reload_a"] / "STOP"
    try:
        stop.write_text("stop\n")
        with pytest.raises(ValueError, match="incomplete or mismatched common-fork endpoint"):
            _compare(common_paths)
    finally:
        stop.unlink(missing_ok=True)


def test_row_phase_label(common_paths):
    paths = common_paths[0]
    trace = paths["reload_a"] / "trace-0.jsonl"
    with restored(trace):
        rows = _rows(trace)
        rows[0]["phase"] = "reload_b"
        _write_rows(trace, rows)
        with pytest.raises(ValueError, match="invalid runtime restart trace invariants"):
            _compare(common_paths)


def test_weights_changed_consistently_everywhere(common_paths):
    paths = common_paths[0]
    files = [paths[p] / n for p in C.PHASES for n in ("trace-0.jsonl", "metrics.jsonl")]
    with restored(*files):
        for f in files:
            rows = _rows(f)
            for r in rows:
                if r.get("update", r.get("step")) > 2:
                    r["contrast_rnc_weight"] = 0.123
            _write_rows(f, rows)
        with pytest.raises(ValueError, match="inactive or changed weights"):
            _compare(common_paths)


def test_trace_disagrees_with_metrics(common_paths):
    paths = common_paths[0]
    metrics = paths["reload_a"] / "metrics.jsonl"
    with restored(metrics):
        rows = _rows(metrics)
        for r in rows:
            if r["step"] > 2:
                r["contrast_rnc_windows"] = r["contrast_rnc_windows"] + 1
        _write_rows(metrics, rows)
        with pytest.raises(ValueError, match="disagrees with production metrics"):
            _compare(common_paths)


def test_validate_fork_four_ranks_without_cuda_rng(common_paths, tmp_path):
    paths, cfg, initial_sha, initial, fork_sha, provenance = common_paths
    state = torch.load(str(paths["continuous"] / "checkpoint_step0000002.pt"), map_location="cpu", weights_only=False)
    state["world_size"] = 4
    state["rank_rng"] = [{"torch_rng": torch.get_rng_state(), "cuda_rng": None} for _ in range(4)]
    synthetic = tmp_path / "fork4.pt"
    torch.save(state, synthetic)
    with pytest.raises(ValueError, match="lacks separate CPU/CUDA RNG"):
        C.validate_fork(synthetic, C.file_sha(synthetic), cfg, provenance, initial_sha,
                        fork_step=2, end_step=4, ranks=4)


def test_prepare_reload_detects_corrupted_copy(common_paths, tmp_path, monkeypatch):
    paths, cfg, initial_sha, initial, fork_sha, provenance = common_paths
    fork = paths["continuous"] / "checkpoint_step0000002.pt"

    def corrupt(src, dst):
        data = bytearray(src.read())
        data[len(data) // 2] ^= 0xFF          # flip one byte in the middle (last byte of a zip is often 0)
        dst.write(bytes(data))
    monkeypatch.setattr(C.shutil, "copyfileobj", corrupt)
    with pytest.raises(ValueError, match="not byte-identical"):
        C.prepare_reload(fork, fork_sha, tmp_path / "bad_reload", cfg, provenance, initial_sha,
                         phase="reload_a", fork_step=2, end_step=4, ranks=1)
