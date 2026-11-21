"""Verify collected bytes and describe traces, without loading any checkpoint."""
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CELLS = ("CN_A", "CN_B", "CR_A", "CR_B")
FORKS = {
    "CN": "aa5952f0c4b7a8f07b5965b096877d568f3b5662bf08e6d8a4fc56f2a9e37ce6",
    "CR": "a9a5b1deb69c46c0238480500605f24fe9415c41f2a964d91d0cd2f7f5608e1a",
}
SOURCE_SHA = "12e215a978b9d30ea6d670f58af1a0f97387d477"


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flattened(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flattened(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def compare(left, right, arm, label):
    result = {"arm": arm, "pair": label, "ranks": []}
    for rank in range(4):
        a, b = rows(left / f"trace-{rank}.jsonl"), rows(right / f"trace-{rank}.jsonl")
        assert [x["update"] for x in a] == [x["update"] for x in b] == list(range(545, 577))
        keys = ("batch_sha256", "input_sha256", "rng_cpu_before_sha256", "rng_cuda_before_sha256",
                "rng_cpu_after_sha256", "rng_cuda_after_sha256")
        matches = {key: all(x[key] == y[key] for x, y in zip(a, b)) for key in keys}
        first = None
        for x, y in zip(a, b):
            dx, dy = dict(flattened(x)), dict(flattened(y))
            changed = [key for key in sorted(set(dx) | set(dy)) if key != "phase" and dx.get(key) != dy.get(key)]
            if changed:
                key = changed[0]
                first = {"step": x["update"], "rank": rank, "field": key, "left": dx.get(key), "right": dy.get(key)}
                break
        result["ranks"].append({"rank": rank, "matches": matches, "first_trace_difference_excluding_phase": first})
    a = [x for x in rows(left / "metrics.jsonl") if 545 <= x["step"] <= 576]
    b = [x for x in rows(right / "metrics.jsonl") if 545 <= x["step"] <= 576]
    assert [x["step"] for x in a] == [x["step"] for x in b] == list(range(545, 577))
    first_numeric = None
    for x, y in zip(a, b):
        dx, dy = dict(flattened(x)), dict(flattened(y))
        for key in dx:
            if key.startswith("t_") or "elapsed" in key:
                continue
            u, v = dx[key], dy.get(key)
            if isinstance(u, (int, float)) and not isinstance(u, bool) and isinstance(v, (int, float)) and u != v:
                first_numeric = {"step": x["step"], "rank": "rank0 metrics", "field": key, "left": u, "right": v,
                                 "absolute_difference": abs(u - v)}
                break
        if first_numeric:
            break
    result["first_numeric_metric_difference_excluding_timing"] = first_numeric
    return result


def main():
    final_receipt = (HERE / "final_remote_receipt.txt").read_text()
    remote_hashes = dict((path, digest) for digest, path in re.findall(r"^([a-f0-9]{64})  (/.+)$", final_receipt, re.M))
    remote_sizes = dict((path, int(size)) for size, path in re.findall(r"^(\d+) (/.+)$", final_receipt + (HERE / "logs_remote_sizes.txt").read_text(), re.M))
    for path, digest in remote_hashes.items():
        if "/logs/" in path:
            local = HERE / "logs" / Path(path).name
            assert hashlib.sha256(local.read_bytes()).hexdigest() == digest, local
            assert local.stat().st_size == remote_sizes[path], local
    for arm, digest in FORKS.items():
        path = f"/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/benchmark/{arm}/checkpoint_step0000544.pt"
        assert remote_hashes[path] == digest and remote_sizes[path] == 463142486
    cells = {}
    for cell in CELLS:
        arm, repeat = cell.split("_")
        folder = HERE / cell
        remote = (HERE / f"{cell}_remote_integrity_and_log.txt").read_text()
        hashes = {Path(path).name: digest for digest, path in re.findall(r"^([a-f0-9]{64})  (/.+)$", remote, re.M)}
        sizes = {Path(path).name: int(size) for size, path in re.findall(r"^(\d+) (/.+)$", remote, re.M)}
        files = list(folder.iterdir())
        assert set(hashes) == set(sizes) == {p.name for p in files} | {"checkpoint.pt"}
        for path in files:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name], path
            assert path.stat().st_size == sizes[path.name], path
        start, summary = read_json(folder / "start.json"), read_json(folder / "summary.json")
        assert start["phase"] == "reload_" + repeat.lower()
        assert start["copied_checkpoint_sha256"] == start["source"]["sha256"] == FORKS[arm]
        assert start["source"]["step"] == 544 and start["source"]["end_step"] == 576
        assert summary["steps"] == 576 and summary["stopped"] == "max_steps"
        assert summary["code"]["sha"] == SOURCE_SHA and summary["world_size"] == 4
        assert [x["step"] for x in rows(folder / "metrics.jsonl")] == list(range(545, 577))
        assert len(list(folder.glob("numeric_profile.rank*.json"))) == 4
        for rank in range(4):
            for suffix in (".jsonl", ".ddp.jsonl"):
                trace = rows(folder / f"trace-{rank}{suffix}")
                assert [x["update"] for x in trace] == list(range(545, 577))
                assert all(x["rank"] == rank for x in trace)
        cells[cell] = {"job_id": read_json(HERE / f"{cell}.state.json")["job_slurm"],
                       "verified_files": len(files), "steps": [545, 576], "phase": start["phase"],
                       "checkpoint": {"path": str(Path(start["source"]["path"]).parents[2] / "reloads" / cell / "checkpoint.pt"),
                                      "bytes": sizes["checkpoint.pt"], "sha256": hashes["checkpoint.pt"]},
                       "summary_sha256": hashes["summary.json"], "gpu_max_memory_gib": summary["gpu_max_memory_gib"]}
    contrasts = []
    for arm in FORKS:
        left, right = HERE / (arm + "_A"), HERE / (arm + "_B")
        contrast = compare(left, right, arm, "reload_A vs reload_B")
        contrast["final_checkpoint_byte_identical"] = cells[arm + "_A"]["checkpoint"]["sha256"] == cells[arm + "_B"]["checkpoint"]["sha256"]
        contrasts.append(contrast)
        for repeat in ("A", "B"):
            continuous, reload = BASE / "benchmark_execution_20261121" / arm, HERE / (arm + "_" + repeat)
            contrast = compare(continuous, reload, arm, "continuous vs reload_" + repeat)
            x, y = read_json(continuous / "summary.json")["contrastive"], read_json(reload / "summary.json")["contrastive"]
            contrast["final_model_hashes_from_summaries"] = {key: {"continuous": x[key], "reload": y[key], "equal": x[key] == y[key]}
                                                           for key in ("final_student_sha256", "final_teacher_sha256")}
            contrasts.append(contrast)
    accounting = list(csv.DictReader((HERE / "allocation.sacct.tsv").read_text().splitlines(), delimiter="|"))
    assert len(accounting) == 5 and all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in accounting)
    gpu_hours = sum(int(r["ElapsedRaw"]) * 4 / 3600 for r in accounting if "gres/gpu=4" in r["AllocTRES"])
    core_hours = sum(int(r["ElapsedRaw"]) * int(r["AllocCPUS"]) / 3600 for r in accounting)
    timeline = list(csv.DictReader((HERE / "execution_timeline.sacct.tsv").read_text().splitlines(), delimiter="|"))
    ids = {str(cells[cell]["job_id"]) for cell in cells}
    intervals = [row for row in timeline if row["JobIDRaw"] in ids]
    assert len(intervals) == 4
    events = sorted([(r["Start"], 1) for r in intervals] + [(r["End"], -1) for r in intervals])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak <= 2 and active == 0
    report = {"status": "COLLECTED_AND_BYTE_VERIFIED", "source_sha": SOURCE_SHA, "cells": cells,
              "trace_comparisons": contrasts, "accounting": {"gpu_hours": gpu_hours, "core_hours": core_hours,
              "requested_gpu_hours": 8, "requested_core_hours": 64.26666666666667, "peak_concurrent_reloads": peak,
              "envelope_gpu_hours": 9.6, "envelope_core_hours": 77.46666666666667,
              "cohort_gpu_hours_including_previous": 3.747778 + gpu_hours,
              "cohort_core_hours_including_previous": 39.067778 + core_hours},
              "scope": "Operational collection and trace description only; no checkpoint tensors loaded and no training release."}
    (HERE / "collection_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "accounting": report["accounting"]}))


if __name__ == "__main__":
    main()
