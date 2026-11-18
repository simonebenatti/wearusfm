"""Independent, CPU-only verification of the completed technical preflight.

Usage: python verify_scaling_preflight_63526191.py ARTIFACT_DIRECTORY
Reads either original files or losslessly compressed *.gz copies. Prints JSON;
does not train, import the preflight implementation, or authorize a next run.
"""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/infra/scaling_300m_20261104"
ARMS = ("T100-G-30M", "T100-G-300M")


def audit(directory):
    directory = Path(directory)
    def raw(name):
        path = directory / name
        return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())

    def read(name):
        return json.loads(raw(name))

    hashes = {}
    for line in (BASE / "preflight_63526191/remote_sha256.txt").read_text().splitlines():
        sha, path = line.split(maxsplit=1)
        name = Path(path).name
        assert hashlib.sha256(raw(name)).hexdigest() == sha, name
        hashes[name] = sha
    assert len(hashes) == 13
    plan = json.loads((BASE / "execution_plan.json").read_text())
    source = json.loads((BASE / "training_source_contract.json").read_text())
    report = read("report.json")
    assert report["complete"] is True and report["code_sha"] == plan["code_sha"]
    assert report["protocol"] == "scaling-300m-preflight-20261104-v1"
    assert report["world_size"] == 4 and report["batches_per_arm"] == 128
    assert report["native_warmup_batches"] == report["peak_stress_batches"] == 64
    assert report["windows_per_arm"] == 4096
    assert report["paired_data_masks_queries_targets"] is True
    assert report["scientific_checkpoints_written"] is False
    assert report["plan_sha256"] == plan["file_sha256"]["results/infra/scaling_300m_20261104/preflight_plan.json"]
    for key, value in source["inputs"].items():
        assert report["inputs"][key]["sha256"] == value["sha256"]
    parts = [read(f"rank{i}.json") for i in range(4)]
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    for i, part in enumerate(parts):
        assert part["complete"] is True and part["rank"] == i and finite(part)
        assert part["protocol"] == report["protocol"]
        assert report["rank_reports"][i]["sha256"] == hashes[f"rank{i}.json"]
        first, second = (part["arms"][arm] for arm in ARMS)
        assert first["delivered"] == second["delivered"] and len(first["delivered"]) == 128
        assert first["initial"]["frontend_and_gain_sha256"] == second["initial"]["frontend_and_gain_sha256"]
        assert first["exposure_windows"] == second["exposure_windows"]
        for arm in ARMS:
            cell = part["arms"][arm]
            assert cell["initial"] == report["arms"][arm]["initial"]
            rows = cell["records"]
            assert len(rows) == 128 and cell["sources_unchanged_after_first_seen"] is True
            stream = [json.loads(line) for line in raw(f"{arm}.rank{i}.jsonl").splitlines()]
            assert len(stream) == 128
            assert sum(cell["exposure_windows"].values()) == 1024
            assert not any(k.split('/')[0] in ("epn612", "uci_emg") for k in cell["exposure_windows"])
            for j, row in enumerate(rows):
                assert stream[j] == {"arm": arm, "rank": i, "fingerprint": cell["delivered"][j], **row}
                assert row["step"] == j + 1 and row["scientific_update"] is False
                assert row["phase"] == ("native_warmup" if j < 64 else "peak_lr_stress")
                assert math.isclose(row["lr"], 1e-4 * (j+1) / 1962 if j < 64 else 1e-4, rel_tol=1e-12)
                assert row["support"] == first["records"][j]["support"]
                assert math.isclose(row["clip_factor"], min(1., 1/(row["grad_norm_before_clip"] + 1e-6)), rel_tol=1e-12)
            for group in cell["initial"]["diagnostic_parameter_groups"]:
                assert any(row["gradient_norms_before_clip"][group] > 0 for row in rows)
                assert any(row["update_norms"][group] > 0 for row in rows)
    measured = list(range(8, 64)) + list(range(72, 128))
    cells = {}
    for arm in ARMS:
        records = [p["arms"][arm]["records"] for p in parts]
        peak = max(row["memory_peak_allocated_bytes"] for rows in records for row in rows)
        total = min(p["total_device_memory_bytes"] for p in parts)
        times = [sum(max(rows[j][key] for rows in records) for key in ("forward_backward_s", "clip_optimizer_ema_s")) for j in range(128)]
        assert np.allclose(times, report["arms"][arm]["normal_update_rank_max_seconds"], rtol=0, atol=0)
        assert peak == report["arms"][arm]["peak_allocated_bytes"] and peak <= .9*total
        phases = {}
        for phase, indices in (("native_warmup", list(range(8, 64))), ("peak_lr_stress", list(range(72, 128)))):
            rows = [r[j] for r in records for j in indices]
            expected = report["arms"][arm]["phases"][phase]
            values = {"normal_update_mean_s": float(np.mean([times[j] for j in indices])),
                      "grad_norm_median": float(np.median([x["grad_norm_before_clip"] for x in rows])),
                      "clip_factor_median": float(np.median([x["clip_factor"] for x in rows])),
                      "fraction_clipped": float(np.mean([x["clip_factor"] < 1 for x in rows]))}
            for key, value in values.items():
                assert math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12)
            phases[phase] = {**values, "loss_total_mean": float(np.mean([x["losses"]["total"] for x in rows])),
                             "loss_pose_mean": float(np.mean([x["losses"]["pose"] for x in rows]))}
        cells[arm] = {"peak_allocated_gib": peak/2**30, "device_gib": total/2**30,
                      "measured_updates": len(measured), "mean_update_s": float(np.mean([times[j] for j in measured])),
                      "phases": phases}
    historical = json.loads((ROOT / "results/infra/tokenizer_execution_20261103/5000/T100-G/summary.json").read_text())
    multiplier = max(1, historical["gpu_max_memory_gib"] / cells[ARMS[0]]["peak_allocated_gib"])
    proxy = cells[ARMS[1]]["peak_allocated_gib"] * multiplier
    limit = .9*cells[ARMS[1]]["device_gib"]
    return {"complete": True, "job_id": 63526191, "checked_artifacts": hashes, "paired_rank_updates": 512,
            "records_checked": 1024, "cells": cells, "production_memory_proxy_gib": proxy,
            "production_memory_proxy_limit_gib": limit, "production_memory_proxy_passed": proxy <= limit,
            "real_production_memory_exercised": False, "time_ratio_300m_30m": cells[ARMS[1]]["mean_update_s"]/cells[ARMS[0]]["mean_update_s"],
            "allocated_gpu_hours": 4*546/3600, "allocated_core_hours": 32*546/3600,
            "scientific_5000_ready": False, "scope": "Technical completion, pairing, finite updates and sampled memory; no downstream result or learning-rate optimality."}


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2, allow_nan=False))
