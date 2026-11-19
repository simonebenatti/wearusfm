from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet.resources import (GIB, HEADS, ResourceEvidenceError, gate_and_pack,
                                      matrix_counts, percentile95)


def receipts():
    numeric = {"threads": 8, "interop_threads": 1, "precision": "float32_no_autocast",
               "deterministic_algorithms": True, "cudnn_deterministic": True,
               "cudnn_benchmark": False, "cudnn_tf32": False, "matmul_tf32": False,
               "cublas_workspace": ":4096:8"}
    population = {"train_windows": 1907, "fold_windows": [953, 954], "fit_fold_windows": [953, 954],
                  "fit_train_windows": 1907, "excluded_fitting_windows": [0, 0], "validation_windows": 18548}
    common = {"schema": "pose-temponet-cost-v1", "train_only": True, "measured": True,
              "test_only": False, "input_identity": "pinned-input", "numeric_profile": numeric,
              "hardware": {"host": "fixture"}, "population": population, "elapsed_seconds": 100.}
    cpu = {**deepcopy(common), "kind": "cpu", "fold_projection_seconds": [400., 420.],
           "report_projection_seconds": 100., "all_patterns_measured": True,
           "all_alphas_measured": True, "grid_combinations_measured": [1936, 1936],
           "memory": {"peak_rss_bytes": GIB, "projected_peak_bytes": 2*GIB}}
    cells = {}
    for head in HEADS:
        cells[head] = {}
        for part, n, held in (("fold0", 953, 954), ("fold1", 954, 953), ("full", 1907, 1907)):
            cell = {"train_seconds": [.1]*10, "eval_seconds": [.01]*10,
                    "save_seconds": [.001]*10, "load_state_seconds": [.001]*10,
                    "artifact_write_seconds": [.002]*10, "artifact_read_seconds": [.002]*10,
                    "artifact_write_bytes": 1024,
                    "startup_seconds": .01, "warmup_epochs": 1, "timed_epochs": 10,
                    "train_windows": n, "eval_windows": held}
            if part == "full":
                cell["inference_batches"] = {"full_seconds": [.001]*10, "last_seconds": [.0008]*10,
                                               "last_size": 18548 % 32}
            cells[head][part] = cell
    gpu = {**deepcopy(common), "kind": "gpu", "cells": cells,
           "io": {"input_load_seconds": 1., "input_bytes": 300*1024**2,
                  "output_write_seconds": .1, "output_write_bytes": 20*1024**2,
                  "reference_read_seconds": .1, "reference_read_bytes": 1024},
           "memory": {"peak_rss_bytes": 2*GIB, "fixed_host_bytes": GIB,
                      "validation_view_bytes": 5*GIB, "batch_bytes": 12*1024**2,
                      "peak_vram_bytes": GIB}}
    return cpu, gpu


def gate(cpu=None, gpu=None, **kwargs):
    baseline_cpu, baseline_gpu = receipts()
    return gate_and_pack(cpu or baseline_cpu, gpu or baseline_gpu,
                         work_free_bytes=kwargs.pop("work_free_bytes", 12*GIB),
                         pending_write_bytes=kwargs.pop("pending_write_bytes", 0),
                         persistent_output_bytes=kwargs.pop("persistent_output_bytes", 7*GIB), **kwargs)


def test_full_formal_matrix_count_not_attainable_five_lr_shortcut():
    result = matrix_counts([953, 954], 1907)
    assert result["trajectories"] == 147
    assert result["cv_trajectories"] == 132
    assert result["held_evaluations"] == 40128
    assert result["fitting_evaluations"] == 37
    assert result["updates"] == 1458000
    assert matrix_counts([960, 947], 1907)["updates"] <= 1477800
    assert matrix_counts([961, 946], 1907)["updates"] == 1477800


def test_p95_uses_linear_interpolation():
    assert percentile95(list(range(10))) == pytest.approx(8.55)


def test_pack_all_groups_and_exact_inclusive_cpu_allocations():
    report = gate()
    assert report["all_passed"]
    assert report["projection"]["gpu_jobs"] == 18
    assert report["projection"]["cpu_jobs"] == 6
    training = [j for j in report["jobs"] if j["phase"] == "neural"]
    assert len(training) == 16
    assert sum(j["stage"] == "controls" for j in training) == 6
    assert sum(j["stage"] == "selection" for j in training) == 5
    assert sum(j["stage"] == "refit" for j in training) == 5
    assert len({j["name"] for j in report["jobs"]}) == 24
    references = [j for j in report["jobs"] if j["phase"] == "references"]
    assert all(j["requested_seconds"] == 14220 and j["allocation_seconds"] == 14400 for j in references)
    assert sum(j["core_hours"] for j in references) == 64
    assert report["projection"]["cpu_only_core_hours"] == 77
    assert all(j["allocation_seconds"] == j["requested_seconds"]+180 for j in report["jobs"])


@pytest.mark.parametrize("damage", [
    lambda c,g: g["cells"]["TCN_post"]["fold0"].pop("eval_seconds"),
    lambda c,g: g["cells"]["L0"]["full"]["inference_batches"].pop("last_seconds"),
    lambda c,g: c.pop("report_projection_seconds"),
    lambda c,g: c.update(all_alphas_measured=False),
    lambda c,g: c.update(grid_combinations_measured=[1936, 1815]),
    lambda c,g: c.update(test_only=True),
    lambda c,g: g["memory"].pop("peak_vram_bytes"),
    lambda c,g: g["io"].pop("reference_read_seconds"),
    lambda c,g: g["cells"]["TCN_post"]["fold0"].pop("artifact_write_seconds"),
    lambda c,g: g["cells"]["TCN_post"]["full"].pop("artifact_read_seconds"),
    lambda c,g: g["cells"]["TCN_post"]["full"].pop("artifact_write_bytes"),
])
def test_no_missing_or_partial_measurement_gets_optimistic_default(damage):
    cpu,gpu=receipts(); damage(cpu,gpu)
    with pytest.raises((ResourceEvidenceError, KeyError)):
        gate(cpu,gpu)


def test_all_regimes_blocked_when_one_fold_cpu_over_budget():
    cpu,gpu=receipts(); cpu["fold_projection_seconds"][1]=7110.001
    result=gate(cpu,gpu)
    assert not result["all_passed"]
    assert not result["gates"]["job_limits"]
    assert gate({**cpu, "fold_projection_seconds": [7110.,7110.]},gpu)["all_passed"]


def test_gpu_training_validation_io_and_save_all_count():
    cpu,gpu=receipts()
    small=gate(cpu,gpu)["projection"]["gpu_hours"]
    for head in HEADS:
        for c in gpu["cells"][head].values():
            c["eval_seconds"]=[10.]*10
    result=gate(cpu,gpu)
    assert result["projection"]["gpu_hours"] > small
    assert not result["gates"]["neural_gpu_hours"]
    assert not result["all_passed"]


def test_memory_quota_and_output_boundaries_are_independent():
    cpu,gpu=receipts()
    assert gate(cpu,gpu, work_free_bytes=13*GIB,pending_write_bytes=GIB)["gates"]["quota"]
    assert not gate(cpu,gpu,work_free_bytes=13*GIB-1,pending_write_bytes=GIB)["gates"]["quota"]
    cpu["memory"]["projected_peak_bytes"]=24*GIB/1.25
    assert gate(cpu,gpu)["gates"]["cpu_host_memory"]
    cpu["memory"]["projected_peak_bytes"]+=1
    assert not gate(cpu,gpu)["gates"]["cpu_host_memory"]
    gpu["memory"]["peak_vram_bytes"]=48*GIB+1
    assert not gate(cpu,gpu)["gates"]["vram"]
    assert not gate(persistent_output_bytes=8*GIB+1)["gates"]["persistent_output"]
    assert not gate(persistent_output_bytes=GIB)["gates"]["persistent_output"]


def test_full_and_partial_inference_batches_are_charged_and_two_cache_loads(monkeypatch):
    import math
    from pose_temponet import resources
    observed = []
    original = resources._job
    def capture(name, phase, raw_seconds, **kwargs):
        if phase == "inference":
            observed.append(raw_seconds)
        return original(name, phase, raw_seconds, **kwargs)
    monkeypatch.setattr(resources, "_job", capture)
    cpu,gpu=receipts()
    before=next(j for j in gate(cpu,gpu)["jobs"] if j["phase"]=="inference")["projected_required_seconds"]
    gpu["io"]["input_load_seconds"]+=10
    after=next(j for j in gate(cpu,gpu)["jobs"] if j["phase"]=="inference")["projected_required_seconds"]
    scale=max(18548/1907,5*GIB/(300*1024**2))
    assert observed[1]-observed[0] == pytest.approx(2*10*scale)
    assert before == 60*math.ceil(math.ceil(2*observed[0])/60)
    assert after == 60*math.ceil(math.ceil(2*observed[1])/60)
    gpu["cells"]["TCN_pre"]["full"]["inference_batches"]["last_seconds"]=[600.]*10
    assert not gate(cpu,gpu)["gates"]["inference_gpu_hours"]


def test_numeric_mismatch_rejected_even_when_fast():
    cpu,gpu=receipts(); cpu["numeric_profile"]["threads"]=1
    with pytest.raises(ResourceEvidenceError,match="threads"):
        gate(cpu,gpu)


def test_zero_support_only_reduces_fitting_updates_not_evaluation_population():
    cpu,gpu=receipts()
    for receipt in (cpu,gpu):
        receipt["population"].update(fit_fold_windows=[952,953],fit_train_windows=1905,
                                     excluded_fitting_windows=[1,1])
    for cells in gpu["cells"].values():
        cells["fold0"]["train_windows"]=952
        cells["fold1"]["train_windows"]=953
        cells["full"]["train_windows"]=1905
    result=gate(cpu,gpu)
    assert result["all_passed"]
    assert result["matrix"]["updates"]==matrix_counts([952,953],1905)["updates"]
    gpu["cells"]["L0"]["fold0"]["eval_windows"]=953
    with pytest.raises(ResourceEvidenceError,match="geometry"):
        gate(cpu,gpu)
