"""Skeptic 1, R1-N4: which numeric receipts are actually read (read-only probes)."""
import copy, json
from pathlib import Path
import pytest
import contrastive_benchmark_readout as B
from contrastive_numeric_profile import numeric_profile, numeric_flags
from test_contrastive_targets_readout import readout_fixture, write


def test_calibration_receipts_are_never_read(tmp_path, monkeypatch):
    kwargs, _, _, _, _ = readout_fixture(tmp_path, monkeypatch)
    cal = kwargs["calibration_run_dir"]
    base = B.read_preflight(**kwargs)
    assert base["status"] == "PASSED_FOR_REVIEW"
    # calibration receipts with a context that the 7-run gate would reject
    with numeric_profile(require_cuda=False) as flags:
        for rank in range(4):
            f = dict(flags); f["bf16_reduced_precision_reduction"] = not f["bf16_reduced_precision_reduction"]
            f["torch_version"] = "9.9.9-other"; f["cublas_workspace_config"] = ":4096:8"
            write(cal / f"numeric_profile.rank{rank}.json", {"rank": rank, "world_size": 4, "phase": None,
                  "source_sha256": "x" * 64, "flags": f})
    with_bad = B.read_preflight(**kwargs)
    print("calibration with divergent receipts:", with_bad["status"], with_bad["numeric_context_across_runs"]["status"],
          sorted(with_bad["numeric_context_across_runs"]["profile_receipts_sha256"]))
    assert with_bad["status"] == "PASSED_FOR_REVIEW"
    for rank in range(4):
        (cal / f"numeric_profile.rank{rank}.json").unlink()
    assert B.read_preflight(**kwargs)["status"] == "PASSED_FOR_REVIEW"
    # control: the same divergence in one of the 7 runs is caught
    folder = kwargs["resumed_dirs"]["CR_A"]
    for rank in range(4):
        p = folder / f"numeric_profile.rank{rank}.json"; row = json.loads(p.read_text())
        row["flags"]["torch_version"] = "9.9.9-other"; write(p, row)
    ctrl = B.read_preflight(**kwargs)
    print("same divergence in CR_A:", ctrl["status"], [d["fields"] for d in ctrl["numeric_context_across_runs"]["differences"]])
    assert ctrl["status"] == "FAILED_REQUIRES_REVIEW"


def test_cuda_evidence_checks_only_minimal_contract_per_rank(tmp_path):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    import hashlib
    calibration_cfg = {'contrastive': {'calibration_only': True, 'nce_weight': 1., 'rnc_weight': 1.}}
    canonical = hashlib.sha256(json.dumps(calibration_cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cohort = {'source_sha256': 's'*64, 'initial_state_sha256': 'i'*64}
    inputs = {k: k+'-pin' for k in ('manifest', 'scales', 'window_seconds', 'pose_stats')}
    keys = ('total', 'pose', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base', 'contrast_global_grad_nce',
            'contrast_global_grad_rnc', 'gradient', 'student', 'teacher')
    precisions = {}
    for name in ('fp32', 'bf16'):
        limits = {k: (.01 if name == 'bf16' or k in ('student', 'teacher') else 1e-5) for k in keys}
        precisions[name] = {'passed': True, 'updates': [{'update': s, 'passed': True, 'errors': dict.fromkeys(keys, 0.), 'limits': limits} for s in (1, 2)]}
    report = {'passed': True, 'source_sha256': cohort['source_sha256'], 'initial_sha256': cohort['initial_state_sha256'],
              'protocol': PROTOCOL, 'criteria': CRITERIA, 'technical_overrides': TECHNICAL_OVERRIDES, 'world_size': 4,
              'inputs_unchanged': True, 'inputs': {**inputs, 'config': 'x'}, 'pose_stats_sha256': inputs['pose_stats'],
              'config_canonical_sha256': canonical,
              'ranks': [{'rank': r, 'passed': True, 'stop_checks': {'coverage': True, 'numeric': True, 'passed': True},
                         'arms': {'CN': copy.deepcopy(precisions), 'CR': copy.deepcopy(precisions)}} for r in range(4)]}
    with numeric_profile(require_cuda=False) as flags:
        for r, row in enumerate(report['ranks']):
            f = dict(flags)
            if r == 2:  # differs from other ranks in fields outside the minimal contract
                f["torch_version"] = "9.9.9-other"; f["bf16_reduced_precision_reduction"] = not f["bf16_reduced_precision_reduction"]
            row['numeric_profile'] = f
    path = tmp_path/'cuda.json'; path.write_text(json.dumps(report))
    assert B._cuda_evidence(path, cohort, inputs, calibration_cfg)['passed']
    print("oracle with rank 2 divergent extra fields: accepted")
    # control: minimal contract violated -> rejected
    bad = copy.deepcopy(report); bad['ranks'][1]['numeric_profile']['matmul_tf32'] = True
    path.write_text(json.dumps(bad))
    with pytest.raises(Exception) as exc:
        B._cuda_evidence(path, cohort, inputs, calibration_cfg)
    print("oracle with TF32 on rank 1:", type(exc.value).__name__, exc.value)


def test_recorded_fields():
    with numeric_profile(require_cuda=False):
        keys = sorted(numeric_flags())
    print("recorded keys:", keys)
    for k in ("cudnn_version", "nccl_version", "driver_version", "device_name", "gpu"):
        assert k not in keys
