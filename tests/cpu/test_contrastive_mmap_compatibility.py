"""Execute every production mmap call against the cluster's filename contract.

The development Torch accepts pathlib.Path with mmap; Torch 2.2 on Leonardo
explicitly requires str. The shim reproduces only that observed argument check
and delegates actual checkpoint decoding/mapping to the installed Torch.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SITES = (
    ("contrastive_cuda_check.py", "main", 0),
    ("contrastive_run.py", "validate_resume", 0),
    ("contrastive_benchmark_readout.py", "compare_resume", 0),
    ("contrastive_benchmark_readout.py", "compare_resume", 1),
    ("contrastive_benchmark_readout.py", "_run_evidence", 0),
    ("contrastive_benchmark_readout.py", "_initial_from_calibration", 0),
)


def _load_calls(tree):
    return sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Attribute)
         and isinstance(node.func.value, ast.Name)
         and node.func.value.id == "torch" and node.func.attr == "load"),
        key=lambda node: (node.lineno, node.col_offset),
    )


def test_all_six_production_mmap_sites_are_covered():
    actual = []
    for filename in sorted({row[0] for row in SITES}):
        module = ast.parse((ROOT / "scripts" / filename).read_text())
        for function in module.body:
            if isinstance(function, ast.FunctionDef):
                for index, call in enumerate(_load_calls(function)):
                    if any(k.arg == "mmap" and isinstance(k.value, ast.Constant)
                           and k.value.value is True for k in call.keywords):
                        actual.append((filename, function.name, index))
    assert sorted(actual) == sorted(SITES)


@pytest.mark.parametrize("filename,function_name,call_index", SITES)
def test_production_load_accepts_cluster_contract_and_preserves_checkpoint(
        tmp_path, filename, function_name, call_index):
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {
        "step": 544,
        "student": {
            "weight": torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
            "gain": torch.tensor(548.6939187567021, dtype=torch.float64),
            "bf16": torch.tensor([-.5, 0., 1.25], dtype=torch.bfloat16),
        },
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([1., -2.])}}},
        "queue": {"last_step": 543, "keys": torch.arange(5, dtype=torch.int64)},
        "rank_rng": [torch.get_rng_state()],
    }
    torch.save(payload, checkpoint)
    reference = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    observed = []

    def load_with_torch22_filename_contract(path, **kwargs):
        if kwargs.get("mmap") and not isinstance(path, str):
            raise ValueError("f must be a string filename in order to use mmap argument")
        observed.append((path, kwargs.copy()))
        return torch.load(path, **kwargs)

    with pytest.raises(ValueError, match="must be a string filename"):
        load_with_torch22_filename_contract(
            checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert observed == []

    module = ast.parse((ROOT / "scripts" / filename).read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    call = _load_calls(function)[call_index]
    expression = ast.Expression(body=call)
    context = {
        "torch": SimpleNamespace(load=load_with_torch22_filename_contract),
        "args": SimpleNamespace(calibration_checkpoint=checkpoint),
        "checkpoint": checkpoint, "full_path": checkpoint, "resumed_path": checkpoint,
        "folder": tmp_path, "run_dir": tmp_path, "Path": Path,
    }
    actual = eval(compile(expression, filename, "eval"), context)
    assert observed == [(str(checkpoint), {
        "map_location": "cpu", "weights_only": False, "mmap": True})]

    def exact_equal(left, right):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.stride() == right.stride()
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact_equal(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for first, second in zip(left, right):
                exact_equal(first, second)
        else:
            assert left == right

    exact_equal(actual, reference)
