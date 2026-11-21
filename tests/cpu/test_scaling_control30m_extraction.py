"""Control-only adapter, original extraction immutability and real launcher guards."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_control30m_evaluation as A
import scaling_evaluation49046 as E
import test_scaling_evaluation49046_launcher as H

harness = H.harness
CONTROL_LAUNCHER = ROOT / "scripts/slurm/scaling_control30m_evaluation.sbatch"
ORIGINAL_LAUNCHER_SHA = "e335d7adde52637eb47e4b5a7e521138b5471655a46826e362c66331f415090b"
CONTROL_CELLS = [cell for cell in H.CELLS if cell[0] == E.ARMS[0]]


def test_frozen_evaluation_and_launcher_remain_exactly_87da975():
    assert hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest() == A.FROZEN_EVALUATION_SHA256
    assert hashlib.sha256(H.LAUNCHER.read_bytes()).hexdigest() == ORIGINAL_LAUNCHER_SHA
    original = inspect.getsource(E.load_context)
    derived = A.derived_loader_source()
    assert original.count(A._ORIGINAL_GUARD) == 1
    assert derived.count(A._CONTROL_GUARD) == 1
    assert derived.replace(A._CONTROL_GUARD, A._ORIGINAL_GUARD, 1) == original
    assert A.load_context.__globals__["validate_checkpoint"] is E.validate_checkpoint


def test_launcher_has_only_declared_control_schema_and_entrypoint_changes():
    original = H.LAUNCHER.read_text()
    substitutions = [
        ("# Two terminal arms, one pose/P1/P2 extraction each. Same historical selection.\n"
         "# Maximum14GPUh plus0.3 scheduler margin. No training of the30M control.",
         "# Existing 30M terminal control only: pose/P1/P2, same historical selection.\n"
         "# Maximum7GPUh plus0.15 scheduler margin, part of the unchanged six-cell budget."),
        ("usage: scaling_evaluation49046.sbatch", "usage: scaling_control30m_evaluation.sbatch"),
        ('case "$ARM" in T100-G-30M|T100-G-300M)', 'case "$ARM" in T100-G-30M)'),
        ("unregistered extraction: only the six registered terminal cells",
         "unregistered extraction: only the three registered 30M terminal cells"),
        ("arms = ['T100-G-30M', 'T100-G-300M']", "arms = ['T100-G-30M']"),
        ("or plan.get('arms') != arms or set(plan.get('checkpoints', {})) != set(arms):",
         "or plan.get('control_only') is not True or arm != arms[0] or plan.get('arms') != arms or set(plan.get('checkpoints', {})) != set(arms):"),
        ('exec python3 scripts/scaling_evaluation49046.py "${ARGS[@]}"',
         'exec python3 scripts/scaling_control30m_evaluation.py "${ARGS[@]}"'),
    ]
    for before, after in substitutions:
        assert original.count(before) == 1
        original = original.replace(before, after, 1)
    assert original == CONTROL_LAUNCHER.read_text()
    assert sum(cell[4] for cell in CONTROL_CELLS) == 7 * 3600


@pytest.fixture
def control_harness(harness, monkeypatch):
    run, record, code, work = harness
    monkeypatch.setattr(H, "LAUNCHER", CONTROL_LAUNCHER)
    wrapper = work.parent / "bin/python3"
    text = wrapper.read_text()
    assert text.count("scripts/scaling_evaluation49046.py") == 1
    wrapper.write_text(text.replace("scripts/scaling_evaluation49046.py", "scripts/scaling_control30m_evaluation.py"))
    plans = code.parent / "metadata"
    original = json.loads((plans / "plan.json").read_text())
    plan = {**original, "control_only": True, "arms": [E.ARMS[0]],
            "checkpoints": {E.ARMS[0]: original["checkpoints"][E.ARMS[0]]}}
    (plans / "control.json").write_text(json.dumps(plan))
    return run, record, code, work, plans, plan


@pytest.mark.parametrize("cell", CONTROL_CELLS)
def test_three_control_jobs_run_real_shell_guards_and_only_the_adapter(control_harness, cell):
    run, record, code, _, _, _ = control_harness
    result = run(cell=cell, plan_name="control.json")
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(record.read_text())
    argv = observed["argv"]
    args = dict(zip(argv[1::2], argv[2::2]))
    assert argv[0] == "scripts/scaling_control30m_evaluation.py"
    assert args["--arm"] == E.ARMS[0] and args["--step"] == "49046"
    assert args["--stage"] == cell[1] and args.get("--task", "none") == cell[2]
    assert args["--plan"] == str(code.parent / "metadata/control.json")
    assert args["--expected-code-sha"] == H.CODE and args["--device"] == "cuda"


MUTATIONS = [
    ("control_only", None), ("control_only", False), ("control_only", 1), ("control_only", "true"),
    ("arms", list(E.ARMS)), ("arms", [E.ARMS[1]]), ("arms", []),
    ("checkpoints", {}), ("checkpoints", {E.ARMS[1]: {}}),
    ("checkpoints", {arm: {} for arm in E.ARMS}),
    ("step", 19618), ("step", True), ("protocol", "unregistered"),
]


@pytest.mark.parametrize("key,value", MUTATIONS)
def test_control_launcher_rejects_implicit_or_extra_arms_before_extraction(control_harness, key, value):
    run, record, _, _, plans, plan = control_harness
    plan[key] = copy.deepcopy(value)
    (plans / "bad.json").write_text(json.dumps(plan))
    result = run(plan_name="bad.json")
    assert result.returncode != 0 and not record.exists()


def test_control_launcher_rejects_the_300m_arm_and_wrong_plan_hash(control_harness):
    run, record, _, _, _, _ = control_harness
    assert run(cell=H.CELLS[3], plan_name="control.json").returncode != 0
    assert not record.exists()
    assert run(plan_name="control.json", environment={"SCALING_EXTRACTION_PLAN_SHA": "f" * 64}).returncode != 0
    assert not record.exists()


class ReachedPinnedInputs(Exception):
    pass


@pytest.fixture
def context_plan(tmp_path, monkeypatch):
    path = ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261106/evaluation/extraction_plan.json"
    plan = json.loads(path.read_text())
    plan.update(protocol=E.PROTOCOL, step=E.STEP, control_only=True, arms=[E.ARMS[0]],
                control_checkpoint_sha256=E.CONTROL_CHECKPOINT_SHA256,
                checkpoints={E.ARMS[0]: {"sha256": E.CONTROL_CHECKPOINT_SHA256}})
    target = tmp_path / "plan.json"
    monkeypatch.setattr(E.P, "code_identity", lambda expected: H.CODE)

    def stop_at_pinned_inputs(pin):
        assert pin == plan["comparison"]
        raise ReachedPinnedInputs

    monkeypatch.setitem(A.load_context.__globals__, "pin_file", stop_at_pinned_inputs)
    return plan, target


def test_control_loader_requires_only_real_control_pin_then_continues_original_checks(context_plan):
    plan, path = context_plan
    path.write_text(json.dumps(plan))
    with pytest.raises(ReachedPinnedInputs):
        A.load_context(path, E.ARMS[0], E.STEP, H.CODE)


@pytest.mark.parametrize("key,value", MUTATIONS + [("control_checkpoint_sha256", "f" * 64)])
def test_control_loader_rejects_bad_shape_before_input_or_model_reads(context_plan, key, value):
    plan, path = context_plan
    plan[key] = copy.deepcopy(value)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        A.load_context(path, E.ARMS[0], E.STEP, H.CODE)


def test_control_loader_cannot_load_candidate_even_with_a_control_plan(context_plan):
    plan, path = context_plan
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="control-only"):
        A.load_context(path, E.ARMS[1], E.STEP, H.CODE)


@pytest.mark.parametrize("fails", [False, True])
def test_adapter_uses_original_execute_and_restores_its_loader_even_after_error(monkeypatch, tmp_path, fails):
    seen = []

    def original_execute(*args):
        assert E.load_context is A.load_context
        seen.append(args)
        if fails:
            raise RuntimeError("extraction failed")
        return {"receipt": "written by frozen execute"}

    original_loader = E.load_context
    monkeypatch.setattr(E, "execute", original_execute)
    args = ("extract-classification", tmp_path / "plan.json", E.ARMS[0], E.STEP,
            H.CODE, tmp_path / "output", "p1", "cpu")
    if fails:
        with pytest.raises(RuntimeError, match="extraction failed"):
            A.execute(*args)
    else:
        assert A.execute(*args) == {"receipt": "written by frozen execute"}
    assert seen == [args] and E.load_context is original_loader
    with pytest.raises(ValueError, match="cannot select"):
        A.execute(*args[:2], E.ARMS[1], *args[3:])
    assert len(seen) == 1 and E.load_context is original_loader


def test_adapter_refuses_a_previously_rebound_loader(monkeypatch, tmp_path):
    alien = lambda *a: None
    monkeypatch.setattr(E, "load_context", alien)
    with pytest.raises(ValueError, match="already rebound"):
        A.execute("validate", tmp_path / "plan.json", E.ARMS[0], E.STEP, H.CODE, tmp_path / "out")
    assert E.load_context is alien
