"""The 19,618 evaluation chain is exactly the declared derivation of the reviewed 5,000 chain.

The frozen 5,000 files are read, never written. Every 19,618 file must equal its source with
the replacements declared in scripts/scaling_derive19618.py, each matching its exact count.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_derive19618 as D

REVIEWED = "97125bd8bb8b14590958501ba91986c8456037f9"  # delivery covered by Review 10
FROZEN = "aef665ba7ea66fc80291846487907bb3ef9107df"  # frozen 5,000 reader code
BASE = ROOT / "results/infra/scaling_300m_20261104"
CODE = sorted(t for t in D.DERIVATIONS if t.startswith("scripts/"))
TESTS = sorted(t for t in D.DERIVATIONS if t.startswith("tests/"))
LEFTOVERS = ("import scaling_evaluation as", "import scaling_probe as", "import scaling_pose_readout as",
             "import scaling_reading as", "import scaling_compare\n", "import scaling_compare;",
             "scripts/scaling_evaluation.py", "scripts/scaling_probe.py", "scripts/scaling_pose_readout.py",
             "scripts/scaling_compare.py", "checkpoint_step0005000", "STEP = 5000", "== 5000", "!= 5000",
             "scaling-evaluation-20261104-v1", "scaling-training-20261104-v1", "scaling-pose-readout-20261104-v1",
             "scaling-probe-20261104-v1", "scaling-probe-resources-20261104-v1", "scaling-pose-resources-20261104-v1",
             "scaling-comparison-5000-20261104-v1", "scaling-comparison-inputs-20261104-v1",
             "aef665ba7ea66fc80291846487907bb3ef9107df")


def test_every_target_is_exactly_its_declared_derivation():
    assert D.check() == []
    assert len(CODE) == 7 and len(TESTS) == 8 and len(D.DERIVATIONS) == 15
    assert D.main(["--check"]) == 0


@pytest.mark.parametrize("target", sorted(D.DERIVATIONS))
def test_frozen_source_is_the_reviewed_file_and_is_never_a_target(target):
    source, pinned, replacements = D.DERIVATIONS[target]
    reviewed = subprocess.check_output(["git", "show", REVIEWED + ":" + source], cwd=ROOT)
    assert hashlib.sha256(reviewed).hexdigest() == pinned == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    assert source not in D.DERIVATIONS and "19618" not in Path(source).name and "19618" in Path(target).name
    assert replacements and all(len(entry) in (2, 3) for entry in replacements)
    if "recovery" not in source:  # the recovery launcher and its test were added after the frozen reader code
        assert subprocess.check_output(["git", "show", FROZEN + ":" + source], cwd=ROOT) == reviewed


def copy_sources_and_targets(root):
    for target, (source, _, _) in D.DERIVATIONS.items():
        for name in (source, target):
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    return root


def test_edited_or_missing_target_and_changed_frozen_source_are_reported(tmp_path):
    root = copy_sources_and_targets(tmp_path)
    assert D.check(root) == []
    target = "scripts/scaling_probe19618.py"
    (root / target).write_text((root / target).read_text() + "# edited by hand\n")
    assert D.check(root) == ["derived file missing or different: " + target]
    (root / target).unlink()
    assert D.check(root) == ["derived file missing or different: " + target]
    D.write(root)
    assert D.check(root) == []
    source = D.DERIVATIONS[target][0]
    before = (ROOT / source).read_bytes()
    (root / source).write_bytes(before + b"\n")
    assert D.check(root) == ["frozen source changed: " + source]
    with pytest.raises(ValueError, match="frozen source changed"):
        D.write(root)
    assert (ROOT / source).read_bytes() == before  # only the copy was touched


def test_replacement_must_match_exactly_its_declared_count(monkeypatch):
    target = "scripts/scaling_reading19618.py"
    source, pinned, _ = D.DERIVATIONS[target]
    for bad in ([("text that is absent", "x")], [("require(", "need(")], [("PROTOCOL", "PROTOCOL")],
                [("PROTOCOL", "NAME", 1)], [("", "x")]):
        monkeypatch.setitem(D.DERIVATIONS, target, (source, pinned, bad))
        with pytest.raises(ValueError, match="replacement 0 expected"):
            D.derive(target)
    with pytest.raises(ValueError, match="unregistered"):
        D.derive(source)


@pytest.mark.parametrize("target", CODE)
def test_derived_code_keeps_no_5000_module_dose_protocol_or_code_pin(target):
    text = (ROOT / target).read_text()
    assert [name for name in LEFTOVERS if name in text] == []
    assert "scaling_derive19618.py" in text or target.endswith(".sbatch")


def test_protocols_and_dose_differ_while_recipes_grids_and_shared_modules_do_not():
    import scaling_compare as C5
    import scaling_compare19618 as C19
    import scaling_continue
    pairs = [(C5.PROTOCOL, C19.PROTOCOL), (C5.INPUT_PROTOCOL, C19.INPUT_PROTOCOL), (C5.C.PROTOCOL, C19.C.PROTOCOL),
             (C5.C.RESOURCE_PROTOCOL, C19.C.RESOURCE_PROTOCOL), (C5.P.PROTOCOL, C19.P.PROTOCOL),
             (C5.C.E.PROTOCOL, C19.C.E.PROTOCOL), (C5.C.E.TRAIN_PROTOCOL, C19.C.E.TRAIN_PROTOCOL),
             (C5.S8.PROTOCOL, C19.S8.PROTOCOL)]
    assert all(old != new and "19618" not in old for old, new in pairs)
    assert all("19618" in new for _, new in pairs if new != scaling_continue.PROTOCOL)
    assert len({name for pair in pairs for name in pair}) == 16
    assert C19.C.E.TRAIN_PROTOCOL == scaling_continue.PROTOCOL
    assert (C5.C.STEP, C5.P.STEP, C5.C.E.STEP) == (5000, 5000, 5000)
    assert (C19.STEP, C19.C.STEP, C19.P.STEP, C19.C.E.STEP, C19.C.E.START) == (19618, 19618, 19618, 19618, 5000)
    assert (scaling_continue.START, scaling_continue.END) == (C19.C.E.START, C19.C.E.STEP)
    # The fixed projection and the training source contract have no dose: one file for both chains.
    assert C19.SP is C5.SP and C19.ST is C5.ST and C19.C.SP is C5.C.SP and C19.C.T is C5.C.T and C19.P.Q is C5.P.Q
    old, new = C5.C.recipe(), C19.C.recipe()  # classification recipe: only the dose field differs
    assert {key: (old.get(key), new.get(key)) for key in set(old) | set(new) if old.get(key) != new.get(key)} == {
        "checkpoint_step": (5000, 19618)}
    assert C19.P.recipe() == C5.P.recipe()  # pose recipe has no dose field
    assert C19.C.CELLS == C5.C.CELLS and C19.P.CELLS == C5.P.CELLS and C19.C.TASKS == C5.C.TASKS
    assert (C19.C.C_GRID, C19.C.MAX_ITER, C19.C.ALIASES) == (C5.C.C_GRID, C5.C.MAX_ITER, C5.C.ALIASES)
    assert C19.C.E.EXTRACTION_SOURCE_SHA256 == C5.C.E.EXTRACTION_SOURCE_SHA256 and C19.C.E.ARMS == C5.C.E.ARMS
    assert (C19.PRIMARY, C19.CONTROL, C19.CANDIDATE) == (C5.PRIMARY, C5.CONTROL, C5.CANDIDATE)


def test_registered_control_is_the_one_of_the_budget_freeze_and_of_the_training_plan():
    import scaling_evaluation19618 as E
    freeze = json.loads((BASE / "continuation19618_budget_freeze_20261105_v2.json").read_text())
    plan = json.loads((BASE / "continuation19618_20261106/training_plan.json").read_text())
    assert E.CONTROL_CHECKPOINT_SHA256 == freeze["control19618"]["checkpoint_sha256"] == plan["control"]["checkpoint"]["sha256"]
    assert freeze["control19618"]["historical_arm_name"] == "T100-G" and freeze["control19618"]["arm"] == E.ARMS[0]
    assert (freeze["start_step"], freeze["end_step"], freeze["additional_steps"]) == (E.START, E.STEP, E.STEP - E.START)
    assert freeze["extractions"]["new_30m_extractions_included"] is False and freeze["training"]["new_30m_training_included"] is False
