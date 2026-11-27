"""Output roots of the diagnostics after 27/11 (Simone: «le prossime cose salvale su scratch»).

D0-D2 keep the root they used on $WORK; every later stage writes under the scratch root, reads from the old root only the D0
receipt, and writes a small marker first so that an unusable root stops the job before any fit. Fixtures of the other
diagnostic tests; no number is a result.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(ROOT / sub))

import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401  (module fixture of the first test file)
from test_tcn_temporal_diag_variant import _variant_context, linear_runs  # noqa: F401
from test_tcn_temporal_diag_d3_rows import _run
from test_tcn_temporal_diag_confirmation import _patch_b
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _under(pin, root):
    return pin["path"].startswith(str(root) + "/")


def test_exploration_writes_under_the_new_root_and_only_reads_the_d0_receipt_of_phase_one(world, linear_runs, tmp_path,
                                                                                         monkeypatch):
    old, new = tmp_path.resolve() / "work", tmp_path.resolve() / "scratch"
    TD._patch_entry(monkeypatch, world, old)
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(new))  # phase 1 stays under `old`, as on Leonardo
    monkeypatch.setattr(entry, "setup_variant", lambda: _variant_context(world, linear_runs))
    monkeypatch.setattr(core, "VARIANT_DOSES", (0.3,))
    assert entry.stage_root("d0") == old == entry.stage_root("d2") and entry.stage_root("d3t") == new == entry.stage_root("bg")
    assert _run("d0", monkeypatch) == 0
    before = _tree(old)
    assert not new.exists() and not (old / "started").exists()  # a phase-1 stage writes no marker and nothing elsewhere
    assert _run("d3s", monkeypatch) == 0
    assert _tree(old) == before  # the exploration adds nothing under the old root
    receipt = json.loads((new / "receipts" / (entry.JOBS["d3s"] + ".json")).read_text())
    assert receipt["output_root"] == str(new) and receipt["status"] == "completed"
    assert receipt["predecessors"][entry.JOBS["d0"]]["path"] == str(old / "receipts" / (entry.JOBS["d0"] + ".json"))
    assert receipt["artifacts"] and all(_under(pin, new) for pin in receipt["artifacts"].values())
    started = receipt["details"]["started"]
    assert started["path"] == str(new / "started" / (entry.JOBS["d3s"] + ".json"))
    marker = json.loads(Path(started["path"]).read_text())
    assert marker["job_id"] == entry.JOBS["d3s"] and marker["stage"] == "d3s" and marker["output_root"] == str(new)
    assert (new / "d3s" / "variant_summary.json").exists()
    d0 = json.loads((old / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert d0["output_root"] == str(old) and "started" not in d0["details"]


def test_confirmation_stages_write_under_the_new_root_too(world, tmp_path, monkeypatch):
    old, new = tmp_path.resolve() / "work", tmp_path.resolve() / "scratch"
    _patch_b(monkeypatch, world, old, {"chosen": 0.3, "confirm": True, "rows": []})
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(new))
    assert _run("d0", monkeypatch) == 0
    before = _tree(old)
    stage = next(iter(entry.B_CONTROLS))
    for name in (*entry.B_CALIBRATIONS, stage):
        assert _run(name, monkeypatch) == 0
        receipt = json.loads((new / "receipts" / (entry.JOBS[name] + ".json")).read_text())
        assert receipt["output_root"] == str(new) and all(_under(pin, new) for pin in receipt["artifacts"].values())
        assert not _under(receipt["predecessors"][entry.JOBS["d0"]], new)
    assert all(_under(receipt["predecessors"][entry.JOBS[c]], new) for c in entry.B_CALIBRATIONS)
    assert _tree(old) == before


def test_an_unusable_root_stops_a_later_stage_before_any_setup(world, linear_runs, tmp_path, monkeypatch):
    old, new = tmp_path.resolve() / "work", tmp_path.resolve() / "scratch"
    TD._patch_entry(monkeypatch, world, old)
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(new))
    assert _run("d0", monkeypatch) == 0
    new.write_text("a file where the output root should be")
    monkeypatch.setattr(entry, "setup_variant", lambda: pytest.fail("no setup when the root cannot be written"))
    monkeypatch.setattr(entry, "setup", lambda: pytest.fail("no setup when the root cannot be written"))
    with pytest.raises(ValueError, match="non-directory path component"):
        _run("d3t", monkeypatch)
    assert new.is_file()
