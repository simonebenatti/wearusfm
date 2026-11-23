"""Confirmation on a new world (B): seeded generator, calibration, control searches, frozen gate, entry point.

The fixtures are the small synthetic controls of test_tcn_temporal_diag; no number here is a result.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(ROOT / sub))
from pose_temponet import controls as C, provenance as P, runner as R, train as T, workflow as W

import test_pose_temponet_workflow as WF
import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry

RECIPE = TD.RECIPE


def small_inputs(n=8):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n, 160, 384)).astype(np.float16)
    meta = [{"user": "user%d" % (i % 4), "window": i} for i in range(n)]
    support = np.ones((n, 148, 20), dtype=bool)
    support[1, 40:60, 3:] = False
    fit = np.array([i for i in range(n) if i % 4 in (0, 2)])
    held = np.array([i for i in range(n) if i % 4 in (1, 3)])
    return x, meta, support, fit, held, ["user0", "user1", "user2", "user3"]


def _same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_seeded_generator_is_the_frozen_generator_at_the_frozen_seed():
    x, meta, support, fit, held, users = small_inputs()
    for fold in (0, 1):
        frozen = C.generate_components(x, meta, support, fit, held, users, outer_fold=fold, block_windows=3)
        ours = core.generate_components_seeded(x, meta, support, fit, held, users, outer_fold=fold,
                                               base_seed=20261118 + 1000 * fold, block_windows=3)
        assert _same(frozen, ours)
        other = core.generate_components_seeded(x, meta, support, fit, held, users, outer_fold=fold,
                                                base_seed=core.new_world_seed(fold), block_windows=3)
        assert other["seed"] == core.new_world_seed(fold) == 20261122 + 1000 * fold
        assert not np.array_equal(other["T"], frozen["T"]) and not np.array_equal(other["epsilon"], frozen["epsilon"])
        assert other["generator_hashes"] != frozen["generator_hashes"]
        np.testing.assert_array_equal(other["epsilon"], C.window_noise(core.new_world_seed(fold) + 2, meta, 160, 20))
        assert other["partition"] == frozen["partition"]
    with pytest.raises(ValueError):
        core.new_world_seed(2)
    with pytest.raises(ValueError, match="base seed"):
        core.generate_components_seeded(x, meta, support, fit, held, users, outer_fold=0, base_seed=1.5, block_windows=3)


def test_new_world_calibration_is_the_frozen_calibration_with_seeded_components(monkeypatch):
    cache = WF.cache()
    b = WF.bundle(cache)
    captured = {}

    def fake(data, fit, held, users, *, outer_fold, block_windows, components):
        captured.update(data=data, fit=fit, held=held, users=users, fold=outer_fold, block=block_windows,
                        components=components)
        return {"outer_fold": outer_fold, "components": components}

    monkeypatch.setattr(C, "build_positive_control_fold", fake)
    fit, held = b.fitting[1], b.held[1]
    result = core.calibrate_new_world(cache, fit, held, b.manifest["users"], 1, block_windows=2)
    assert captured["fold"] == 1 and captured["block"] == 2 and captured["data"] is cache
    assert captured["components"]["seed"] == core.new_world_seed(1) and result["outer_fold"] == 1
    features, real, _, support = C._data_arrays(cache)
    expected = core.generate_components_seeded(features, cache["meta"], support, fit, held, b.manifest["users"],
                                               outer_fold=1, base_seed=core.new_world_seed(1), block_windows=2)
    assert _same(captured["components"], expected)


def _row(dose, world, head, recovery, status="passed", reproduced=True):
    gate = ({"recovery": dict(zip(core.PANELS, recovery)), "status": "x"} if world == "temporal"
            else {"status": status, "recovery": {}})
    return {"weight_decay": dose, "world": world, "head": head, "gate": gate, "reproduced": reproduced,
            "late": False, "selected": {"exposure": 31.0}}


def test_d3_results_mark_incomplete_or_unreproduced_doses_invalid():
    rows = {}
    for dose in core.VARIANT_DOSES:
        rows["t%s" % dose] = _row(dose, "temporal", "TCN_post", (.6, .6, .6) if dose == 1.0 else (.3, .3, .3))
        for head in W.CONTROL_HEADS:
            rows["l%s%s" % (dose, head)] = _row(dose, "linear", head, None)
    decision = core.choose_dose(core.d3_results(rows))
    assert decision["chosen"] == 1.0 and decision["confirm"] is True
    missing = {k: v for k, v in rows.items() if k != "l1.0MLP"}
    decision = core.choose_dose(core.d3_results(missing))
    assert decision["chosen"] != 1.0 and decision["confirm"] is False
    assert next(r for r in decision["rows"] if r["dose"] == 1.0)["valid"] is False
    broken = copy.deepcopy(rows)
    broken["t1.0"]["reproduced"] = False
    assert core.choose_dose(core.d3_results(broken))["confirm"] is False
    absent = {k: v for k, v in rows.items() if not k.startswith("t3.0")}
    results = core.d3_results(absent)
    assert results[3.0]["valid"] is False and results[3.0]["temporal"]["recovery"]["pooled"] is None


def _new_world_artifacts(world_fixture):
    artifacts = copy.deepcopy(world_fixture["artifacts"])
    for fold, artifact in enumerate(artifacts):
        artifact["components"]["seed"] = core.new_world_seed(fold)
    return artifacts


def _patch_b(monkeypatch, world, root, decision):
    TD._patch_entry(monkeypatch, world, root)
    artifacts = _new_world_artifacts(world)
    cache = world["bundle"].cache
    calibration_context = {"plan": {}, "plan_pin": {}, "code": {"code_sha": "c" * 40}, "numeric": {"kind": "cpu"},
                           "pins": {}, "cache": cache, "folds": None, "users": world["bundle"].manifest["users"]}
    monkeypatch.setattr(entry, "allocation_cpu", lambda cpus: {"slurm_job_id": "2", "scontrol": "test"})
    monkeypatch.setattr(entry, "setup_calibration", lambda: calibration_context)
    monkeypatch.setattr(entry.D, "alternating_user_folds", lambda data, expected=None: world["bundle"].manifest["user_folds"])
    monkeypatch.setattr(core, "calibrate_new_world",
                        lambda cache, fit, held, users, fold, block_windows: artifacts[fold])
    context = TD._context(world)
    context["plan"] = {"producers": {"train": {"users": world["bundle"].manifest["users"]}}}
    context["folds"] = None
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setattr(entry.R, "_input_identity", lambda plan: world["bundle"].manifest["source_identity"])
    monkeypatch.setattr(entry.W, "prepare_control_folds",
                        lambda cache, arts, w, **kw: W._prepare_control_folds(cache, arts, w, expected_user_count=4, **kw))
    monkeypatch.setattr(entry, "d3_decision", lambda root_, predecessors: decision)
    monkeypatch.setattr(entry, "reviewed_plan", lambda: ({}, {}, {"code_sha": "c" * 40}))
    monkeypatch.setattr(entry, "RECIPE", RECIPE)
    return artifacts


def test_confirmation_chain_writes_calibrations_six_searches_and_the_frozen_gate(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    artifacts = _patch_b(monkeypatch, world, root, {"chosen": 0.3, "confirm": True, "rows": []})
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    first = next(iter(entry.B_CONTROLS))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[first])
    with pytest.raises(FileNotFoundError):
        entry.main([first])  # no completed calibration yet: stops before any search
    root = tmp_path.resolve() / "diag2"
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(root))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    for stage in entry.B_CALIBRATIONS:
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        assert entry.main([stage]) == 0
        receipt = json.loads((root / "receipts" / (entry.JOBS[stage] + ".json")).read_text())
        assert receipt["protocol"] == entry.PHASE3_PROTOCOL and receipt["gates"] == {"calibrated": True, "new_world_seed": True}
        saved = torch.load(root / stage / "calibration.pt", map_location="cpu", weights_only=False)
        assert saved["components"]["seed"] == core.new_world_seed(entry.B_CALIBRATIONS[stage])
        assert "components" not in json.loads((root / stage / "calibration.json").read_text())
    for stage, (world_name, head) in entry.B_CONTROLS.items():
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        assert entry.main([stage]) == 0
        receipt = json.loads((root / "receipts" / (entry.JOBS[stage] + ".json")).read_text())
        assert receipt["status"] == "completed" and set(receipt["artifacts"]) == {"selection", "states", "decision"}
        assert set(receipt["predecessors"]) >= {entry.JOBS["bc0"], entry.JOBS["bc1"]}
        selection = json.loads((root / stage / "selection.json").read_text())
        assert selection["world"] == world_name and selection["head"] == head
        assert selection["recipe"]["weight_decay"] == 0.3 and selection["recipe"]["test_only"] is True
        assert selection["bundle"]["artifacts_sha256"] == W._identity(sorted(artifacts, key=lambda a: a["outer_fold"]))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[entry.B_GATE])
    assert entry.main([entry.B_GATE]) == 0
    gate = json.loads((root / entry.B_GATE / "gate.json").read_text())
    assert gate["complete"] is True and gate["status"] in ("passed", "recipe_not_admitted", "optimization_unresolved")
    assert set(gate["linear_head_gates"]) == set(W.CONTROL_HEADS) and "recovery" in gate["temporal_head_gate"]
    selections = {w: {h: json.loads((root / s / "selection.json").read_text()) for s, (w2, h) in entry.B_CONTROLS.items()
                      if w2 == w} for w in W.WORLDS}
    expected = W.qualify_controls(artifacts, selections)
    assert gate["gate_sha256"] == expected["gate_sha256"] and gate["status"] == expected["status"]
    receipt = json.loads((root / "receipts" / (entry.JOBS[entry.B_GATE] + ".json")).read_text())
    assert set(receipt["predecessors"]) >= {entry.JOBS[s] for s in entry.B_CONTROLS}
    with pytest.raises(ValueError, match="do not duplicate or retry"):
        entry.main([entry.B_GATE])


def test_confirmation_searches_stop_when_the_rule_does_not_propose_it(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_b(monkeypatch, world, root, {"chosen": 0.3, "confirm": False, "rows": []})
    for stage in ("d0", *entry.B_CALIBRATIONS):
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        assert entry.main([stage]) == 0
    stage = next(iter(entry.B_CONTROLS))
    monkeypatch.setattr(entry.W, "_run_selection", lambda *a, **k: pytest.fail("no search without the rule"))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
    assert entry.main([stage]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS[stage] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and "does not propose" in receipt["details"]["error"]


def test_d3_decision_reads_the_authenticated_summaries(world, tmp_path, monkeypatch):
    import test_tcn_temporal_diag_variant as TV
    lin = WF.control_bundle(WF.bundle(), world["artifacts"], "linear")
    linear = {"bundle": lin, "runs": {h: W._run_selection(lin, h, recipe=RECIPE) for h in W.CONTROL_HEADS}}
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setattr(entry, "setup_variant", lambda: TV._variant_context(world, linear))
    monkeypatch.setattr(core, "VARIANT_DOSES", (0.1, 1.0))
    for stage in ("d0", *entry.D3_CELLS):
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        assert entry.main([stage]) == 0
    predecessors = {}
    decision = entry.d3_decision(root, predecessors)
    assert set(predecessors) == {entry.JOBS[s] for s in entry.D3_CELLS}
    assert [row["dose"] for row in decision["rows"]] == [0.1, 1.0]
    rows = {}
    for stage in entry.D3_CELLS:
        rows.update(json.loads((root / stage / "variant_summary.json").read_text())["rows"])
    assert decision == core.choose_dose(core.d3_results(rows, doses=(0.1, 1.0)))
