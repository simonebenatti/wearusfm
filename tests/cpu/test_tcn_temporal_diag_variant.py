"""Weight-decay exploration of the temporal control (D3): recipe, paired fit, frozen gates, dose rule, entry point.

Same small synthetic control and private test recipe as test_tcn_temporal_diag; no number here is a result.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(ROOT / sub))
from pose_temponet import controls as C, provenance as P, selection as S, train as T, workflow as W

import test_pose_temponet_workflow as WF
import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401  (module fixture of the first test file)
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry

RECIPE = TD.RECIPE


@pytest.fixture(scope="module")
def linear_runs(world):
    # WF.bundle() is deterministic: the same cache and folds as the temporal world of the first test file.
    linear = WF.control_bundle(WF.bundle(), world["artifacts"], "linear")
    return {"bundle": linear, "runs": {head: W._run_selection(linear, head, recipe=RECIPE) for head in W.CONTROL_HEADS}}


def test_variant_recipe_changes_only_the_weight_decay_and_cannot_enter_production():
    frozen = core.variant_recipe(0.3)
    assert {k for k, v in asdict(frozen).items() if v != asdict(T.Recipe())[k]} == {"weight_decay", "test_only"}
    assert frozen.weight_decay == 0.3 and frozen.initial_epochs == 150 and frozen.maximum_epochs == 300
    assert frozen.batch_size == 32 and frozen.last_epochs == 5 and frozen.clip_norm == 1.0
    with pytest.raises(ValueError):
        T._check_recipe(frozen, production=True)
    test = core.variant_recipe(1.0, RECIPE)
    assert {k for k, v in asdict(test).items() if v != asdict(RECIPE)[k]} == {"weight_decay"}
    assert core.VARIANT_DOSES == (0.1, 0.3, 1.0, 3.0)


def test_variant_fit_at_the_frozen_weight_decay_is_the_archived_curve_of_that_learning_rate(world):
    cb, selection, states, refs = TD._tcn(world)
    for curve in selection["curves"]:
        lr = curve["lr"]
        fit = core.variant_fit(cb, selection, refs, device="cpu", recipe=core.variant_recipe(RECIPE.weight_decay, RECIPE),
                               lr=lr, head=core.HEAD, fitting_exposures=(1, 3))
        ours = [{**q, "fold_steps": list(q["fold_steps"])} for q in fit["curve"]["points"]]
        assert ours == curve["points"] and fit["horizon"] == curve["horizon"] and fit["selected_reproduced"]
        assert fit["start_parity_level"] == "exact" and fit["start_parity"]["max_relative_difference"] == 0.0
        pair = states["candidate_states"][lr]
        for fold in (0, 1):
            assert fit["states"]["final_states"][fold]["model"].keys() == pair["current_states"][fold]["model"].keys()
            assert T.payload_sha256(fit["states"]["final_states"][fold]["model"]) == \
                T.payload_sha256(pair["current_states"][fold]["model"])
            assert T.payload_sha256(fit["states"]["best_states"][fold]["model"]) == \
                T.payload_sha256(pair["best_states"][fold]["model"])
    best = next(c for c in selection["curves"] if c["lr"] == selection["selected"]["lr"])
    fit = core.variant_fit(cb, selection, refs, device="cpu", recipe=core.variant_recipe(RECIPE.weight_decay, RECIPE),
                           lr=best["lr"], head=core.HEAD)
    assert fit["gate"]["recovery"] == pytest.approx(core.frozen_gate(fit["selected_summary"], cb)["recovery"], abs=0)
    archived = C.temporal_head_gate(cb.reference_gate["temporal_records"],
                                    {k: v["E_dyn"] for k, v in selection["selected_outer_values"].items()})
    assert fit["gate"]["recovery"] == archived["recovery"]


def test_variant_fit_changes_the_trajectory_only_through_the_weight_decay(world):
    cb, selection, _, refs = TD._tcn(world)
    lr = selection["selected"]["lr"]
    base = core.variant_fit(cb, selection, refs, device="cpu", recipe=core.variant_recipe(RECIPE.weight_decay, RECIPE),
                            lr=lr, head=core.HEAD, fitting_exposures=(1,))
    strong = core.variant_fit(cb, selection, refs, device="cpu", recipe=core.variant_recipe(3.0, RECIPE),
                              lr=lr, head=core.HEAD, fitting_exposures=(1,))
    assert strong["initial_sha256_per_fold"] == base["initial_sha256_per_fold"] == selection["initial_sha256_per_fold"]
    assert strong["training_set_sha256"] == base["training_set_sha256"]
    assert strong["curve"]["points"][0] == base["curve"]["points"][0]
    assert strong["start_parity_level"] == "exact"  # weight decay acts from the first update only
    assert strong["curve"]["points"][1:] != base["curve"]["points"][1:]
    assert strong["recipe"]["weight_decay"] == 3.0 and strong["weight_decay"] == 3.0
    # Fitting users are read at the declared exposures only; every checkpoint has the held users.
    full = [r for r in strong["records"] if "summary" in r]
    assert [r["point"]["exposure"] for r in full] == [1.0]
    assert all("held_recovery" in r for r in strong["records"] if "summary" not in r)
    assert len(strong["records"]) == len(strong["curve"]["points"])
    json.dumps(core.plain({k: v for k, v in strong.items() if k != "states"}), allow_nan=False)


def test_variant_fit_in_the_linear_world_uses_the_frozen_linear_gate(linear_runs):
    linear = linear_runs["bundle"]
    for head in W.CONTROL_HEADS:
        selection = linear_runs["runs"][head]["selection"]
        lr = selection["selected"]["lr"]
        fit = core.variant_fit(linear, selection, None, device="cpu",
                               recipe=core.variant_recipe(RECIPE.weight_decay, RECIPE), lr=lr, head=head)
        assert fit["world"] == "linear" and fit["selected_summary"] is None
        ours = [{**q, "fold_steps": list(q["fold_steps"])} for q in fit["curve"]["points"]]
        assert ours == next(c for c in selection["curves"] if c["lr"] == lr)["points"]
        values = {k: {"MSE": v["MSE"], "E_dyn": v["E_dyn"]} for k, v in selection["selected_outer_values"].items()}
        expected = C.linear_head_gate(linear.reference_gate["linear_records"], values)
        assert fit["gate"]["recovery"] == expected["recovery"] and fit["gate"]["status"] == expected["status"]
    with pytest.raises(ValueError, match="another cell"):
        core.variant_fit(linear, linear_runs["runs"]["L0"]["selection"], None, device="cpu",
                         recipe=core.variant_recipe(0.3, RECIPE), lr=0.001, head="MLP")


def _gates(temporal, linear=("passed", "passed", "passed"), late=False, valid=True):
    return {"temporal": {"recovery": dict(zip(core.PANELS, temporal))},
            "linear": {head: {"status": status} for head, status in zip(W.CONTROL_HEADS, linear)},
            "late": late, "valid": valid}


def test_dose_rule_is_the_pre_registered_one():
    results = {0.1: _gates((.30, .31, .305)), 0.3: _gates((.52, .55, .53)), 1.0: _gates((.60, .58, .59),
               linear=("passed", "recipe_not_admitted", "passed")), 3.0: _gates((.40, .41, .405))}
    decision = core.choose_dose(results)
    assert decision["chosen"] == 0.3 and decision["confirm"] is True
    results[0.3] = _gates((.52, .49, .51))
    decision = core.choose_dose(results)
    assert decision["chosen"] == 0.3 and decision["confirm"] is False  # one panel below the frozen 0.50
    pair = (0.3, 1.0)
    tie = {0.3: _gates((.4, .4, .4)), 1.0: _gates((.4, .4, .4))}
    assert core.choose_dose(tie, doses=pair)["chosen"] == 0.3
    none = {0.3: _gates((.9, .9, .9), linear=("recipe_not_admitted",) * 3), 1.0: _gates((.9, .9, .9), valid=False)}
    assert core.choose_dose(none, doses=pair) == {"rows": core.choose_dose(none, doses=pair)["rows"], "chosen": None,
                                                  "confirm": False}
    nan = {0.3: _gates((.9, float("nan"), .9)), 1.0: _gates((.6, .6, .6))}
    assert core.choose_dose(nan, doses=pair)["chosen"] == 0.3 and core.choose_dose(nan, doses=pair)["confirm"] is False
    # Declared on purpose (8.3): only the chosen dose opens the confirmation, even if another admissible dose passes.
    other = {0.1: _gates((.60, .49, .55)), 0.3: _gates((.52, .51, .515)), 1.0: _gates((.2, .2, .2)),
             3.0: _gates((.1, .1, .1))}
    assert core.choose_dose(other) == {"rows": core.choose_dose(other)["rows"], "chosen": 0.1, "confirm": False}
    exact = {0.1: _gates((.50, .70, .60)), 0.3: _gates((.4, .4, .4)), 1.0: _gates((.3, .3, .3)), 3.0: _gates((.2, .2, .2))}
    assert core.choose_dose(exact)["confirm"] is True  # exactly the frozen 0.50 reaches the threshold
    with pytest.raises(ValueError, match="registered doses"):
        core.choose_dose(tie)  # two doses where four are registered
    assert C.TEMPORAL_HEAD_RECOVERY == 0.5


def _variant_context(world, linear_runs):
    context = TD._context(world)
    context.update(folds=None, linear_bundle=linear_runs["bundle"],
                   linear_controls={head: {"receipt": {"job_id": head}, "selection": linear_runs["runs"][head]["selection"]}
                                    for head in W.CONTROL_HEADS})
    return context


def test_exploration_stages_write_every_dose_and_need_a_completed_d0(world, linear_runs, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setattr(entry, "setup_variant", lambda: _variant_context(world, linear_runs))
    monkeypatch.setattr(core, "VARIANT_DOSES", (0.1, 1.0))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d3t"])
    with pytest.raises(FileNotFoundError):
        entry.main(["d3t"])
    root = tmp_path.resolve() / "diag2"
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(root))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    results = {}
    for stage, cells in entry.D3_CELLS.items():
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        assert entry.main([stage]) == 0
        receipt = json.loads((root / "receipts" / (entry.JOBS[stage] + ".json")).read_text())
        assert receipt["status"] == "completed" and all(receipt["gates"].values())
        summary = json.loads((root / stage / "variant_summary.json").read_text())
        assert summary["protocol"] == entry.PHASE2_PROTOCOL and len(summary["rows"]) == 2 * len(cells)
        for head, world_name in cells:
            for dose in (0.1, 1.0):
                key = "%s_%s_wd%s" % (world_name, head, repr(dose))
                data = json.loads((root / stage / (key + ".json")).read_text())
                assert data["weight_decay"] == dose and data["head"] == head and data["world"] == world_name
                assert (root / stage / (key + "_states.pt")).exists() == (head == core.HEAD)
                assert data["start_parity_level"] == "exact"
                r = results.setdefault(dose, {"linear": {}, "valid": True})
                r["valid"] = r["valid"] and data["selected_reproduced"] and data["start_parity_level"] != "failed"
                r.update({"temporal": data["gate"]} if world_name == "temporal" else {})
                if world_name == "linear":
                    results[dose]["linear"][head] = data["gate"]
        with pytest.raises(ValueError, match="do not duplicate or retry"):
            entry.main([stage])
    decision = core.choose_dose(results)
    assert [row["dose"] for row in decision["rows"]] == [0.1, 1.0] and isinstance(decision["confirm"], bool)
    assert all(set(row["linear_status"]) == set(W.CONTROL_HEADS) for row in decision["rows"])
