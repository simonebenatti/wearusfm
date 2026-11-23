"""D3 rows and rule: the rows that summarize.py and the driver read are the per-cell results, every production dose runs,
summarize.d3 applies the registered rule to exactly those rows, late and step-zero parity are the frozen notions, and the
production recipe flips to the variant without changing a number.

Adapted from the logic review of b521122e (23/11); same fixtures as test_tcn_temporal_diag_variant. No number is a result.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(ROOT / sub))
from pose_temponet import selection as S, workflow as W

import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401  (module fixture of the first test file)
from test_tcn_temporal_diag_variant import _variant_context, linear_runs  # noqa: F401
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry


def _summarize():
    spec = importlib.util.spec_from_file_location(
        "tcnd_summarize", ROOT / "results/infra/tcn_temporal_diag_20261122/summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(stage, monkeypatch):
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
    return entry.main([stage])


def test_rows_equal_the_cell_files_for_every_production_dose(world, linear_runs, tmp_path, monkeypatch, capsys):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setattr(entry, "setup_variant", lambda: _variant_context(world, linear_runs))
    assert _run("d0", monkeypatch) == 0
    expected = {}
    for stage, cells in entry.D3_CELLS.items():
        assert _run(stage, monkeypatch) == 0
        receipt = json.loads((root / "receipts" / (entry.JOBS[stage] + ".json")).read_text())
        assert receipt["protocol"] == entry.PHASE2_PROTOCOL and receipt["campaign"] == entry.CAMPAIGN_D3
        assert receipt["linear_bundle_sha256"] == linear_runs["bundle"].manifest["bundle_sha256"]
        summary = json.loads((root / stage / "variant_summary.json").read_text())
        assert summary["doses"] == list(core.VARIANT_DOSES) == [0.1, 0.3, 1.0, 3.0]
        assert summary["gates"]["start_points_reproduced"] is True
        assert len(summary["rows"]) == len(cells) * len(core.VARIANT_DOSES)
        assert {(r["head"], r["world"]) for r in summary["rows"].values()} == set(cells)
        assert {r["weight_decay"] for r in summary["rows"].values()} == set(core.VARIANT_DOSES)
        for key, row in summary["rows"].items():
            cell = json.loads((root / stage / (key + ".json")).read_text())
            assert key == "%s_%s_wd%s" % (cell["world"], cell["head"], repr(cell["weight_decay"]))
            assert (row["head"], row["world"], row["weight_decay"]) == (cell["head"], cell["world"], cell["weight_decay"])
            assert cell["recipe"]["weight_decay"] == cell["weight_decay"]
            assert row["gate"] == cell["gate"] and row["selected"] == cell["selected"] and row["late"] == cell["late"]
            assert row["reproduced"] is cell["selected_reproduced"]
            assert row["start_parity_level"] == cell["start_parity_level"] == "exact" and row["start_reproduced"] is True
            r = expected.setdefault(cell["weight_decay"], {"linear": {}, "valid": True})
            r["valid"] = r["valid"] and cell["selected_reproduced"] and cell["start_parity_level"] != "failed"
            if cell["world"] == "temporal":
                r["temporal"], r["late"] = cell["gate"], cell["late"]
            else:
                r["linear"][cell["head"]] = cell["gate"]
    decision = core.choose_dose(expected)
    assert decision == entry.d3_decision(root, {})
    capsys.readouterr()
    _summarize().d3(root)
    assert "dose scelta %s, conferma su un mondo nuovo: %s" % (decision["chosen"], decision["confirm"]) in \
        capsys.readouterr().out


def test_a_selected_point_that_does_not_reproduce_reaches_the_rows(world, linear_runs, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setattr(entry, "setup_variant", lambda: _variant_context(world, linear_runs))
    monkeypatch.setattr(core, "VARIANT_DOSES", (0.3,))
    assert _run("d0", monkeypatch) == 0
    original = core.score_splits

    def drifting(model, bundle, fold, device, depth=core.DEPTH):
        scores = original(model, bundle, fold, device, depth)
        scores["held"]["observed"] = {**scores["held"]["observed"],
                                      "mse_sum": scores["held"]["observed"]["mse_sum"] * (1 + 1e-9)}
        return scores

    monkeypatch.setattr(core, "score_splits", drifting)
    assert _run("d3s", monkeypatch) == 3
    summary = json.loads((root / "d3s" / "variant_summary.json").read_text())
    assert summary["gates"]["selected_points_reproduced"] is False and summary["gates"]["start_points_reproduced"] is True
    assert [row["reproduced"] for row in summary["rows"].values()] == [False, False]


def test_step_zero_parity_has_the_two_registered_tiers(world):
    _, selection, _, _ = TD._tcn(world)
    lr = selection["selected"]["lr"]
    archived = next(c for c in selection["curves"] if c["lr"] == lr)
    curve = S.curve_from_dict(archived)
    assert core.parity_level([core.start_point_parity(curve, selection, lr)]) == "exact"

    def level(change):
        moved = copy.deepcopy(selection)
        change(next(c for c in moved["curves"] if c["lr"] == lr)["points"][0])
        return core.parity_level([core.start_point_parity(curve, moved, lr)])

    assert level(lambda p: p.update(mse=p["mse"] * (1 + 1e-6))) == "numerical"
    assert level(lambda p: p.update(edyn=p["edyn"] * (1 + 1e-3))) == "failed"
    assert level(lambda p: p.update(fold_steps=[1, 0])) == "failed"
    with pytest.raises(ValueError, match="archived curve"):
        core.start_point_parity(curve, selection, 0.123)


def _gate(recovery, status="passed"):
    return {"recovery": recovery, "status": status}


def _write_rows(root, temporal, linear_reproduced=None, missing=(), statuses=None):
    """Crafted variant summaries: temporal[dose] = (fold0, fold1, pooled); every linear gate passes at 0.95.

    Each stage also gets a receipt with ``statuses[stage]`` (completed by default), as collected from Leonardo.
    """
    linear_reproduced = linear_reproduced or {}
    statuses = statuses or {}
    rows = {"d3t": {}, "d3l": {}, "d3s": {}}
    for dose, values in temporal.items():
        rec = dict(zip(core.PANELS, values))
        rows["d3t"]["temporal_TCN_post_wd%r" % dose] = {
            "head": "TCN_post", "world": "temporal", "weight_decay": dose, "gate": _gate(rec, "recipe_not_admitted"),
            "selected": {"exposure": 31.0}, "late": False, "extended": False, "reproduced": True,
            "start_parity_level": "exact", "start_reproduced": True}
        for stage, head in (("d3l", "TCN_post"), ("d3s", "L0"), ("d3s", "MLP")):
            if (dose, head) in missing:
                continue
            panel = {name: {"MSE": 0.95, "E_dyn": 0.95} for name in core.PANELS}
            rows[stage]["linear_%s_wd%r" % (head, dose)] = {
                "head": head, "world": "linear", "weight_decay": dose, "gate": _gate(panel), "selected": {},
                "late": False, "extended": False, "reproduced": linear_reproduced.get((dose, head), True),
                "start_parity_level": "exact", "start_reproduced": True}
    (root / "receipts").mkdir(parents=True)
    for stage, content in rows.items():
        (root / stage).mkdir(parents=True)
        (root / stage / "variant_summary.json").write_text(json.dumps({"rows": content}))
        (root / "receipts" / (entry.JOBS[stage] + ".json")).write_text(
            json.dumps({"job_id": entry.JOBS[stage], "status": statuses.get(stage, "completed")}))


def test_summarize_applies_the_registered_rule_to_the_collected_rows(tmp_path, capsys):
    four = {0.1: (.30, .31, .305), 0.3: (.60, .62, .61), 1.0: (.55, .56, .555), 3.0: (.20, .21, .205)}
    # The best dose loses admissibility through one linear cell that does not reproduce its selected point.
    _write_rows(tmp_path / "a", four, linear_reproduced={(0.3, "MLP"): False})
    _summarize().d3(tmp_path / "a")
    out = capsys.readouterr().out
    assert "wd 0.3 (valida False)" in out and "dose scelta 1.0, conferma su un mondo nuovo: True" in out
    assert "non riprodotti: linear MLP: punto scelto" in out
    # A missing cell makes its dose invalid; the rule chooses among the complete doses.
    _write_rows(tmp_path / "b", four, missing={(3.0, "L0")})
    _summarize().d3(tmp_path / "b")
    out = capsys.readouterr().out
    assert "wd 3 (valida False)" in out and "dose scelta 0.3, conferma su un mondo nuovo: True" in out
    # A whole stage missing (a job that did not finish): no dose is admissible and nothing is confirmed.
    _write_rows(tmp_path / "c", four)
    (tmp_path / "c" / "d3s" / "variant_summary.json").unlink()
    _summarize().d3(tmp_path / "c")
    assert "dose scelta None, conferma su un mondo nuovo: False" in capsys.readouterr().out
    # Exactly 0.50 in a panel reaches the frozen threshold.
    _write_rows(tmp_path / "d", {0.1: (.50, .70, .60), 0.3: (.4, .4, .4), 1.0: (.3, .3, .3), 3.0: (.2, .2, .2)})
    _summarize().d3(tmp_path / "d")
    assert "dose scelta 0.1, conferma su un mondo nuovo: True" in capsys.readouterr().out
    # A job that closed gate_failed still wrote its summary: like the driver, summarize.py does not count it.
    _write_rows(tmp_path / "e", four, statuses={"d3l": "gate_failed"})
    _summarize().d3(tmp_path / "e")
    out = capsys.readouterr().out
    assert "sintesi escluse, job non completati: d3l (gate_failed)" in out
    assert "dose scelta None, conferma su un mondo nuovo: False" in out


def test_late_flag_is_the_frozen_alarm_of_the_same_curve(world):
    cb, selection, _, refs = TD._tcn(world)
    flags = []
    for curve in selection["curves"]:
        fit = core.variant_fit(cb, selection, refs, device="cpu", lr=curve["lr"], head=core.HEAD,
                               recipe=core.variant_recipe(TD.RECIPE.weight_decay, TD.RECIPE))
        alarms = S.assessment([S.curve_from_dict(curve)], maximum_epochs=TD.RECIPE.maximum_epochs,
                              last_epochs=TD.RECIPE.last_epochs)["optimizer_alarms"]
        assert fit["late"] is ("minimum_in_last_five_maximum_epochs" in alarms)
        assert fit["extended_by_frozen_rule"] is (curve["horizon"] == TD.RECIPE.maximum_epochs)
        flags.append(fit["late"])
    assert True in flags and False in flags


def test_frozen_weight_decay_variant_trains_bit_for_bit_like_the_frozen_recipe():
    """The production flip test_only False -> True changes no number (the archived-curve test uses a test base)."""
    import torch
    from pose_temponet import heads, train as T
    g = torch.Generator().manual_seed(1)
    n = 70
    data = T.TensorTrainingSet(torch.randn(n, 160, 384, generator=g).half(), torch.randn(n, 160, 20, generator=g),
                               torch.rand(n, 148, 20, generator=g) > 0.2, ["w%03d" % i for i in range(n)])
    for head in ("L0", "MLP", "TCN_post"):
        runs = []
        for recipe in (T.Recipe(), core.variant_recipe(0.01), core.variant_recipe(0.3)):
            t = T.Trajectory(lambda seed, fold: heads.make_head(head, seed=seed), data, lr=0.003, seed=0, fold=0,
                             device="cpu", _recipe=recipe)
            t.train_to_step(7)
            runs.append(t)
        frozen, same, strong = runs
        params = [m.model.state_dict().values() for m in runs]
        assert all(torch.equal(a, b) for a, b in zip(params[0], params[1])) and frozen.epoch_losses == same.epoch_losses
        assert not all(torch.equal(a, b) for a, b in zip(params[0], params[2]))
