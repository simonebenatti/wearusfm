"""Fifteen-epoch campaign: fresh fits, provenance, fixed dose and no held leakage."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_finetune15 as C
from wearusfm.harness import finetuning as F


def result(epochs=15, best=3):
    return {"stopped": "fixed_budget", "val_bacc": .8, "best_epoch": best,
            "updates": epochs, "planned_updates": epochs,
            "history": [{"epoch": i, "step": i, "val_bacc": .8} for i in range(1, epochs + 1)]}


def test_complete_dose_guard_rejects_five_epoch_or_truncated_reports():
    roles = {"train": np.arange(8)}
    plan = {"epochs": 15, "batch": 8}
    C.require_full_dose(result(), roles, plan)
    with pytest.raises(ValueError, match="dose"):
        C.require_full_dose(result(5), roles, plan)
    incomplete = result()
    incomplete["history"].pop(4)
    with pytest.raises(ValueError, match="complete epochs"):
        C.require_full_dose(incomplete, roles, plan)
    incomplete = result(best=16)
    with pytest.raises(ValueError, match="selected epoch"):
        C.require_full_dose(incomplete, roles, plan)


def test_candidates_start_fresh_and_receive_full_fifteen_epoch_budget(tmp_path, monkeypatch):
    initial = {"teacher": "unchanged-19618"}
    created, received = [], []
    def build(state, regime, classes, seed):
        assert state is initial and seed == 0
        model = torch.nn.Linear(2, classes)
        created.append(model)
        return model
    def fit(model, *args, **kwargs):
        assert kwargs["epochs"] == 15 and kwargs["batch"] == 8 and kwargs["seed"] == 0
        assert kwargs.get("max_updates") is None
        received.append(model)
        return result()
    monkeypatch.setattr(F, "build_classifier", build)
    monkeypatch.setattr(F, "fit_candidate", fit)
    args = (initial, None, None, {"train": np.arange(8)}, None, np.array(["rest", "move"]), "finetune")
    plan = {"seed": 0, "batch": 8, "epochs": 15}
    for number in (0, 1):
        C.run_candidate(*args, number, plan, "cpu", tmp_path, {"protocol": C.PROTOCOL})
    assert created == received and created[0] is not created[1]
    assert initial == {"teacher": "unchanged-19618"}
    for number in (0, 1):
        saved = json.loads((tmp_path / f"candidate_{number}.json").read_text())
        assert saved["complete"] and len(saved["history"]) == 15
        assert saved["head_lr"] == pytest.approx((1e-3, 1e-2)[number])


def test_divergence_is_persisted_but_oom_remains_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "build_classifier", lambda *a, **k: torch.nn.Linear(2, 2))
    def diverge(*a, **k):
        raise F.NumericalDivergence("loss", step=7, epoch=8, history=result(7)["history"])
    monkeypatch.setattr(F, "fit_candidate", diverge)
    args = ({}, None, None, {"train": np.arange(8)}, None, np.array(["a", "b"]), "scratch")
    plan = {"seed": 0, "batch": 8, "epochs": 15}
    saved = C.run_candidate(*args, 0, plan, "cpu", tmp_path, {})
    assert saved["stopped"] == "nonfinite" and saved["val_bacc"] is None and len(saved["history"]) == 7
    assert not (tmp_path / "candidate_0.pt").exists()
    def oom(*a, **k):
        raise torch.OutOfMemoryError("not a scientific candidate outcome")
    monkeypatch.setattr(F, "fit_candidate", oom)
    with pytest.raises(torch.OutOfMemoryError):
        C.run_candidate(*args, 1, plan, "cpu", tmp_path, {})
    assert not (tmp_path / "candidate_1.json").exists()


def test_prior_candidate_rejects_five_epoch_provenance_and_changed_state(tmp_path):
    first, second = tmp_path / "candidate0", tmp_path / "candidate1"
    first.mkdir()
    provenance = {"protocol": C.PROTOCOL, "fold": 0, "regime": "frozen", "recipe_sha256": "fifteen"}
    checkpoint = first / "candidate_0.pt"
    checkpoint.write_bytes(b"selected-model")
    candidate = {**result(), "candidate": 0, "complete": True, "provenance": provenance,
                 "checkpoint": str(checkpoint), "checkpoint_sha256": C.T.R.sha256_file(checkpoint)}
    path = first / "candidate_0.json"
    path.write_text(json.dumps(candidate))
    receipt = {"complete": True, "candidate_only": 0, "candidate_provenance": provenance,
               "candidate_file_sha256": {"0": C.T.R.sha256_file(path)}}
    (first / "report.json").write_text(json.dumps(receipt))
    assert C.prior_candidate(second, provenance)["best_epoch"] == 3
    with pytest.raises(ValueError, match="wrong first"):
        C.prior_candidate(second, {**provenance, "protocol": "consolidation-finetune-epn-20261104-v1"})
    checkpoint.write_bytes(b"modified-model")
    with pytest.raises(ValueError, match="checkpoint changed"):
        C.prior_candidate(second, provenance)


def test_source_identity_covers_runtime_helpers_and_launcher(monkeypatch):
    files = []
    def identity(path):
        files.append(Path(path).relative_to(ROOT).as_posix())
        return "a" * 64
    monkeypatch.setattr(C.T.R, "sha256_file", identity)
    C.source_identity()
    assert {"scripts/consolidation_finetune15.py", "scripts/calibration_probe.py", "scripts/tokenizer_train.py",
            "scripts/tokenizer_preflight.py", "scripts/slurm/consolidation_finetune15.sbatch",
            "scripts/slurm/lib_code_dir.sh", "src/wearusfm/harness/finetuning.py"} <= set(files)
    assert "scripts/consolidation_finetune.py" in files


def test_selection_is_validation_only_and_ties_favor_first_candidate():
    candidates = [{**result(), "candidate": number, "held_bacc": held}
                  for number, held in ((1, 1.), (0, .1))]
    assert C.select_candidate(candidates)["candidate"] == 0
    assert not hasattr(C, "scratch_still_improving")


def test_launcher_rejects_invalid_regime_before_cluster_setup(tmp_path):
    launcher = ROOT / "scripts/slurm/consolidation_finetune15.sbatch"
    failed = subprocess.run(["bash", str(launcher), "fit", str(tmp_path / "out"), "0", "unknown", "0"],
                            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert failed.returncode == 2 and "unknown fit regime" in failed.stderr
    assert "WORK" not in failed.stderr


def published_plan():
    return json.loads((ROOT / 'results/infra/finetune15_20261105/plan.json').read_text())


def test_published_recipe_and_operational_policy_are_explicit():
    plan = published_plan()
    assert C.recipe_identity(plan) == plan['recipe_sha256']
    assert plan['epochs'] == 15 and plan['reading_rule15'] == C.READING_RULE15
    assert C.fit_resource_policy(plan) == {'frozen': 21300, 'finetune': 57300, 'scratch': 57300}
    changed = copy.deepcopy(plan)
    changed['epochs'] = 5
    assert C.recipe_identity(changed) != plan['recipe_sha256']
    changed = copy.deepcopy(plan)
    changed['fit_packaging_policy']['mode'] = 'joint'
    with pytest.raises(ValueError, match='packaging policy'):
        C.fit_resource_policy(changed)
    changed = copy.deepcopy(plan)
    changed['resource_projection_policy']['elapsed_multiplier'] = 1.
    with pytest.raises(ValueError, match='projection policy'):
        C.fit_resource_policy(changed)


def test_reference_verifies_all_twentyfour_fits_without_opening_oof(monkeypatch):
    def no_predictions(*args, **kwargs):
        raise AssertionError('held predictions must not enter the resource gate')
    monkeypatch.setattr(np, 'load', no_predictions)
    reference = C.reference_five(published_plan())
    assert reference['max_elapsed_seconds_by_regime'] == {'frozen': 4519, 'finetune': 13359, 'scratch': 13446}
    monkeypatch.setattr(C.FIVE, 'source_identity', lambda: 'changed-source')
    with pytest.raises(ValueError, match='source equivalence'):
        C.reference_five(published_plan())


def test_complete_job_projection_never_discounts_faster_microbenchmark():
    reference = C.reference_five(published_plan())
    measurements = copy.deepcopy(reference['benchmark'])
    expected = {'frozen': 17547, 'finetune': 50697, 'scratch': 51023}
    projection = C.resource_projection(measurements, reference)
    assert projection['projected_one_candidate_seconds'] == expected
    assert projection['additional_fixed_seconds'] == 0
    for timing in measurements['timings'].values():
        timing['seconds'] *= .5
        timing['seconds_per_update_steady'] *= .5
    faster = C.resource_projection(measurements, reference)
    assert faster['projected_one_candidate_seconds'] == expected
    for timing in measurements['timings'].values():
        timing['seconds'] *= 4
        timing['seconds_per_update_steady'] *= 4
    measurements['load_seconds'] += 20
    measurements['projected_full_preprocessing_seconds'] += 10
    slower = C.resource_projection(measurements, reference)
    assert slower['additional_fixed_seconds'] == pytest.approx(40.)
    assert slower['slowdown_by_regime'] == dict.fromkeys(F.REGIMES, 2.)
    assert slower['projected_one_candidate_seconds']['scratch'] > 57300
    measurements['timings']['frozen']['updates'] = 16
    with pytest.raises(ValueError, match='benchmark timing'):
        C.resource_projection(measurements, reference)


def benchmark_fixture(plan, reference):
    gate = copy.deepcopy(reference['benchmark'])
    gate.update(protocol=C.PROTOCOL, source_sha256='fifteen-controller', recipe_sha256=plan['recipe_sha256'],
                fit_allocation_by_regime=plan['fit_allocation_by_regime'],
                fit_packaging_policy=plan['fit_packaging_policy'],
                resource_projection_policy=plan['resource_projection_policy'],
                reference_five=plan['reference_five'], reference_source_sha256=plan['reference_source_sha256'],
                fit_limit_seconds_by_regime=C.fit_resource_policy(plan))
    gate['resource_projection'] = C.resource_projection(gate, reference)
    gate['fit_packaging'] = dict.fromkeys(F.REGIMES, 'split')
    return gate


def test_gate_recomputes_rule_and_rejects_five_epoch_receipt_or_changed_limits():
    plan = published_plan()
    reference = C.reference_five(plan)
    gate = benchmark_fixture(plan, reference)
    assert C.benchmark_packaging(gate, plan, 'fifteen-controller', reference) == dict.fromkeys(F.REGIMES, 'split')
    with pytest.raises(ValueError, match='provenance'):
        C.benchmark_packaging(reference['benchmark'], plan, 'fifteen-controller', reference)
    bad = copy.deepcopy(gate)
    bad['resource_projection']['projected_one_candidate_seconds']['scratch'] -= 1
    with pytest.raises(ValueError, match='projection differs'):
        C.benchmark_packaging(bad, plan, 'fifteen-controller', reference)
    bad = copy.deepcopy(gate)
    bad['fit_limit_seconds_by_regime']['scratch'] += 1
    with pytest.raises(ValueError, match='resource policy'):
        C.benchmark_packaging(bad, plan, 'fifteen-controller', reference)
    bad = copy.deepcopy(gate)
    bad['resource_gate_passed'] = False
    with pytest.raises(ValueError, match='memory gate'):
        C.benchmark_packaging(bad, plan, 'fifteen-controller', reference)
    # Slower new hardware may stop a regime; it never triggers a dose reduction.
    for timing in gate['timings'].values():
        timing['seconds'] *= 2
        timing['seconds_per_update_steady'] *= 2
    gate['resource_projection'] = C.resource_projection(gate, reference)
    gate['fit_packaging'] = dict.fromkeys(F.REGIMES, 'blocked')
    assert C.benchmark_packaging(gate, plan, 'fifteen-controller', reference) == dict.fromkeys(F.REGIMES, 'blocked')
