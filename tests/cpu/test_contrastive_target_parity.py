"""Independent mutations of the real-state gate plus real small CPU backwards."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import contrastive_target_parity as P
from contrastive_numeric_profile import numeric_profile
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader
from wearusfm.training.contrastive_memory import CandidateQueue, WindowIdentity
from wearusfm.training.pairing import state_sha256


def case():
    with numeric_profile(require_cuda=False) as flags:
        flags = {**flags, "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}
    return {"before": {key: "a" * 64 for key in P.MATCH_FIELDS},
            "after_rng": {key: "b" * 64 for key in ("cpu_rng_sha256", "cuda_rng_sha256", "python_rng_sha256", "numpy_rng_sha256")},
            "gradient_sha256": "c" * 64, "gradient_norm": 1., "loss": 1.,
            "nce_weight": 0., "rnc_weight": 0., "nce_eligible": 4., "rnc_windows": 3., "rnc_pairs": 20.,
            "global_windows": 32., "backward_seconds": .1, "flags": flags,
            "memory": {"cuda_measured": True, "peak_allocated_bytes": 100}}


def point(update=128):
    return {"update": update, "state_completed_steps": update - 1, "ranks": [
        {"rank": rank, "profile_status": "AVAILABLE", "global_batch_size": 32,
         "global_ids_sha256": "d" * 64, "live_before": case()["before"], "live_state_restored": True,
         "cases": {name: case() for name in P.CASES}}
        for rank in range(4)]}


def test_gate_requires_both_real_control_states_and_all_reduced_gradients():
    report = {"schema": 1, "profile": P.PROFILE, "points": [point(128), point(256)]}
    assert P.validate_parity_report(report)["status"] == "PASSED"
    report["points"][1]["ranks"][2]["cases"]["CNb"]["gradient_sha256"] = "f" * 64
    # Serialized gates cannot override evidence.
    report["status"] = "PASSED"
    assert P.validate_parity_report(report)["status"] == "FAILED"
    assert P.validate_parity_report({**report, "points": [point()]})["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("key", ["nce_eligible", "rnc_windows", "rnc_pairs"])
def test_missing_loss_support_is_inconclusive_not_negative_or_pass(key):
    record = point()
    record["ranks"][0]["cases"]["CRa"][key] = 0.
    result = P.assess_point(record)
    assert result["status"] == "INCONCLUSIVE"
    assert any("loss support" in why for why in result["inconclusive"])


def test_missing_profile_rank_or_repetition_is_inconclusive():
    record = point()
    record["ranks"][0]["cases"]["CNa"]["flags"]["math_sdp_enabled"] = False
    assert P.assess_point(record)["status"] == "INCONCLUSIVE"
    record = point()
    record["ranks"].pop()
    assert P.assess_point(record)["status"] == "INCONCLUSIVE"
    record = point()
    del record["ranks"][0]["cases"]["C0b"]
    assert P.assess_point(record)["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("field", P.MATCH_FIELDS)
def test_same_gradient_never_excuses_changed_live_state_or_mask(field):
    record = point()
    record["ranks"][2]["cases"]["CRb"]["before"][field] = "changed"
    assert P.assess_point(record)["status"] == "FAILED"


def test_consistent_replayed_state_must_also_equal_live_control():
    record = point()
    for row in record["ranks"]:
        for c in row["cases"].values():
            c["before"]["teacher_sha256"] = "invented initial teacher"
    assert P.assess_point(record)["status"] == "FAILED"


def test_byte_hash_distinguishes_signed_zero_gradients():
    assert state_sha256({"w": torch.tensor([0.])}) != state_sha256({"w": torch.tensor([-0.])})


@pytest.mark.parametrize("what", ["rng", "weight", "batch", "memory", "timing", "live"])
def test_gates_check_rng_weights_production_size_measurements_and_restoration(what):
    record = point()
    row = record["ranks"][0]
    c = row["cases"]["CNb"]
    if what == "rng": c["after_rng"]["cuda_rng_sha256"] = "different"
    elif what == "weight": c["nce_weight"] = .0001
    elif what == "batch": row["global_batch_size"] = 8
    elif what == "memory": c["memory"]["cuda_measured"] = False
    elif what == "timing": c["backward_seconds"] = float("nan")
    else: row["live_state_restored"] = False
    assert P.assess_point(record)["status"] == ("INCONCLUSIVE" if what == "memory" else "FAILED")


def test_real_cpu_replays_use_actual_distinct_teacher_and_restore_dropout_state(tmp_path):
    root, manifest, pose = _pose_setup(tmp_path)
    cfg = _config(pose)
    cfg = replace(cfg, model=replace(cfg.model, muscle_dropout=.4),
                  contrastive={**cfg.contrastive, "fill_steps": 256})
    loader = PretrainLoader(P.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
    batch = loader.batch(4, np.random.default_rng(42))
    micros = [P.trainer._prepare(batch, "cpu")]
    student = P.attach_contrastive_head(P.WearUsFM(cfg.model))
    teacher = P.make_teacher(student)
    # EMA and student genuinely differ: a fixture reconstruction from student would be detected.
    with torch.no_grad():
        next(teacher.parameters()).add_(.001)
    runtime = P.ContrastiveRuntime(cfg.contrastive)
    runtime.expected_datasets = ("emg2pose",)
    keydim = student.contrastive_head.layers[-1].out_features
    key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
    ident = WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)
    runtime.queue.commit(key, [ident], step=0)
    saved = runtime.state_dict()
    saved["queue"]["last_step"] = 126
    saved["queue"]["entries"][0]["step"] = 126
    initial_rng = P.rng_state(torch.device("cpu"))
    observations = []
    with numeric_profile(require_cuda=False):
        for name in P.CASES:
            measured = P._measure_case(name, cfg, student.state_dict(), teacher.state_dict(), saved, batch, micros,
                                      initial_rng, P.trainer.DistContext(), torch.device("cpu"),
                                      P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, require_cuda=False)
            assert measured["before"]["teacher_sha256"] == state_sha256(teacher.state_dict())
            assert measured["before"]["teacher_sha256"] != measured["before"]["student_sha256"]
            assert measured["nce_weight"] == measured["rnc_weight"] == 0.
            assert measured["nce_eligible"] > 0 and measured["rnc_pairs"] > 0
            observations.append(measured)
    assert len({r["gradient_sha256"] for r in observations}) == 1
    assert all(r["after_rng"] == observations[0]["after_rng"] for r in observations)
    assert all(r["before"] == observations[0]["before"] for r in observations)
    # CPU evidence is intentionally not sufficient to pass the production gate.
    assert all(r["memory"]["cuda_measured"] is False for r in observations)


def test_gradient_difference_without_required_support_is_inconclusive():
    record = point()
    record["ranks"][0]["cases"]["CNa"]["nce_eligible"] = 0.
    record["ranks"][0]["cases"]["CNa"]["gradient_sha256"] = "f" * 64
    assert P.assess_point(record)["status"] == "INCONCLUSIVE"


def test_missing_separate_rng_records_is_not_accepted():
    record = point()
    for row in record["ranks"]:
        for c in row["cases"].values():
            c["after_rng"] = {}
    assert P.assess_point(record)["status"] == "FAILED"


def test_observer_restores_live_generators_gradients_and_queue_after_six_probes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, manifest, pose = _pose_setup(tmp_path)
    cfg = replace(_config(pose), batch_size=32,
                  contrastive={**_config(pose).contrastive, "fill_steps": 256})
    loader = PretrainLoader(P.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
    batch = loader.batch(32, np.random.default_rng(42))
    batch.index = 127
    micros = [P.trainer._prepare(batch, "cpu")]
    student = P.attach_contrastive_head(P.WearUsFM(cfg.model))
    teacher = P.make_teacher(student)
    runtime = P.ContrastiveRuntime(cfg.contrastive)
    runtime.expected_datasets = ("emg2pose",)
    runtime.queue.last_step = 126
    runtime.prepare(batch, 127)
    observer = P.ControlParityObserver(tmp_path / "out")
    observer.captured[id(runtime)] = batch
    saved_rng = state_sha256(P.rng_state("cpu"))
    saved_queue = state_sha256(runtime.state_dict())
    called = []

    def mock_case(name, *args, **kwargs):
        # Deliberately consume all host generators; observer must restore them.
        torch.rand(1)
        np.random.random()
        P.random.random()
        called.append(name)
        return case()

    monkeypatch.setattr(P, "_measure_case", mock_case)
    monkeypatch.setattr(P, "assert_numeric_profile", lambda **kw: {})
    ctx = SimpleNamespace(world_size=4, active=False, rank=0)
    observer.measure(None, student, teacher, micros, cfg, ctx, "cpu", None,
                     P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, contrastive=runtime)
    assert called == list(P.CASES)
    assert state_sha256(P.rng_state("cpu")) == saved_rng
    assert state_sha256(runtime.state_dict()) == saved_queue
    assert observer.points[0]["ranks"][0]["live_state_restored"]
    assert all(p.grad is None for p in student.parameters())
    assert observer.points[0]["state_completed_steps"] == 127


def test_observer_does_not_instrument_other_updates_or_active_arms(tmp_path):
    observer = P.ControlParityObserver(tmp_path)
    for arm, step in (("CN", 127), ("CR", 255), ("C0", 128), ("C0", 254)):
        from types import SimpleNamespace
        runtime = SimpleNamespace(cfg=SimpleNamespace(arm=arm), step=step)
        observer.measure(None, None, None, None, None, None, None, None, None, None, contrastive=runtime)
    assert observer.points == []
