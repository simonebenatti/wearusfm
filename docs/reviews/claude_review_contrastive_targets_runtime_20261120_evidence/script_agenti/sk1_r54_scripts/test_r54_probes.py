"""Sonde negative dello scettico 1 (R5-4), fuori dalla copia: ogni guardia del rilievo, isolata.

Si eseguono contro i moduli del clone temporaneo (PYTHONPATH). Su codice non mutato devono passare tutte;
con la guardia corrispondente disattivata deve fallire la sonda relativa.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import test_contrastive_target_parity as T  # point(), case() e il modulo P del clone
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader
from wearusfm.training.pairing import state_sha256

P = T.P


# ---------- assess_point: identita' replicate fra rank (215-216) ----------
@pytest.mark.parametrize("field", ["student_sha256", "teacher_sha256", "queue_sha256", "coverage_sha256", "global_ids"])
def test_cross_rank_identity_only(field):
    record = T.point()
    row = record["ranks"][1]
    if field == "global_ids":
        row["global_ids_sha256"] = "e" * 64
    else:
        # coerente entro il rank: live_before e tutti i casi -> 178 e 189 non scattano
        row["live_before"][field] = "e" * 64
        for c in row["cases"].values():
            c["before"][field] = "e" * 64
    result = P.assess_point(record)
    assert result["status"] == "FAILED"
    assert result["errors"] == ["model/teacher/queue/global identities differ across ranks"]


# ---------- validate_parity_report: schema e profilo (227-228) ----------
@pytest.mark.parametrize("what", ["schema", "profile"])
def test_report_schema_profile(what):
    report = {"schema": 1, "profile": P.PROFILE, "points": [T.point(128), T.point(256)]}
    assert P.validate_parity_report(report)["status"] == "PASSED"
    if what == "schema":
        report["schema"] = 2
    else:
        report["profile"] = "bf16_auto"
    result = P.validate_parity_report(report)
    assert result["status"] == "INCONCLUSIVE"
    assert result["inconclusive"] == ["required parity report/profile missing"]


# ---------- osservatore: banco come nel test esistente ----------
def _bench(tmp_path):
    root, manifest, pose = _pose_setup(tmp_path)
    cfg = replace(_config(pose), batch_size=32, contrastive={**_config(pose).contrastive, "fill_steps": 256})
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
    ctx = SimpleNamespace(world_size=4, active=False, rank=0)
    return cfg, batch, micros, student, teacher, runtime, observer, ctx


def _measure(observer, cfg, student, teacher, micros, ctx, runtime):
    observer.measure(None, student, teacher, micros, cfg, ctx, "cpu", None,
                     P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, contrastive=runtime)


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(P, "assert_numeric_profile", lambda **kw: {})
    monkeypatch.setattr(P, "_measure_case", lambda name, *a, **k: T.case())


# 294 + 300-301: un probe che altera lo stato vivo deve essere rilevato e fermare
@pytest.mark.parametrize("what", ["param", "teacher", "grad", "queue"])
def test_observer_detects_live_modification(tmp_path, monkeypatch, what):
    cfg, batch, micros, student, teacher, runtime, observer, ctx = _bench(tmp_path)
    monkeypatch.setattr(P, "assert_numeric_profile", lambda **kw: {})

    def tamper(name, *a, **k):
        if name == "CRb":
            with torch.no_grad():
                if what == "param":
                    next(student.parameters()).add_(1e-3)
                elif what == "teacher":
                    next(teacher.parameters()).add_(1e-3)
                elif what == "grad":
                    p = next(student.parameters())
                    p.grad = torch.ones_like(p)
                else:
                    runtime.coverage["updates"] += 1
        return T.case()

    monkeypatch.setattr(P, "_measure_case", tamper)
    with pytest.raises(RuntimeError, match="parity probe modified the live control state"):
        _measure(observer, cfg, student, teacher, micros, ctx, runtime)
    assert observer.points[0]["ranks"][0]["live_state_restored"] is False
    assert observer.points[0]["gate"]["status"] == "FAILED"


# 262-263: batch.index
def test_observer_rejects_wrong_batch_index(tmp_path, quiet):
    cfg, batch, micros, student, teacher, runtime, observer, ctx = _bench(tmp_path)
    batch.index = 126
    with pytest.raises(ValueError, match="live batch is not immediately before"):
        _measure(observer, cfg, student, teacher, micros, ctx, runtime)


# 255-256: cfg di produzione
@pytest.mark.parametrize("what", ["batch16", "accum2", "fp32", "ws2"])
def test_observer_rejects_non_production_cfg(tmp_path, quiet, what):
    cfg, batch, micros, student, teacher, runtime, observer, ctx = _bench(tmp_path)
    if what == "batch16": cfg = replace(cfg, batch_size=16)
    elif what == "accum2": cfg = replace(cfg, accum_steps=2)
    elif what == "fp32": cfg = replace(cfg, amp_bf16=False)
    else: ctx = SimpleNamespace(world_size=2, active=False, rank=0)
    with pytest.raises(ValueError, match="parity requires production C0"):
        _measure(observer, cfg, student, teacher, micros, ctx, runtime)


# 252-253: cfg cambiato fra i punti
def test_observer_rejects_config_change_between_points(tmp_path, quiet):
    cfg, batch, micros, student, teacher, runtime, observer, ctx = _bench(tmp_path)
    _measure(observer, cfg, student, teacher, micros, ctx, runtime)
    assert [p["update"] for p in observer.points] == [128]
    # secondo punto prescritto (256): isola la guardia dal controllo del duplicato
    runtime.step, batch.index = 255, 255
    observer.captured[id(runtime)] = batch
    with pytest.raises(ValueError, match="configuration changed between prescribed points"):
        _measure(observer, replace(cfg, lr=2e-4), student, teacher, micros, ctx, runtime)


# 257-258: punto duplicato (stesso cfg, catturato di nuovo il batch)
def test_observer_rejects_duplicate_point(tmp_path, quiet):
    cfg, batch, micros, student, teacher, runtime, observer, ctx = _bench(tmp_path)
    _measure(observer, cfg, student, teacher, micros, ctx, runtime)
    observer.captured[id(runtime)] = batch
    with pytest.raises(ValueError, match="duplicate parity control update"):
        _measure(observer, cfg, student, teacher, micros, ctx, runtime)
