"""Riproduzione indipendente di R1-N3 (solo lettura della copia; output in tmp)."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import contrastive_target_parity as P
import test_contrastive_target_parity as TP
from contrastive_numeric_profile import NumericProfileUnavailable
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader

WORK = Path(sys.argv[1])


def readout_reason(assessment):
    # Replica testuale di contrastive_benchmark_readout.py:517-521 (fbfde797).
    status = assessment if isinstance(assessment, str) else assessment.get("status")
    if status == "INCONCLUSIVE":
        return "inconclusive: production zero-weight comparison lacks required profile/support"
    if status != "PASSED":
        return "failure: production zero-weight gradients/invariants not byte-identical"
    return "passed"


def show(label, mutate):
    rec = TP.point(128)
    mutate(rec)
    res = P.assess_point(copy.deepcopy(rec))
    full = P.validate_parity_report({"schema": 1, "profile": P.PROFILE, "points": [rec, TP.point(256)]})
    print(f"[A] {label}: point={res['status']} errors={res['errors']} inconclusive={res['inconclusive']} "
          f"byte_identical={res['within_and_between_byte_identical']} report={full['status']} -> {readout_reason(full)}")


print("=== Parte A: campi assenti ===")
show("controllo integro", lambda r: None)
show("after_rng assente (chiave tolta) in rank0/CNb", lambda r: r["ranks"][0]["cases"]["CNb"].pop("after_rng"))
show("after_rng={} ovunque (test esistente)", lambda r: [c.__setitem__("after_rng", {}) for row in r["ranks"] for c in row["cases"].values()])
show("live_state_restored assente in rank0", lambda r: r["ranks"][0].pop("live_state_restored"))
show("global_batch_size assente in rank0", lambda r: r["ranks"][0].pop("global_batch_size"))
show("[confronto] flags assenti in rank0/CNa", lambda r: r["ranks"][0]["cases"]["CNa"].pop("flags"))
show("[confronto] nce_eligible assente in rank0/CRa", lambda r: r["ranks"][0]["cases"]["CRa"].pop("nce_eligible"))
show("[confronto] memory assente in rank0/CRa", lambda r: r["ranks"][0]["cases"]["CRa"].pop("memory"))
show("[confronto] rank3 profile UNAVAILABLE", lambda r: r["ranks"][3].update(profile_status="UNAVAILABLE", cases={}))

print("=== Parte B: profilo assente nel probe (riga 121, _measure_case reale su CPU) ===")
root, manifest, pose = _pose_setup(WORK)
cfg = replace(_config(pose), batch_size=32, contrastive={**_config(pose).contrastive, "fill_steps": 256})
print("cfg.amp_bf16 =", cfg.amp_bf16, "accum_steps =", cfg.accum_steps)
loader = PretrainLoader(P.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
batch = loader.batch(32, np.random.default_rng(42))
batch.index = 127
micros = [P.trainer._prepare(batch, "cpu")]
student = P.attach_contrastive_head(P.WearUsFM(cfg.model))
teacher = P.make_teacher(student)
runtime = P.ContrastiveRuntime(cfg.contrastive)
runtime.expected_datasets = ("emg2pose",)
import torch
from wearusfm.training.contrastive_memory import WindowIdentity
keydim = student.contrastive_head.layers[-1].out_features
key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
runtime.queue.commit(key, [WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)], step=0)
qstate = runtime.queue.state_dict()
qstate["last_step"] = 126
qstate["entries"][0]["step"] = 126
runtime.queue.load_state_dict(qstate)
runtime.prepare(batch, 127)
print("runtime.step =", runtime.step, "queue.last_step =", runtime.queue.last_step)
observer = P.ControlParityObserver(WORK / "out")
observer.captured[id(runtime)] = batch
real_assert = P.assert_numeric_profile
calls = []


def selective(**kw):
    calls.append(dict(kw))
    if "require_autocast" in kw:
        # Riga 121: verifica REALE (sul Mac manca CUDA -> NumericProfileUnavailable autentica).
        return real_assert(**kw)
    return {}  # Riga 280 simulata come superata, come in produzione.


P.assert_numeric_profile = selective
ctx = SimpleNamespace(world_size=4, active=False, rank=0)
try:
    observer.measure(None, student, teacher, micros, cfg, ctx, "cpu", None,
                     P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, contrastive=runtime)
    print("[B] measure terminato senza eccezione; punti:", len(observer.points))
except NumericProfileUnavailable as exc:
    print("[B] measure PROPAGA NumericProfileUnavailable:", exc)
    print("[B] chiamate:", calls)
    print("[B] punti registrati:", len(observer.points), "| out_dir esiste:", (WORK / "out").exists())
finally:
    P.assert_numeric_profile = real_assert

print("=== Parte B2: eccezione dentro observe_control -> finish() non eseguito ===")
out2 = WORK / "out2"
try:
    with P.observe_control(out2):
        raise NumericProfileUnavailable("simulata dal corpo del trainer")
except NumericProfileUnavailable as exc:
    print("[B2] propagata:", exc, "| out2 esiste:", out2.exists(),
          "| zero_weight_parity.json:", (out2 / "zero_weight_parity.json").exists())
print("[B2] patch rimosse:", P.trainer._accumulate_terms.__name__, P.ContrastiveRuntime.prepare.__name__)
