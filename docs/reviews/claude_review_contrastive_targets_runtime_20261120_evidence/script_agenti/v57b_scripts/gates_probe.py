"""Probe R5-7 (verifier 2): first execution of the two production gates that no test runs,
contrastive_release.validate_cuda_prerequisite and contrastive_run.validate_target_fork, on
fixtures shaped like their documented producers (CUDA report as in the repo's _cuda_evidence
test, rank-exit receipt from the repo's fake_receipt, a 544 fork as validate_fork reads it)."""
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TMP = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests/cpu")]
import torch
import contrastive_release as C
import contrastive_run as RUN
from contrastive_prepare import build_review_configs, BASELINE, _read, source_digest
from contrastive_numeric_profile import numeric_profile
from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
from test_contrastive_rank_exit import fake_receipt
from test_contrastive_runtime import _config
from wearusfm.training import run as R


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attempt(label, fn):
    try:
        value = fn()
        print(label, "OK")
        return value
    except Exception as error:
        print(label, type(error).__name__, str(error)[:150])


TMP.mkdir(parents=True, exist_ok=False)
source = source_digest()
cohort = {"source_sha256": source, "initial_state_sha256": "i" * 64, "calibration_sha256": "c" * 64}
inputs = {k: k[0] * 64 for k in ("manifest", "scales", "window_seconds", "pose_stats")}
calrun = TMP / "calibration"
calrun.mkdir()
(calrun / "checkpoint.pt").write_bytes(b"calibration checkpoint bytes")
rank_exit = fake_receipt(TMP / "rank_exit")  # repo fixture; source "a"*64
report_path = TMP / "rank_exit/report.json"
data = json.loads(report_path.read_text()); data["source_sha256"] = source; report_path.write_text(json.dumps(data))
keys = ("total", "pose", "contrast_nce", "contrast_rnc", "contrast_global_grad_base", "contrast_global_grad_nce",
        "contrast_global_grad_rnc", "gradient", "student", "teacher")
precisions = {}
for name in ("fp32", "bf16"):
    limits = {k: (.01 if name == "bf16" or k in ("student", "teacher") else 1e-5) for k in keys}
    precisions[name] = {"passed": True, "updates": [{"update": s, "passed": True, "errors": dict.fromkeys(keys, 0.),
                                                     "limits": limits} for s in (1, 2)]}
cal_cfg = build_review_configs(_read(BASELINE))["calibration"]
cuda = {"passed": True, "source_sha256": source, "initial_sha256": cohort["initial_state_sha256"],
        "protocol": PROTOCOL, "criteria": CRITERIA, "technical_overrides": TECHNICAL_OVERRIDES, "world_size": 4,
        "inputs_unchanged": True, "inputs": {**inputs, "calibration_checkpoint": C.file_sha(calrun / "checkpoint.pt")},
        "pose_stats_sha256": inputs["pose_stats"], "config_canonical_sha256": canonical(cal_cfg),
        "rank_exit_report_path": str(report_path), "rank_exit_report_sha256": C.file_sha(report_path),
        "ranks": [{"rank": r, "passed": True, "stop_checks": {"coverage": True, "numeric": True, "passed": True},
                   "arms": {"CN": copy.deepcopy(precisions), "CR": copy.deepcopy(precisions)}} for r in range(4)]}
with numeric_profile(require_cuda=False) as flags:
    for row in cuda["ranks"]:
        row["numeric_profile"] = flags
cuda_path = TMP / "cuda.json"
cuda_path.write_text(json.dumps(cuda))
spec = {"path": str(cuda_path), "sha256": C.file_sha(cuda_path), "bytes": cuda_path.stat().st_size}
attempt("validate_cuda_prerequisite positive", lambda: C.validate_cuda_prerequisite(spec, cohort, inputs, calrun))
attempt("validate_cuda_prerequisite wrong calibration dir (must refuse)",
        lambda: C.validate_cuda_prerequisite(spec, cohort, inputs, TMP / "rank_exit"))
attempt("validate_cuda_prerequisite changed pin (must refuse)",
        lambda: C.validate_cuda_prerequisite({**spec, "sha256": "0" * 64}, cohort, inputs, calrun))

# validate_target_fork: one complete continuous576 folder with the named 544 save.
from contrastive_prepare import config_from_dict
base = config_from_dict(_read(ROOT / "results/infra/contrastive_targets_20261118/review_configs/CN.config.json"))
cfg = replace(base, max_steps=576, schedule=replace(base.schedule, keep_checkpoints=(544, 5000, 19618)))
print("CN review config: max_steps", base.max_steps, "keep", base.schedule.keep_checkpoints, "arm", base.contrastive["arm"])
full = TMP / "full_CN"
full.mkdir()
release = {"arm": "CN", "cohort_sha256": "b" * 64, "inputs": inputs}
provenance = {"source_sha256": source, "calibration_sha256": cohort["calibration_sha256"],
              "cohort_sha256": release["cohort_sha256"], "input_sha256": inputs}
state = {"step": 544, "world_size": 4, "config": R.config_to_dict(cfg),
         "contrastive_initial_sha256": cohort["initial_state_sha256"], "contrastive_provenance": provenance,
         "contrastive_state": {"queue": {"last_step": 543}},
         "rank_rng": [{"torch_rng": torch.tensor([r]), "cuda_rng": torch.tensor([r])} for r in range(4)]}
torch.save(state, full / "checkpoint_step0000544.pt")
summary = {"steps": 576, "stopped": "max_steps", "world_size": 4,
           "contrastive": {"config": {"arm": "CN"}, "initial_sha256": cohort["initial_state_sha256"], "provenance": provenance}}
(full / "summary.json").write_text(json.dumps(summary))
release["fork_checkpoint"] = C.pin(full / "checkpoint_step0000544.pt")
release["continuous_summary"] = C.pin(full / "summary.json")
attempt("validate_target_fork positive", lambda: RUN.validate_target_fork(release, cfg, cohort))
bad = copy.deepcopy(release); bad["cohort_sha256"] = "d" * 64
attempt("validate_target_fork other cohort (must refuse)", lambda: RUN.validate_target_fork(bad, cfg, cohort))
attempt("validate_target_fork changed dose (must refuse)",
        lambda: RUN.validate_target_fork(release, replace(cfg, max_steps=544), cohort))
