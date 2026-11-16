"""One conditional P1 re-extraction with audited output finalization, no submissions.

The existing 19,618 context, numerical extractor, cache validator and immutable
input guards are reused unchanged. Only newly produced output files use the
bounded timestamp-finalization policy. Historical FAILED jobs are never adopted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_evaluation19618 as V
import scaling_output_identity as O

PROTOCOL = "scaling-p1-recovery-20261116-v1"
CPU_TEST_PROTOCOL = "scaling-p1-recovery-cpu-tests-20261116-v1"
DIAGNOSTIC_PROTOCOL = "SCALING-P1-IDENTITY-DIAG-v1"
ARM, STEP, TASK = "T100-G-300M", 19618, "p1"
LOGICAL_ID = "SCALING-300M-eval19618-p1-r2-20261116"
FAILED_ATTEMPTS = [{"job_id": "64572000", "state": "FAILED", "exit_code": "1:0"},
                   {"job_id": "64596915", "state": "FAILED", "exit_code": "1:0"}]
OUTPUT = "/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M/p1-r2"
ALLOCATION = {"nodes": 1, "gpus": 1, "cpus": 8, "memory_gib": 100, "walltime_seconds": 7200}
BUDGET = {"p1_cumulative_gpu_hours_cap": 3.10, "extractions_gpu_hours_cap": 7.30,
          "p1_max_attempts": 3, "retry_authorized": False}
SOURCE_PATHS = ("scripts/scaling_p1_recovery.py", "scripts/scaling_p1_recovery_plans.py",
                "scripts/scaling_output_identity.py", "tests/cpu/test_scaling_p1_recovery.py",
                "tests/cpu/test_scaling_p1_recovery_plans.py", "tests/cpu/test_scaling_output_identity.py")
require = V.require


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(root=ROOT):
    return {name: digest(Path(root) / name) for name in SOURCE_PATHS}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "sha256", "bytes"}, "invalid recovery evidence pin")
    require(Path(pin["path"]).is_absolute() and _sha(pin["sha256"]) and
            type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid recovery evidence pin fields")
    state = V.P.file_identity(pin["path"])
    require(state["sha256"] == pin["sha256"] and state["bytes"] == pin["bytes"], "recovery evidence pin mismatch")
    return json.loads(Path(pin["path"]).read_text()), state


def validate_manifest(manifest, diagnostic, tests, sources):
    """Pure gate: neither a null diagnostic nor a changed policy permits inference."""
    expected = {"protocol": PROTOCOL, "arm": ARM, "step": STEP, "task": TASK,
                "logical_job_id": LOGICAL_ID, "out": OUTPUT, "failed_attempts": FAILED_ATTEMPTS,
                "allocation": ALLOCATION, "budget": BUDGET, "output_barrier": "none"}
    for key, value in expected.items():
        require(manifest.get(key) == value and type(manifest.get(key)) is type(value), "recovery manifest differs: " + key)
    require(manifest.get("source_sha256") == sources and set(sources) == set(SOURCE_PATHS), "recovery source pins differ")
    require(all(_sha(v) for v in sources.values()), "invalid recovery source hash")
    require(diagnostic.get("complete") is True and diagnostic.get("outcome") == "METADATA_ONLY_REPRODUCED" and
            diagnostic.get("readers_status") == "READERS_CLEAR_IN_TEST", "diagnostic does not release P1 recovery")
    require(manifest.get("diagnostic_protocol") == DIAGNOSTIC_PROTOCOL and
            diagnostic.get("protocol") == DIAGNOSTIC_PROTOCOL, "diagnostic protocol mismatch")
    declared = manifest.get("diagnostic_source_sha256")
    require(isinstance(declared, dict) and declared and all(_sha(v) for v in declared.values()) and
            diagnostic.get("source_sha256") == declared, "diagnostic source pins mismatch")
    helper = "scripts/scaling_output_identity.py"
    require(declared.get(helper) == sources[helper], "output policy changed after diagnostic")
    require(tests.get("protocol") == CPU_TEST_PROTOCOL and tests.get("complete") is True and tests.get("passed") is True and
            type(tests.get("counts", {}).get("passed")) is int and tests["counts"]["passed"] > 0 and
            type(tests.get("failed")) is int and tests["failed"] == 0, "complete passing CPU test receipt required")
    require(isinstance(tests.get("source_sha256"), dict) and
            all(tests["source_sha256"].get(k) == v for k, v in sources.items()), "CPU tests cover other source bytes")
    return True


def load_manifest(path, expected_sha256):
    state = V.P.file_identity(path)
    require(_sha(expected_sha256) and state["sha256"] == expected_sha256, "recovery manifest SHA mismatch")
    manifest = json.loads(Path(path).read_text())
    diagnostic, diag_state = _load_pin(manifest["diagnostic_report"])
    tests, test_state = _load_pin(manifest["cpu_tests"])
    validate_manifest(manifest, diagnostic, tests, source_hashes())
    extraction, extraction_state = _load_pin(manifest["extraction_plan"])
    require(extraction.get("protocol") == V.PROTOCOL and extraction.get("step") == STEP and
            set(extraction.get("checkpoints", {})) == {ARM}, "not the original candidate extraction plan")
    states = {"recovery_manifest": state, "recovery_diagnostic": diag_state, "recovery_cpu_tests": test_state,
              "recovery_extraction_plan": extraction_state}
    states.update({"recovery_source_" + name: V.P.file_identity(ROOT / name) for name in SOURCE_PATHS})
    return manifest, states


def execute(manifest_path, manifest_sha256, expected_code_sha, device="cuda"):
    started = time.perf_counter()
    manifest, recovery_states = load_manifest(manifest_path, manifest_sha256)
    plan, states, guard, code = V.load_context(manifest["extraction_plan"]["path"], ARM, STEP, expected_code_sha)
    require(states["plan"]["sha256"] == manifest["extraction_plan"]["sha256"], "extraction plan changed")
    states.update(recovery_states)
    checkpoint, out = plan["checkpoints"][ARM]["path"], Path(manifest["out"])
    require(out.is_absolute() and not out.exists() and not out.is_symlink() and out.resolve() == out,
            "recovery output must be new and canonical")
    require(not any(out.is_relative_to(p) for p in (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"),
            Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))), "recovery output inside code")
    common = {"protocol": V.PROTOCOL, "arm": ARM, "which": "teacher", "checkpoint_step": STEP,
              "checkpoint_sha256": guard["checkpoint_sha256"], "code_sha": code,
              "plan_sha256": states["plan"]["sha256"], "postload_recipe_passed": True,
              "model_config": guard["model_config"], "stats_sha256": states["pose_stats"]["sha256"],
              "training_source_contract_sha256": states["training_source_contract"]["sha256"],
              "recovery_protocol": PROTOCOL, "recovery_manifest_sha256": manifest_sha256,
              "recovery_source_sha256": manifest["source_sha256"], "output_barrier": "none"}
    out.mkdir(parents=True, exist_ok=False)
    commands = V.extraction_commands("extract-classification", plan, checkpoint, out, TASK, device)
    require(list(commands) == [TASK], "exactly one unchanged P1 extraction command required")
    raw_before = V.E.classification_sources(plan, TASK)
    for dataset in V.DATASETS[TASK]:
        states["reference_" + dataset] = V.pin_file(plan["classification_references"][dataset])
    with (out / "p1.log").open("x") as log:
        subprocess.run(commands[TASK], cwd=ROOT, env={**os.environ, "WEARUSFM_CODE_SHA": code},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    require(raw_before == V.E.source_states([Path(p) for p in raw_before]), "raw source changed during extraction")
    require(V.E.classification_sources(plan, TASK) == raw_before, "classification source inventory changed")
    outputs = {}
    with (out / "output_identity_events.jsonl").open("x") as events:
        def event(record):
            events.write(json.dumps(record, allow_nan=False) + "\n")
            events.flush()

        for dataset in V.DATASETS[TASK]:
            path = out / (dataset + ".npz")
            def validate(candidate, ds=dataset):
                return V.verify_classification_cache(candidate, plan["classification_references"][ds], ds,
                    checkpoint, guard["checkpoint_sha256"], code, states["splits"], ARM)
            result = O.finalized_output_identity(path, validate, event=event)
            identity, details = result["identity"], result["details"]
            provenance = {**common, **details, "cache": str(path), "cache_sha256": identity["sha256"],
                          "cache_bytes": identity["bytes"], "output_identity_audit": result["audit"]}
            pp = Path(str(path) + ".provenance.json")
            V.P.write_json(pp, provenance)
            outputs[dataset] = {"path": str(path), "sha256": identity["sha256"], "bytes": identity["bytes"],
                                "provenance_path": str(pp), "provenance_sha256": digest(pp),
                                "output_identity_audit": result["audit"]}
    V.P.check_unchanged(states)
    import scaling_train as ST
    ST.check_source_contract(states["training_source_contract"]["path"])
    require(V.P.code_identity(code) == code and not (Path(checkpoint).parent / "STOP").exists(),
            "code/STOP changed during extraction")
    runtime = {"python": platform.python_version(), "numpy": V.np.__version__, "torch": V.torch.__version__,
               "cuda": V.torch.version.cuda, "device": device, "torch_threads": V.torch.get_num_threads(),
               "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    V.P.write_json(out / "receipt.json", {**common, "complete": True, "stage": "extract-classification",
        "checkpoint_unchanged": True, "outputs": outputs, "checkpoint_guard": guard, "source_states": states,
        "raw_inputs": {TASK: {"file_states": raw_before, "unchanged": True}}, "commands": commands,
        "runtime": runtime, "elapsed_seconds": time.perf_counter() - started,
        "logical_job_id": LOGICAL_ID, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "failed_attempts_preserved": FAILED_ATTEMPTS,
        "output_identity_events_sha256": digest(out / "output_identity_events.jsonl")})
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--expected-code-sha", required=True)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    args = p.parse_args(argv)
    execute(args.manifest, args.manifest_sha256, args.expected_code_sha, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
