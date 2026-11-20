"""Local checks of the second TCN delivery (20/11/2026). No cluster calls, no job.

Runs the 20 TCN test files and the 5 historical pose files, checks that the
committed plan is exactly what prepare-plan derives from the current sources,
and writes the logs with a small receipt. Same layout as revision_runtime/validate.py
of the first delivery.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]
from pose_temponet import execution as E, provenance as P  # noqa: E402

HERE = Path(__file__).resolve().parent
PLAN = HERE/"plan.json"
BASELINE_FAILURE = ["tests/cpu/test_consolidation_long_common.py::"
                    "test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned"]
HISTORICAL = ("pose_dynamics", "tokenizer_pose_readout", "consolidation_long_pose",
              "consolidation_long_pose_runtime", "consolidation_long_common")


def pin(path):
    data = path.resolve().read_bytes()
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = P.source_identity()
    runtime = sorted(ROOT.glob("tests/cpu/test_pose_temponet*.py"))
    historical = [ROOT/"tests/cpu"/("test_"+name+".py") for name in HISTORICAL]
    before = [pin(p) for p in runtime+historical]
    settings = {"PYTHONPATH": str(ROOT/"src")+os.pathsep+str(ROOT/"scripts"), "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment = {**os.environ, **settings}
    runs = []
    for name, files in (("runtime", runtime), ("historical", historical)):
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short",
                   "-o", "tmp_path_retention_policy=none", *[str(p.relative_to(ROOT)) for p in files]]
        log = destination/(name+".log")
        started = time.monotonic()
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text()
        failures = re.findall(r"^FAILED (.+?)(?: - |$)", text, re.M)
        summary = re.findall(r"^(?:=+ )?(.*(?:passed|failed).* in [0-9.]+s.*?)(?: =+)?$", text, re.M)
        accepted = (result.returncode == 0 if name == "runtime"
                    else result.returncode in (0, 1) and failures in ([], BASELINE_FAILURE))
        runs.append({"name": name, "command": command, "returncode": result.returncode,
                     "elapsed_s": time.monotonic()-started, "log": pin(log), "failures": failures,
                     "summary": summary[-1] if summary else None,
                     "expected_baseline_failure_only": name == "historical" and failures == BASELINE_FAILURE,
                     "accepted": accepted})
    if source != P.source_identity() or before != [pin(p) for p in runtime+historical]:
        raise RuntimeError("sources or tests changed during validation")
    # The committed plan must be exactly the public entry point applied to these sources.
    committed = json.loads(PLAN.read_text())
    rebuilt = P.build_plan(Path("results/infra/pose_temponet_proposal_20261117/proposal.json"), jobs=E.job_graph())
    plan_matches = P.canonical_bytes(committed) == P.canonical_bytes(rebuilt)
    data = {"base_commit": "31877ad85c619ec69214cd5e9fb53df50e452148",
            "first_delivery_reviewed_code": "01252b6c3023c49046b312e003b7915d8f3f6119",
            "source_sha256": source, "sources": len(P.source_manifest()), "cwd": str(ROOT),
            "environment": settings, "test_files": before, "runs": runs, "cuda_tested": False,
            "plan": pin(PLAN), "plan_matches_prepare_plan": plan_matches,
            "plan_status": committed["status"], "plan_submission_released": committed["submission_released"],
            "no_new_regression": all(r["accepted"] for r in runs) and plan_matches,
            "no_jobs": True, "no_remote_operations": True}
    (destination/"local_validation.json").write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps({"receipt": str(destination/"local_validation.json"),
                      "no_new_regression": data["no_new_regression"]}))
    raise SystemExit(0 if data["no_new_regression"] else 1)


if __name__ == "__main__":
    main()
