"""Reproduce a queue guard error locally. Never imports/runs the operational CLI."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "37632d18e0078069c54d131c9ba235891712c1b5"
BASE = "results/infra/scaling_300m_20261104/final_stages_20261106/"


def blob(path):
    return subprocess.check_output(["git", "show", COMMIT + ":" + path], cwd=ROOT)


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def audit():
    source = blob(BASE + "final_stages.py")
    assert hashlib.sha256(source).hexdigest() == "9533c82fa4e4cde93860fc62218bd1fd586f6919021ac2870b1b27330d60b9b8"
    plan = json.loads(blob(BASE + "aggregate_submission_plan.json"))
    state = json.loads(blob(BASE + "live_state_20261106_fits_complete.json"))
    fit_plan = json.loads((ROOT / "results/infra/scaling_300m_20261104/cpu_fit_release_20261105/submission_plan.json").read_text())
    funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in ("checks", "cmd_check")]
    assert len(funcs) == 2
    namespace = {
        "json": json, "Path": Path, "require": require,
        "fits": lambda: [j for j in fit_plan["jobs"] if j["stage"] == "probe-fit"],
        "CELLS": sorted({j["cell"] for j in plan["jobs"]}),
        "TASKS": {"p1": ["ninapro_db2", "ninapro_db3", "ninapro_db6"], "p2": ["epn612", "uci_emg"]},
        "PROBE_PLAN": plan["probe_plan"]["path"], "PROBE_PLAN_SHA": plan["probe_plan"]["sha256"],
        "CODE_SHA": plan["runtime_code_sha"], "DISPATCHER_SHA": plan["dispatcher"]["sha256"],
        "SUBMITTER_SHA": plan["submitter"]["sha256"],
        "SHARD_FILES": ["audit.jsonl", "audit_progress.jsonl", "shard.json", "shard.oof.npz"],
    }
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "reviewed_guard_functions", "exec"), namespace)
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        p, s = Path(tmp) / "plan.json", Path(tmp) / "state.json"
        p.write_text(json.dumps(plan))
        cases = [("original_9_jobs", state), ("failed_squeue", copy.deepcopy(state))]
        cases[1][1]["squeue_serial"] = {"exit": 1, "stdout": "", "stderr": "synthetic timeout"}
        for label, value in cases:
            s.write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                try:
                    namespace["cmd_check"](p, s)
                    accepted = True
                except SystemExit:
                    accepted = False
            result[label] = {"accepted": accepted, "last_output": output.getvalue().splitlines()[-1]}
    assert not result["original_9_jobs"]["accepted"]
    assert result["failed_squeue"]["accepted"]
    return {"reviewed_commit": COMMIT, "scope": "Local synthetic evidence only; no SSH, submitter or operational CLI execution", "reproduction": result}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
