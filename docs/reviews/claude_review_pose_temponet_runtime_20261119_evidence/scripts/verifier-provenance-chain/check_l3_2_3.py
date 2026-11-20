"""Read-only checks for L3-2 (walltime rounding) and L3-3 (--agente choices).

No submission: submit_job.main is never called; its choices are read from the AST.
"""
import ast
import copy
import json
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path.cwd()
from pose_temponet import execution as E, resources as RS, runner as R  # noqa: E402

# ---------- L3-3: --agente value against submit_job choices (AST only)
tree = ast.parse((ROOT / "scripts/submit_job.py").read_text())
choices = None
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--agente":
            for kw in node.keywords:
                if kw.arg == "choices":
                    choices = ast.literal_eval(kw.value)
plan = json.loads((ROOT / "results/infra/pose_temponet_execution_20261119/plan.json").read_text())
released = copy.deepcopy(plan)
released["submission_released"] = True  # in-memory only
job = E.find_job(released, "TCNH-preflight-20261119")
cmd = E.submit_command(released, job, checkout_sha="0" * 40, plan_path="/x/plan.json", plan_sha256="0" * 64)
agent = cmd[cmd.index("--agente") + 1]
print("L3-3 submit_job --agente choices:", choices)
print("L3-3 generated --agente:", agent, "-> accepted:", agent in choices)
export = [a for a in cmd if a.startswith("--export=")][0]
print("L3-3 export string:", export)

# ---------- L3-2: non-minute neural walltime and the TimeLimit equality
row = RS._job("select-L0-linear", "neural", 2718.4, gpus=1)
sec = row["requested_seconds"]
print("L3-2 requested_seconds:", sec, "multiple of 60:", sec % 60 == 0)
sent = "%02d:%02d:%02d" % (sec // 3600, sec // 60 % 60, sec % 60)
mins = -(-sec // 60)  # Slurm time_str2mins rounds seconds up to whole minutes
shown = "%02d:%02d:00" % (mins // 60, mins % 60)
print("L3-2 sbatch --time sent:", sent, "| Slurm-stored TimeLimit:", shown)
# Build a minimal released neural job with these seconds and run _allocation with mocked scontrol.
neural = copy.deepcopy(E.find_job(released, "TCNH-controls-linear-L0-20261119"))
neural["resources"]["requested_seconds"] = sec
neural["resource_release"] = True
fake = (f"JobId=1 JobName=x TimeLimit={shown} Partition=boost_usr_prod NumNodes=1 NumCPUs=8 "
        f"AllocTRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1")
env = {"SLURM_JOB_ID": "1", "SLURM_RESTART_COUNT": "0", "SLURM_MEM_PER_NODE": str(32 * 1024)}
with mock.patch.dict(os.environ, env), \
     mock.patch.object(R.subprocess, "check_output", return_value=fake), \
     mock.patch.object(R.P, "validate_job_allocation", return_value={}), \
     mock.patch.object(R.torch.cuda, "device_count", return_value=1):
    try:
        R._allocation(neural)
        print("L3-2 _allocation: ACCEPTED")
    except Exception as exc:  # noqa: BLE001
        print("L3-2 _allocation: REFUSED:", exc)
    # Control: same job with a whole-minute limit is accepted by the same check.
    whole = copy.deepcopy(neural)
    whole["resources"]["requested_seconds"] = mins * 60
    try:
        R._allocation(whole)
        print("L3-2 control (whole minutes", mins * 60, "s): ACCEPTED")
    except Exception as exc:  # noqa: BLE001
        print("L3-2 control: REFUSED:", exc)
