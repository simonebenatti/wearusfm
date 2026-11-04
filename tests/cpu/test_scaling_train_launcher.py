"""Run real launcher/allocation guards, intercept only the final scientific dispatch."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/slurm/scaling_train.sbatch"


@pytest.fixture
def launch(tmp_path):
    work = tmp_path / "work"
    code = work / "wearusfm_campaigns/test"
    (code / "scripts/slurm").mkdir(parents=True)
    (code / "scripts/slurm/lib_code_dir.sh").write_text('code_dir_setup() { cd "$CODE_DIR"; }\n')
    (code / "scripts/scaling_train.py").symlink_to(ROOT / "scripts/scaling_train.py")
    (work / "venv_wearusfm/bin").mkdir(parents=True)
    (work / "venv_wearusfm/bin/activate").write_text("")
    out = tmp_path / "receipts"
    plan = {"protocol": "scaling-training-20261104-v1", "arm": "T100-G-300M", "checkpoint_step": 5000,
            "allocation": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "480G", "walltime_seconds": 19800},
            "receipt_dir": str(out)}
    commands = tmp_path / "bin"
    commands.mkdir()
    python = shlex.quote(sys.executable)
    # The real inline Python and frozen verify_allocation run on CPU; only CUDA
    # hardware discovery is replaced. No fake readiness/pins are passed to train.
    bootstrap = "import os,sys,torch; torch.cuda.device_count=lambda:int(os.environ['TEST_GPUS']); exec(compile(sys.stdin.read(),'<launcher>','exec'))"
    dispatch = "import json,os,sys; print(json.dumps({'argv':sys.argv[1:],'threads':os.environ['OMP_NUM_THREADS']}))"
    for name, body in {
        "module": "exit 0\n", "nvidia-smi": "exit 0\n",
        "scontrol": 'printf "%s\\n" "$TEST_SCONTROL"\n',
        "python3": f'if [ "$1" = - ]; then shift; exec {python} -c {shlex.quote(bootstrap)} "$@"; fi\n'
                   f'exec {python} -c {shlex.quote(dispatch)} "$@"\n',
    }.items():
        path = commands / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    fields = {"JobId": "123", "JobState": "RUNNING", "NumNodes": "1", "NumCPUs": "32",
              "TimeLimit": "05:30:00", "AllocTRES": "cpu=32,node=1,gres/gpu=4,mem=480G"}

    def run(*, args=None, env=None, allocation=None, change=None, receipt=None, plan_arg="plan.json"):
        current = json.loads(json.dumps(plan))
        if change: change(current)
        (code / "plan.json").write_text(json.dumps(current))
        environment = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                       "PYTHONPATH": str(ROOT / "src"), "WORK": str(work), "CODE_DIR": str(code),
                       "EXPECTED_SHA": "a" * 40, "SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1",
                       "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "32", "SLURM_MEM_PER_NODE": "491520",
                       "SLURM_JOB_PARTITION": "boost_usr_prod", "TEST_GPUS": "4",
                       "TEST_SCONTROL": " ".join(f"{k}={v}" for k, v in {**fields, **(allocation or {})}.items()),
                       **(env or {})}
        argv = [plan_arg, str(receipt or out)] if args is None else args
        return subprocess.run(["bash", str(LAUNCHER), *argv], env=environment, text=True, capture_output=True)

    return run, code, out


def test_only_guarded_fresh_scientific_wrapper_is_dispatched(launch):
    run, code, out = launch
    result = run()
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got["argv"] == ["scripts/scaling_train.py", "run", "--plan", str(code / "plan.json"),
                           "--expected-code-sha", "a" * 40, "--out", str(out)]
    assert got["threads"] == "1" and not out.exists()


@pytest.mark.parametrize("args", [[], ["plan.json"], ["plan.json", "/tmp/receipt", "--resume", "/tmp/checkpoint.pt"],
                                   ["plan.json", "/tmp/receipt", "--init-from", "/tmp/checkpoint.pt"],
                                   ["plan.json", "/tmp/receipt", "--max-steps", "19618"]])
def test_no_extra_recipe_or_resume_arguments(launch, args):
    result = launch[0](args=args)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("env", [{"SLURM_JOB_NUM_NODES": "2"}, {"SLURM_NTASKS": "4"},
    {"SLURM_CPUS_PER_TASK": "16"}, {"SLURM_MEM_PER_NODE": "524288"},
    {"SLURM_JOB_PARTITION": "boost_usr_dbg"}, {"TEST_GPUS": "8"}])
def test_environment_overrides_stop_before_training(launch, env):
    result = launch[0](env=env)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("allocation", [{"TimeLimit": "06:00:00"}, {"TimeLimit": "01:00:00"},
    {"JobState": "PENDING"}, {"JobId": "321"}, {"NumNodes": "2"}, {"NumCPUs": "64"},
    {"AllocTRES": "cpu=32,node=1,gres/gpu=8,mem=480G"},
    {"AllocTRES": "cpu=32,node=1,gres/gpu=4,mem=512G"}])
def test_actual_scheduler_allocation_overrides_stop_before_training(launch, allocation):
    result = launch[0](allocation=allocation)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("bad", ["dose", "arm", "time", "receipt", "absolute", "escape", "symlink", "existing", "code"])
def test_unregistered_plan_or_output_stops_before_training(launch, bad):
    run, code, out = launch
    kwargs = {}
    if bad == "dose": kwargs["change"] = lambda p: p.update(checkpoint_step=19618)
    if bad == "arm": kwargs["change"] = lambda p: p.update(arm="T100-L-300M")
    if bad == "time": kwargs["change"] = lambda p: p["allocation"].update(walltime_seconds=21600)
    if bad == "receipt": kwargs["receipt"] = out.parent / "other"
    if bad == "absolute": kwargs["plan_arg"] = str(code / "plan.json")
    if bad in ("escape", "symlink"):
        external = code.parent / "outside.json"
        external.write_text("{}")
        kwargs["plan_arg"] = "../outside.json"
        if bad == "symlink":
            (code / "link.json").symlink_to(external)
            kwargs["plan_arg"] = "link.json"
    if bad == "existing": out.mkdir()
    if bad == "code": kwargs["receipt"] = code / "output"
    result = run(**kwargs)
    assert result.returncode != 0 and '"argv"' not in result.stdout
