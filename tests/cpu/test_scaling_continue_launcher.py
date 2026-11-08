"""Run the real continuation launcher and allocation guards, intercept only the final dispatch."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/slurm/scaling_continue.sbatch"
WRAPPER = ["scripts/scaling_continue.py", "run"]


@pytest.fixture
def launch(tmp_path):
    work = tmp_path / "work"
    code = work / "wearusfm_campaigns/test"
    (code / "scripts/slurm").mkdir(parents=True)
    (code / "scripts/slurm/lib_code_dir.sh").write_text('code_dir_setup() { cd "$CODE_DIR"; }\n')
    (code / "scripts/scaling_continue.py").symlink_to(ROOT / "scripts/scaling_continue.py")
    (work / "venv_wearusfm/bin").mkdir(parents=True)
    (work / "venv_wearusfm/bin/activate").write_text("")
    out = tmp_path / "receipts"
    plan = {"protocol": "scaling-continuation-training-20261106-v1", "arm": "T100-G-300M", "start_step": 5000, "end_step": 19618,
            "allocation": {"account": "iscrb_wearusfm", "partition": "boost_usr_prod", "qos": "normal", "nodes": 1, "gpus": 4,
                           "cpus": 32, "memory": "480G", "walltime_seconds": 53100},
            "receipt_dir": str(out)}
    commands = tmp_path / "bin"
    commands.mkdir()
    python = shlex.quote(sys.executable)
    # The real inline Python and the wrapper's verify_allocation run on CPU; only CUDA hardware
    # discovery is replaced. Nothing after the guards is executed: the dispatch is printed.
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
    fields = {"JobId": "123", "JobState": "RUNNING", "NumNodes": "1", "NumCPUs": "32", "TimeLimit": "14:45:00",
              "Partition": "boost_usr_prod", "QOS": "normal", "Account": "iscrb_wearusfm",
              "AllocTRES": "cpu=32,node=1,gres/gpu=4,mem=480G"}

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


def test_only_the_guarded_continuation_wrapper_is_dispatched(launch):
    run, code, out = launch
    result = run()
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got["argv"] == [*WRAPPER, "--plan", str(code / "plan.json"), "--expected-code-sha", "a" * 40, "--out", str(out)]
    assert got["threads"] == "1" and not out.exists()


def test_requested_resources_are_the_frozen_single_segment():
    options = [line[len("#SBATCH "):] for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    header = dict(option.split("=", 1) if option.startswith("--") else option.split(None, 1) for option in options)
    assert len(header) == len(options) == 12 and header["-J"] == "scaling_continue"
    assert header["-A"] == "IscrB_WearUsFM" and header["-p"] == "boost_usr_prod" and header["--qos"] == "normal"
    assert (header["--nodes"], header["--ntasks"], header["--gres"], header["--cpus-per-task"], header["--mem"]) == \
        ("1", "1", "gpu:4", "32", "480G")
    hours, minutes, seconds = (int(x) for x in header["--time"].split(":"))
    assert hours * 3600 + minutes * 60 + seconds == 53100 and 4 * 53100 / 3600 == 59.0  # the frozen training cap
    assert header["-o"] == header["-e"] and header["-o"].startswith("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/")
    assert "--array" not in header and "--requeue" not in LAUNCHER.read_text()


@pytest.mark.parametrize("args", [[], ["plan.json"], ["plan.json", "/tmp/receipt", "--resume", "/tmp/checkpoint.pt"],
                                   ["plan.json", "/tmp/receipt", "--init-from", "/tmp/checkpoint.pt"],
                                   ["plan.json", "/tmp/receipt", "--max-steps", "49046"]])
def test_no_extra_recipe_dose_or_warm_start_arguments(launch, args):
    result = launch[0](args=args)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("env", [{"SLURM_JOB_NUM_NODES": "2"}, {"SLURM_NTASKS": "4"},
    {"SLURM_CPUS_PER_TASK": "16"}, {"SLURM_MEM_PER_NODE": "524288"},
    {"SLURM_JOB_PARTITION": "boost_usr_dbg"}, {"TEST_GPUS": "8"}, {"SLURM_JOB_ID": ""}, {"EXPECTED_SHA": ""}])
def test_environment_overrides_stop_before_training(launch, env):
    result = launch[0](env=env)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("allocation", [{"TimeLimit": "1-00:00:00"}, {"TimeLimit": "05:30:00"},
    {"JobState": "PENDING"}, {"JobId": "321"}, {"NumNodes": "2"}, {"NumCPUs": "64"},
    {"AllocTRES": "cpu=32,node=1,gres/gpu=8,mem=480G"}, {"AllocTRES": "cpu=32,node=1,gres/gpu=4,mem=512G"},
    {"QOS": "boost_qos_bprod"}, {"QOS": "boost_qos_dbg"}, {"Partition": "boost_usr_dbg"}, {"Account": "another_account"}])
def test_actual_scheduler_allocation_overrides_stop_before_training(launch, allocation):
    result = launch[0](allocation=allocation)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("bad", ["dose", "start", "arm", "time", "qos", "phase5000", "receipt", "absolute", "escape", "symlink",
                                  "existing", "code"])
def test_unregistered_plan_or_output_stops_before_training(launch, bad):
    run, code, out = launch
    kwargs = {}
    if bad == "dose": kwargs["change"] = lambda p: p.update(end_step=49046)
    if bad == "start": kwargs["change"] = lambda p: p.update(start_step=0)
    if bad == "arm": kwargs["change"] = lambda p: p.update(arm="T100-L-300M")
    if bad == "time": kwargs["change"] = lambda p: p["allocation"].update(walltime_seconds=86400)
    if bad == "qos": kwargs["change"] = lambda p: p["allocation"].update(qos="boost_qos_bprod")
    if bad == "phase5000": kwargs["change"] = lambda p: p.update(protocol="scaling-training-20261104-v1", checkpoint_step=5000)
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


def _first_imports(script):
    """Top-level imports of torch, numpy, scipy and wearusfm in order of first appearance."""
    import ast
    seen = []
    for node in ast.parse(Path(script).read_text()).body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in ("torch", "numpy", "scipy", "wearusfm") and root not in seen:
                seen.append(root)
    return seen


def test_worker_imports_torch_before_numpy_and_wearusfm_like_the_trainer_entry_point():
    """Job 63696864 (08/11/2026) failed at resume because the recomputed fixed-set identity differed from the seeded one with
    identical windows, data, code and configuration: the worker imported wearusfm.training.run (hence numpy) before torch, so numpy
    bound its `64_`-suffixed BLAS/LAPACK symbols to its own OpenBLAS; with torch first (as `torchrun scripts/train_jepa.py`, the 5000
    run) MKL, published globally by libtorch_global_deps, provides them, and sosfiltfilt initial conditions (np.linalg.solve) and anchor
    targets (power @ bands) come out bit-identical to the 5000 run (diagnosis jobs 63861166 and 63864244). The order is a contract."""
    for script in ("scripts/scaling_continue.py", "scripts/train_jepa.py"):
        order = _first_imports(ROOT / script)
        assert order and order[0] == "torch", (script, order)
