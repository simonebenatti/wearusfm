"""Run the real continuation launchers (training and identity pre-check) and allocation guards, intercept only the final dispatch."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/slurm/scaling_continue.sbatch"
PRECHECK_LAUNCHER = ROOT / "scripts/slurm/scaling_identity_precheck.sbatch"
WRAPPER = ["scripts/scaling_continue.py", "run"]
PRECHECK_WRAPPER = ["scripts/scaling_continue.py", "identity-precheck"]
TRAINING_PLAN = {"protocol": "scaling-continuation-training-20261106-v1", "arm": "T100-G-300M", "start_step": 5000, "end_step": 19618,
                 "allocation": {"account": "iscrb_wearusfm", "partition": "boost_usr_prod", "qos": "normal", "nodes": 1, "gpus": 4,
                                "cpus": 32, "memory": "480G", "walltime_seconds": 53100}}
PRECHECK = {"protocol": "scaling-continuation-identity-precheck-20261109-v2",
            "allocation": {"partition": "lrd_all_serial", "cpus": 4, "memory": "24G", "walltime_seconds": 1200}}
TRAINING_SLURM = {"SLURM_CPUS_PER_TASK": "32", "SLURM_MEM_PER_NODE": "491520", "SLURM_JOB_PARTITION": "boost_usr_prod"}
PRECHECK_SLURM = {"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "24576", "SLURM_JOB_PARTITION": "lrd_all_serial"}
TRAINING_FIELDS = {"JobId": "123", "JobState": "RUNNING", "NumNodes": "1", "NumCPUs": "32", "TimeLimit": "14:45:00",
                   "Partition": "boost_usr_prod", "QOS": "normal", "Account": "iscrb_wearusfm",
                   "AllocTRES": "cpu=32,node=1,gres/gpu=4,mem=480G"}
PRECHECK_FIELDS = {"JobId": "123", "JobState": "RUNNING", "NumNodes": "1", "NumCPUs": "4", "TimeLimit": "00:20:00",
                   "Partition": "lrd_all_serial", "QOS": "normal", "Account": "iscrb_wearusfm", "AllocTRES": "cpu=4,mem=24G,node=1"}


def wrapper_module():
    spec = importlib.util.spec_from_file_location("scaling_continue_launcher_test", ROOT / "scripts/scaling_continue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher(tmp_path, launcher, plan_for, slurm, fields):
    """A fake frozen checkout under a fake $WORK, stub commands, SLURM environment and scontrol answer of one launcher."""
    work = tmp_path / "work"
    code = work / "wearusfm_campaigns/test"
    (code / "scripts/slurm").mkdir(parents=True)
    (code / "scripts/slurm/lib_code_dir.sh").write_text('code_dir_setup() { cd "$CODE_DIR"; }\n')
    (code / "scripts/scaling_continue.py").symlink_to(ROOT / "scripts/scaling_continue.py")
    (work / "venv_wearusfm/bin").mkdir(parents=True)
    (work / "venv_wearusfm/bin/activate").write_text("")
    out = tmp_path / "receipts"
    plan = plan_for(out)
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

    def run(*, args=None, env=None, allocation=None, change=None, receipt=None, plan_arg="plan.json"):
        current = json.loads(json.dumps(plan))
        if change: change(current)
        (code / "plan.json").write_text(json.dumps(current))
        environment = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                       "PYTHONPATH": str(ROOT / "src"), "WORK": str(work), "CODE_DIR": str(code),
                       "EXPECTED_SHA": "a" * 40, "SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1",
                       "SLURM_NTASKS": "1", **slurm, "TEST_GPUS": "4",
                       "TEST_SCONTROL": " ".join(f"{k}={v}" for k, v in {**fields, **(allocation or {})}.items()),
                       **(env or {})}
        argv = [plan_arg, str(receipt or out)] if args is None else args
        return subprocess.run(["bash", str(launcher), *argv], env=environment, text=True, capture_output=True)

    return run, code, out


@pytest.fixture
def launch(tmp_path):
    return _launcher(tmp_path, LAUNCHER, lambda out: {**TRAINING_PLAN, "receipt_dir": str(out)}, TRAINING_SLURM, TRAINING_FIELDS)


@pytest.fixture
def precheck(tmp_path):
    """The identity pre-check launcher: the receipt directory of the plan's `identity_precheck` is the second argument."""
    return _launcher(tmp_path, PRECHECK_LAUNCHER,
                     lambda out: {**TRAINING_PLAN, "receipt_dir": str(tmp_path / "training_receipts"),
                                  "identity_precheck": {**PRECHECK, "receipt_dir": str(out)}}, PRECHECK_SLURM, PRECHECK_FIELDS)


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


# ---------------------------------------------------------------- the identity pre-check launcher (v2 gate, 0 GPU)


def test_only_the_identity_precheck_of_the_wrapper_is_dispatched(precheck):
    run, code, out = precheck
    result = run()
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got["argv"] == [*PRECHECK_WRAPPER, "--plan", str(code / "plan.json"), "--expected-code-sha", "a" * 40, "--out", str(out)]
    assert got["threads"] == "1" and not out.exists()


def test_precheck_requests_the_small_serial_allocation_of_the_plan_gate():
    options = [line[len("#SBATCH "):] for line in PRECHECK_LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    header = dict(option.split("=", 1) if option.startswith("--") else option.split(None, 1) for option in options)
    allocation = wrapper_module().PRECHECK_ALLOCATION
    assert len(header) == len(options) == 10 and header["-J"] == "scaling_identity_precheck"
    assert header["-A"] == "IscrB_WearUsFM" and header["-p"] == allocation["partition"] == "lrd_all_serial"
    assert (header["--nodes"], header["--ntasks"], header["--cpus-per-task"], header["--mem"]) == ("1", "1", str(allocation["cpus"]), allocation["memory"])
    hours, minutes, seconds = (int(x) for x in header["--time"].split(":"))
    assert hours * 3600 + minutes * 60 + seconds == allocation["walltime_seconds"] == 1200  # 4 CPU x 20 min: 1.33 core-hours cap
    assert "--gres" not in header and "--qos" not in header and "--array" not in header and "--requeue" not in PRECHECK_LAUNCHER.read_text()
    assert header["-o"] == header["-e"] and header["-o"].startswith("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_identity_precheck_")
    assert "identity-precheck" in PRECHECK_LAUNCHER.read_text() and " run " not in PRECHECK_LAUNCHER.read_text().split("exec python3")[-1]


@pytest.mark.parametrize("args", [[], ["plan.json"], ["plan.json", "/tmp/receipt", "--device", "cuda"],
                                   ["plan.json", "/tmp/receipt", "--max-steps", "49046"]])
def test_precheck_takes_no_extra_arguments(precheck, args):
    result = precheck[0](args=args)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("env", [{"SLURM_JOB_NUM_NODES": "2"}, {"SLURM_NTASKS": "4"}, {"SLURM_CPUS_PER_TASK": "32"},
    {"SLURM_MEM_PER_NODE": "491520"}, {"SLURM_JOB_PARTITION": "boost_usr_prod"}, {"SLURM_JOB_ID": ""}, {"EXPECTED_SHA": ""}])
def test_precheck_environment_overrides_stop_before_computing(precheck, env):
    result = precheck[0](env=env)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("allocation", [{"TimeLimit": "01:00:00"}, {"TimeLimit": "00:30:00"}, {"JobState": "PENDING"}, {"JobId": "321"},
    {"NumNodes": "2"}, {"NumCPUs": "8"}, {"AllocTRES": "cpu=4,mem=24G,node=1,gres/gpu=1"}, {"AllocTRES": "cpu=4,mem=32G,node=1"},
    {"Partition": "boost_usr_prod"}, {"Account": "another_account"}])
def test_precheck_actual_scheduler_allocation_overrides_stop_before_computing(precheck, allocation):
    result = precheck[0](allocation=allocation)
    assert result.returncode != 0 and '"argv"' not in result.stdout


@pytest.mark.parametrize("bad", ["no_gate", "gate_protocol", "gate_cpus", "gate_partition", "receipt", "dose", "arm", "phase5000",
                                  "absolute", "escape", "existing", "code"])
def test_precheck_unregistered_plan_or_output_stops_before_computing(precheck, bad):
    run, code, out = precheck
    kwargs = {}
    if bad == "no_gate": kwargs["change"] = lambda p: p.pop("identity_precheck")
    if bad == "gate_protocol": kwargs["change"] = lambda p: p["identity_precheck"].update(protocol="scaling-diag-fixedeval-20261108-v1")
    if bad == "gate_cpus": kwargs["change"] = lambda p: p["identity_precheck"]["allocation"].update(cpus=8)
    if bad == "gate_partition": kwargs["change"] = lambda p: p["identity_precheck"]["allocation"].update(partition="boost_usr_prod")
    if bad == "receipt": kwargs["receipt"] = out.parent / "other"
    if bad == "dose": kwargs["change"] = lambda p: p.update(end_step=49046)
    if bad == "arm": kwargs["change"] = lambda p: p.update(arm="T100-L-300M")
    if bad == "phase5000": kwargs["change"] = lambda p: p.update(protocol="scaling-training-20261104-v1", checkpoint_step=5000)
    if bad == "absolute": kwargs["plan_arg"] = str(code / "plan.json")
    if bad == "escape":
        (code.parent / "outside.json").write_text("{}")
        kwargs["plan_arg"] = "../outside.json"
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
