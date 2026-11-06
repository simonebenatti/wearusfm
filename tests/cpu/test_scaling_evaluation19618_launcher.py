"""Execute the 19618 launcher with fake scheduler/GPU commands, never the controller.

Derived from test_scaling_evaluation_launcher.py by the replacements declared in scaling_derive19618.py.

The real lib_code_dir.sh and embedded Python guards run. Git/SLURM/module/CUDA
observations are fixtures; reaching the final command only writes its argv.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/slurm/scaling_evaluation19618.sbatch"
CODE = "c" * 40
CELLS = [("T100-G-300M", "extract-pose", "none", "02:30:00", 9000),
         ("T100-G-300M", "extract-classification", "p1", "02:00:00", 7200),
         ("T100-G-300M", "extract-classification", "p2", "02:30:00", 9000)]


@pytest.fixture
def harness(tmp_path):
    root = tmp_path.resolve()
    work, bins = root/"work", root/"bin"
    code = work/"wearusfm_campaigns/frozen"
    (code/"scripts/slurm").mkdir(parents=True)
    bins.mkdir()
    shutil.copyfile(ROOT/"scripts/slurm/lib_code_dir.sh", code/"scripts/slurm/lib_code_dir.sh")
    plan = {"protocol": "scaling-evaluation-19618-20261106-v1", "step": 19618,
            "arms": ["T100-G-30M", "T100-G-300M"], "checkpoints": {"T100-G-300M": {"pin": "validated by controller"}}}
    for name, value in (("plan.json", plan), ("wrong_step.json", {**plan, "step": 5000}),
                        ("legacy.json", {**plan, "protocol": "scaling-evaluation-20261104-v1"}),
                        ("missing_pin.json", {**plan, "checkpoints": {}}),
                        ("control_pin.json", {**plan, "checkpoints": {a: {} for a in ("T100-G-30M", "T100-G-300M")}})):
        (code/name).write_text(json.dumps(value))
    outside = work/"outside.json"
    outside.write_text(json.dumps(plan))
    (code/"escaped.json").symlink_to(outside)
    (work/"venv_wearusfm/bin").mkdir(parents=True)
    (work/"venv_wearusfm/bin/activate").write_text("# no environment changes in the fixture\n")

    def executable(name, body):
        path = bins/name
        path.write_text("#!"+sys.executable+"\n"+body)
        path.chmod(0o755)
    executable("module", "import sys\nsys.exit(0)\n")
    executable("nvidia-smi", "print('FAKE GPU, no GPU operations')\n")
    executable("scontrol", "import os,sys\nassert sys.argv[1:]==['show','job','-o',os.environ['SLURM_JOB_ID']]\nprint(os.environ['TEST_JOB_INFO'])\n")
    executable("git", """import os,sys
if sys.argv[3:] == ['rev-parse','--is-inside-work-tree']: print('true')
elif sys.argv[3:] == ['rev-parse','HEAD']: print(os.environ.get('TEST_GIT_HEAD', 'c'*40))
elif sys.argv[3:] == ['status','--porcelain','--untracked-files=all']:
    if os.environ.get('TEST_DIRTY'): print(' M tracked_file')
else: sys.exit(3)
""")
    executable("python3", """import json,os,sys,types
argv=sys.argv[1:]
if argv[0]=='-':
    sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda:int(os.environ.get('TEST_CUDA_COUNT','1'))))
    sys.argv=argv
    exec(compile(sys.stdin.read(),'<launcher-guards>','exec'))
elif argv[0]=='scripts/scaling_evaluation19618.py':
    with open(os.environ['TEST_RECORD'],'x') as f:
        json.dump({'argv':argv,'cwd':os.getcwd(),'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH','WEARUSFM_CODE_SHA']}},f)
else: raise AssertionError('Unexpected Python command: '+repr(argv))
""")
    for path in code.rglob('*'):
        if not path.is_symlink(): path.chmod(0o555 if path.is_dir() else 0o444)
    code.chmod(0o555)
    record = root/"controller_argv.json"
    env = {**os.environ, "PATH": str(bins)+os.pathsep+os.environ["PATH"], "WORK": str(work), "CODE_DIR": str(code),
           "EXPECTED_SHA": CODE, "SLURM_JOB_ID": "12345", "SLURM_JOB_NUM_NODES": "1", "SLURM_CPUS_PER_TASK": "8", "TEST_RECORD": str(record)}
    def run(cell=CELLS[0], plan_name="plan.json", out=None, fields=None, environment=None, extra=()):
        arm, stage, task, time_limit, _ = cell
        info = {"JobId": "12345", "JobState": "RUNNING", "NumNodes": "1", "NumCPUs": "8", "NumTasks": "1",
                "TimeLimit": time_limit, "AllocTRES": "cpu=8,mem=100G,node=1,gres/gpu=1"}
        info.update(fields or {})
        args = [str(LAUNCHER), plan_name, arm, stage, task, str(out or root/"new_output"), *extra]
        return subprocess.run(["/bin/bash", *args], cwd=root, env={**env, "TEST_JOB_INFO": " ".join(k+'='+v for k,v in info.items()), **(environment or {})},
                              capture_output=True, text=True, timeout=20)
    yield run, record, code, work
    for path in [code, *code.rglob('*')]:
        if not path.is_symlink(): path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize("cell", CELLS)
def test_each_registered_extraction_runs_guards_and_invokes_fixed19618_once(harness, cell):
    run, record, code, _ = harness
    got = run(cell)
    assert got.returncode == 0, got.stdout+got.stderr
    result = json.loads(record.read_text())
    argv = result["argv"]
    args = dict(zip(argv[1::2], argv[2::2]))
    assert argv[0] == "scripts/scaling_evaluation19618.py"
    assert args["--step"] == "19618" and args["--arm"] == cell[0] and args["--stage"] == cell[1]
    assert args["--device"] == "cuda" and args["--expected-code-sha"] == CODE
    assert args["--plan"] == str(code/"plan.json") and result["cwd"] == str(code)
    assert args.get("--task", "none") == cell[2]
    assert all(result["environment"][k] == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"))
    assert result["environment"]["PYTHONPATH"].split(os.pathsep)[0] == str(code/"src")
    assert result["environment"]["WEARUSFM_CODE_SHA"] == CODE


def test_three_cell_gpu_budget_of_the_freeze_and_no_separate_validation():
    assert sum(c[4] for c in CELLS)/3600 == 7. and all(c[0] == "T100-G-300M" for c in CELLS)
    freeze = json.loads((ROOT/"results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json").read_text())
    assert freeze["extractions"]["gpu_hours_cap"] == 7. and not freeze["extractions"]["new_30m_extractions_included"]
    assert [(j["arm"], j["stage"], j["task"] or "none", j["walltime_seconds"], j["checkpoint_step"], j["gpus"], j["cpus"], j["memory"])
            for j in freeze["extractions"]["jobs"]] == [(c[0], c[1], c[2], c[4], 19618, 1, 8, "100G") for c in CELLS]
    assert sum(c[1] == "extract-pose" for c in CELLS) == 1
    assert not any(c[1] == "validate" for c in CELLS)
    text = LAUNCHER.read_text()
    for directive in ('--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8', '--mem=100G', '--time=02:30:00'):
        assert '#SBATCH '+directive in text
    assert subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], capture_output=True).returncode == 0


@pytest.mark.parametrize("arm,stage,task", [("T100-G-30M", "extract-pose", "none"),
    ("T100-G-30M", "extract-classification", "p1"), ("T100-G-30M", "extract-classification", "p2"),
    ("T100-G-300M", "validate", "none"), ("T100-G-300M", "extract-pose", "p1"),
    ("T100-G-300M", "extract-classification", "none"), ("T25-L", "extract-classification", "p1")])
def test_unregistered_cells_never_reach_controller(harness, arm, stage, task):
    run, record, _, _ = harness
    result = run((arm, stage, task, "02:30:00", 9000))
    assert result.returncode != 0 and not record.exists()
    assert 'unregistered extraction' in result.stderr


@pytest.mark.parametrize("fields", [{"TimeLimit": "02:29:59"}, {"TimeLimit": "UNLIMITED"}, {"TimeLimit": "02:60:00"},
    {"NumCPUs": "16"}, {"NumNodes": "2"}, {"NumTasks": "2"}, {"JobId": "999"}, {"JobState": "PENDING"},
    {"AllocTRES": "cpu=8,mem=100G,node=1,gres/gpu=2"}, {"AllocTRES": "cpu=8,mem=101G,node=1,gres/gpu=1"},
    {"AllocTRES": "cpu=7,mem=100G,node=1,gres/gpu=1"}, {"AllocTRES": ""}])
def test_actual_scheduler_allocations_and_walltime_must_match(harness, fields):
    run, record, _, _ = harness
    result = run(fields=fields)
    assert result.returncode != 0 and not record.exists()


@pytest.mark.parametrize("environment", [{"TEST_CUDA_COUNT": "0"}, {"TEST_CUDA_COUNT": "2"}, {"SLURM_JOB_NUM_NODES": "2"},
    {"SLURM_CPUS_PER_TASK": "4"}, {"TEST_GIT_HEAD": "a"*40}, {"TEST_DIRTY": "1"}])
def test_environment_and_frozen_checkout_guards(harness, environment):
    run, record, _, _ = harness
    assert run(environment=environment).returncode != 0 and not record.exists()


@pytest.mark.parametrize("name", ["wrong_step.json", "legacy.json", "missing_pin.json", "control_pin.json", "escaped.json",
                                  "../../outside.json"])
def test_plan_dose_protocol_pins_and_containment(harness, name):
    run, record, _, _ = harness
    assert run(plan_name=name).returncode != 0 and not record.exists()


@pytest.mark.parametrize("kind", ["existing", "inside_code", "inside_canonical", "symlink", "relative", "extra_endpoint_arg"])
def test_output_new_canonical_external_and_no_endpoint_override(harness, tmp_path, kind):
    run, record, code, work = harness
    out, extra = tmp_path/"output", ()
    if kind == "existing": out.mkdir()
    if kind == "inside_code": out = code/"output"
    if kind == "inside_canonical": out = work/"wearusfm/output"
    if kind == "symlink": out.symlink_to(tmp_path/"absent")
    if kind == "relative": out = Path("relative-output")
    if kind == "extra_endpoint_arg": extra = ("5000",)
    assert run(out=out, extra=extra).returncode != 0 and not record.exists()


def test_equivalent_slurm_memory_and_day_clock_encoding_are_accepted(harness):
    run, record, _, _ = harness
    result = run(fields={"TimeLimit": "0-02:30:00", "AllocTRES": "cpu=8,mem=102400M,node=1,gres/gpu=1"})
    assert result.returncode == 0, result.stderr
    assert record.exists()
