"""Exercise launcher dispatch and fail-closed guards without running any fits."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/slurm/tokenizer_probe.sbatch"


@pytest.fixture
def launch(tmp_path):
    work = tmp_path / "work"
    code = work / "wearusfm_campaigns/test"
    (code / "scripts/slurm").mkdir(parents=True)
    (code / "scripts/slurm/lib_code_dir.sh").write_text('code_dir_setup() { cd "$CODE_DIR"; }\n')
    (code / "plan.json").write_text("{}\n")
    plan_sha = hashlib.sha256((code / "plan.json").read_bytes()).hexdigest()
    (work / "venv_wearusfm/bin").mkdir(parents=True)
    (work / "venv_wearusfm/bin/activate").write_text("")
    commands = tmp_path / "bin"
    commands.mkdir()
    for name, body in {
        "module": "exit 0\n",
        "scontrol": 'echo "JobId=123 TimeLimit=$TEST_WALLTIME"\n',
        "python3": 'if [ "$1" = - ]; then exec ' + shlex.quote(sys.executable) + ' "$@"; fi\n'
                   + 'exec ' + shlex.quote(sys.executable) + ' -c '
                   + shlex.quote("import json,os,sys; print(json.dumps({'argv':sys.argv[1:], 'threads':os.environ['OMP_NUM_THREADS'], 'cuda':os.environ['CUDA_VISIBLE_DEVICES']}))")
                   + ' "$@"\n',
    }.items():
        path = commands / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run(stage="fit-shard", args=("T25-L", "db2-01"), cpus=8, minutes=60, env_overrides=None, out=None, plan=None, digest=None):
        env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
               "WORK": str(work), "CODE_DIR": str(code), "EXPECTED_SHA": "a" * 40,
               "SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "lrd_all_serial", "SLURM_JOB_NUM_NODES": "1",
               "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": str(cpus), "SLURM_MEM_PER_NODE": "30720",
               "TEST_WALLTIME": f"{minutes // 60:02}:{minutes % 60:02}:00", **(env_overrides or {})}
        return subprocess.run(["bash", str(LAUNCHER), plan or "plan.json", digest or plan_sha, stage,
                               str(out or tmp_path / "output"), *map(str, args)], env=env, text=True, capture_output=True)

    return run, tmp_path, code


@pytest.mark.parametrize("key,task,datasets,folds,minutes", [
    ("db2-01", "p1", ["ninapro_db2"], ["0", "1"], 60),
    ("db2-23", "p1", ["ninapro_db2"], ["2", "3"], 60),
    ("db3", "p1", ["ninapro_db3"], ["0", "1", "2", "3"], 30),
    ("db6", "p1", ["ninapro_db6"], ["0", "1", "2", "3"], 30),
    ("p2", "p2", ["epn612", "uci_emg"], ["0", "1", "2", "3"], 60),
])
def test_registered_shards_reach_cli(launch, key, task, datasets, folds, minutes):
    run, _, _ = launch
    result = run(args=("T100-G", key), minutes=minutes)
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    expected = ["--arm", "T100-G", "--task", task, "--datasets", *datasets, "--folds", *folds]
    assert dispatched["argv"][-len(expected):] == expected
    assert dispatched["threads"] == "8" and dispatched["cuda"] == ""


@pytest.mark.parametrize("stage,task,count,minutes", [("aggregate", "p1", 4, 20), ("aggregate", "p2", 1, 20), ("compare", None, 6, 30)])
def test_completed_reports_reach_cli(launch, stage, task, count, minutes):
    run, tmp, _ = launch
    paths = [tmp / f"report_{i}.json" for i in range(count)]
    for path in paths:
        path.write_text("{}")
    args = ["T25-L", task, *paths] if task else paths
    result = run(stage=stage, args=args, cpus=2, minutes=minutes)
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    assert dispatched["argv"][-count:] == list(map(str, paths))
    assert dispatched["threads"] == "2"


@pytest.mark.parametrize("overrides", [
    {"args": ("T25-L", "db3"), "minutes": 60},
    {"cpus": 16},
    {"env_overrides": {"SLURM_MEM_PER_NODE": "32768"}},
    {"env_overrides": {"SLURM_JOB_PARTITION": "boost_usr_prod"}},
    {"args": ("T25-L", "all")},
    {"args": ("unregistered", "db2-01")},
    {"stage": "benchmark"},
    {"digest": "b" * 64},
    {"plan": "../../../absent.json"},
])
def test_reject_before_science(launch, overrides):
    run, _, _ = launch
    result = run(**overrides)
    assert result.returncode != 0
    assert '"argv"' not in result.stdout


def test_reject_output_in_code_or_existing(launch):
    run, _, code = launch
    for out in (code / "result", code / "plan.json"):
        result = run(out=out)
        assert result.returncode != 0 and '"argv"' not in result.stdout
