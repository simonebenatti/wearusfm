"""B must start from zero: rejection happens before module setup, data loading or torchrun."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("case", ["existing", "branch", "resume", "init", "branch_arg"])
def test_new_run_refuses_existing_state_before_loading_cluster_environment(tmp_path, case):
    run = tmp_path / "run"
    if case == "existing":
        run.mkdir()
        (run / "checkpoint.pt").write_bytes(b"preserve me")
    env = {**os.environ, "REQUIRE_NEW_RUN": "1", "WORK": str(tmp_path), "SCRATCH": str(tmp_path),
           "BRANCH_FROM": "", "RESUME_TO": "", "EXTRA_ARGS": ""}
    env.update({"branch": {"BRANCH_FROM": "/some/checkpoint.pt"}, "resume": {"RESUME_TO": "5000"},
                "init": {"EXTRA_ARGS": "--init-from=/some/checkpoint.pt"},
                "branch_arg": {"EXTRA_ARGS": "--branch-from /some/checkpoint.pt"}}.get(case, {}))
    script = Path(__file__).resolve().parents[2] / "scripts/slurm/train_ddp.sbatch"
    result = subprocess.run(["bash", str(script), "/manifest", "/scales", str(run), "5000"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "REQUIRE_NEW_RUN" in result.stdout
    assert "module" not in result.stderr and "torchrun" not in result.stderr
    if case == "existing":
        assert (run / "checkpoint.pt").read_bytes() == b"preserve me"
    else:
        assert not run.exists()
