import importlib.util
import shlex
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mac_launcher_command.py"
spec = importlib.util.spec_from_file_location("mac_launcher_command", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    tree = tmp_path / "campagna con spazi"
    local = tree / "scripts/submit_job.py"
    shared = tmp_path / "shared/submit_job.py"
    for path in (local, shared):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"versioned")
    monkeypatch.setattr(launcher, "SUBMITTER", shared)

    def git_show(command):
        assert command == ["git", "-C", str(tree), "show", "a"*40 + ":scripts/submit_job.py"]
        return b"versioned"

    monkeypatch.setattr(launcher.subprocess, "check_output", git_show)
    return tree


@pytest.mark.parametrize("line", [1, 2, 3])
def test_preserves_frozen_args_cwd_and_shell_quoting(checkout, line):
    prefix = {
        1: ["python3", str(checkout / "scripts/submit_job.py")],
        2: ["python3", "scripts/submit_job.py"],
        3: [str(launcher.PYTHON), str(launcher.SUBMITTER)],
    }[line]
    tail = ["--id", "job", "--dopo", "previous", "--script-stdin", "scripts/job.sbatch",
            "--", "--export=ALL,PATH_WITH_SPACES=x y", "arg'with $characters"]
    rendered = launcher.render(checkout, "a"*40, prefix + tail)
    assert rendered["workdir"] == str(checkout)
    assert shlex.split(rendered["cmd"]) == [str(launcher.PYTHON), str(launcher.SUBMITTER), *tail]


@pytest.mark.parametrize("change_shared", [True, False])
def test_refuses_submitter_different_from_frozen_version(checkout, change_shared):
    source = launcher.SUBMITTER if change_shared else checkout / "scripts/submit_job.py"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="coincidere"):
        launcher.render(checkout, "a"*40, ["python3", "scripts/submit_job.py", "--id", "job", "--", "script"])
