"""Launcher delle sonde P1/P2 (`scripts/slurm/probe_p{1,2}{,_probe}.sbatch`) con la regola dei checkout di campagna (critica del punto 5,
16/10/2026): codice da CODE_DIR con SHA e pulizia verificati (`lib_code_dir.sh`), uscite assolute e fuori dal codice, Python del venv."""

import os
import subprocess
from pathlib import Path

import pytest

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "scripts" / "slurm"


def _env(root):
    venv = root / "venv_wearusfm" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    fakebin = root / "venvpy"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "python3").write_text("#!/bin/sh\necho VENV_PYTHON \"$@\"\n")
    (fakebin / "python3").chmod(0o755)
    (venv / "activate").write_text(f"export PATH={fakebin}:$PATH\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA", "RAW_ROOT")}
    return {**env, "WORK": str(root), "SCRATCH": str(root / "scratch")}


@pytest.mark.parametrize("name,args,expected", [
    ("probe_p1", ["/C/ckpt.pt", "{root}/out/f1"], "VENV_PYTHON scripts/probe_p1.py --stage extract --checkpoint /C/ckpt.pt"),
    ("probe_p1_probe", ["/C/ckpt.pt", "{root}/out/f1", "{root}/out/p1.json"], "VENV_PYTHON scripts/probe_p1.py --stage probe"),
    ("probe_p2", ["/C/ckpt.pt", "{root}/out/f2"], "VENV_PYTHON scripts/probe_p2.py --stage extract --checkpoint /C/ckpt.pt"),
    ("probe_p2_probe", ["/C/ckpt.pt", "{root}/out/f2", "{root}/out/p2.json"], "VENV_PYTHON scripts/probe_p2.py --stage probe"),
])
def test_probe_launchers_use_the_code_dir_rule(work, name, args, expected):  # noqa: F811
    root, canon, first, second = work
    env = _env(root)
    launcher = SLURM / f"{name}.sbatch"
    args = [a.format(root=root) for a in args]
    r = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True)
    assert expected in r.stdout, r.stdout + r.stderr  # canonico pulito: accettato
    bad = list(args)
    bad[1] = "rel/feat"
    r = subprocess.run(["bash", str(launcher), *bad], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "relativa" in r.stdout and "VENV_PYTHON" not in r.stdout
    bad[1] = f"{root}/wearusfm/feat"  # dentro il codice
    r = subprocess.run(["bash", str(launcher), *bad], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "dentro il codice" in r.stdout
    if len(args) == 3:
        bad = [args[0], args[1], "out.json"]
        r = subprocess.run(["bash", str(launcher), *bad], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "relativo" in r.stdout
    other = root / "altro_checkout"  # un checkout di campagna senza EXPECTED_SHA: rifiutato da lib_code_dir.sh
    subprocess.run(["git", "-C", str(canon), "worktree", "add", "-q", "--detach", str(other), second], check=True, capture_output=True)
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(other)}, capture_output=True, text=True)
    assert r.returncode == 2 and "senza EXPECTED_SHA" in r.stderr and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(other), "EXPECTED_SHA": first}, capture_output=True,
                       text=True)
    assert r.returncode == 2 and "atteso" in r.stderr  # SHA diverso da quello del checkout


def test_p2_extraction_reads_the_raw_root_override(work):  # noqa: F811
    root = work[0]
    env = _env(root)
    r = subprocess.run(["bash", str(SLURM / "probe_p2.sbatch"), "/C/ckpt.pt", f"{root}/out/f2"], env={**env, "RAW_ROOT": "/R/raw"},
                       capture_output=True, text=True)
    assert "--epn-root /R/raw/epn612/EMG-EPN612 Dataset/trainingJSON" in r.stdout and "--uci-root /R/raw/uci_emg" in r.stdout
