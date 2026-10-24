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


def test_gate_d8_launcher_uses_the_code_dir_rule_and_forwards_the_window(work):  # noqa: F811
    """Gate D8 riaperto per il banco da 150 ms (R5 del pilot degli obiettivi, 24/10/2026): stesso launcher, codice da CODE_DIR con SHA verificato
    (prima faceva `cd $WORK/wearusfm` e basta), argomenti passati allo script, esito fuori dal codice."""
    root, canon, first, second = work
    env = {**_env(root), "KAIFOSH_ROOT": str(root / "kaifosh"), "SLURM_JOB_ID": "7"}
    (root / "kaifosh").mkdir()
    launcher = SLURM / "gate_d8.sbatch"
    r = subprocess.run(["bash", str(launcher), "--window-ms", "150"], env=env, capture_output=True, text=True)
    assert "VENV_PYTHON scripts/gate_d8.py" in r.stdout and "--window-ms 150" in r.stdout, r.stdout + r.stderr
    assert f"--out {root}/wearusfm_runs/results/step3/gate_7.json" in r.stdout and "codice:" in r.stdout
    r = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert "VENV_PYTHON scripts/gate_d8.py" in r.stdout and "--window-ms" not in r.stdout  # il banco firmato, come prima
    other = root / "altro_checkout"
    subprocess.run(["git", "-C", str(canon), "worktree", "add", "-q", "--detach", str(other), second], check=True, capture_output=True)
    r = subprocess.run(["bash", str(launcher), "--window-ms", "150"], env={**env, "CODE_DIR": str(other)}, capture_output=True, text=True)
    assert r.returncode == 2 and "senza EXPECTED_SHA" in r.stderr and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), "--window-ms", "150"], env={**env, "CODE_DIR": str(other), "EXPECTED_SHA": first},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "atteso" in r.stderr and "VENV_PYTHON" not in r.stdout


def test_p2_extraction_reads_the_raw_root_override(work):  # noqa: F811
    root = work[0]
    env = _env(root)
    r = subprocess.run(["bash", str(SLURM / "probe_p2.sbatch"), "/C/ckpt.pt", f"{root}/out/f2"], env={**env, "RAW_ROOT": "/R/raw"},
                       capture_output=True, text=True)
    assert "--epn-root /R/raw/epn612/EMG-EPN612 Dataset/trainingJSON" in r.stdout and "--uci-root /R/raw/uci_emg" in r.stdout


def test_spectral_diagnostic_launcher_reads_the_2c_caches_and_writes_outside_code_and_caches(work):  # noqa: F811
    """Diagnostica dei rami (docs/rami_diagnostica_v1.md): radici delle cache del 2C fisse, rapporti fuori dal codice e dalle cache."""
    root = work[0]
    env = _env(root)
    launcher = SLURM / "spectral_diagnostic.sbatch"
    out = f"{root}/out/diag"
    r = subprocess.run(["bash", str(launcher), out, "probe", "--task", "p1", "--arm", "B_seed0", "--sets", "R"], env=env, capture_output=True,
                       text=True)
    assert ("VENV_PYTHON scripts/spectral_diagnostic.py probe --eval-root "
            "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410 --downstream-root "
            "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_downstream_0910 --out-dir " + out) in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), out, "compare"], env=env, capture_output=True, text=True)
    assert f"VENV_PYTHON scripts/spectral_diagnostic.py compare --out-dir {out} --out {out}/esito.json" in r.stdout  # esito fisso
    for extra in (["--out", "esito.json"], ["--out=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410/e.json"],
                  ["--out-dir", "rel"], ["--eval-root", "/x"], ["--downstream-root=/y"], ["--splits", "s.json"]):
        r = subprocess.run(["bash", str(launcher), out, "compare", *extra], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "non ammesso" in r.stdout and "VENV_PYTHON" not in r.stdout  # review del critico del 20/10
    for bad, msg in (("rel/diag", "relativa"), (f"{root}/wearusfm/diag", "dentro il codice"),
                     ("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step2c_1410/diag", "dentro le cache")):
        r = subprocess.run(["bash", str(launcher), bad, "probe"], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and msg in r.stdout and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), out, "extract"], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "sconosciuta" in r.stdout
