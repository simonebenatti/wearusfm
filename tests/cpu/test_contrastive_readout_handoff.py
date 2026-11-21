"""Exercise the login check at the real reader's argument-parsing boundary."""
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "results/infra/contrastive_targets_20261118"
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "contrastive_readout_handoff", PACKAGE / "readout_handoff_20261121/validate_readout.py"
)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


@pytest.fixture
def readout_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    root = tmp_path / "run archive"
    full = {arm: root / "benchmark" / arm for arm in ("C0", "CN", "CR")}
    reloads = {cell: root / "reloads" / cell for cell in ("CN_A", "CN_B", "CR_A", "CR_B")}
    command = [
        sys.executable, str(Path(handoff.reader.__file__).resolve()),
        "--cohort", str(PACKAGE / "releases/benchmark_20261121/cohort/cohort.json"),
        "--cuda-report", str(root / "cuda.json"),
        "--out", str(root / "preflight_readout.json"),
        "--cpu-audit", str(root / "audit.json"),
        "--cpu-audit-sacct", str(PACKAGE / "audit_execution_20261120/audit_accounting.json"),
        "--calibration-report", str(root / "calibration_report.json"),
        "--calibration-run-dir", str(root / "calibration"),
    ]
    for flag, paths in (("--full", full), ("--resumed", reloads)):
        for arm, folder in paths.items():
            command.extend([flag, f"{arm}={folder}"])
    return command, root, full, reloads


def test_real_reader_accepts_all_seven_paths_without_writing_a_report(readout_cli):
    command, root, full, reloads = readout_cli
    handoff.validate_reader_command(command, root, full, reloads)
    assert not (root / "preflight_readout.json").exists()


def test_real_reader_rejects_swapped_paths_with_unchanged_replica_labels(readout_cli):
    command, root, full, reloads = readout_cli
    a = command.index(f"CN_A={reloads['CN_A']}")
    b = command.index(f"CN_B={reloads['CN_B']}")
    command[a] = f"CN_A={reloads['CN_B']}"
    command[b] = f"CN_B={reloads['CN_A']}"
    with pytest.raises(ValueError, match="arm/receipt paths"):
        handoff.validate_reader_command(command, root, full, reloads)
    assert not (root / "preflight_readout.json").exists()


def test_real_reader_rejects_wrong_audit_receipt(readout_cli):
    command, root, full, reloads = readout_cli
    command[command.index("--cpu-audit") + 1] = str(root / "other_audit.json")
    with pytest.raises(ValueError, match="arm/receipt paths"):
        handoff.validate_reader_command(command, root, full, reloads)
    assert not (root / "preflight_readout.json").exists()
