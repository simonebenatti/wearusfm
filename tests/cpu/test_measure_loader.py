"""scripts/measure_loader.py lanciato come processo, sull'albero sintetico del test del dataloader: scale di sessione e misura del ritmo."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_pretraining_loader import SIGMA, _tree

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_loader.py"


def test_scales_and_rate_end_to_end(tmp_path):
    root, mpath = _tree(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out), "--workers", "2",
                        "--batches", "2", "--batch-size", "3"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.loads((out / "session_scales.json").read_text())
    assert sc["n_sessions"] == 3 and not sc["errors"] and set(sc["scales"]) == {"emg2pose/u1/sessA", "ninapro_db2/s01/session1", "capgmyo/s01/s"}
    for key, fs in (("emg2pose/u1/sessA", 2000.0), ("ninapro_db2/s01/session1", 2000.0), ("capgmyo/s01/s", 1000.0)):
        kept = (min(450.0, 0.45 * fs) - 20.0) / (fs / 2.0)  # rumore bianco filtrato 20-450 Hz: resta questa frazione di potenza
        assert sc["scales"][key] == pytest.approx(0.6745 * SIGMA * kept ** 0.5, rel=0.1)  # MAD = 0,6745 sigma (meno un poco per i notch)
    th = json.loads((out / "throughput.json").read_text())
    assert [run["filter"] for run in th["runs"]] == [None, [20.0, 450.0]]
    for run in th["runs"]:
        assert len(run["per_process_windows_per_s"]) == 2 and all(v > 0 for v in run["per_process_windows_per_s"])
        assert abs(sum(run["stage_fraction"].values()) - 1.0) < 1e-9 and "lettura_e_filtro" in run["stage_fraction"]
    assert "ritmo, filtro" in r.stdout


def test_scales_from_a_previous_run_are_reused(tmp_path):
    root, mpath = _tree(tmp_path)
    prev = tmp_path / "prev.json"
    prev.write_text(json.dumps({"scales": {"emg2pose/u1/sessA": 123.0}}))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out), "--workers", "1",
                        "--batches", "1", "--batch-size", "2", "--scales-from", str(prev)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.loads((out / "session_scales.json").read_text())
    assert sc["reused"] == 1 and sc["scales"]["emg2pose/u1/sessA"] == 123.0 and len(sc["scales"]) == 3


def test_bipolar_scales_of_virtual_montages(tmp_path):
    """--bipolar-scales: per le sessioni di classe C anche le scale delle derivazioni bipolari (D6a), uguali a quelle che il dataloader
    calcolerebbe al volo; quelle gia' note non si ricalcolano."""
    import numpy as np

    from wearusfm.data import pretraining_loader as L
    from wearusfm.data import virtual_montage as VM

    root, mpath = _tree(tmp_path)
    prev = tmp_path / "prev.json"
    known = {"emg2pose/u1/sessA": 1.0, "ninapro_db2/s01/session1": 1.0, "capgmyo/s01/s": 1.0}
    prev.write_text(json.dumps({"scales": known}))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out), "--workers", "1",
                        "--scales-from", str(prev), "--bipolar-scales", "--skip-rate"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.loads((out / "session_scales.json").read_text())
    bip = {k: v for k, v in sc["scales"].items() if "|bip:" in k}
    assert sc["reused"] == 2 and not sc["errors"]
    assert set(bip) == {f"capgmyo/s01/s|bip:{a}:{s}" for a in VM.AXES for s in (1, 2)}
    idx = L.ManifestIndex.load(mpath, [root])
    row = next(r for r in idx.rows if r["dataset"] == "capgmyo")
    loader = L.PretrainLoader(idx, L.signed_config((20.0, 450.0)))
    view = loader.view(row)
    vm = VM.draw(view.montage, view.qc_valid, VM.VirtualSpec(p=1.0, p_bipolar=1.0), np.random.default_rng(0))
    assert loader.scale(row, view, np.random.default_rng(9), vm) == pytest.approx(bip[f"capgmyo/s01/s|{vm.scale_key}"], rel=1e-6)
    mono = loader.scale(row, view, np.random.default_rng(9))
    assert bip[f"capgmyo/s01/s|{vm.scale_key}"] == pytest.approx(mono * 2 ** 0.5, rel=0.1)  # rumore indipendente: la differenza ha sigma*sqrt(2)
