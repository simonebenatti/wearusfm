import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_processed import _write  # noqa: E402

from wearusfm.data.processed import validate_session  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qc_relative_check.py"
_spec = importlib.util.spec_from_file_location("qc_relative_check", SCRIPT)
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)


def _signal(n=4000, c=4, dead=(), amp=2000.0, dead_amp=0.0, seed=0):
    """(n, c) int16: rumore forte, con i canali `dead` ridotti a dead_amp (0 = piatto, >0 = quasi morto ma con un po' di rumore)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=amp, size=(n, c))
    for d in dead:
        x[:, d] = rng.normal(scale=dead_amp, size=n) if dead_amp else 0.0
    return np.round(x).astype(np.int16)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_channel_std_matches_numpy_over_chunks_layouts_and_scale(tmp_path):
    rng = np.random.default_rng(1)
    x2 = rng.integers(-3000, 3000, size=(1003, 5)).astype(np.int16)
    x3 = rng.integers(-3000, 3000, size=(7, 141, 5)).astype(np.int16)
    for arr in (x2, x3):
        ref = arr.reshape(-1, 5).astype(np.float64).std(axis=0)
        assert np.allclose(Q.channel_std(arr, chunk_elements=37), ref, rtol=1e-10)  # molti blocchi piccoli
        assert np.allclose(Q.channel_std(arr, chunk_elements=10**9), ref, rtol=1e-10)  # un solo blocco
    sc = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert np.allclose(Q.channel_std(x2, sc), (x2.astype(np.float64) / sc).std(axis=0), rtol=1e-10)
    assert np.allclose(Q.channel_std(x2, 4.0), x2.astype(np.float64).std(axis=0) / 4.0, rtol=1e-10)


def test_analyze_flags_dead_channel_that_absolute_threshold_missed(tmp_path):
    # canale 2 quasi morto (rumore 0,3 codici contro 2000): l'ingest con soglia assoluta 1e-6 lo lascia valido
    d = _write(tmp_path / "s", _signal(dead=(2,), dead_amp=0.3), qc=(1, 1, 1, 1))
    r = Q.analyze_session(d)
    assert r["new_flat"] == [2] and r["relative_flat"] == [2] and r["currently_invalid"] == []
    assert r["ratio_min"] < 1e-3


def test_analyze_ignores_already_invalid_and_healthy_channels(tmp_path):
    d = _write(tmp_path / "s", _signal(dead=(1,)), qc=(1, 0, 1, 1))  # il canale 1 e' gia' segnato non valido
    r = Q.analyze_session(d)
    assert r["relative_flat"] == [1] and r["new_flat"] == [] and r["currently_invalid"] == [1]
    d2 = _write(tmp_path / "t", _signal(), qc=(1, 1, 1, 1))  # tutti sani: nessun falso allarme
    assert Q.analyze_session(d2)["new_flat"] == []


def test_low_but_alive_channel_is_not_flagged(tmp_path):
    # un canale a 1/20 dell'ampiezza tipica e' debole ma vivo: la soglia e' 1e-3 x mediana, non lo tocca
    x = _signal()
    x[:, 3] = (x[:, 3] / 20).astype(np.int16)
    assert Q.analyze_session(_write(tmp_path / "s", x, qc=(1, 1, 1, 1)))["new_flat"] == []


def test_apply_revision_documents_backs_up_and_keeps_data_untouched(tmp_path):
    d = _write(tmp_path / "s", _signal(dead=(3,), dead_amp=0.3), qc=(1, 1, 1, 1))  # gruppi da 2+2: colonna 3 = secondo canale del 2o gruppo
    before_data, before_meta = _sha(d / "data_int16.npy"), (d / "metadata.json").read_text()
    Q.apply_revision(d, [3], "2026-09-30T12:00:00Z")
    meta = json.loads((d / "metadata.json").read_text())
    flags = [c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]]
    assert flags == [True, True, True, False] and meta["n_channels_discarded_by_qc"] == 1
    rev = meta["qc_revisions"][0]
    assert rev["added_invalid_columns"] == [3] and rev["date"] == "2026-09-30T12:00:00Z" and "Simone" in rev["decision"]
    assert _sha(d / "data_int16.npy") == before_data  # i dati non si toccano
    assert (d / Q.BACKUP_NAME).read_text() == before_meta  # reversibile
    assert validate_session(d) == []
    # idempotente: dopo la correzione il canale non e' piu' new_flat
    assert Q.analyze_session(d)["new_flat"] == []
    # un secondo intervento non sovrascrive il backup originale
    Q.apply_revision(d, [0], "2026-10-01T00:00:00Z")
    assert (d / Q.BACKUP_NAME).read_text() == before_meta and len(json.loads((d / "metadata.json").read_text())["qc_revisions"]) == 2


def test_apply_revision_restores_sidecar_if_result_is_not_conforming(tmp_path):
    # sessione GIA' non conforme (shape del sidecar sbagliata): dopo la correzione validate_session fallisce e il sidecar torna com'era
    d = _write(tmp_path / "s", _signal(dead=(2,), dead_amp=0.3), qc=(1, 1, 1, 1), shape=[9, 9])
    original = (d / "metadata.json").read_text()
    with pytest.raises(RuntimeError, match="non e' conforme"):
        Q.apply_revision(d, [2], "2026-09-30T12:00:00Z")
    assert (d / "metadata.json").read_text() == original
    assert "qc_revisions" not in json.loads((d / "metadata.json").read_text())


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_cli_read_only_reports_and_writes_nothing(tmp_path):
    _write(tmp_path / "ds" / "s01" / "a", _signal(dead=(2,), dead_amp=0.3), qc=(1, 1, 1, 1))
    _write(tmp_path / "ds" / "s02" / "a", _signal(seed=5), qc=(1, 1, 1, 1))
    before = {p: p.read_bytes() for p in (tmp_path / "ds").rglob("*") if p.is_file()}
    rep = tmp_path / "rep.json"
    r = _run("--scan", tmp_path, "--report", rep)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(rep.read_text())["datasets"]["ds"]
    assert d["n_sessions"] == 2 and d["n_sessions_with_new_flat"] == 1 and d["n_new_flat_channels"] == 1
    assert d["sessions_with_new_flat"]["s01/a"]["new_flat"] == [2]
    assert before == {p: p.read_bytes() for p in (tmp_path / "ds").rglob("*") if p.is_file()}  # sola lettura davvero


def test_cli_guard_blocks_apply_then_force_allows_it(tmp_path):
    # 1 sessione su 2 con nuovi canali piatti: oltre il 5% delle sessioni -> il freno scatta
    _write(tmp_path / "ds" / "s01" / "a", _signal(dead=(2,), dead_amp=0.3), qc=(1, 1, 1, 1))
    _write(tmp_path / "ds" / "s02" / "a", _signal(seed=5), qc=(1, 1, 1, 1))
    before = (tmp_path / "ds" / "s01" / "a" / "metadata.json").read_text()
    rep = tmp_path / "rep.json"
    r = _run("--scan", tmp_path, "--apply", "--report", rep)
    assert r.returncode == 2 and "NON APPLICATO" in r.stdout
    assert (tmp_path / "ds" / "s01" / "a" / "metadata.json").read_text() == before
    r2 = _run("--scan", tmp_path, "--apply", "--force-dataset", "ds", "--report", rep)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    meta = json.loads((tmp_path / "ds" / "s01" / "a" / "metadata.json").read_text())
    assert [c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]] == [True, True, False, True]
    r3 = _run("--scan", tmp_path, "--report", rep)  # dopo la correzione: niente piu' da segnalare
    assert json.loads(rep.read_text())["datasets"]["ds"]["n_new_flat_channels"] == 0


def test_cli_apply_below_guard_and_only_filter(tmp_path):
    for i in range(30):  # 1 sessione su 30 (3,3% < 5%) con un canale su 4 (25%, non oltre il 25%): il freno non scatta
        dead = (2,) if i == 0 else ()
        _write(tmp_path / "ds" / f"s{i:02d}" / "a", _signal(n=800, dead=dead, dead_amp=0.3, seed=i), qc=(1, 1, 1, 1))
    _write(tmp_path / "altro" / "s01" / "a", _signal(dead=(1,), dead_amp=0.3), qc=(1, 1, 1, 1))
    rep = tmp_path / "rep.json"
    r = _run("--scan", tmp_path, "--only", "ds", "--apply", "--report", rep)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(rep.read_text())["datasets"]
    assert list(d) == ["ds"] and d["ds"]["apply"] == "applicato a 1 sessioni"
    other = json.loads((tmp_path / "altro" / "s01" / "a" / "metadata.json").read_text())
    assert "qc_revisions" not in other  # il filtro --only lascia stare gli altri dataset
