"""Sonda e verdetto del gate D8, senza torch (scikit-learn e torch insieme vanno in segfault su macOS)."""

import numpy as np

from wearusfm.gate import consistency as G


def _xy(n_subj=10, per=10, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, us = [], [], []
    for s in range(n_subj):
        for _ in range(per):
            base = rng.normal(size=20)
            for label in (0, 1):
                xs.append(base + rng.normal(scale=0.01, size=20) + label * shift)
                ys.append(label)
                us.append(f"s{s}")
    return np.array(xs), np.array(ys), np.array(us)


def test_probe_at_chance_when_versions_are_identical_and_high_when_shifted():
    x, y, u = _xy()
    p = G.probe_case(x, y, u)
    assert p["chance"] == 0.5 and p["balanced_accuracy"] <= 0.65
    x2, y2, u2 = _xy(shift=1.0)
    assert G.probe_case(x2, y2, u2)["balanced_accuracy"] > 0.9


def test_probe_stage_records_the_bank_from_the_features_not_from_its_own_flags(tmp_path):
    """Gate riaperto per il banco da 150 ms (R5, 24/10/2026): il JSON dell'esito dice quale banco e' stato provato, letto da errors.json della
    fase features (torch), non dal flag della fase probe lanciata da sola; un flag in contrasto ferma la sonda."""
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("gate_d8_script", Path(__file__).resolve().parents[2] / "scripts" / "gate_d8.py")
    GD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GD)
    x, y, u = _xy()
    rel = {"total": 0.01, "fourier": 0.02, "spline": 0.0, "mlp": 0.0}
    for bank, label in (({"window_ms": 150.0, "n_harmonics": 156, "n_knots": 120, "mlp_omega0": 36.0}, "finestra del kernel di 150 ms"),
                        (None, "firmato")):
        work = tmp_path / ("w150" if bank else "w25")
        work.mkdir()
        np.savez(work / "case_seed0.npz", x=x, y=y, units=u)
        e = {"errors": {"case_seed0": rel}, "n_windows": 3, "elapsed_s": 1.0}
        if bank:
            e["frontend"] = bank
        (work / "errors.json").write_text(json.dumps(e))
        out = work / "gate.json"
        args = ["--processed-root", str(tmp_path), "--out", str(out), "--work", str(work), "--stage", "probe"]
        assert GD.main(args + (["--window-ms", "150"] if bank else [])) in (0, 1)
        res = json.loads(out.read_text())
        assert res["bank"].startswith(label) and res["frontend"]["window_ms"] == (bank["window_ms"] if bank else None)
        if bank:
            assert res["frontend"]["n_harmonics"] == 156 and "case_seed0" in res["cases"]
    with __import__("pytest").raises(SystemExit, match="banco"):  # flag della probe in contrasto con le feature
        GD.main(["--processed-root", str(tmp_path), "--out", str(tmp_path / "x.json"), "--work", str(tmp_path / "w25"), "--stage", "probe",
                 "--window-ms", "150"])


def test_verdict_rules():
    ok_err = {"total": 0.01, "fourier": 0.02, "spline": 0.0, "mlp": 0.0}
    ok_probe = {"balanced_accuracy": 0.52, "ci95": [0.45, 0.58]}
    assert G.verdict(ok_err, ok_probe)["passes"]
    assert not G.verdict({**ok_err, "spline": 0.051}, ok_probe)["passes"]  # una famiglia oltre il 5%: non passa (4a firmata)
    assert not G.verdict(ok_err, {"balanced_accuracy": 0.56, "ci95": [0.50, 0.62]})["passes"]
    assert not G.verdict(ok_err, {"balanced_accuracy": 0.54, "ci95": [0.51, 0.57]})["passes"]  # l'intervallo non contiene 0,5
