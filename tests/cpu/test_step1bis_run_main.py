"""`scripts/step1bis_run.py main()` eseguita senza GPU: runner e pipeline finti. Il 30/09/2026 un collaudo su Leonardo e' fallito in 13 s per una
variabile che nascondeva `functools.partial` dentro `main()`: questo test esegue il percorso fino alla chiamata di `run_all`."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_processed import _data, _write  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "step1bis_run.py"


def _load():
    spec = importlib.util.spec_from_file_location("step1bis_run", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_builds_loaders_and_calls_run_all(tmp_path, monkeypatch):
    mod = _load()
    seen = {}

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

    def fake_run_all(runner, cfg, emg2pose, datasets, save_arrays_dir=None, on_progress=None):
        seen["datasets"] = {name: len(items) for name, (items, loader) in datasets.items()}
        name, (items, loader) = next(iter(datasets.items()))
        s = loader(*items[0])  # il loader costruito con functools.partial deve funzionare
        seen["loaded_shape"] = s.segments[0].shape
        on_progress({"progress": "prova"})
        return {"config": cfg.__dict__.copy()}

    monkeypatch.setattr(mod, "NeuroRVQRunner", FakeRunner)
    monkeypatch.setattr(mod, "run_all", fake_run_all)
    monkeypatch.setattr(mod, "_sha256", lambda p: "0" * 64)
    root = tmp_path / "processed"
    _write(root / "dsA" / "s01" / "session1", _data((100, 4)), scale=2.0)
    _write(root / "dsA" / "s02" / "session1", _data((100, 4)), scale=2.0)
    em = tmp_path / "em"
    em.mkdir()
    (em / "rec-1_left.hdf5").write_bytes(b"")  # main() cerca solo i file: non li apre (lo fa il loader, finto qui)
    csvp = tmp_path / "meta.csv"
    csvp.write_text("filename,user\nrec-1_left.hdf5,u1\n")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "out" / "r.json"
    monkeypatch.setattr(sys, "argv", ["step1bis_run.py", "--repo-dir", str(tmp_path), "--checkpoint", str(ckpt), "--processed-root", str(root),
                                      "--datasets", "dsA", "--emg2pose-dir", str(em), "--emg2pose-csv", str(csvp), "--out", str(out), "--smoke"])
    assert mod.main() == 0
    assert seen["datasets"] == {"dsA": 2} and seen["loaded_shape"] == (100, 4)
    assert json.loads(out.with_name("r.partial.json").read_text()) == {"progress": "prova"}
    rep = json.loads(out.read_text())
    assert rep["provenance"]["smoke"] is True and np.isclose(rep["config"]["n_groups"], 78)
