"""scripts/make_splits_draft.py: regole della bozza D9 (d) sugli split dei soggetti."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("make_splits_draft", Path(__file__).resolve().parents[2] / "scripts" / "make_splits_draft.py")
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)

E2Q_TEST = sorted(f"u{i}" for i in json.loads((MS.OFFICIAL / "emg2qwerty_users.json").read_text())["test_users"].values())


def _by():
    return {"zhang2026": [f"HG_{i:02d}" for i in range(62)], "csl_hdemg": [f"s{i:02d}" for i in range(1, 6)],
            "ninapro_db7": [f"s{i:02d}" for i in range(1, 23)], "ninapro_db10": ["s010", "s108"],
            "emg2qwerty": E2Q_TEST + [f"u9{i:07d}" for i in range(100)], "kaifosh": [f"u{i:03d}" for i in range(100)]}


def test_deterministic_and_disjoint():
    a, b = MS.make(_by(), 0), MS.make(_by(), 0)
    assert a == b
    for r in a.values():
        assert not set(r["pretraining"]) & set(r["test"])
    assert MS.make(_by(), 1)["zhang2026"]["test"] != a["zhang2026"]["test"]


def test_fraction_minimum_and_stratification():
    r = MS.make(_by(), 0)
    assert len(r["zhang2026"]["test"]) == 13 and len(r["csl_hdemg"]["test"]) == 1  # ceil(0,2 x 62); CSL: 1 su 5 (decisione di Simone)
    assert len(MS.make({"hyser": [f"s{i:02d}" for i in range(1, 6)]}, 0)["hyser"]["test"]) == 2  # altrove almeno 2 anche con 5 soggetti
    db10 = r["ninapro_db10"]  # dai 45 attesi, non dai 2 gia' ingeriti
    assert len(db10["pretraining"]) + len(db10["test"]) == 45
    assert len(db10["test"]) == 9
    for seed in range(10):  # stratificato: sempre 3 amputati su 9 (20% di 15 e 20% di 30), con ogni seme
        t = MS.make(_by(), seed)["ninapro_db10"]["test"]
        assert sum(int(s[1:]) >= 101 for s in t) == 3


def test_official_splits_ninapro_overlap_and_benchmark():
    r = MS.make(_by(), 0)
    assert r["emg2qwerty"]["test"] == E2Q_TEST and r["emg2qwerty"]["rule"] == "split ufficiale"
    assert r["ninapro_db7"]["test"] == [] and len(r["ninapro_db7"]["pretraining"]) == 22
    assert r["kaifosh"]["role"] == "benchmark" and r["kaifosh"]["pretraining"] == [] and len(r["kaifosh"]["benchmark"]) == 100


def test_nested_subsets_are_nested_prefixes():
    n = MS.make(_by(), 0)["zhang2026"]["nested"]
    assert set(n["0.125"]) <= set(n["0.25"]) <= set(n["0.5"])
    assert [len(n[k]) for k in ("0.125", "0.25", "0.5")] == [7, 13, 25]
