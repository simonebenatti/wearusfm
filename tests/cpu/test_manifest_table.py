"""scripts/manifest_table.py: passaggi per classe e ripartizione per dataset con tetto e ridistribuzione."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("manifest_table", Path(__file__).resolve().parents[2] / "scripts" / "manifest_table.py")
MT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MT)


def _rows():
    return [{"key": "big", "class": "B", "hours": 900.0}, {"key": "small", "class": "B", "hours": 10.0},
            {"key": "a1", "class": "A", "hours": 60.0}, {"key": "a2", "class": "A", "hours": 30.0}]


def test_passes():
    assert MT.passes(0.4, 110.0, 1100.0) == pytest.approx(16.0)


def test_allocate_caps_small_dataset_and_redistributes_within_class():
    a = MT.allocate(_rows(), {"A": 0.1, "B": 0.9}, alpha=0.5, max_passes=8.0, hours_total=1000.0)  # A: 90 h, al massimo 18%
    pd = a["per_dataset"]
    assert pd["small"][1] == pytest.approx(8.0)  # al tetto: 8 x 10 / 4000 = 2%, il resto della classe va a "big"
    assert pd["small"][0] + pd["big"][0] == pytest.approx(0.9) and pd["a1"][0] + pd["a2"][0] == pytest.approx(0.1)
    assert a["unused_quota"] == {"A": pytest.approx(0.0), "B": pytest.approx(0.0)}
    assert all(p <= 8.0 + 1e-9 for _, p in pd.values())


def test_allocate_reports_unusable_quota_when_whole_class_is_capped():
    a = MT.allocate(_rows(), {"A": 0.5, "B": 0.5}, alpha=1.0, max_passes=8.0, hours_total=1000.0)
    assert a["unused_quota"]["A"] == pytest.approx(0.5 - 8.0 * 90.0 / 4000.0)  # A ha 90 h: al massimo 8 x 90 / 4000 = 18%


def test_alpha_zero_is_uniform_when_no_cap_binds():
    a = MT.allocate(_rows()[2:], {"A": 0.01}, alpha=0.0, max_passes=8.0, hours_total=1000.0)
    assert a["per_dataset"]["a1"][0] == pytest.approx(a["per_dataset"]["a2"][0])
