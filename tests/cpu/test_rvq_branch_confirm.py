import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_branch_probe import _base_without_branch0_identity, _new_dataset  # noqa: E402

from wearusfm.tokenizer_checks.run_io import CHECKPOINT_SHA256  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rvq_branch_confirm.py"
_spec = importlib.util.spec_from_file_location("rvq_branch_confirm", SCRIPT)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _report(factor=21.72, em=0.0352, v1=1.23, sha=CHECKPOINT_SHA256, v2=None):
    return {"scale_calibration": {"factor": factor}, "v1": {"ratio_single_over_multi": v1},
            "v2": {"emg2pose": {"v2_median_nmse": em}, **(v2 or {})}, "provenance": {"checkpoint_sha256": sha}}


def test_comparability_checks():
    base = _report()
    assert all(ok for *_, ok in S.comparability(base, _report()).values())
    assert not S.comparability(base, _report(factor=20.04))["scale_factor"][2]
    assert not S.comparability(base, _report(em=0.0360))["emg2pose_median_nmse"][2]
    assert S.comparability(base, _report(em=0.03521))["emg2pose_median_nmse"][2]
    assert not S.comparability(base, _report(sha="0" * 64))["checkpoint_sha256"][2]


def _save(dirp, per):
    dirp.mkdir(parents=True)
    for n, d in per.items():
        np.savez(dirp / f"{n}.npz", **d)


def test_end_to_end_selects_by_v2_and_stops_if_not_comparable(tmp_path):
    per, cb = _base_without_branch0_identity()
    _save(tmp_path / "base", per)
    new = {"ninapro_db2": _new_dataset(np.random.default_rng(12), identity_in_branch0=True),
           "ninapro_db6": _new_dataset(np.random.default_rng(13), identity_in_branch0=False)}
    _save(tmp_path / "new", new)
    np.savez(tmp_path / "cb.npz", codebooks=cb, checkpoint_sha256=CHECKPOINT_SHA256)
    (tmp_path / "base.json").write_text(json.dumps(_report()))
    v2 = {"ninapro_db2": {"v2_ratio_vs_emg2pose": 1.5, "v2_passes": True}, "ninapro_db6": {"v2_ratio_vs_emg2pose": 2.1, "v2_passes": False}}
    (tmp_path / "step1bis_1.json").write_text(json.dumps(_report(v2=v2)))
    common = ["--new-arrays", str(tmp_path / "new"), "--base-report", str(tmp_path / "base.json"), "--base-arrays", str(tmp_path / "base"),
              "--codebooks", str(tmp_path / "cb.npz")]
    out = tmp_path / "o.json"
    assert S.main(["--new-report", str(tmp_path / "step1bis_1.json"), "--out", str(out), *common]) == 0
    r = json.loads(out.read_text())
    assert list(r["confirm"]) == ["ninapro_db2"]  # db6 ha V2 > 2: non si prova nemmeno
    assert r["confirm"]["ninapro_db2"]["enters"] is False and "ninapro_db2" not in r["final_enabled"]
    # run non confrontabile: si ferma senza provare nessun dataset
    (tmp_path / "step1bis_2.json").write_text(json.dumps(_report(factor=20.04, v2=v2)))
    out2 = tmp_path / "o2.json"
    assert S.main(["--new-report", str(tmp_path / "step1bis_2.json"), "--out", str(out2), *common]) == 2
    assert "confirm" not in json.loads(out2.read_text())
