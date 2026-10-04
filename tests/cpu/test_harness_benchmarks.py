"""Benchmark del passo 5 (EPN-612, UCI-EMG) nel pacchetto: stessi dati per la replica di NeuroRVQ e per la sonda sul nostro FM."""

import json
from pathlib import Path

import numpy as np

from wearusfm.harness import benchmarks as B


def _epn_user(d: Path, name: str, n=2, t=1000):
    rng = np.random.default_rng(0)
    samples = {}
    for i, g in enumerate(("fist", "open") * n):
        samples[f"idx_{i}"] = {"gestureName": g, "emg": {f"ch{c}": rng.normal(size=t).tolist() for c in range(1, 9)}}
    (d / name).mkdir(parents=True)
    (d / name / f"{name}.json").write_text(json.dumps({"generalInfo": {"samplingFrequencyInHertz": 200}, "trainingSamples": samples,
                                                        "testingSamples": {"idx_0": {"emg": {}}}}))


def test_epn_users_of_both_folders_are_distinct_subjects(tmp_path):
    tr, te = tmp_path / "trainingJSON", tmp_path / "testingJSON"
    _epn_user(tr, "user1")
    _epn_user(te, "user1")  # stesso numero, persona diversa
    w, y, s, fs = B.load_epn612([tr, te])
    assert w.shape == (8, 1000, 8) and fs == 200.0 and set(s) == {"trainingJSON/user1", "testingJSON/user1"}
    assert set(y) == {"fist", "open"}


def test_uci_windows_one_second_no_overlap_classes_1_to_6(tmp_path):
    d = tmp_path / "EMG_data" / "01"
    d.mkdir(parents=True)
    lab = np.repeat([0, 1, 7, 2], 2000)  # 2 s per classe a 1 kHz
    rows = ["time\tch1\tch2\tch3\tch4\tch5\tch6\tch7\tch8\tclass"]
    rows += ["\t".join([str(i)] + ["0.1"] * 8 + [str(int(c))]) for i, c in enumerate(lab)]
    (d / "1_raw_data.txt").write_text("\n".join(rows) + "\n")
    w, y, s, fs = B.load_uci_emg(tmp_path)
    assert fs == 1000.0 and w.shape == (4, 1000, 8) and sorted(y.tolist()) == [1, 1, 2, 2] and set(s) == {"01"}  # niente 0 e niente 7
