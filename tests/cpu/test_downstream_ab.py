"""Paired inference must pool class counts, preserve pairing, and reject cache drift."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch  # before sklearn

from wearusfm.harness.downstream_ab import bacc, compare_reports, fingerprint, paired_component, subject_counts


def _stats(gain):
    y = np.tile([0, 0, 1, 1], 4)
    p = y.copy()
    if not gain:
        p[::4] = 1
    return subject_counts(y, p, np.repeat(["u0", "u1", "u2", "u3"], 4))


def _reports():
    reports = {"p1": {}, "p2": {}}
    for task, datasets in (("p1", ("ninapro_db2", "ninapro_db3", "ninapro_db6")), ("p2", ("epn612", "uci_emg"))):
        for arm in ("A_seed0", "B_seed0", "A_seed1", "B_seed1"):
            gain = arm.startswith("B")
            score = 1. if gain else .75
            row = {"test_bacc": score, "paired_counts": _stats(gain)}
            ds = {d: copy.deepcopy(row) if task == "p1" else {
                "test_bacc": score, "splits": [dict(copy.deepcopy(row), split_seed=s) for s in (0, 1, 2)]} for d in datasets}
            reports[task][arm] = {task: score, "datasets": ds}
    return reports


def test_pooled_bacc_is_not_mean_subject_bacc():
    n = np.array([[90, 10], [10, 90]])
    hit = np.array([[90, 0], [0, 90]])
    assert bacc(n, hit) == pytest.approx(.9)
    assert np.mean([bacc(n[i:i+1], hit[i:i+1]) for i in range(2)]) == pytest.approx(.5)


def test_paired_bootstrap_and_guards():
    reports = _reports()
    out = compare_reports(reports, n_boot=40)
    assert out["both_practical_guards_pass"]
    for task in ("p1", "p2"):
        assert out["tasks"][task]["delta"] == pytest.approx(.25)
        assert out["tasks"][task]["ci95"] == pytest.approx([.25, .25])
        reports[task]["A_seed0"], reports[task]["B_seed0"] = reports[task]["B_seed0"], reports[task]["A_seed0"]
        reports[task]["A_seed1"], reports[task]["B_seed1"] = reports[task]["B_seed1"], reports[task]["A_seed1"]
    assert not compare_reports(reports, n_boot=40)["both_practical_guards_pass"]


def test_identical_models_have_zero_paired_uncertainty():
    stats = _stats(False)
    stats["subjects"]["u0"]["correct"] = [0, 0]
    result, draws = paired_component([stats]*4, np.random.default_rng(3), 100)
    assert result["delta"] == 0
    assert np.array_equal(draws, np.zeros(100))


@pytest.mark.parametrize("change", ["counts", "classes", "subjects", "accuracy", "split"])
def test_reject_unpaired_reports(change):
    reports = _reports()
    row = reports["p2"]["B_seed1"]["datasets"]["epn612"]["splits"][0]
    if change == "counts":
        row["paired_counts"]["subjects"]["u0"]["n"][0] += 1
    elif change == "classes":
        row["paired_counts"]["classes"] = [1, 0]
    elif change == "subjects":
        row["paired_counts"]["subjects"]["extra"] = row["paired_counts"]["subjects"].pop("u0")
    elif change == "accuracy":
        row["test_bacc"] -= .1
    else:
        row["split_seed"] = 5
    with pytest.raises(ValueError):
        compare_reports(reports, n_boot=5)


def test_fingerprint_sensitive_to_signal_order_labels_and_montage():
    x = np.arange(24, dtype=np.float32).reshape(3, 8)
    original = fingerprint(x, [0, 1], {"fs": 200})
    assert original == fingerprint(x.copy(), [0, 1], {"fs": 200})
    assert original != fingerprint(x[::-1], [0, 1], {"fs": 200})
    assert original != fingerprint(x, [1, 0], {"fs": 200})
    assert original != fingerprint(x, [0, 1], {"fs": 201})


def test_p2_uncertainty_does_not_shrink_for_repeated_splits():
    reports = _reports()
    for arm in ("B_seed0", "B_seed1"):
        for ds in ("epn612", "uci_emg"):
            for row in reports["p2"][arm]["datasets"][ds]["splits"]:
                row["paired_counts"]["subjects"]["u0"]["correct"] = [0, 2]
                row["test_bacc"] = .875
            reports["p2"][arm]["datasets"][ds]["test_bacc"] = .875
        reports["p2"][arm]["p2"] = .875
    result = compare_reports(reports, n_boot=100)["tasks"]["p2"]["datasets"]["epn612"]
    ses = [r["paired_subject_se"] for r in result["splits"]]
    assert result["paired_subject_se"] == pytest.approx(np.sqrt(np.mean(np.square(ses))))
    assert result["paired_subject_se"] >= min(ses) > 0


def test_wrapper_extract_probe_and_cache_tampering(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import step1_downstream as cli
    ckpt = tmp_path / "runs/A_seed0/checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"test checkpoint")
    checksum = cli.sha256_file(ckpt)
    monkeypatch.setitem(cli.CHECKPOINTS, "A_seed0", checksum)
    monkeypatch.setattr(cli, "check_source", lambda _: None)
    (ckpt.parent / "run_provenance.json").write_text(json.dumps({"source_commit": cli.TRAIN_COMMIT, "checkpoint_sha256": checksum}))
    args = SimpleNamespace(expected_commit="test", runs_root=tmp_path / "runs", eval_root=tmp_path / "eval", arm="A_seed0",
                           task="p2", stage="extract", epn_root=[tmp_path], uci_root=tmp_path, root=[], compare_after=False)
    folder = args.eval_root / "p2/A_seed0"
    def fake(argv):
        assert argv[argv.index("--seed")+1] == "0"
        if "--audit-inputs" in argv:
            folder.mkdir(parents=True)
            for ds in cli.DATASETS["p2"]:
                np.savez(folder / f"{ds}.npz", x=np.ones((4, 768)), y=np.array([0, 1, 0, 1]),
                         s=np.array(["u0", "u0", "u1", "u1"]), input_sha256=np.asarray("a"*64))
        else:
            Path(argv[argv.index("--out")+1]).write_text('{"p2": 0.5}')
        return 0
    monkeypatch.setattr(cli.probe_p2, "main", fake)
    cli.run(args)
    with pytest.raises(ValueError, match="NEW"):
        cli.run(args)
    args.stage = "probe"
    cli.run(args)
    report = args.eval_root / "p2/A_seed0.json"
    assert json.loads(report.read_text())["provenance"]["identity"]["checkpoint_sha256"] == checksum
    report.unlink()
    (folder / "uci_emg.npz").write_bytes((folder / "uci_emg.npz").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        cli.run(args)


def test_final_comparison_checks_artifact_pairing(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import step1_downstream as cli
    monkeypatch.setattr(cli, "check_source", lambda _: None)
    spectral = tmp_path / "spectral.json"
    spectral.write_text('{}')
    monkeypatch.setattr(cli, "SPECTRAL_SHA", cli.sha256_file(spectral))
    inventory = {"data": {"sha256": "physical", "input_sha256": "inputs", "rows": {"n": 16}}}
    monkeypatch.setattr(cli, "cache_inventory", lambda *_: copy.deepcopy(inventory))
    reports = _reports()
    for task in reports:
        (tmp_path / task).mkdir()
        for arm, report in reports[task].items():
            identity = {"protocol": cli.PROTOCOL, "source_commit": "test", "training_commit": cli.TRAIN_COMMIT,
                        "checkpoint_sha256": cli.CHECKPOINTS[arm], "arm": arm, "task": task,
                        "which": "teacher", "probe_seed": 0, "split_sha256": cli.SPLIT_SHA}
            report["provenance"] = {"identity": identity, "cache": inventory}
            (tmp_path / task / f"{arm}.json").write_text(json.dumps(report))
    args = SimpleNamespace(expected_commit="test", eval_root=tmp_path, spectral_report=spectral)
    cli.compare(args)
    assert json.loads((tmp_path / "comparison.json").read_text())["both_practical_guards_pass"]
    (tmp_path / "comparison.json").unlink()
    path = tmp_path / "p2/B_seed1.json"
    r = json.loads(path.read_text())
    r["provenance"]["identity"]["checkpoint_sha256"] = "changed"
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="provenance"):
        cli.compare(args)
