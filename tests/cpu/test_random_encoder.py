import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wearusfm.harness.fm_features import load_model
from wearusfm.harness.random_encoder import compare_to_random, initial_teacher
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training.pairing import state_sha256
from wearusfm.training.spectral_calibration import sha256_file

CFG = FMConfig(dim=16, n_heads=2, k_latents=2, local_layers=1, backbone_layers=1, decoder_layers=1,
               muscle_dropout=0., multiscale_anchor=True, keep_readout=True)


def _reference(tmp_path, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = WearUsFM(CFG)
    model.spectral_keep.fit_target_stats(np.arange(128).reshape(4, 32), np.ones((4, 32), bool))
    initial = state_sha256(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.)  # simulate learned parameters, including the keep head
    path = tmp_path / f"reference{seed}.pt"
    torch.save({"config": {"seed": seed, "model": asdict(CFG), "keep_calibration_sha256": "c"*64},
                "teacher": model.state_dict()}, path)
    return path, sha256_file(path), initial


def test_initial_state_not_learned_state_rng_restored_and_frozen(tmp_path):
    path, checksum, initial = _reference(tmp_path)
    rng = torch.get_rng_state().clone()
    model, cfg = initial_teacher(path, reference_sha256=checksum, seed=0, initial_sha256=initial)
    assert torch.equal(torch.get_rng_state(), rng)
    assert state_sha256(model.state_dict()) == initial
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert cfg["model"] == asdict(CFG)
    ckpt = tmp_path / "initial_teacher.pt"
    torch.save({"config": cfg, "teacher": model.state_dict(), "artifact_role": "untrained_random_encoder", "step": 0}, ckpt)
    assert state_sha256(load_model(str(ckpt)).state_dict()) == initial
    assert state_sha256(torch.load(path, weights_only=False)["teacher"]) != initial


@pytest.mark.parametrize("drift", ["file", "seed", "initial", "buffer", "schema"])
def test_reconstruction_rejects_drift(tmp_path, drift):
    path, checksum, initial = _reference(tmp_path)
    seed = 0
    if drift == "file":
        checksum = "a"*64
    elif drift == "seed":
        seed = 1
    elif drift == "initial":
        initial = "a"*64
    else:
        state = torch.load(path, weights_only=False)
        if drift == "buffer":
            state["teacher"]["spectral_keep.target_mean"] += 1
        else:
            state["config"]["model"]["unrecognized"] = True
        torch.save(state, path)
        checksum = sha256_file(path)
    with pytest.raises(ValueError):
        initial_teacher(path, reference_sha256=checksum, seed=seed, initial_sha256=initial)


def test_two_seeds_are_paired_deterministically_and_distinct(tmp_path):
    hashes = []
    for seed in (0, 1):
        path, checksum, initial = _reference(tmp_path, seed)
        for _ in range(2):
            model, _ = initial_teacher(path, reference_sha256=checksum, seed=seed, initial_sha256=initial)
            assert state_sha256(model.state_dict()) == initial
        hashes.append(initial)
    assert hashes[0] != hashes[1]


def test_random_comparison_reuses_estimator_without_adoption_guard():
    from test_downstream_ab import _reports
    reports = _reports()
    random = {t: {s: reports[t][f"A_seed{s}"] for s in (0, 1)} for t in reports}
    learned = {t: {s: reports[t][f"B_seed{s}"] for s in (0, 1)} for t in reports}
    result = compare_to_random(random, learned, n_boot=40)
    assert "guard" not in json.dumps(result)
    for task in result["tasks"].values():
        assert task["random"] == pytest.approx(.75) and task["learned"] == pytest.approx(1.)
        assert task["delta"] == pytest.approx(.25)
    bad = copy.deepcopy(learned)
    bad["p1"][0]["datasets"]["ninapro_db2"]["paired_counts"]["subjects"]["u0"]["n"][0] += 1
    with pytest.raises(ValueError, match="unpaired"):
        compare_to_random(random, bad, n_boot=5)


def test_wrapper_reuses_frozen_probe_and_rejects_changed_cache(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import step1_random_downstream as cli
    monkeypatch.setattr(cli, "check_pipeline", lambda _: None)
    runs = tmp_path / "runs/A_seed0"
    runs.mkdir(parents=True)
    path, checksum, initial = _reference(runs)
    path.rename(runs / "checkpoint.pt")
    monkeypatch.setitem(cli.old.CHECKPOINTS, "A_seed0", checksum)
    monkeypatch.setitem(cli.INITIAL_SHA, 0, initial)
    args = SimpleNamespace(expected_commit="test", task="p2", model_seed=0, runs_root=tmp_path / "runs",
                           eval_root=tmp_path / "eval", stage="extract", root=[], epn_root=[tmp_path],
                           uci_root=tmp_path, compare_after=False)
    folder = args.eval_root / "p2/Random_seed0"

    def fake(argv):
        assert argv[argv.index("--seed")+1] == "0"
        state = torch.load(argv[argv.index("--checkpoint")+1], weights_only=False)
        assert state["step"] == 0 and "optimizer" not in state and "student" not in state
        assert state_sha256(state["teacher"]) == initial
        if "--audit-inputs" in argv:
            for ds in cli.old.DATASETS["p2"]:
                np.savez(folder / f"{ds}.npz", x=np.ones((4, 768)), y=np.array([0, 1, 0, 1]),
                         s=np.array(["u0", "u0", "u1", "u1"]), input_sha256=np.asarray("a"*64))
        else:
            Path(argv[argv.index("--out")+1]).write_text('{"p2": 0.5}')
        return 0

    monkeypatch.setattr(cli.old.probe_p2, "main", fake)
    cli.run(args)
    with pytest.raises(ValueError, match="NEW"):
        cli.run(args)
    args.stage = "probe"
    cli.run(args)
    report = args.eval_root / "p2/Random_seed0.json"
    assert json.loads(report.read_text())["provenance"]["identity"]["optimization_updates"] == 0
    report.unlink()
    cache = folder / "uci_emg.npz"
    cache.write_bytes(cache.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        cli.run(args)


def test_pipeline_is_still_historical_except_unused_loss_code(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import step1_random_downstream as cli
    monkeypatch.setattr(cli.old, "check_source", lambda _: None)
    cli.check_pipeline("test")


def test_pairing_inventory_checks_real_inputs_not_feature_hash():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import step1_random_downstream as cli
    a = {"ds": {"sha256": "a", "input_sha256": "same", "rows": {"n": 4}}}
    b = copy.deepcopy(a)
    b["ds"]["sha256"] = "b"
    assert cli.paired_inventory(a) == cli.paired_inventory(b)
    b["ds"]["input_sha256"] = "changed"
    assert cli.paired_inventory(a) != cli.paired_inventory(b)
