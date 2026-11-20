"""The real, unpatched preflight body on small files written in the real cache format; no cluster I/O.

The first preflight (job 64931573) failed on a difference that no fixture reproduced: the producer receipt names storage
directories ("u"+ID), the cache metadata stores the ID. Here nothing in provenance, data or the historical readers is
replaced. Real: user IDs, the two producer receipts with their storage directories, config_json, metadata keys, dtypes,
the historical writer and readers. Synthetic: array values and sizes, so only the parity constants are rebound.
Adapted from the dry run of the independent LOGIC reviewer of the second delivery (20/11/2026).
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import data as D, execution as E, metrics as M, provenance as P

REPO = Path(__file__).resolve().parents[2]
USERS = json.loads((REPO/"tests/cpu/data/pose_temponet_cache_users_20261120.json").read_text())
LAYOUT = json.loads((REPO/"tests/cpu/data/pose_temponet_cache_layout_20261120.json").read_text())
DETAILS = ("targets", "key_time_valid_sha256", "dim", "n_recordings", "n_users", "n_windows")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(path, **extra):
    return {"path": str(path), "sha256": _sha(path), "bytes": Path(path).stat().st_size, **extra}


def _cache(split, users, windows, seed):
    """One cache with the member names, dtypes, metadata keys and config_json of the real file."""
    rng = np.random.default_rng(seed)
    n = len(users)*windows
    lengths = np.full(n, 160)
    lengths[3::7] = 150
    available = np.arange(160)[None, :] < lengths[:, None]
    lat = rng.normal(size=(n, 160, 384)).astype(np.float16)
    pre = rng.normal(size=(n, 160, 384)).astype(np.float16)
    lat[~available] = 0
    pre[~available] = 0
    scale = np.where(np.arange(n) % 3 == 0, .05, 1.)[:, None, None]  # some static windows
    target = (rng.normal(size=(n, 160, 20))*scale).astype(np.float32)
    mask = np.repeat(available[..., None], 20, axis=2) & (rng.random((n, 160, 20)) > .05)
    mask[1] = False  # like the real caches, some windows carry no valid pose at all
    pred = (.6*target + .3*scale*rng.normal(size=target.shape)).astype(np.float32)
    meta = [{"fs": 2000.0, "n_patches": int(lengths[i]), "recording": i//windows,
             "session": "2022-04-13-1649836800-9aadd-cv-emg-pose-train@2", "side": "left" if i % 2 else "right",
             "stage": "WristFlexionAbduction", "start": 400*(i % windows),
             "stem": "2022-04-13-1649836800-9aadd-cv-emg-pose-train@2-recording-%d_left" % (i//windows),
             "user": users[i//windows]} for i in range(n)]
    assert sorted(meta[0]) == LAYOUT["metadata_keys"]
    data = {"pre": pre, "lat": lat, "key_time_valid": available, "pred": pred, "target": target, "mask": mask,
            "std": np.full((n, 20), .3, np.float32), "pose_var": M.dynamic_variance(target, mask).astype(np.float32),
            "has_head": True, "dim": 384, "n_recordings": len(users), "meta": meta,
            "config": copy.deepcopy(LAYOUT[split]["config"])}
    for name, member in LAYOUT[split]["members"].items():
        if name in ("pre", "lat", "key_time_valid", "pred", "target", "mask", "std", "pose_var"):
            assert data[name].dtype.str == member["descr"] and list(data[name].shape[1:]) == member["shape"][1:], name
    return data


@pytest.fixture
def real_layout(tmp_path, monkeypatch):
    """A plan built by the real build_plan over small real-format files and the two real producer receipts."""
    repo = tmp_path/"repo"
    for relative in ("src/example.py", "scripts/legacy.py", "scripts/pose_temponet/example.py"):
        (repo/relative).parent.mkdir(parents=True, exist_ok=True)
        (repo/relative).write_text("# immutable source\n")
    proposal = json.loads((REPO/"results/infra/pose_temponet_proposal_20261117/proposal.json").read_text())
    receipts, caches = {}, {}
    for index, (split, windows) in enumerate((("train", 2), ("val", 4))):
        users = USERS[split]["users"]
        data = _cache(split, users, windows, 100+index)
        cache, reference = tmp_path/(split+".npz"), tmp_path/(split+"_reference.npz")
        D.historical.save_cache(cache, data)  # the historical writer
        with reference.open("wb") as stream:  # the target reference is a pose cache of another model
            np.savez_compressed(stream, lat=data["lat"], pred=data["pred"], target=data["target"], mask=data["mask"],
                                std=data["std"], pose_var=data["pose_var"], has_head=True, dim=384,
                                meta_json=np.asarray(json.dumps(data["meta"])), n_recordings=data["n_recordings"],
                                config_json=np.asarray(json.dumps({"which": "student"})))
        count = len(data["meta"])
        proposal["caches"][split] = _pin(cache, users=len(users), windows=count)
        reference_pin = _pin(reference, n_recordings=data["n_recordings"], n_users=len(users), n_windows=count)
        proposal["scratch_dependencies"]["pose_references"][split] = reference_pin
        # The REAL receipt, with its real storage directories; only the artifact pin and the six details are re-bound.
        receipt = json.loads((REPO/P.HISTORICAL_RECEIPTS[split]["local_copy"]).read_text())
        receipt["artifacts"]["cache"] = {**receipt["artifacts"]["cache"], "sha256": _sha(cache), "bytes": cache.stat().st_size}
        targets = D.reference.read_targets(str(reference))
        receipt["details"] = D.historical.validate_cache(D.historical.read_cache(str(cache), targets, reference_pin),
                                                        targets, reference_pin)  # as the historical runtime computed them
        path = repo/"receipts"/(split+".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(receipt))
        receipts[split] = {"path": str(path.relative_to(repo)), "sha256": _sha(path), "bytes": path.stat().st_size}
        caches[split] = data
    stats = tmp_path/"pose_stats.json"
    stats.write_text("{}\n")
    proposal["scratch_dependencies"]["pose_stats"] = {"path": str(stats), "sha256": _sha(stats)}
    held = caches["val"]
    np.savez_compressed(tmp_path/"post_ridge.predictions.npz", prediction=(.5*held["target"][:, 6:-6]).astype(np.float64))
    np.savez_compressed(tmp_path/"trained_head.predictions.npz", prediction=held["pred"][:, 6:-6].astype(np.float64))
    np.savez_compressed(tmp_path/"post_ridge.coefficients.npz", weights=np.zeros((384, 20)), bias=np.zeros(20))
    for key, name in (("post_ridge_coefficients", "post_ridge.coefficients.npz"),
                      ("post_ridge_predictions", "post_ridge.predictions.npz"),
                      ("trained_head_predictions", "trained_head.predictions.npz")):
        proposal["references"][key] = _pin(tmp_path/name)
    proposal_path = repo/"results/infra/pose_temponet_proposal_20261117/proposal.json"
    proposal_path.parent.mkdir(parents=True)
    proposal_path.write_text(json.dumps(proposal))
    plan = P.build_plan("results/infra/pose_temponet_proposal_20261117/proposal.json", receipts, source_root=repo,
                        jobs=E.job_graph())
    # The parity constants belong to the real arrays: rebind only them to the synthetic values.
    probe = P.read_verified_inputs(plan, "val", stage="preflight", depth="post")
    rows = {name: M.evaluate(D.read_predictions(plan["inputs"][name+"_predictions"]), probe)["groups"]["dynamic"]
            for name in ("post_ridge", "trained_head")}
    centered = rows["post_ridge"]["centered"]
    monkeypatch.setattr(M, "HISTORICAL_REFERENCE", {
        "post_ridge_E_dyn": centered["centered_mse_ratio"],
        "trained_head_E_dyn": rows["trained_head"]["centered"]["centered_mse_ratio"],
        "dynamic_selected_windows": rows["post_ridge"]["n_selected_windows"],
        "dynamic_centered_windows": centered["n_windows"], "dynamic_centered_elements": centered["n_valid_elements"],
        "dynamic_window_joint_pairs": centered["n_window_joints"], "dynamic_users": centered["n_users"]})
    return plan


def test_unpatched_preflight_completes_on_the_real_cache_layout(real_layout):
    plan = real_layout
    for split in ("train", "val"):
        assert plan["producers"][split]["users"] == USERS[split]["users"]
        assert plan["producers"][split]["storage_directories"] == ["u"+user for user in USERS[split]["users"]]
    result = P.preflight(plan)
    assert result["integrity"] is True and result["historical_R1"]["passed"] is True
    assert [len(fold) for fold in result["train"]["folds"]] == [20, 20]
    assert sorted(result["train"]["folds"][0]+result["train"]["folds"][1]) == USERS["train"]["users"]
    assert set(result["inputs"]) == set(P.ALL_INPUTS) and len(result["inputs"]) == 8
    assert 0 < result["train"]["supported_windows"] < 80  # one window was written without any valid pose
    assert result["historical_R1"]["observed"]["trained_head"]["users"] == 15
    # What the runner stores in the receipt must be plain JSON.
    payload = {"details": {"preflight": copy.deepcopy(result)}, "gates": {"integrity": True, "historical_R1": True}}
    assert json.loads(P.canonical_bytes(payload))["details"]["preflight"]["integrity"] is True


def test_first_delivery_user_list_fails_exactly_like_job_64931573(real_layout):
    first = copy.deepcopy(real_layout)
    for split in ("train", "val"):
        first["producers"][split]["users"] = list(first["producers"][split]["storage_directories"])
    with pytest.raises(ValueError, match="^cache user population differs from original producer$"):
        P.read_verified_inputs(first, "train", stage="preflight", depth="post")
    with pytest.raises(ValueError, match="^cache user population differs from original producer$"):
        P.preflight(first)


def test_real_read_path_refuses_drifted_directories_and_each_changed_detail(real_layout):
    plan = real_layout
    drifted = copy.deepcopy(plan)
    drifted["producers"]["val"]["storage_directories"][0] = "uffffffffff"
    with pytest.raises(ValueError, match="do not map onto the producer storage directories"):
        P.read_verified_inputs(drifted, "val", stage="preflight")
    for key in DETAILS:
        changed = copy.deepcopy(plan)
        changed["producers"]["train"]["details"][key] = "x"
        with pytest.raises(ValueError, match="historical cache details changed: "+key):
            P.read_verified_inputs(changed, "train", stage="preflight")
