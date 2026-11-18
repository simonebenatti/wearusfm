from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/diagnostics"))
import contrastive_common_state_cuda as probe
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader
from wearusfm.training.contrastive_memory import CandidateQueue, WindowIdentity


def test_vector_comparison_resolves_direction_at_equal_norm_and_unused_parameters():
    a = {"backbone.blocks.0.q": torch.tensor([3.0, 4.0]), "decoder.x": None}
    b = {"backbone.blocks.0.q": torch.tensor([-3.0, 4.0]), "decoder.x": torch.zeros(2)}
    out = probe.gradient_delta(a, b)
    assert out["grad_presence_changed"] == ["decoder.x"]
    assert out["groups"]["ALL"]["relative_vector_difference"] == pytest.approx(6 / 5)
    assert out["groups"]["backbone.blocks.0"]["cosine"] == pytest.approx(7 / 25)


def test_rng_and_input_mismatch_are_rejected():
    hashes = {k: "common" for k in probe.MATCH_FIELDS}
    for k in ("rng", "queue", "micro", "batch", "initial_teacher", "initial_student"):
        with pytest.raises(ValueError, match=k):
            probe.check_match(hashes, {**hashes, k: "changed"})


def test_only_known_determinism_errors_are_reported_as_unsupported():
    assert probe.unsupported(RuntimeError("example does not have a deterministic implementation"))
    assert not probe.unsupported(RuntimeError("Out of memory, deterministic enabled"))


def test_calibration_keys_keep_age_order_and_values():
    keys = [{"key": torch.tensor(k), "step": k} for k in (255, 258, 510, 511)]
    saved = {"last_step": 511, "entries": keys}
    remapped = probe.warm_queue(saved)
    assert [e["step"] for e in remapped["entries"]] == [0, 252, 253]
    for new, orig in zip(remapped["entries"], keys[1:]):
        assert remapped["last_step"] - new["step"] == saved["last_step"] - orig["step"]
        assert torch.equal(new["key"], orig["key"])
    assert saved["entries"][1]["step"] == 258


def test_real_cpu_backward_replays_dropout_and_supports_both_losses(tmp_path):
    root, manifest, pose = _pose_setup(tmp_path)
    cfg = _config(pose)
    cfg = replace(
        cfg, model=replace(cfg.model, muscle_dropout=0.4), contrastive={**cfg.contrastive, "fill_steps": 256}
    )
    loader = PretrainLoader(probe.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
    batch = loader.batch(4, np.random.default_rng(42))
    device = torch.device("cpu")
    state = probe.attach_contrastive_head(probe.WearUsFM(cfg.model)).state_dict()
    # A genuine detached FIFO key, physically independent of this loader batch.
    fifo = CandidateQueue(capacity_per_dataset=cfg.contrastive["capacity_per_dataset"], max_age=256)
    ident = WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.0)
    feature_dim = (
        probe.attach_contrastive_head(probe.WearUsFM(cfg.model)).contrastive_head.layers[-1].out_features
    )
    key = torch.nn.functional.normalize(torch.ones(1, feature_dim), dim=-1)
    fifo.commit(key, [ident], step=0)
    queue = fifo.state_dict()
    queue["last_step"], queue["entries"][0]["step"] = 253, 253
    rng = probe.rng_state(device)
    observations = []
    for arm in ("C0", "CN", "CR", "C0"):
        obs, grad = probe.snapshot(
            arm, cfg, state, queue, batch, rng, probe.trainer.DistContext(), device, False
        )
        assert obs["nce_eligible"] > 0 and obs["rnc_windows"] > 0
        if observations:
            first, base = observations[0]
            probe.check_match(first, obs)
            assert (
                first["after_rng"] == obs["after_rng"] and first["gradient_sha256"] == obs["gradient_sha256"]
            )
            norm = probe.gradient_delta(base, grad)["groups"]["ALL"]
            assert norm["relative_vector_difference"] == 0
            assert norm["left_norm"] == pytest.approx(first["trainer_norm"], rel=1e-5)
        observations.append((obs, grad))


def test_cross_precision_compares_same_rank_and_same_state(tmp_path):
    def profile(grad):
        hashes = {k: "fixed" for k in probe.MATCH_FIELDS}
        ranks = [
            {
                "rank": i,
                "global_batch_sha256": "global",
                "cases": {"C0a": {**hashes, "rng": "seed" + str(i), "after_rng": "after" + str(i)}},
            }
            for i in range(4)
        ]
        return {"ranks": ranks}

    profiles = {key: profile(1.0) for key in ("bf16_auto", "bf16_math", "fp32_math")}
    for key in profiles:
        grad = [1.0, 2.0] if key == "fp32_math" else [2.0, 1.0]
        torch.save({"encoder.weight": torch.tensor(grad)}, tmp_path / (key + ".pt"))
    result = probe.aggregate(profiles, tmp_path)
    assert result["fp32_math_bf16_math"]["groups"]["ALL"]["cosine"] == pytest.approx(0.8)
    assert result["bf16_auto_bf16_math"]["groups"]["ALL"]["relative_vector_difference"] == 0.0
    assert all(result["fp32_math_bf16_math"]["post_backward_rng_equal_by_rank"])
    profiles["fp32_math"]["ranks"][1]["cases"]["C0a"]["rng"] = "changed"
    with pytest.raises(ValueError, match="rng"):
        probe.aggregate(profiles, tmp_path)
