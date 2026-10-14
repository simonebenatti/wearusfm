"""Reconstruct the paired INITIAL teacher, never load learned encoder weights."""
from dataclasses import fields

import torch

from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training.pairing import state_sha256
from wearusfm.training.spectral_calibration import sha256_file

# These are fixed calibration buffers, not optimized parameters. The full state
# hash below proves that even these equal their values before the first update.
CALIBRATION_BUFFERS = ("target_mean", "target_std", "target_active", "stats_fitted")


def initial_teacher(reference, *, reference_sha256, seed, initial_sha256):
    """Return a CPU, frozen initial model and config from a hash-checked template.

    Only architecture and immutable calibration buffers are read from the
    trained checkpoint. No state_dict load, optimizer, forward or EMA update.
    Historical training constructed the model with one CPU thread. The
    tokenizer's fixed float64 gain depends on reduction order, so reproduce
    that environment, then restore inference parallelism and CPU RNG.
    Do not touch CUDA RNG or initialize CUDA.
    """
    if sha256_file(reference) != reference_sha256:
        raise ValueError("reference checkpoint changed")
    state = torch.load(reference, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg["seed"] != seed or set(cfg["model"]) != {f.name for f in fields(FMConfig)}:
        raise ValueError("initialization seed or model schema changed")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = WearUsFM(FMConfig(**cfg["model"]))
    finally:
        torch.set_num_threads(threads)
    if model.spectral_keep is not None:
        with torch.no_grad():
            for name in CALIBRATION_BUFFERS:
                getattr(model.spectral_keep, name).copy_(state["teacher"]["spectral_keep." + name])
    if state_sha256(model.state_dict()) != initial_sha256:
        raise ValueError("reconstructed teacher differs from the frozen INITIAL state")
    if sha256_file(reference) != reference_sha256:
        raise ValueError("reference changed during reconstruction")
    return model.eval().requires_grad_(False), cfg


def compare_to_random(random_reports, learned_reports, *, n_boot=1000, seed=20261009):
    """Keep the signed estimator, label delta learned-minus-random, no adoption gate."""
    from wearusfm.harness.downstream_ab import compare_reports

    paired = {t: {f"{side}_seed{s}": reports[t][s]
                  for side, reports in (("A", random_reports), ("B", learned_reports)) for s in (0, 1)}
              for t in ("p1", "p2")}
    result = compare_reports(paired, n_boot=n_boot, seed=seed)
    result.pop("both_practical_guards_pass")
    result.pop("guard_note")

    def relabel(value):
        if isinstance(value, dict):
            return {{"A": "random", "B": "learned"}.get(k, k): relabel(v)
                    for k, v in value.items() if k != "practical_guard_pass"}
        if isinstance(value, list):
            return [relabel(v) for v in value]
        return value

    result = relabel(result)
    result["uncertainty"] = result["uncertainty"].replace("two trained models", "two paired initializations per condition")
    result["delta_definition"] = "learned minus random; descriptive, no newly tuned practical threshold"
    return result
