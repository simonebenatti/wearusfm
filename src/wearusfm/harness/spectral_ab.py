"""Prespecified two-seed spectral A/B pilot, paired subject uncertainty.

The bootstrap resamples the SAME test subjects in both arms and both model
seeds. It is conditional on these trained models, not a training-seed CI.
"""
import copy
import numpy as np

from wearusfm.model.spectral_step1 import FAMILIES

BOOTSTRAP_SEED = 20261008
BOOTSTRAP_DRAWS = 2000
MIN_SHAPE_GAIN = .02
MAX_FAMILY_DROP = .01


def validate_training_pair(a, b, expected_steps=4000):
    """Inputs: config, summary, pairing and metric dictionaries (no checkpoint IO)."""
    ca, cb = copy.deepcopy(a["config"]), copy.deepcopy(b["config"])
    if ca["jepa"].pop("keep_weight") != 0 or cb["jepa"].pop("keep_weight") != .05 or ca != cb:
        raise ValueError("A/B configs must differ ONLY in keep_weight 0/0.05")
    for run in (a, b):
        if run["summary"]["steps"] != expected_steps or run["summary"]["stopped"] != "max_steps":
            raise ValueError("incomplete arm: no scientific comparison or automatic resume")
        if run["config"]["max_steps"] != expected_steps or not run["config"].get("pairing_audit"):
            raise ValueError("fixed endpoint and pairing audit required")
        if run["pairing"].get("resume") is not False or run["pairing"]["seed"] != run["config"]["seed"]:
            raise ValueError("fresh-run/seed provenance mismatch")
        if any(len(run["pairing"].get(k, "")) != 64 for k in ("student_initial_sha256", "teacher_initial_sha256")):
            raise ValueError("initial state fingerprints required")
        if [r["step"] for r in run["metrics"]] != list(range(1, expected_steps+1)):
            raise ValueError("missing/duplicate/out-of-order training updates")
        for r in run["metrics"]:
            if len(r.get("batch_sha256", "")) != 64 or "alarm" in r:
                raise ValueError("missing batch audit or training alarm")
            for value in r.values():
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    raise ValueError("nonfinite training metric")
    if a["pairing"] != b["pairing"]:
        raise ValueError("initial student/teacher weights or workers are not paired")
    if any(x["batch_sha256"] != y["batch_sha256"] for x, y in zip(a["metrics"], b["metrics"])):
        raise ValueError("delivered data/targets/masks differ between arms")
    return {"updates": expected_steps, "all_batches_identical": True, "initial_states_identical": True}


def paired_spectral_comparison(pairs, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """pairs is two (A, B) spectral_probe outputs from the SAME selected view.

    Uses pooled coordinate R2, not the mean of subject R2. All 32 coordinates
    and all subject target sufficient statistics must match; no post-hoc drops.
    """
    if len(pairs) != 2 or draws < 2:
        raise ValueError("exactly two paired model seeds and >=2 draws required")
    subjects = sorted(pairs[0][0]["coordinates"][0]["per_subject"])
    if len(subjects) < 2:
        raise ValueError("at least two test subjects required")
    stats = np.empty((2, 2, 32, len(subjects), 4), dtype=np.float64)
    for model_seed, arms in enumerate(pairs):
        for arm, report in enumerate(arms):
            if [r["coordinate"] for r in report["coordinates"]] != list(range(32)):
                raise ValueError("32 ordered coordinates required")
            for j, row in enumerate(report["coordinates"]):
                if row["status"] != "ok" or row["r2"] is None or sorted(row["per_subject"]) != subjects:
                    raise ValueError("complete identical subject support required (no omitted coordinates)")
                for k, subject in enumerate(subjects):
                    sub = row["per_subject"][subject]
                    stats[model_seed, arm, j, k] = [sub[key] for key in ("n", "target_mean", "target_m2", "sse")]
    if not np.isfinite(stats).all() or (stats[..., 0] < 1).any() or (stats[..., 2:] < 0).any():
        raise ValueError("invalid sufficient statistics")
    target = stats[0, 0, :, :, :3]
    if not np.all(stats[..., :3] == target):
        raise ValueError("A/B/seeds have different test targets or coordinate support")

    def scores(weights):
        n, mean, m2, sse = (stats[..., k] for k in range(4))
        wn = n * weights
        grand = (wn * mean).sum(-1) / wn.sum(-1)
        sst = (weights * (m2 + n * np.square(mean-grand[..., None]))).sum(-1)
        if (sst <= 1e-12).any():
            raise ValueError("constant resampled test target: comparison not assessable")
        coord = 1 - (weights*sse).sum(-1)/sst
        family = np.stack([coord[..., cols].mean(-1) for cols in FAMILIES.values()], axis=-1)
        return family, family[:, 1]-family[:, 0]

    point, delta = scores(np.ones(len(subjects)))
    # Check the stored coordinate scores against the same pooled estimand.
    n, mean, m2, sse = (stats[..., k] for k in range(4))
    grand = (n*mean).sum(-1)/n.sum(-1)
    expected = 1-sse.sum(-1)/(m2+n*np.square(mean-grand[..., None])).sum(-1)
    observed = np.array([[[row["r2"] for row in arm["coordinates"]] for arm in arms] for arms in pairs])
    if not np.allclose(observed, expected, rtol=1e-10, atol=1e-10):
        raise ValueError("stored R2 inconsistent with sufficient statistics")
    boot = np.empty((draws, 4))
    rng = np.random.default_rng(seed)
    for i in range(draws):
        weights = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        _, difference = scores(weights)
        boot[i] = difference.mean(0)  # same draw across seeds, NOT independently resampled seeds
    families = {}
    for j, name in enumerate(FAMILIES):
        families[name] = {
            "a_r2": float(point[:, 0, j].mean()), "b_r2": float(point[:, 1, j].mean()),
            "delta": float(delta[:, j].mean()), "delta_by_model_seed": delta[:, j].tolist(),
            "paired_subject_ci95": np.quantile(boot[:, j], [.025, .975]).tolist(),
        }
    shape_indices = [list(FAMILIES).index(name) for name in ("fast_shape", "slow_shape")]
    primary = delta[:, shape_indices].mean(1)
    ci = np.quantile(boot[:, shape_indices].mean(1), [.025, .975])
    gates = {
        "mean_shape_gain_at_least_0.02": bool(primary.mean() >= MIN_SHAPE_GAIN),
        "paired_subject_ci_lower_positive": bool(ci[0] > 0),
        "positive_shape_gain_both_model_seeds": bool((primary > 0).all()),
        "neither_shape_family_mean_decreases": bool((delta[:, shape_indices].mean(0) >= 0).all()),
        "no_family_seed_drop_over_0.01": bool((delta >= -MAX_FAMILY_DROP).all()),
    }
    return {
        "families": families,
        "primary": {"name": "equal_mean_fast_slow_shape_r2", "delta": float(primary.mean()),
                    "delta_by_model_seed": primary.tolist(), "paired_subject_ci95": ci.tolist()},
        "bootstrap": {"unit": "test_subject", "draws": draws, "seed": seed, "subjects": subjects,
                      "shared_across_arms_and_model_seeds": True,
                      "interpretation": "conditional on two trained models; NOT training-seed uncertainty"},
        "gates": gates,
        "decision": "promising_run_fresh_P1_P2" if all(gates.values()) else "not_promising_at_this_weight_and_endpoint",
        "downstream_status": "not_yet_evaluated; no adoption claim",
    }
