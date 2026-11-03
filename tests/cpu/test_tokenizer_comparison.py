"""Comparable tokenizer arms: common query support and immutable checkpoint semantics."""
import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_jepa import CFG as BASE, _heterogeneous_targets, _inputs
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.jepa import JEPAConfig, jepa_loss_terms, loss_counts, make_teacher, masked_query_radius

CFG = replace(BASE, rvq_codes=None, muscle_dropout=0., pose_head=True, multiscale_anchor=True)
ARMS = ((None, "identity"), (100., "identity"), (100., "gelu"))


def scene():
    inp = _inputs()
    inp.signals[1] = inp.signals[1][:, :300]  # 12 real patches, against 32 in the first sample
    inp.qc_valid[3] = False
    visible = torch.ones(sum(inp.counts), 32, dtype=torch.bool)
    visible[:, 4:24] = False
    y = torch.randn(2, 32, 20, generator=torch.Generator().manual_seed(13))
    valid = torch.ones_like(y, dtype=torch.bool)  # deliberately true also on padded patches
    valid[0, 3] = False
    valid[1, 1, 10:] = False
    return inp, visible, _heterogeneous_targets(inp), (y, valid, torch.ones(2, 20))


@pytest.mark.parametrize("per_window", [False, True])
@pytest.mark.parametrize("radius", [0, 2])
def test_three_arms_decode_the_same_queries_and_preserve_every_loss_count(per_window, radius, monkeypatch):
    inp, visible, anchors, pose = scene()
    jcfg = JEPAConfig("b", .2, .99, loss_per_window=per_window, jepa_weight=0., pose_weight=1., masked_loss_radius=radius)
    # Independent oracle: hidden 4..23 at radius zero, 6..21 at radius two.
    # The shorter sample has no visible neighbours after patch 11.
    first, end, sample0_queries, n_queries = {0: (4, 24, 300, 388), 2: (6, 22, 240, 306)}[radius]
    expected_queries = [(ch, t) for ch in range(27) if ch != 3
                        for t in (range(first, end) if ch < 16 else range(first, 12))]
    counts_by_arm = []
    for window, activation in ARMS:
        model_cfg = replace(CFG, window_ms=window, tokenizer_activation=activation)
        torch.manual_seed(41)
        student = WearUsFM(model_cfg).eval()
        teacher = make_teacher(student)
        calls, decode = [], student.decode

        def capture(enc, ch, time, offsets):
            calls.append((list(zip(ch.tolist(), time.tolist())), offsets.tolist()))
            return decode(enc, ch, time, offsets)

        monkeypatch.setattr(student, "decode", capture)
        rng = torch.random.get_rng_state().clone()
        counts = loss_counts(inp, visible, jcfg, anchors, model_cfg=model_cfg, pose_valid=pose[1])
        assert torch.equal(rng, torch.random.get_rng_state())
        with torch.no_grad():
            terms = jepa_loss_terms(student, teacher, inp, visible, jcfg, anchor_targets=anchors, pose=pose)
        assert calls == [(expected_queries, [0, sample0_queries, n_queries])]
        assert set(counts) == set(terms.den) == set(terms.num)
        assert {"jepa", "pose", "log_rms", "log_env", "band_shape"} < set(counts)  # includes multiscale anchors
        for key in counts:
            assert counts[key].dtype == terms.den[key].dtype == torch.int64
            assert torch.equal(counts[key], terms.den[key]), (window, activation, key)
            assert torch.isfinite(terms.num[key])
        assert int(counts["jepa"]) == (2 if per_window else n_queries * CFG.dim)
        assert terms.weight["jepa"] == 0.  # query support also governs anchors when JEPA is disabled
        assert int(counts["pose"]) == (2 if per_window else 31 * 20 + 12 * 20 - 10)
        counts_by_arm.append({k: int(v) for k, v in counts.items()})
    assert counts_by_arm[0] == counts_by_arm[1] == counts_by_arm[2]
    # Radius two changes the 25-ms support; explicit zero preserves it. Pose
    # support does not follow masked queries and stays fixed for either radius.
    legacy = loss_counts(inp, visible, replace(jcfg, masked_loss_radius=None), anchors, model_cfg=CFG, pose_valid=pose[1])
    assert int(legacy["jepa"]) == (2 if per_window else 388 * CFG.dim)
    assert int(legacy["pose"]) == counts_by_arm[0]["pose"]


@pytest.mark.parametrize("window,radius", [(None, 0), (100., 2), (150., 3)])
@pytest.mark.parametrize("jepa_weight", [0., 1.])
def test_omitted_radius_matches_legacy_nominal_support_bit_for_bit(window, radius, jepa_weight):
    inp, visible, anchors, pose = scene()
    model_cfg = replace(CFG, window_ms=window)
    torch.manual_seed(17)
    student = WearUsFM(model_cfg).eval()
    teacher = make_teacher(student)
    default = JEPAConfig("a", .2, .99, pose_weight=1., jepa_weight=jepa_weight)
    assert masked_query_radius(default, model_cfg) == radius
    explicit = replace(default, masked_loss_radius=radius)
    with torch.no_grad():
        a = jepa_loss_terms(student, teacher, inp, visible, default, anchor_targets=anchors, pose=pose)
        b = jepa_loss_terms(student, teacher, inp, visible, explicit, anchor_targets=anchors, pose=pose)
    assert a.num.keys() == b.num.keys()
    for key in a.num:
        assert torch.equal(a.num[key], b.num[key]) and torch.equal(a.den[key], b.den[key]), key
    assert a.weight == b.weight


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
@pytest.mark.parametrize("jepa_weight", [0., 1.])
def test_radius_rejects_nonintegers_boolean_and_negative_values(bad, jepa_weight):
    with pytest.raises(ValueError, match="masked_loss_radius"):
        JEPAConfig("a", .2, .99, masked_loss_radius=bad, jepa_weight=jepa_weight)


@pytest.mark.parametrize("window,radius", [(100., 0), (100., 1), (150., 2)])
@pytest.mark.parametrize("jepa_weight", [.25, 1.])
def test_override_cannot_weaken_nominal_kernel_protection_with_jepa_active(window, radius, jepa_weight, monkeypatch):
    inp, visible, _, _ = scene()
    cfg = replace(CFG, window_ms=window)
    jcfg = JEPAConfig("a", .2, .99, masked_loss_radius=radius, jepa_weight=jepa_weight)
    with pytest.raises(ValueError, match="inferiore.*core_radius"):
        masked_query_radius(jcfg, cfg)
    with pytest.raises(ValueError, match="inferiore.*core_radius"):
        loss_counts(inp, visible, jcfg, model_cfg=cfg)
    student = WearUsFM(cfg)
    teacher = make_teacher(student)

    def forbidden_encode(*args, **kwargs):
        pytest.fail("invalid radius must fail before encoding")

    monkeypatch.setattr(student, "encode", forbidden_encode)
    with pytest.raises(ValueError, match="inferiore.*core_radius"):
        jepa_loss_terms(student, teacher, inp, visible, jcfg)


@pytest.mark.parametrize("per_window", [False, True])
def test_legacy_25ms_b_and_explicit_zero_have_identical_losses_and_gradients(per_window):
    inp, visible, anchors, pose = scene()
    torch.manual_seed(71)
    student = WearUsFM(CFG).eval()
    # Exercise a trained pose head, so the pose gradient reaches the backbone;
    # its usual zero initialization would only test the output layer here.
    with torch.no_grad():
        student.pose_head.out.weight.normal_(std=.1)
        student.pose_head.out.bias.normal_(std=.1)
    teacher = make_teacher(student)
    default = JEPAConfig("b", .2, .99, loss_per_window=per_window, jepa_weight=0., pose_weight=1.)

    def forward_backward(jcfg, targets):
        student.zero_grad(set_to_none=True)
        terms = jepa_loss_terms(student, teacher, inp, visible, jcfg, anchor_targets=anchors, pose=targets)
        loss = terms.backward_loss()
        loss.backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone() for name, p in student.named_parameters()}
        return loss.detach(), {k: v.detach() for k, v in terms.num.items()}, terms.den, gradients

    legacy = forward_backward(default, pose)
    explicit = forward_backward(replace(default, masked_loss_radius=0), pose)
    assert torch.equal(legacy[0], explicit[0])
    assert legacy[1].keys() == explicit[1].keys()
    for key in legacy[1]:
        assert torch.equal(legacy[1][key], explicit[1][key]) and torch.equal(legacy[2][key], explicit[2][key]), key
    for key, grad in legacy[3].items():
        other = explicit[3][key]
        assert (grad is None and other is None) or (grad is not None and other is not None and torch.equal(grad, other)), key
    for prefix in ("tokenizer.frontend.", "backbone.", "pose_head."):
        selected = [g for k, g in explicit[3].items() if k.startswith(prefix) and g is not None]
        assert selected and all(torch.isfinite(g).all() for g in selected) and any(g.abs().sum() > 0 for g in selected)
    # Invalid annotated entries and annotations on real-data padding must not
    # affect B after the explicit-zero change, including its gradients.
    excluded = ~pose[1].clone()
    excluded[1, 12:] = True
    altered_y = pose[0].clone()
    altered_y[excluded] += 1000.
    altered = forward_backward(replace(default, masked_loss_radius=0), (altered_y, pose[1], pose[2]))
    assert torch.equal(explicit[0], altered[0])
    for key, grad in explicit[3].items():
        other = altered[3][key]
        assert (grad is None and other is None) or (grad is not None and other is not None and torch.equal(grad, other)), key


def test_same_seed_initializes_all_shared_backbone_weights_identically_across_arms():
    states, rng_states = [], []
    for window, activation in ARMS:
        torch.manual_seed(61)
        model = WearUsFM(replace(CFG, window_ms=window, tokenizer_activation=activation))
        states.append({k: v.clone() for k, v in model.state_dict().items()
                       if not k.startswith("tokenizer.frontend.") and k != "tokenizer.fe_gain"})
        rng_states.append(torch.random.get_rng_state().clone())
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert any(k.startswith("backbone.") for k in states[0])
    assert any(k.startswith("tokenizer.proj.") for k in states[0])
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]) and torch.equal(states[1][key], states[2][key]), key
    assert all(torch.equal(rng_states[0], state) for state in rng_states[1:])


def test_cli_builds_three_comparable_arms_and_keeps_legacy_defaults():
    spec = importlib.util.spec_from_file_location("train_jepa_comparison", Path(__file__).resolve().parents[2] / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]

    def build(extra):
        return cli.build_config(cli.build_parser().parse_args(base + extra))

    legacy = build([])
    assert legacy.model.window_ms is None and legacy.model.tokenizer_activation == "identity"
    assert legacy.jepa.masked_loss_radius is None
    for radius in (0, 2):
        for window, activation in ARMS:
            args = ["--masked-loss-radius", str(radius), "--tokenizer-activation", activation, "--jepa-weight", "0"]
            if window is not None:
                args += ["--window-ms", str(window)]
            cfg = build(args)
            assert cfg.model.window_ms == window and cfg.model.tokenizer_activation == activation
            assert cfg.jepa.jepa_weight == 0. and cfg.jepa.masked_loss_radius == masked_query_radius(cfg.jepa, cfg.model) == radius
    default_b = build(["--window-ms", "100", "--jepa-weight", "0"])
    assert default_b.jepa.masked_loss_radius is None and masked_query_radius(default_b.jepa, default_b.model) == 2
    active_jepa = build(["--window-ms", "100", "--masked-loss-radius", "2", "--jepa-weight", "1"])
    assert masked_query_radius(active_jepa.jepa, active_jepa.model) == 2
    for window, radius in [(100., 0), (100., 1), (150., 2), (100., -1)]:
        with pytest.raises(ValueError, match="--masked-loss-radius"):
            build(["--window-ms", str(window), "--masked-loss-radius", str(radius)])
    with pytest.raises(ValueError, match="--masked-loss-radius"):
        build(["--window-ms", "100", "--masked-loss-radius", "-1", "--jepa-weight", "0"])


def legacy_config(cfg):
    old = R.config_to_dict(cfg)
    del old["model"]["tokenizer_activation"]
    del old["jepa"]["masked_loss_radius"]
    return old


def test_resume_normalizes_legacy_fields_but_cannot_reinterpret_activation_or_radius():
    cfg = R.small_config()
    old = legacy_config(cfg)
    before = copy.deepcopy(old)
    assert R._config_with_defaults(old) == R.config_to_dict(cfg)
    assert old == before  # normalization must not modify the loaded checkpoint
    R._check_resume_recipe(cfg, {"config": old})
    R._check_resume_recipe(replace(cfg, max_steps=5), {"config": old})
    nonlinear = replace(cfg, model=replace(cfg.model, tokenizer_activation="gelu"))
    common = replace(cfg, jepa=replace(cfg.jepa, masked_loss_radius=2))
    for changed, field in [(nonlinear, "model"), (common, "jepa")]:
        with pytest.raises(ValueError, match=field):
            R._check_resume_recipe(changed, {"config": old})
        R._check_resume_recipe(changed, {"config": R.config_to_dict(changed)})
        with pytest.raises(ValueError, match=field):
            R._check_resume_recipe(cfg, {"config": R.config_to_dict(changed)})
        # Branching changes schedule, not the meaning of existing parameters.
        with pytest.raises(ValueError, match=field):
            R._check_branch(changed, {"config": old}, Path("legacy.pt"))


def test_frozen_encoder_accepts_old_identity_and_rejects_activation_reinterpretation(tmp_path):
    cfg = R.small_config()
    torch.manual_seed(22)
    source = WearUsFM(cfg.model)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"config": legacy_config(cfg), "teacher": source.state_dict(), "step": 7}, checkpoint)
    loaded = WearUsFM(cfg.model)
    receipt = R._load_frozen_encoder(loaded, cfg, checkpoint)
    assert receipt["weights"] == "teacher" and receipt["step"] == 7
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in source.state_dict().items())
    nonlinear = replace(cfg, model=replace(cfg.model, tokenizer_activation="gelu"))
    with pytest.raises(ValueError, match="tokenizer_activation"):
        R._load_frozen_encoder(WearUsFM(nonlinear.model), nonlinear, checkpoint)
