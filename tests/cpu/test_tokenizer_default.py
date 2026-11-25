"""Tokenizer dei nuovi training: kernel di 100 ms (passo 25 ms) e GELU senza opzioni; i run storici restano a 25 ms e identity, scritti in
modo esplicito (Simone, 24/11/2026: «Sì, fallo» alla domanda «Rendo 100 ms + GELU il default anche del tokenizer?»)."""
import copy
import importlib.util
import json
import shlex
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_pretraining_loader import _tree  # noqa: E402
from wearusfm.harness import fm_features as FF  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import JEPAConfig, masked_query_radius  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
PRESETS = ("sanity", "small", "30m", "100m", "300m")
BASE = ["--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
EXECUTED = ROOT / "results/infra/tokenizer_execution_20261103/19618"


def _cli():
    spec = importlib.util.spec_from_file_location("train_jepa_tokenizer_default", ROOT / "scripts" / "train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _build(cli, extra, preset="small"):
    return cli.build_config(cli.build_parser().parse_args(["--preset", preset] + BASE + list(extra)))


def _tokenizer(cfg):
    return cfg.model.window_ms, cfg.model.tokenizer_activation


def _preset(name):
    return {"sanity": R.sanity_config, "small": R.small_config}.get(name, lambda: R.ladder_config(name))()


@pytest.mark.parametrize("preset", PRESETS)
def test_every_new_training_preset_uses_100ms_and_gelu(preset):
    cli = _cli()
    for cfg in (_preset(preset), R.with_window1_rules(_preset(preset)), _build(cli, [], preset), _build(cli, ["--window1-rules"], preset)):
        assert _tokenizer(cfg) == (100.0, "gelu")
        assert cfg.model.patch_ms == cfg.loader.patch_ms == 25.0  # il passo non cambia: un token ogni 25 ms
        assert cfg.model.core_radius == 2
    # cambia solo il tokenizer: con la ricetta storica tutto il resto della configurazione e' identico
    new, old = R.config_to_dict(_preset(preset)), R.config_to_dict(R.with_historical_tokenizer(_preset(preset)))
    assert {k: v for k, v in new.items() if k != "model"} == {k: v for k, v in old.items() if k != "model"}
    assert {k for k in new["model"] if new["model"][k] != old["model"][k]} == {"window_ms", "tokenizer_activation"}
    assert (old["model"]["window_ms"], old["model"]["tokenizer_activation"]) == (None, "identity")


def test_the_new_default_is_a_real_100ms_gelu_model_on_the_25ms_grid():
    new, old = R.small_config().model, R.with_historical_tokenizer(R.small_config()).model
    torch.manual_seed(0)
    model = WearUsFM(new)
    torch.manual_seed(0)
    historical = WearUsFM(old)
    assert model.tokenizer.window_s == pytest.approx(0.1) and model.tokenizer.patch_s == pytest.approx(0.025)
    assert historical.tokenizer.window_s == historical.tokenizer.patch_s == pytest.approx(0.025)
    assert (model.tokenizer.tokenizer_activation, historical.tokenizer.tokenizer_activation) == ("gelu", "identity")
    # le basi scalano con la finestra: 5.184 parametri in piu', tutti nel front-end (docs/tokenizer_window_nonlinear_protocol_20261102.md)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(model) - count(historical) == 5184
    assert count(model.tokenizer.frontend) - count(historical.tokenizer.frontend) == 5184
    x = torch.randn(3, 2000, generator=torch.Generator().manual_seed(1))
    tokens, patch_valid, _ = model.tokenizer([x], [2000.0])
    assert tokens.shape[:2] == (3, 40) and patch_valid.all()  # 40 token al secondo, come prima


def test_parameter_count_of_the_30m_rung():
    assert R.count_parameters(R.ladder_config("30m").model) == 28_341_995
    assert R.count_parameters(R.with_historical_tokenizer(R.ladder_config("30m")).model) == 28_336_811


def test_fmconfig_itself_stays_historical_so_old_configs_and_checkpoints_do_too():
    """Il default nuovo sta nei preset: un `model` salvato senza i due campi si ricostruisce storico con qualunque strumento."""
    assert {f.name: f.default for f in fields(FMConfig) if f.name in ("window_ms", "tokenizer_activation")} == R.HISTORICAL_TOKENIZER
    saved = R.config_to_dict(R.with_historical_tokenizer(R.small_config()))["model"]
    for key in ("window_ms", "tokenizer_activation"):
        del saved[key]
    for cfg in (FMConfig(**saved), FMConfig(**{f.name: saved[f.name] for f in fields(FMConfig) if f.name in saved})):
        assert (cfg.window_ms, cfg.tokenizer_activation) == (None, "identity")
        assert cfg == R.with_historical_tokenizer(R.small_config()).model
    stored = R.config_to_dict(R.with_historical_tokenizer(R.small_config()))
    for key in ("window_ms", "tokenizer_activation"):
        del stored["model"][key]
    assert R._config_with_defaults(stored)["model"] == R.config_to_dict(R.with_historical_tokenizer(R.small_config()))["model"]


@pytest.mark.parametrize("extra,expected", [
    (["--window-ms", "25", "--tokenizer-activation", "identity"], (None, "identity")),  # la ricetta storica
    (["--window-ms", "25.0", "--tokenizer-activation=identity"], (None, "identity")),
    (["--window-ms=25"], (None, "gelu")),  # un fattore alla volta
    (["--tokenizer-activation", "identity"], (100.0, "identity")),  # T100-L
    (["--window-ms", "100", "--tokenizer-activation", "gelu"], (100.0, "gelu")),  # T100-G, i comandi di novembre
    (["--window-ms", "150"], (150.0, "gelu")),
    (["--window-ms", "150", "--tokenizer-activation", "identity"], (150.0, "identity")),
])
def test_cli_options_select_the_tokenizer(extra, expected):
    cli = _cli()
    assert _tokenizer(_build(cli, extra)) == expected


@pytest.mark.parametrize("bad", ["10", "0", "-25", "24.999", "nan", "inf"])
def test_cli_rejects_windows_shorter_than_the_step(bad):
    cli = _cli()
    with pytest.raises(ValueError, match="--window-ms"):
        _build(cli, [f"--window-ms={bad}"])


def test_historical_tokenizer_helper_is_idempotent_and_touches_nothing_else():
    for preset in PRESETS:
        cfg = _preset(preset)
        old = R.with_historical_tokenizer(cfg)
        assert _tokenizer(old) == (None, "identity") and old.model.core_radius == 0
        assert R.with_historical_tokenizer(old) == old
        assert replace(old, model=replace(old.model, **R.NEW_TOKENIZER)) == cfg
        assert (old.loader, old.jepa, old.target_version) == (cfg.loader, cfg.jepa, cfg.target_version)
    both = R.with_historical_tokenizer(R.with_historical_targets(R.sanity_config()))
    cli = _cli()
    assert R.config_to_dict(both) == R.config_to_dict(cli.build_config(cli.build_parser().parse_args(
        cli.historical_args(["--preset", "sanity"] + BASE))))


def test_historical_args_write_each_missing_option_and_never_override():
    cli = _cli()
    base = ["--preset", "small"] + BASE
    targets, tokenizer = ["--rms-window-ms", "25", "--ms-fast-window-ms", "50"], ["--window-ms", "25", "--tokenizer-activation", "identity"]
    assert cli.historical_args(base) == base + targets + tokenizer
    assert cli.historical_args(cli.historical_args(base)) == cli.historical_args(base)
    for given, missing in ((["--window-ms", "100"], ["--tokenizer-activation", "identity"]),  # T100-L scritto senza attivazione, R5
                           (["--window-ms=150"], ["--tokenizer-activation", "identity"]),
                           (["--tokenizer-activation", "identity"], ["--window-ms", "25"]),  # T25-L, comando registrato
                           (["--tokenizer-act", "gelu"], ["--window-ms", "25"]),
                           (["--window-ms", "100", "--tokenizer-activation", "gelu"], [])):
        assert cli.historical_args(base + given) == base + given + targets + missing
    historical = cli.build_config(cli.build_parser().parse_args(cli.historical_args(base)))
    assert _tokenizer(historical) == (None, "identity")
    assert (historical.loader.rms_window_ms, historical.loader.ms_fast_window_ms) == (25.0, 50.0)


def _registered_argv(node):
    if isinstance(node, list):
        if any(isinstance(x, str) and x.endswith("train_jepa.py") for x in node):
            return node[[str(x).endswith("train_jepa.py") for x in node].index(True) + 1:]
        node = {str(i): v for i, v in enumerate(node)}
    if isinstance(node, dict):
        for value in node.values():
            found = _registered_argv(value)
            if found:
                return found
    return None


@pytest.mark.parametrize("arm,expected", [("T25-L", (None, "identity")), ("T100-L", (100.0, "identity")), ("T100-G", (100.0, "gelu"))])
def test_registered_november_commands_still_rebuild_their_executed_recipe(arm, expected):
    """I comandi VERI dei tre bracci (T25-L non scrive `--window-ms`) tornano la configurazione eseguita e salvata."""
    cli = _cli()
    argv = _registered_argv(json.loads((EXECUTED / arm / "receipt.json").read_text()))
    saved = json.loads((EXECUTED / arm / "config.json").read_text())
    cfg = cli.build_config(cli.build_parser().parse_args(cli.historical_args(argv)))
    assert _tokenizer(cfg) == expected == (saved["model"]["window_ms"], saved["model"]["tokenizer_activation"])
    assert R.config_to_dict(cfg) == R._config_with_defaults(saved)
    R._check_resume_recipe(cfg, {"config": saved})
    if arm == "T25-L":  # senza la ricetta storica il comando registrato darebbe oggi un altro modello: la ripresa lo rifiuta
        bare = cli.build_config(cli.build_parser().parse_args(argv))
        assert _tokenizer(bare) == (100.0, "identity")
        with pytest.raises(ValueError, match="model"):
            R._check_resume_recipe(bare, {"config": saved})


def test_resume_and_branch_refuse_the_new_default_on_a_historical_run():
    cli = _cli()
    wsd = ["--schedule", "wsd", "--total-steps", "20", "--warmup-steps", "2"]
    targets = ["--rms-window-ms", "25", "--ms-fast-window-ms", "50"]  # isolano il tokenizer
    new, old = _build(cli, wsd + targets), _build(cli, wsd + targets + ["--window-ms", "25", "--tokenizer-activation", "identity"])
    stored = R.config_to_dict(old)
    bare = copy.deepcopy(stored)
    for key in ("window_ms", "tokenizer_activation"):
        del bare["model"][key]
    for saved in (stored, bare):
        R._check_resume_recipe(old, {"config": saved})
        with pytest.raises(ValueError, match="model"):
            R._check_resume_recipe(new, {"config": saved})
        state = {"config": saved, "step": 4, "stream": {"num_workers": 0}}
        R._check_branch(old, state, Path("source.pt"))
        with pytest.raises(ValueError, match="model"):
            R._check_branch(new, state, Path("source.pt"))
    with pytest.raises(ValueError, match="model"):  # nell'altro verso: un run nuovo non si riprende con la ricetta storica
        R._check_resume_recipe(old, {"config": R.config_to_dict(new)})


def test_masked_query_radius_keeps_its_definition():
    """Non cambia la regola: senza override il raggio e' quello nominale del kernel, che con 100 ms vale 2 patch (0 col banco storico)."""
    cli = _cli()
    assert masked_query_radius(R.sanity_config().jepa, R.sanity_config().model) == 2
    historical = R.with_historical_tokenizer(R.sanity_config())
    assert masked_query_radius(historical.jepa, historical.model) == 0
    cfg = _build(cli, ["--jepa-weight", "0", "--masked-loss-radius", "0"])  # la scelta esplicita di T100-G
    assert masked_query_radius(cfg.jepa, cfg.model) == 0 and _tokenizer(cfg) == (100.0, "gelu")
    with pytest.raises(ValueError, match="core_radius"):  # con JEPA accesa il raggio 0 resta rifiutato
        _build(cli, ["--masked-loss-radius", "0"])
    assert JEPAConfig("b", 0.2, 0.996).masked_loss_radius is None


# --- strumenti che rileggono run e piani veri ----------------------------------------------------------------------------------------------

def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tokenizer_default", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registered_pilot_plan_keeps_the_historical_tokenizer():
    """Piano vero del pilot degli obiettivi (24/10): R0-R2 senza opzioni del tokenizer, R5 con `--window-ms 150` e senza attivazione."""
    select = _script("calibration_select")
    plan = json.loads((ROOT / "results/infra/pilot_obiettivi_20261024/piano.json").read_text())
    for run in plan["runs"]:
        model = select.expected_config(plan, run["id"])["model"]
        assert (model["window_ms"], model["tokenizer_activation"]) == (150.0 if "-R5-" in run["id"] else None, "identity"), run["id"]


def test_a_new_plan_must_write_the_four_options():
    """Il trainer e `calibration_select` leggono lo stesso piano in modo diverso se le opzioni non ci sono: con le sole finestre dei target (il
    piano coerente fino al 23/11) la differenza e' esattamente il tokenizer; con le quattro opzioni non c'e'."""
    from test_wsd_chain import W, _plan

    select, cli = _script("calibration_select"), _cli()

    def differing(extra):
        plan = _plan(common_args=_plan()["common_args"] + extra)
        run = W.resolve(plan)[0]
        argv = ["--preset", run["preset"]] + BASE + ["--max-steps", str(run["total_steps"])] + shlex.split(W.extra_args(plan, run))
        trained = select._flatten(R.config_to_dict(cli.build_config(cli.build_parser().parse_args(argv))))
        expected = select._flatten(select.expected_config(plan, run["id"]))
        return {k for k in set(trained) | set(expected) if trained.get(k) != expected.get(k)}

    targets, tokenizer = " --rms-window-ms 100 --ms-fast-window-ms 125", " --window-ms 100 --tokenizer-activation gelu"
    assert differing(targets) == {"model.window_ms", "model.tokenizer_activation"}
    assert differing(targets + tokenizer) == set()
    assert {"loader.rms_window_ms", "loader.ms_fast_window_ms", "model.window_ms", "model.tokenizer_activation"} <= differing("")


def test_library_builders_of_the_historical_step1_stay_historical(tmp_path):
    """A/B dello Step 1, 2C e pilot dei rami partono da `sanity_config` senza passare dalla riga di comando."""
    import branch_pilot as BP
    import step1_ab as SA
    import step2c as S2C

    cal = tmp_path / "calibration.npz"
    cal.write_bytes(b"solo per lo SHA")
    configs = [SA.build_config(arm, 0, cal) for arm in ("A", "B")] + [S2C.build_config("B_seed0", cal)]
    configs += [BP.build_config(arm, "B_seed0", cal) for arm in ("PWl", "PWf", "PWm")]
    for cfg in configs:
        assert _tokenizer(cfg) == (None, "identity") and masked_query_radius(cfg.jepa, cfg.model) == 0
        assert (cfg.loader.rms_window_ms, cfg.loader.ms_fast_window_ms) == (25.0, 50.0)


@pytest.mark.parametrize("window_ms", [None, 100.0, 150.0])
def test_a_saved_checkpoint_without_the_fields_is_never_read_as_gelu(tmp_path, window_ms):
    """Checkpoint di prima dei campi (R5 compreso: finestra scritta, attivazione no): ogni strumento lo rilegge senza attivazione."""
    cfg = replace(R.with_historical_tokenizer(R.small_config()).model, window_ms=window_ms)
    torch.manual_seed(3)
    model = WearUsFM(cfg)
    saved = R.config_to_dict(replace(R.small_config(), model=cfg))["model"]
    del saved["tokenizer_activation"]
    if window_ms is None:
        del saved["window_ms"]
    path = tmp_path / "old.pt"
    torch.save({"config": {"model": saved}, "student": model.state_dict(), "teacher": model.state_dict()}, path)
    restored = FF.load_model(str(path), which="student")
    assert (restored.cfg.window_ms, restored.cfg.tokenizer_activation) == (window_ms, "identity")
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    random_state = _script("random_checkpoint").random_state({"model": saved}, 0)
    assert set(random_state["student"]) == set(model.state_dict())
    assert WearUsFM(FMConfig(**saved)).tokenizer.tokenizer_activation == "identity"


# --- trainer vero: ripresa, pesi iniziali, ancora RVQ, opzioni incompatibili ------------------------------------------------------------------

def _quiet(_):
    return None


def test_a_historical_run_resumes_only_with_the_explicit_recipe(tmp_path):
    """Un checkpoint di prima dei campi: con la CLI di oggi e nessuna opzione la ripresa si ferma senza toccarlo, con la ricetta storica continua."""
    root, manifest = _tree(tmp_path)
    historical = R.with_historical_tokenizer(R.small_config(datasets=None, max_steps=1))
    out = tmp_path / "run"
    R.train(historical, manifest, [root], out, log=_quiet)
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    for key in ("window_ms", "tokenizer_activation"):
        del state["config"]["model"][key]
    torch.save(state, out / "checkpoint.pt")
    before = (out / "checkpoint.pt").read_bytes(), (out / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError, match="'model'"):
        R.train(R.small_config(datasets=None, max_steps=2), manifest, [root], out, log=_quiet)
    assert ((out / "checkpoint.pt").read_bytes(), (out / "metrics.jsonl").read_bytes()) == before
    assert R.train(replace(historical, max_steps=2), manifest, [root], out, log=_quiet)["steps"] == 2


def test_init_from_wants_the_tokenizer_of_the_checkpoint(tmp_path):
    """L'attivazione non sta nei pesi: T100-L (100 ms, identity) si caricherebbe senza errori nel default con GELU. Ora e' un errore, come per
    un checkpoint storico; con il tokenizer del checkpoint si carica, e un config senza i campi vale come storico."""
    root, manifest = _tree(tmp_path)
    new = R.small_config(datasets=None, max_steps=1)
    linear = replace(new, model=replace(new.model, tokenizer_activation="identity"))
    old = R.with_historical_tokenizer(new)
    for name, cfg in (("t100l", linear), ("old", old)):
        R.train(cfg, manifest, [root], tmp_path / name, log=_quiet)
    state = torch.load(tmp_path / "old" / "checkpoint.pt", weights_only=False)
    for key in ("window_ms", "tokenizer_activation"):
        del state["config"]["model"][key]
    (tmp_path / "bare").mkdir()
    torch.save(state, tmp_path / "bare" / "checkpoint.pt")

    def start(cfg, source, name):
        return R.train(replace(cfg, lr=0.0), manifest, [root], tmp_path / name, log=_quiet, init_from=tmp_path / source / "checkpoint.pt")

    with pytest.raises(ValueError, match="tokenizer_activation 'identity' -> 'gelu'"):
        start(new, "t100l", "a")
    for source in ("old", "bare"):
        with pytest.raises(ValueError, match="window_ms None -> 100.0, tokenizer_activation 'identity' -> 'gelu'"):
            start(new, source, f"b_{source}")
        assert start(old, source, f"c_{source}")["steps"] == 1
    assert start(linear, "t100l", "d")["steps"] == 1
    logsquare = replace(linear, model=replace(linear.model, tokenizer_logsquare=True))  # l'epsilon non sta nei pesi: solo la guardia lo vede
    R.train(logsquare, manifest, [root], tmp_path / "lq", log=_quiet)
    with pytest.raises(ValueError, match="tokenizer_logsquare_eps 0.001 -> 0.01"):
        start(replace(logsquare, model=replace(logsquare.model, tokenizer_logsquare_eps=0.01)), "lq", "e")
    assert start(logsquare, "lq", "f")["steps"] == 1
    loaded, source = (torch.load(tmp_path / n / "checkpoint.pt", weights_only=False)["student"] for n in ("d", "t100l"))
    assert all(torch.equal(loaded[k], source[k]) for k in source)


def test_the_rvq_anchor_is_refused_when_its_windows_cannot_stay_hidden(tmp_path, monkeypatch):
    """Con un raggio delle query maggiore di 0 l'ancora RVQ perde senza errori gran parte delle sue finestre da 200 ms. Il trainer lo rifiuta prima
    di leggere i dati; il banco storico, oppure raggio 0 con JEPA spenta, passano la guardia."""
    root, manifest = _tree(tmp_path)

    class Past(Exception):
        pass

    def stop(*_a, **_k):
        raise Past

    monkeypatch.setattr(R, "load_index", stop)
    new = R.with_rvq(R.small_config(datasets=None, max_steps=1), tmp_path / "codes")
    assert new.model.window_ms == 100.0
    with pytest.raises(ValueError, match="ancora RVQ con raggio delle query 2"):
        R.train(new, manifest, [root], tmp_path / "a", log=_quiet)
    off = replace(new, jepa=replace(new.jepa, jepa_weight=0.0, masked_loss_radius=0))
    with pytest.raises(ValueError, match="ancora RVQ con raggio delle query 1"):  # la soglia e' 0, non il raggio del kernel
        R.train(replace(off, jepa=replace(off.jepa, masked_loss_radius=1)), manifest, [root], tmp_path / "b", log=_quiet)
    for k, cfg in enumerate((R.with_historical_tokenizer(new), off, R.small_config(datasets=None, max_steps=1))):
        with pytest.raises(Past):
            R.train(cfg, manifest, [root], tmp_path / f"ok{k}", log=_quiet)


def test_options_that_need_the_historical_bank_say_so(tmp_path):
    cli = _cli()
    cal = tmp_path / "calibration.npz"
    cal.write_bytes(b"solo per lo SHA")
    branches = ["--window1-rules", "--keep-calibration", str(cal), "--spectral-branches", "psd,wavelet"]
    with pytest.raises(ValueError, match="--window-ms 25"):
        _build(cli, branches)
    for extra, expected in ((["--window-ms", "25"], (None, "gelu")), (["--window-ms", "25", "--tokenizer-activation", "identity"], (None, "identity"))):
        cfg = _build(cli, branches + extra)
        assert _tokenizer(cfg) == expected and cfg.model.spectral_branches == ("psd", "wavelet")
    cfg = cli.build_config(cli.build_parser().parse_args(cli.historical_args(["--preset", "small"] + BASE + branches)))
    assert _tokenizer(cfg) == (None, "identity") and cfg.model.spectral_branches
    with pytest.raises(ValueError, match="--tokenizer-activation identity"):
        _build(cli, ["--tokenizer-logsquare"])
    assert _tokenizer(_build(cli, ["--tokenizer-logsquare", "--tokenizer-activation", "identity"])) == (100.0, "identity")
