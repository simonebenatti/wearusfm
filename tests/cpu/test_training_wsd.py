"""Schedule WSD e rami (D14, firmata il 04/10/2026; contato in aggiornamenti dell'optimizer, decisione del 09/10/2026): warmup, fase
stabile, decay lineare a zero sul 20% finale, checkpoint pre-decay e della fase stabile, rami identici a un run fatto da zero con lo schedule
del ramo (pilot sulle epoche: decay da un checkpoint stabile; continuazione a 2D: fase stabile prolungata dal pre-decay)."""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from wearusfm.training import run as R  # noqa: E402

from test_pretraining_loader import _tree  # noqa: E402


def _cfg(total, keep=()):
    return R.with_wsd(R.small_config(datasets=None), total, warmup_steps=2, keep_checkpoints=keep)


def _same_weights(a, b):
    sa, sb = (torch.load(p, weights_only=False) for p in (a, b))
    assert sa["step"] == sb["step"]
    for part in ("student", "teacher"):
        for k in sa[part]:
            assert torch.equal(sa[part][k], sb[part][k]), (part, k)


def _lines(p):
    return [json.loads(x) for x in (p / "metrics.jsonl").read_text().splitlines()]


def test_learning_rate_curve_and_d14_warmup():
    cfg = _cfg(10)
    assert cfg.schedule.decay_start == 8 and cfg.max_steps == 10
    lr = [R._lr_at(s, cfg) / cfg.lr for s in range(10)]
    assert lr == [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]  # warmup di 2, stabile, decay sugli ultimi 2 (verso 0 al passo 10)
    long = _cfg(100)
    assert [round(R._lr_at(s, long) / long.lr, 4) for s in (79, 80, 90, 99)] == [1.0, 1.0, 0.5, 0.05]
    const = R.small_config()
    assert [R._lr_at(s, const) for s in range(4)] == [const.lr * min(1.0, (s + 1) / 2) for s in range(4)]  # il sanity resta com'era
    assert R.wsd_warmup_steps(92_600) == 1852 and R.wsd_warmup_steps(10_000) == 1000  # 2% di D, almeno 1.000
    with pytest.raises(ValueError):  # checkpoint da tenere fuori dalla fase stabile
        R.Schedule("wsd", 10, 0.2, (9,))
    with pytest.raises(ValueError):
        R.Schedule("constant", 10)


def test_runs_too_short_for_the_signed_shape_are_refused():
    with pytest.raises(ValueError, match="nessun passo di decay"):  # 2 - round(0,4) = 2: decay vuoto
        R.with_wsd(R.small_config(), 2, warmup_steps=1)
    with pytest.raises(ValueError, match="prima della fine del warmup"):  # decay da 4, warmup di 5
        R.with_wsd(R.small_config(), 5, warmup_steps=5)


@pytest.mark.parametrize("stop", [7, 8, 9])  # prima del decay, esattamente al suo inizio (pre-decay gia' salvato), dentro il decay
def test_resume_across_the_decay_boundary(tmp_path, stop):
    root, mpath = _tree(tmp_path)
    run = lambda cfg, out: R.train(cfg, mpath, [root], tmp_path / out, log=lambda m: None)  # noqa: E731
    run(_cfg(10), "straight")
    run(replace(_cfg(10), max_steps=stop), "chained")  # primo job della catena, fermato a `stop`
    run(_cfg(10), "chained")  # ripresa fino alla fine
    for name in ("checkpoint.pt", "checkpoint_predecay.pt", "checkpoint_postdecay.pt"):
        _same_weights(tmp_path / "chained" / name, tmp_path / "straight" / name)
    assert torch.load(tmp_path / "straight" / "checkpoint_predecay.pt", weights_only=False)["step"] == 8
    assert torch.load(tmp_path / "straight" / "checkpoint_postdecay.pt", weights_only=False)["step"] == 10
    assert [r["lr"] for r in _lines(tmp_path / "chained")] == [r["lr"] for r in _lines(tmp_path / "straight")]


def test_wsd_run_keeps_predecay_and_stable_checkpoints(tmp_path):
    root, mpath = _tree(tmp_path)
    out = tmp_path / "a"
    s = R.train(_cfg(10, keep=(2,)), mpath, [root], out, log=lambda m: None)
    assert s["steps"] == 10 and s["stopped"] == "max_steps"
    assert torch.load(out / "checkpoint_predecay.pt", weights_only=False)["step"] == 8  # 10 - round(0,2 x 10)
    assert torch.load(out / "checkpoint_step0000002.pt", weights_only=False)["step"] == 2
    lrs = [r["lr"] / 3e-4 for r in _lines(out)]  # il learning rate di ogni aggiornamento eseguito
    assert lrs == pytest.approx([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])  # warmup, stabile, decay sugli ultimi 2


def test_branches_equal_runs_from_scratch(tmp_path):
    root, mpath = _tree(tmp_path)
    run = lambda cfg, out, **kw: R.train(cfg, mpath, [root], tmp_path / out, log=lambda m: None, **kw)  # noqa: E731
    run(_cfg(10, keep=(4,)), "long")  # T = 10: decay da 8; copia stabile al passo 4
    run(_cfg(5), "short")  # T = 5: decay da 4; pre-decay al passo 4
    # continuazione a 2D: dal pre-decay del run corto, stabile fino a 8 e decay: identico al run lungo fatto da zero
    run(_cfg(10), "cont", branch_from=tmp_path / "short" / "checkpoint_predecay.pt")
    _same_weights(tmp_path / "cont" / "checkpoint.pt", tmp_path / "long" / "checkpoint.pt")
    # ramo del pilot: dalla copia stabile del run lungo, decay a T = 5: identico al run corto fatto da zero
    run(_cfg(5), "branch", branch_from=tmp_path / "long" / "checkpoint_step0000004.pt")
    _same_weights(tmp_path / "branch" / "checkpoint.pt", tmp_path / "short" / "checkpoint.pt")
    b = json.loads((tmp_path / "branch" / "branch.json").read_text())
    assert b["step"] == 4 and b["from"].endswith("checkpoint_step0000004.pt") and len(b["sha256"]) == 64
    assert [r["step"] for r in _lines(tmp_path / "branch")] == [5]  # il ramo scrive solo i suoi passi


def test_branch_and_resume_refusals(tmp_path):
    root, mpath = _tree(tmp_path)
    run = lambda cfg, out, **kw: R.train(cfg, mpath, [root], tmp_path / out, log=lambda m: None, **kw)  # noqa: E731
    run(_cfg(10, keep=(4,)), "long")
    stable, final = tmp_path / "long" / "checkpoint_step0000004.pt", tmp_path / "long" / "checkpoint.pt"
    with pytest.raises(ValueError, match="configurazione diversa"):  # cambia anche il picco: non e' piu' un ramo
        run(replace(_cfg(5), lr=1e-3), "x1", branch_from=stable)
    with pytest.raises(ValueError, match="dentro il decay"):  # checkpoint gia' nel decay dell'origine
        run(_cfg(20), "x2", branch_from=final)
    with pytest.raises(ValueError, match="decay cominci dopo"):  # T = 4: decay da 3, prima del passo 4 del checkpoint
        run(_cfg(4), "x3", branch_from=stable)
    with pytest.raises(ValueError, match="schedule diverso"):  # una ripresa non cambia lo schedule
        run(_cfg(12), "long")
    with pytest.raises(ValueError, match="oltre total_steps"):
        run(replace(_cfg(5), max_steps=6), "x4")
    state = torch.load(stable, weights_only=False)
    state.pop("stream")
    torch.save(state, tmp_path / "old.pt")  # un checkpoint di prima della ripresa esatta
    with pytest.raises(ValueError, match="stream"):
        run(_cfg(5), "x5", branch_from=tmp_path / "old.pt")


def test_cli_wsd_and_branch(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root, mpath = _tree(tmp_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_jepa.py"
    base = [sys.executable, str(script), "--preset", "small", "--manifest", str(mpath), "--root", str(root), "--device", "cpu",
            "--datasets", "all", "--schedule", "wsd", "--warmup-steps", "2"]
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "a"), "--total-steps", "5", "--keep-checkpoints", "2"], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg["schedule"] == {"kind": "wsd", "total_steps": 5, "decay_fraction": 0.2, "keep_checkpoints": [2]} and cfg["max_steps"] == 5
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "b"), "--total-steps", "10", "--branch-from",
                               str(tmp_path / "a" / "checkpoint_predecay.pt")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and json.loads((tmp_path / "b" / "summary.json").read_text())["steps"] == 10, r.stderr[-2000:]
    bad = subprocess.run(base + ["--out-dir", str(tmp_path / "c")], capture_output=True, text=True, timeout=300)
    assert bad.returncode != 0 and "--total-steps" in bad.stderr
