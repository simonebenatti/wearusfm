"""Bracci dell'obiettivo W1-OBJ (16/10/2026): `JEPAConfig.jepa_weight` (ancore-only) e `anchor_weight` = 0 (JEPA puro) dal comando di
training. Pesi controllati; totale e contratto delle loss distribuite coi pesi; con la JEPA a peso 0 il gradiente dello studente non dipende
dal teacher (la JEPA e' davvero spenta), con le ancore a peso 0 e' quello della sola JEPA; la ripresa accetta i checkpoint di prima e rifiuta
un peso cambiato; le opzioni del comando arrivano nella configurazione e nel totale registrato."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_jepa import CFG, _heterogeneous_targets, _inputs, _visible  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import JEPAConfig, jepa_loss_terms, jepa_losses, make_teacher  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ANCHORS = ("log_rms", "band_shape", "log_env", "ms_fast", "ms_slow")


def test_weights_are_checked_and_default_to_the_usual_objective():
    assert JEPAConfig("b", 0.2, 0.99).jepa_weight == 1.0
    JEPAConfig("b", 0.2, 0.99, jepa_weight=0.0)  # ancore-only
    JEPAConfig("b", 0.0, 0.99)  # JEPA puro
    for bad in ({"jepa_weight": -1.0}, {"jepa_weight": float("nan")}, {"jepa_weight": float("inf")}):
        with pytest.raises(ValueError, match="jepa_weight"):
            JEPAConfig("b", 0.2, 0.99, **bad)
    with pytest.raises(ValueError, match="anchor_weight"):
        JEPAConfig("b", -0.2, 0.99)
    with pytest.raises(ValueError, match="nessuna perdita"):
        JEPAConfig("b", 0.0, 0.99, jepa_weight=0.0)


def _setup(seed=7):
    torch.manual_seed(seed)
    model_cfg = replace(CFG, multiscale_anchor=True, rvq_codes=0)
    student = WearUsFM(model_cfg).eval()
    teacher = make_teacher(student)
    inp = _inputs()
    return student, teacher, inp, _visible(inp), _heterogeneous_targets(inp)


@pytest.mark.parametrize("jw,aw", [(1.0, 0.2), (0.0, 0.2), (1.0, 0.0), (0.5, 0.3)])
def test_total_and_distributed_terms_use_both_weights(jw, aw):
    student, teacher, inp, vis, tg = _setup()
    cfg = JEPAConfig("b", aw, 0.99, loss_per_window=True, jepa_weight=jw)
    losses = jepa_losses(student, teacher, inp, vis, cfg, anchor_targets=tg)
    anchors = sum(losses[k] for k in ANCHORS if k in losses)
    torch.testing.assert_close(losses["total"], jw * losses["jepa"] + aw * anchors, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(losses["jepa"])  # a peso 0 si calcola e si registra lo stesso
    terms = jepa_loss_terms(student, teacher, inp, vis, cfg, anchor_targets=tg)
    assert terms.weight["jepa"] == jw and all(terms.weight[k] == aw for k in ANCHORS if k in terms.weight)
    torch.testing.assert_close(terms.backward_loss(), losses["total"], rtol=2e-6, atol=1e-7)


def _student_grads(student, teacher, inp, vis, tg, cfg):
    student.zero_grad(set_to_none=True)
    jepa_loss_terms(student, teacher, inp, vis, cfg, anchor_targets=tg).backward_loss().backward()
    return [None if p.grad is None else p.grad.clone() for p in student.parameters()]


def _same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


def test_with_jepa_off_the_teacher_no_longer_drives_the_student():
    student, teacher, inp, vis, tg = _setup()
    other = make_teacher(student)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(torch.randn_like(p))  # un teacher diverso cambia i target della JEPA, non quelli delle ancore
    off = JEPAConfig("b", 0.2, 0.99, loss_per_window=True, jepa_weight=0.0)
    assert _same(_student_grads(student, teacher, inp, vis, tg, off), _student_grads(student, other, inp, vis, tg, off))
    on = JEPAConfig("b", 0.2, 0.99, loss_per_window=True)
    assert not _same(_student_grads(student, teacher, inp, vis, tg, on), _student_grads(student, other, inp, vis, tg, on))


def test_with_anchors_off_the_gradient_is_the_jepa_one():
    student, teacher, inp, vis, tg = _setup()
    pure = JEPAConfig("b", 0.0, 0.99, loss_per_window=True)
    g_pure = _student_grads(student, teacher, inp, vis, tg, pure)
    g_no_targets = _student_grads(student, teacher, inp, vis, None, JEPAConfig("b", 0.2, 0.99, loss_per_window=True))
    for a, b in zip(g_pure, g_no_targets):  # le ancore a peso 0 contribuiscono zero: il gradiente e' quello della sola JEPA
        if a is not None and b is not None:
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)


def test_jepa_off_without_anchor_targets_is_refused():
    student, teacher, inp, vis, _ = _setup()
    with pytest.raises(ValueError, match="senza target delle ancore"):
        jepa_losses(student, teacher, inp, vis, JEPAConfig("b", 0.2, 0.99, jepa_weight=0.0))


def test_resume_accepts_old_checkpoints_and_refuses_a_changed_weight():
    cfg = R.small_config()
    old = R.config_to_dict(cfg)
    old["jepa"].pop("jepa_weight")  # checkpoint di prima del 16/10
    R._check_resume_recipe(cfg, {"config": old})
    with pytest.raises(ValueError, match="jepa"):
        R._check_resume_recipe(replace(cfg, jepa=replace(cfg.jepa, jepa_weight=0.0)), {"config": old})


def _train(tmp_path, *extra):
    from test_pretraining_loader import _tree

    root, mpath = _tree(tmp_path)
    out = tmp_path / "run"
    argv = sys.argv
    sys.argv = ["train_jepa.py", "--preset", "small", "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out), "--device", "cpu",
                "--max-steps", "2", "--datasets", "all", "--num-workers", "0", *extra]
    try:
        with pytest.raises(SystemExit) as e:
            runpy.run_path(str(REPO / "scripts" / "train_jepa.py"), run_name="__main__")
    finally:
        sys.argv = argv
    return e.value.code, out


@pytest.mark.parametrize("extra,jw,aw", [(["--jepa-weight", "0"], 0.0, 0.2), (["--anchor-weight", "0"], 1.0, 0.0)])
def test_training_command_runs_both_arms(tmp_path, extra, jw, aw):
    code, out = _train(tmp_path, *extra)
    assert code in (0, None)
    jepa = json.loads((out / "config.json").read_text())["jepa"]
    assert (jepa["jepa_weight"], jepa["anchor_weight"]) == (jw, aw)
    last = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
    anchors = sum(last[k] for k in ANCHORS if k in last)
    assert last["total"] == pytest.approx(jw * last["jepa"] + aw * anchors, rel=1e-6)
    assert last["jepa"] > 0


@pytest.mark.parametrize("extra", [["--jepa-weight", "-1"], ["--jepa-weight", "0", "--anchor-weight", "0"], ["--anchor-weight", "nan"]])
def test_training_command_refuses_bad_weights(tmp_path, extra):
    code, _ = _train(tmp_path, *extra)
    assert code == 2  # ap.error


def test_chain_planner_accepts_the_weights_as_free_arguments():
    import importlib.util

    spec = importlib.util.spec_from_file_location("wsd_chain", REPO / "scripts" / "wsd_chain.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    dests = W.free_arg_dests("--jepa-weight 0 --anchor-weight 0.2", W._trainer_parser())
    assert dests == {"jepa_weight", "anchor_weight"} and not dests & set(W.PLANNER_DESTS + W.LAUNCHER_DESTS)


@pytest.mark.parametrize("jw,aw", [(0.0, 0.2), (1.0, 0.0)])
def test_every_student_parameter_keeps_a_gradient_in_both_arms(jw, aw):
    """DDP senza `find_unused_parameters` (il caso del default) richiede che ogni parametro usato abbia un gradiente a ogni passo: un termine
    a peso 0 resta nel grafo (`backward_loss` lo moltiplica, non lo salta), quindi gradiente zero, non assente."""
    student, teacher, inp, vis, tg = _setup()
    ref = _student_grads(student, teacher, inp, vis, tg, JEPAConfig("b", 0.2, 0.99, loss_per_window=True))
    arm = _student_grads(student, teacher, inp, vis, tg, JEPAConfig("b", aw, 0.99, loss_per_window=True, jepa_weight=jw))
    assert [g is None for g in arm] == [g is None for g in ref]


@pytest.mark.parametrize("extra", [["--jepa-weight", "0"], ["--anchor-weight", "0"]])
def test_both_arms_train_under_torchrun_without_unused_parameters(tmp_path, extra):
    import socket
    import subprocess

    from test_pretraining_loader import _tree

    root, mpath = _tree(tmp_path)
    out = tmp_path / "ddp"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--master-addr=127.0.0.1", f"--master-port={port}", "--nproc_per_node=2",
           str(REPO / "scripts" / "train_jepa.py"), "--preset", "small", "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out),
           "--device", "cpu", "--batch-size", "4", "--max-steps", "3", "--datasets", "all", "--distributed", "--accum-steps", "2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["steps"] == 3 and s["stopped"] == "max_steps" and s["world_size"] == 2
