"""R3 nel trainer vero (`docs/pilot_posa_v1.md` §6, punto 2, test che il critico del 26/10 ha chiesto): training con `--pose-root` sul loader
vero, pannello fisso con l'errore per giunto, SHA delle statistiche nel checkpoint e nel summary (non nel config), ramo e ripresa che passano
le guardie, ripresa rifiutata se le statistiche cambiano; batch intero = micro-batch con la posa (contratto); passo intero senza posa
attraverso `_accumulate_terms`; standardizzazione per lato dal loader vero; flag del launcher."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ddp_equivalence import _same, _update  # noqa: E402
from test_pose_targets import _pose_tree, _stats  # noqa: E402
from test_pretraining_loader import _cfg as _loader_cfg  # noqa: E402
from test_pretraining_loader import _tree  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _pose_setup(tmp_path, per_side=False):
    root, mpath = _tree(tmp_path)
    _pose_tree(tmp_path / "pose", 24000, invalid_blocks=range(50, 60))
    by_side = {"left": {"mean": [0.0] * 20, "std": [1.0] * 20}, "right": {"mean": [5.0] * 20, "std": [2.0] * 20}} if per_side else None
    _stats(tmp_path / "pose", mean=[0.5] * 20, std=[0.25] * 20, per_side=per_side, by_side=by_side)
    return root, mpath, tmp_path / "pose"


def _r3_config(pose_root, total=6, keep=(3,)):
    cfg = R.with_wsd(R.small_config(datasets=None), total, warmup_steps=2, keep_checkpoints=keep)
    return replace(cfg, loader=replace(cfg.loader, pose_root=str(pose_root)), model=replace(cfg.model, pose_head=True),
                   jepa=replace(cfg.jepa, jepa_weight=0.0, pose_weight=1.0, loss_per_window=True), fixed_eval_every=3, fixed_eval_batches=1)


def test_train_with_pose_logs_branches_and_resumes(tmp_path):
    root, mpath, pose_root = _pose_setup(tmp_path)
    run = lambda cfg, out, **kw: R.train(cfg, mpath, [root], tmp_path / out, log=lambda m: None, **kw)  # noqa: E731
    sha = hashlib.sha256((pose_root / PT.STATS_FILE).read_bytes()).hexdigest()
    s = run(_r3_config(pose_root), "trunk")
    assert s["stopped"] == "max_steps" and s["steps"] == 6 and s["pose_stats_sha256"] == sha
    assert s["environment"]["torch"] == torch.__version__ and "python" in s["environment"]
    recs = [json.loads(x) for x in (tmp_path / "trunk" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [1, 2, 3, 4, 5, 6]
    assert all("pose" in r and "pose_windows" in r and np.isfinite(r["pose"]) and r["pose_windows"] >= 0 for r in recs)
    assert all(abs(r["total"] - (0.2 * sum(r[a] for a in ("log_rms", "band_shape", "log_env")) + r["pose"])) < 1e-5 * max(1.0, r["total"])
               for r in recs)  # composizione: 0 x jepa + 0,2 x ancore + 1 x posa (configurazione piccola, senza multi-scala)
    cfg_json = json.loads((tmp_path / "trunk" / "config.json").read_text())
    assert "pose_stats_sha256" not in cfg_json and cfg_json["loader"]["pose_root"] == str(pose_root) and cfg_json["model"]["pose_head"] is True
    prov = json.loads((tmp_path / "trunk" / "pose_provenance.json").read_text())
    assert prov["pose_stats_sha256"] == sha and prov["pose_unprompted"] is True
    fixed = [json.loads(x) for x in (tmp_path / "trunk" / "fixed_eval.jsonl").read_text().splitlines()]
    assert [f["step"] for f in fixed] == [0, 3, 6]
    assert all("pose_mae" in f and "pose_mae_std" in f and "pose_mae_j00" in f and "pose_windows" in f and "pose_count" in f for f in fixed)
    assert fixed[0]["pose_mae"] == pytest.approx(0.25 * fixed[0]["pose_mae_std"], rel=1e-6)  # deviazione 0,25 per giunto, insieme
    ck = torch.load(tmp_path / "trunk" / "checkpoint.pt", weights_only=False)
    assert ck["pose_stats_sha256"] == sha and "pose_head.out.weight" in ck["student"]
    # ramo a 1 epoca dalla copia stabile (passo 3): la guardia del ramo passa (il config non porta lo SHA; il checkpoint si')
    run(_r3_config(pose_root, total=5, keep=()), "branch", branch_from=tmp_path / "trunk" / "checkpoint_step0000003.pt")
    assert json.loads((tmp_path / "branch" / "branch.json").read_text())["step"] == 3
    # ripresa: interrotto al passo 4 e ripreso fino al 6 = run intero, bit per bit
    run(replace(_r3_config(pose_root), max_steps=4), "resume")
    run(_r3_config(pose_root), "resume")
    a, b = (torch.load(tmp_path / d / "checkpoint.pt", weights_only=False) for d in ("trunk", "resume"))
    for part in ("student", "teacher"):
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part]), part
    # statistiche cambiate dopo il checkpoint: ripresa e ramo si rifiutano prima di scrivere
    run(replace(_r3_config(pose_root), max_steps=4), "resume2")  # interrotto al passo 4
    st = json.loads((pose_root / PT.STATS_FILE).read_text())
    st["all"]["mean"] = [0.6] * 20
    (pose_root / PT.STATS_FILE).write_text(json.dumps(st))
    with pytest.raises(ValueError, match="statistiche della posa"):
        run(_r3_config(pose_root), "resume2")
    with pytest.raises(ValueError, match="statistiche della posa"):
        run(_r3_config(pose_root, total=5, keep=()), "branch2", branch_from=tmp_path / "trunk" / "checkpoint_step0000003.pt")


def _reference_pose(cfg, init, batch):
    """Come `_reference` di test_ddp_equivalence (un processo, batch intero, `jepa_losses`), con i target di posa."""
    from wearusfm.training.jepa import ema_update, jepa_losses, make_teacher

    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    m = R._prepare(batch, "cpu", torch.float64)
    student.train()
    losses = jepa_losses(student, teacher, m.inp, m.visible, cfg.jepa, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    losses["total"].backward()
    grads = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
    opt.step()
    ema_update(teacher, student, cfg.jepa.ema_momentum)
    return {"metrics": {k: float(v.detach()) for k, v in losses.items()}, "grads": grads, "student": student.state_dict(),
            "teacher": teacher.state_dict()}


def test_whole_batch_equals_micro_batches_with_pose_and_a_step_without_pose_is_finite(tmp_path):
    root, mpath, pose_root = _pose_setup(tmp_path)
    cfg = _r3_config(pose_root)
    cfg = replace(cfg, model=replace(cfg.model, muscle_dropout=0.0))
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), cfg.loader, {}, None)
    batch = loader.batch(4, np.random.default_rng(3))
    assert any(item is not None for item in batch.pose)
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    ref = _reference_pose(cfg, init, batch)
    assert "pose" in ref["metrics"] and ref["metrics"]["pose"] > 0 and any(k.startswith("pose_head") for k in ref["grads"])
    one = R.DistContext()
    _same(ref, _update(cfg, init, batch, 2, one, lambda s: None), "posa: 1 processo x 2 micro-batch")
    _same(ref, _update(cfg, init, batch, 4, one, lambda s: None), "posa: 1 processo x 4 micro-batch")
    # passo intero senza finestre di posa: contratto finito, termine e conteggio a zero, testa nel grafo con gradiente nullo
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = R.make_teacher(student)
    micros = [R._prepare(mb, "cpu", torch.float64) for mb in L.split_batch(batch, 2)]
    micros = [replace(m, pose=None) for m in micros]
    student.train()
    failure, metrics = R._accumulate_terms(None, student, teacher, micros, cfg, one, "cpu", nullcontext())
    assert failure is None and metrics["pose"] == 0.0 and metrics["pose_windows"] == 0.0 and np.isfinite(metrics["total"])
    g = student.pose_head.out.weight.grad
    assert g is not None and float(g.abs().max()) == 0.0 and all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)


def test_per_side_standardization_through_the_real_loader(tmp_path):
    root, mpath, pose_root = _pose_setup(tmp_path, per_side=True)
    meta_path = root / "emg2pose" / "u1" / "sessA" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    cfg = _loader_cfg(pose_root=str(pose_root))
    store = PT.PoseStore(pose_root)
    assert store.per_side
    out = {}
    for side in ("left", "right"):
        meta["csv_row"] = {"side": side, "moving_hand": "both"}
        meta_path.write_text(json.dumps(meta))
        view = L.SessionView.open(meta_path.parent, True)  # la vista vera della sessione (senza il controllo dello SHA del manifest)
        win = L.sample_window(view, cfg, np.random.default_rng(4))
        assert win is not None
        out[side] = store.window({"dataset": "emg2pose"}, view, win, cfg.patch_ms)
    (tl, ml, sl), (tr, mr, sr) = out["left"], out["right"]
    assert np.array_equal(ml, mr) and ml.any() and np.allclose(sl, 1.0) and np.allclose(sr, 2.0)
    assert np.allclose(tl[ml], 2 * tr[mr] + 5.0, atol=1e-3)  # sinistra (x - 0) / 1, destra (x - 5) / 2


def test_launcher_flags_for_pose():
    sp = importlib.util.spec_from_file_location("train_jepa_pose", REPO / "scripts" / "train_jepa.py")
    TJ = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TJ)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]
    cfg = TJ.build_config(TJ.build_parser().parse_args(base))
    assert cfg.loader.pose_root is None and cfg.model.pose_head is False and cfg.jepa.pose_weight == 0.0 and cfg.loader.pose_unprompted is True
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--jepa-weight", "0"]))
    assert cfg.loader.pose_root == "/x" and cfg.model.pose_head is True and cfg.jepa.pose_weight == 1.0 and cfg.jepa.jepa_weight == 0.0
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--pose-weight", "0.5", "--pose-unprompted", "off"]))
    assert cfg.jepa.pose_weight == 0.5 and cfg.loader.pose_unprompted is False
    with pytest.raises(ValueError, match="--pose-weight senza --pose-root"):
        TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-weight", "1"]))
    with pytest.raises(ValueError, match="positivo"):
        TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--pose-weight", "0"]))
    assert R.config_to_dict(cfg)["loader"]["pose_root"] == "/x"


def test_fingerprints_and_data_signature_of_runs_without_pose_are_unchanged(tmp_path):
    from wearusfm.training.pairing import batch_sha256
    from wearusfm.training.spectral_calibration import _loader_signature

    root, mpath = _tree(tmp_path)
    cfg = _loader_cfg()
    b = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), cfg).batch(3, np.random.default_rng(1))
    assert b.pose is None
    h = batch_sha256(b)
    # senza posa il campo non entra nell'impronta: uguale a quella di un batch che non avesse il campo
    from dataclasses import fields
    plain = {f.name: getattr(b, f.name) for f in fields(b) if f.name != "pose"}
    import hashlib as _h
    from wearusfm.training import pairing as P
    d = _h.sha256()
    P._update(d, plain)
    assert h == d.hexdigest()
    sig = _loader_signature(cfg)
    assert "pose_root" not in sig and "pose_unprompted" not in sig
    sig2 = _loader_signature(replace(cfg, pose_root="/x"))
    assert sig2["pose_root"] == "/x" and sig2["pose_unprompted"] is True
