"""Verificatore (seconda esecuzione): il ramo di arresto del TRAINER vero (run.py:1669-1683), che il revisore ha solo letto.
Trainer vero `R.train` sulla fixture sintetica dei test consegnati (segnali sintetici, loader, encoder, AdamW, EMA, checkpoint), un processo.
Caso A: cancello che fallisce (min_negatives irraggiungibile), coverage_block piccolo, enforce_coverage True -> ci si aspetta lo stop alla fine
del primo blocco. Caso B: stesso run con enforce_coverage False -> arriva in fondo e registra il blocco fallito. Caso C: cancello raggiungibile
(min_negatives 1, 32 query nel blocco) con enforce True -> arriva in fondo. Nessun dato vero."""
import json, sys, tempfile, shutil
from dataclasses import asdict, replace
from pathlib import Path
import torch
TREE = Path(__import__("sys").argv[1])  # albero a db5ac41 (git archive)
sys.path.insert(0, str(TREE / "tests/cpu"))
from test_pose_training import _pose_setup, _r3_config
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig

def cfg_for(pose_root, steps, **kw):
    base = _r3_config(pose_root, total=steps, keep=())
    c = dict(arm="CN", capacity_per_dataset=8, min_negatives=1, gap_seconds=.001, fill_steps=0, ramp_steps=1, rnc_stride=2)
    c.update(kw)
    return replace(base, max_steps=steps, batch_size=4, fixed_eval_every=0, ckpt_every=3, eval_every=10**6,  # niente allarme di collasso della fixture
                  
                   model=replace(base.model, contrastive_projection=True, muscle_dropout=0.),
                   loader=replace(base.loader, min_window_s=1., max_window_s=1.),
                   contrastive=asdict(ContrastiveConfig(**c)), datasets=("emg2pose",))

work = Path(tempfile.mkdtemp(dir="."))
out = {}
try:
    root, manifest, pose_root = _pose_setup(work)
    cases = {"A_fallisce_enforce_true": dict(steps=12, min_negatives=64, coverage_block=4, enforce_coverage=True),
             "B_fallisce_enforce_false": dict(steps=12, min_negatives=64, coverage_block=4, enforce_coverage=False),
             "C_passa_enforce_true": dict(steps=12, min_negatives=1, coverage_block=8, enforce_coverage=True)}
    for name, kw in cases.items():
        steps = kw.pop("steps")
        cfg = cfg_for(pose_root, steps, **kw)
        folder = work / name
        try:
            s = R.train(cfg, manifest, [root], folder, log=lambda _: None)
            err = None
        except Exception as exc:
            s, err = None, f"{type(exc).__name__}: {exc}"
        recs = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines()] if (folder / "metrics.jsonl").exists() else []
        files = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
        ck = torch.load(folder / "checkpoint.pt", weights_only=False) if (folder / "checkpoint.pt").exists() else None
        row = {"errore": err, "stopped": None if s is None else str(s["stopped"])[:160], "steps": None if s is None else s["steps"],
               "record_metriche": [r["step"] for r in recs], "file": files,
               "STOP": (folder / "STOP").read_text()[:120] if (folder / "STOP").exists() else None,
               "checkpoint_step": None if ck is None else ck.get("step"),
               "blocchi_completi": None if s is None else [(b["through_step"], b["updates"], round(b["nce_fraction"], 3), b["rnc_active_fraction"], b["passed"],
                                                           {d: (v["queries"], v["eligible"]) for d, v in b["datasets"].items()}) for b in s["contrastive"]["coverage_completed_blocks"]],
               "optimizer_updates": None if s is None else s["contrastive"]["optimizer_updates"]}
        out[name] = row
        print(name, json.dumps(row, default=str))
finally:
    shutil.rmtree(work, ignore_errors=True)
json.dump(out, open("w_trainer_stop.json", "w"), indent=1, default=str)
