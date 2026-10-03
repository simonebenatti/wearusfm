"""Ciclo di training su piu' passi (passo 6, sanity JEPA; poi finestra 1): dataloader vero -> `WearUsFM` -> `training.jepa`, con diagnostiche,
allarmi, checkpoint e ripresa. La logica e' qui (provabile su CPU); il comando e' `scripts/train_jepa.py`.

**Valori firmati da Simone il 03/10/2026** (`docs/fogli_firma_d9_d10.md`), nel preset `sanity_config()`: sanity su emg2qwerty (17); frazione
nascosta 0,5 (15); soglie d'allarme (16): collasso da query sotto 0,05 o rango effettivo sotto il 10% della dimensione per 3 valutazioni di fila,
valutate ogni 500 passi su un batch fisso; peso delle ancore 0,2 e momento EMA 0,996 (18); dropout del muscolo 0,4 (19); target (b) (20); patch
25 ms, contesto 1-4 s, maschere D10 a tubi (10-12); salti spezzati (8); filtro nel dataloader (13).

**Scelte di AG, da confermare (non firmate):** taglia del modello del sanity (~30M parametri, i default di lavoro della finestra 1: K = 64, encoder
locale a 2 livelli); AdamW con lr 3e-4, warmup lineare di 1.000 passi poi costante (lo schedule vero e' D14), weight decay 0,05, clip del gradiente
1,0; batch di 32 finestre; batch di validazione per le diagnostiche estratto con un seme fisso dal pretraining (non dal test, che resta intatto);
**ancora RVQ spenta** finche' il tokenizer congelato non e' collegato al ciclo (su Leonardo).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.model.query_decoder import probe_query_collapse
from wearusfm.training import masking as MK
from wearusfm.training.jepa import JEPAConfig, effective_rank, ema_update, make_teacher, jepa_losses


@dataclass(frozen=True)
class AlarmRule:
    collapse_ratio_min: float = 0.05  # firmata (16)
    erank_fraction_min: float = 0.10  # firmata (16)
    patience: int = 3  # valutazioni di fila (16)


@dataclass(frozen=True)
class RunConfig:
    model: FMConfig
    jepa: JEPAConfig
    loader: L.LoaderConfig
    datasets: tuple[str, ...] | None  # None = tutto il pretraining del manifest
    batch_size: int
    lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    max_steps: int
    eval_every: int
    ckpt_every: int
    seed: int
    alarm: AlarmRule = field(default_factory=AlarmRule)


def sanity_config(max_steps: int = 20000) -> RunConfig:
    return RunConfig(
        model=FMConfig(dim=384, n_heads=6, k_latents=64, local_layers=2, backbone_layers=8, decoder_layers=2, muscle_dropout=0.4),
        jepa=JEPAConfig(target="b", anchor_weight=0.2, ema_momentum=0.996),
        loader=L.LoaderConfig(min_window_s=1.0, max_window_s=4.0, split_at_gaps=True, mask=MK.MaskSpec.d10_proposal(0.5), k_neighbors=8,
                              filter_band_hz=(20.0, 450.0)),
        datasets=("emg2qwerty",), batch_size=32, lr=3e-4, warmup_steps=1000, weight_decay=0.05, grad_clip=1.0, max_steps=max_steps,
        eval_every=500, ckpt_every=500, seed=0)


def config_to_dict(cfg: RunConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=list))


# --- dati ----------------------------------------------------------------------------------------------------------------------------

def load_index(manifest: Path, roots: list[Path], datasets: tuple[str, ...] | None) -> L.ManifestIndex:
    idx = L.ManifestIndex.load(manifest, roots)
    if datasets is not None:
        keep = [i for i, r in enumerate(idx.rows) if r["dataset"] in datasets]
        if not keep:
            raise ValueError(f"nessuna riga di pretraining per {datasets}")
        w = idx.weights[keep]
        idx = L.ManifestIndex([idx.rows[i] for i in keep], w / w.sum(), idx.roots)
    return idx


class BatchStream(torch.utils.data.IterableDataset):
    """Batch del dataloader nei processi di torch: un `PretrainLoader` e un generatore per processo (seme = seme del run, processo, ripresa)."""

    def __init__(self, index: L.ManifestIndex, cfg: L.LoaderConfig, batch_size: int, seed: int, scales: dict):
        self.index, self.cfg, self.batch_size, self.seed, self.scales = index, cfg, batch_size, seed, scales

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        loader = L.PretrainLoader(self.index, self.cfg, dict(self.scales))
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield loader.batch(self.batch_size, rng)


# --- allarmi e diagnostiche -----------------------------------------------------------------------------------------------------------

class AlarmMonitor:
    def __init__(self, rule: AlarmRule, dim: int):
        self.rule, self.dim = rule, dim
        self.low_collapse = 0
        self.low_rank = 0

    def update(self, collapse_ratio: float, erank: float) -> str | None:
        self.low_collapse = self.low_collapse + 1 if not collapse_ratio >= self.rule.collapse_ratio_min else 0  # NaN conta come allarme
        self.low_rank = self.low_rank + 1 if not erank >= self.rule.erank_fraction_min * self.dim else 0
        if self.low_collapse >= self.rule.patience:
            return f"collasso da query: rapporto sotto {self.rule.collapse_ratio_min} per {self.low_collapse} valutazioni"
        if self.low_rank >= self.rule.patience:
            return f"rango effettivo sotto {self.rule.erank_fraction_min:.0%} della dimensione per {self.low_rank} valutazioni"
        return None


@torch.no_grad()
def diagnostics(student: WearUsFM, teacher: WearUsFM, val: tuple, n_probe: int = 16) -> dict:
    """Su un batch fisso, senza maschera: collasso da query (stesse query di sonda su tutti i campioni: identita' dei primi canali del primo
    campione a istanti fissi) per studente e teacher, e rango effettivo delle uscite del backbone mediate sui latenti, per istante valido."""
    inp, _, _ = val
    was = student.training
    student.eval()
    out = {}
    for name, model in (("student", student), ("teacher", teacher)):
        enc = model.encode(inp, None)
        p = int(enc.key_time_valid.sum(dim=1).min())
        n_ch = min(inp.counts[0], n_probe)
        probe_t = torch.linspace(0, max(p - 1, 0), steps=n_probe).round().long()
        probe_q = enc.ids[torch.arange(n_probe) % n_ch]
        out[f"{name}_collapse"] = probe_query_collapse(model.decoder, enc.z, probe_q, probe_t, enc.key_time_valid)
        zt = enc.z.mean(dim=2)[enc.key_time_valid]  # (istanti validi, d)
        out[f"{name}_erank"] = effective_rank(zt)
    student.train(was)
    return out


# --- ciclo ---------------------------------------------------------------------------------------------------------------------------

def _lr_at(step: int, cfg: RunConfig) -> float:
    return cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))


def _to_device(inp, visible, rvq_on, device):
    inp.signals = [x.to(device) for x in inp.signals]
    inp.codes = {k: v.to(device) for k, v in inp.codes.items()}
    inp.sets = {k: v.to(device) for k, v in inp.sets.items()}
    inp.qc_valid = inp.qc_valid.to(device)
    return inp, visible.to(device), rvq_on.to(device)


def train(cfg: RunConfig, manifest: Path, roots: list[Path], out_dir: Path, *, scales: dict | None = None, device: str = "cpu",
          num_workers: int = 0, time_limit_s: float | None = None, log=print) -> dict:
    """Allena fino a `max_steps`, al limite di tempo o a un allarme; riprende da `out_dir/checkpoint.pt` se c'e'. Scrive `metrics.jsonl`,
    `checkpoint.pt` e `summary.json` in `out_dir` (fuori dal repo)."""
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir / "STOP"
    if stop_file.exists():  # un allarme o una perdita non finita hanno fermato il run: i job successivi della catena non riprendono
        reason = stop_file.read_text().strip()
        log(f"run fermato in precedenza ({reason}): nessun passo")
        return {"stopped": f"fermato in precedenza: {reason}", "steps": None, "elapsed_s": 0.0}
    torch.manual_seed(cfg.seed)
    index = load_index(manifest, roots, cfg.datasets)
    student = WearUsFM(cfg.model).to(device)
    teacher = make_teacher(student).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, monitor = 0, AlarmMonitor(cfg.alarm, cfg.model.dim)
    ckpt = out_dir / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        opt.load_state_dict(state["optimizer"])
        step = int(state["step"])
        monitor.low_collapse, monitor.low_rank = state["alarm_counts"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        log(f"ripresa dal passo {step}")
    (out_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
    scales = dict(scales or {})
    val_loader = L.PretrainLoader(index, cfg.loader, dict(scales))
    val = _to_device(*L.to_model_inputs(val_loader.batch(min(cfg.batch_size, 16), np.random.default_rng([cfg.seed, 999]))), device)
    stream = torch.utils.data.DataLoader(BatchStream(index, cfg.loader, cfg.batch_size, cfg.seed * 1000 + step, scales), batch_size=None,
                                         num_workers=num_workers, persistent_workers=num_workers > 0)
    metrics = (out_dir / "metrics.jsonl").open("a")
    reason, it = "max_steps", iter(stream)

    def save():
        torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "alarm_counts": (monitor.low_collapse, monitor.low_rank), "config": config_to_dict(cfg),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   ckpt.with_suffix(".tmp"))
        ckpt.with_suffix(".tmp").replace(ckpt)

    while step < cfg.max_steps:
        if time_limit_s is not None and time.time() - t_start > time_limit_s:
            reason = "limite di tempo"
            break
        t0 = time.time()
        batch = next(it)
        t_data = time.time() - t0
        inp, visible, rvq_on = _to_device(*L.to_model_inputs(batch), device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        student.train()
        opt.zero_grad(set_to_none=True)
        losses = jepa_losses(student, teacher, inp, visible, cfg.jepa, anchor_targets=batch.anchor_targets)
        if not torch.isfinite(losses["total"]):
            reason = f"perdita non finita al passo {step}"
            break
        losses["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(teacher, student, cfg.jepa.ema_momentum)
        step += 1
        rec = {"step": step, "lr": _lr_at(step - 1, cfg), "grad_norm": float(gnorm), "t_data_s": t_data, "t_step_s": time.time() - t0,
               "windows": len(batch.signals), "skipped_sessions": batch.skipped_sessions, **{k: float(v.detach()) for k, v in losses.items()}}
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            d = diagnostics(student, teacher, val)
            rec.update({"student_collapse_ratio": d["student_collapse"]["ratio"], "teacher_collapse_ratio": d["teacher_collapse"]["ratio"],
                        "student_erank": d["student_erank"], "teacher_erank": d["teacher_erank"]})
            alarm = monitor.update(min(d["student_collapse"]["ratio"], d["teacher_collapse"]["ratio"]), d["student_erank"])
            if alarm:
                rec["alarm"] = alarm
                metrics.write(json.dumps(rec) + "\n")
                reason = "allarme: " + alarm
                break
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        if step % cfg.ckpt_every == 0:
            save()
        if step % 50 == 0 or step == 1:
            log(f"passo {step}: totale {rec['total']:.4f}, jepa {rec['jepa']:.4f}, {rec['t_step_s']:.2f} s/passo (dati {t_data:.2f} s)")
    save()
    metrics.close()
    if reason.startswith("allarme") or reason.startswith("perdita non finita"):
        stop_file.write_text(reason + "\n")  # stop definitivo per i job successivi della catena
    summary = {"stopped": reason, "steps": step, "elapsed_s": time.time() - t_start, "parameters": sum(p.numel() for p in student.parameters()),
               "device": device}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"fine: {reason}, passo {step}")
    return summary


def small_config(**overrides) -> RunConfig:
    """Una configurazione minuscola per le prove (stessi valori firmati per maschere, ancore, EMA e target)."""
    base = sanity_config(max_steps=4)
    model = replace(base.model, dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1)
    return replace(base, model=model, batch_size=3, eval_every=2, ckpt_every=2, warmup_steps=2, **overrides)


def count_parameters(cfg: FMConfig) -> int:
    return sum(p.numel() for p in WearUsFM(cfg).parameters())

