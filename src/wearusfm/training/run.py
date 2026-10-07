"""Ciclo di training su piu' passi (passo 6, sanity JEPA; poi finestra 1): dataloader vero -> `WearUsFM` -> `training.jepa`, con diagnostiche,
allarmi, checkpoint e ripresa. La logica e' qui (provabile su CPU); il comando e' `scripts/train_jepa.py`.

**Valori firmati da Simone il 03/10/2026** (`docs/fogli_firma_d9_d10.md`), nel preset `sanity_config()`: sanity su emg2qwerty (17); frazione
nascosta 0,5 (15); soglie d'allarme (16): collasso da query sotto 0,05 o rango effettivo sotto il 10% della dimensione per 3 valutazioni di fila,
valutate ogni 500 passi su un batch fisso; peso delle ancore 0,2 e momento EMA 0,996 (18); dropout del muscolo 0,4 (19); target (b) (20); patch
25 ms, contesto 1-4 s, maschere D10 a tubi (10-12); salti spezzati (8); filtro nel dataloader (13).

**Scelte di AG, da confermare (non firmate):** taglia del modello del sanity (~30M parametri, i default di lavoro della finestra 1: K = 64, encoder
locale a 2 livelli); AdamW con lr 3e-4, warmup lineare di 1.000 passi poi costante (lo schedule vero e' D14), weight decay 0,05, clip del gradiente
1,0; batch di 32 finestre; batch di validazione per le diagnostiche estratto con un seme fisso dal pretraining (non dal test, che resta intatto);
**ancora RVQ spenta** finche' il tokenizer congelato non e' collegato al ciclo (su Leonardo). Su GPU: **bf16** (autocast; il front-end resta in
float64) e **activation checkpointing** per blocco (v10 §5.5): senza, il preset chiedeva ~190 GB (review del 03/10, job 59276285).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.data import rvq_codes as RC
from wearusfm.data import virtual_montage as VM
from wearusfm.model.anchors import N_RVQ_CODES
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.model.spectral_targets import TARGET_VERSION
from wearusfm.model.spectral_step1 import READOUT_VERSION
from wearusfm.training.spectral_calibration import calibration_signature, read_calibration, sha256_file
from wearusfm.model.query_decoder import probe_query_collapse
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
    amp_bf16: bool = True  # autocast bf16, solo su CUDA (v10 §5.5)
    target_version: str = TARGET_VERSION
    readout_version: str = READOUT_VERSION
    keep_calibration: str | None = None
    keep_calibration_sha256: str | None = None
    keep_grad_every: int = 0  # optional diagnostic, NOT additional optimizer steps


def sanity_config(max_steps: int = 20000) -> RunConfig:
    return RunConfig(
        model=FMConfig(dim=384, n_heads=6, k_latents=64, local_layers=2, backbone_layers=8, decoder_layers=2, muscle_dropout=0.4,
                       grad_checkpoint=True),
        jepa=JEPAConfig(target="b", anchor_weight=0.2, ema_momentum=0.996),
        loader=L.signed_config((20.0, 450.0)),
        datasets=("emg2qwerty",), batch_size=32, lr=3e-4, warmup_steps=1000, weight_decay=0.05, grad_clip=1.0, max_steps=max_steps,
        eval_every=500, ckpt_every=500, seed=0)


def with_window1_rules(cfg: RunConfig) -> RunConfig:
    """Le decisioni del 04/10/2026, dalla finestra 1: quote nel tempo (1), perdita per finestra (2), nessun target delle ancore entro 100 ms dal
    bordo di un tratto (3) («approvo tutto»); ancora multi-scala («ok per l'ancora multi-scala nella finestra 1»); montaggi virtuali dalle griglie
    HD coi parametri del foglio D6a (firmato il 05/10/2026). Il preset del sanity resta quello collaudato. L'ancora RVQ non c'e' (decisione del
    04/10): `with_rvq` non va applicato."""
    return replace(cfg, jepa=replace(cfg.jepa, loss_per_window=True), model=replace(cfg.model, multiscale_anchor=True),
                   loader=replace(cfg.loader, time_weighted=True, anchor_edge_guard_s=0.1, multiscale_anchor=True, virtual=VM.VirtualSpec()))


def with_rvq(cfg: RunConfig, codes_root: str | Path) -> RunConfig:
    """Ancora RVQ accesa (D5b): testa RVQ da 8192 codici e codici precalcolati dal dataloader."""
    return replace(cfg, model=replace(cfg.model, rvq_codes=N_RVQ_CODES), loader=replace(cfg.loader, rvq_codes_root=str(codes_root)))


def with_spectral_keep(cfg: RunConfig, calibration: str | Path, weight: float) -> RunConfig:
    """Both A (weight=0) and B instantiate the SAME head, preserving encoder initialization."""
    return replace(cfg, model=replace(cfg.model, keep_readout=True), jepa=replace(cfg.jepa, keep_weight=weight),
                   keep_calibration=str(calibration), keep_calibration_sha256=sha256_file(calibration))


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

    def __init__(self, index: L.ManifestIndex, cfg: L.LoaderConfig, batch_size: int, seed: int, scales: dict, window_s: dict | None = None):
        self.index, self.cfg, self.batch_size, self.seed, self.scales, self.window_s = index, cfg, batch_size, seed, scales, window_s

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        loader = L.PretrainLoader(self.index, self.cfg, dict(self.scales), self.window_s)
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
        dev = enc.z.device
        probe_t = torch.linspace(0, max(p - 1, 0), steps=n_probe, device=dev).round().long()
        probe_q = enc.ids[torch.arange(n_probe, device=dev) % n_ch]
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
          num_workers: int = 0, time_limit_s: float | None = None, log=print, window_s: dict | None = None,
          init_from: Path | None = None) -> dict:
    """Allena fino a `max_steps`, al limite di tempo o a un allarme; riprende da `out_dir/checkpoint.pt` se c'e'. Scrive `metrics.jsonl`,
    `checkpoint.pt` e `summary.json` in `out_dir` (fuori dal repo). Uno stop numerico preserva l'ultimo checkpoint gia' salvato;
    il passo difettoso non aggiorna optimizer/EMA e non viene contato."""
    t_start = time.time()
    if cfg.target_version != TARGET_VERSION:
        raise ValueError("target_version incompatible with this code")
    if cfg.readout_version != READOUT_VERSION:
        raise ValueError("readout_version incompatible with this code")
    if cfg.keep_grad_every < 0:
        raise ValueError("keep_grad_every must be nonnegative")
    if cfg.model.keep_readout and torch.device(device).type not in ("cpu", "cuda"):
        raise ValueError("Step 1 supports CPU/CUDA only")
    if cfg.jepa.keep_weight > 0 and not (cfg.model.keep_readout and cfg.loader.multiscale_anchor):
        raise ValueError("keep requires a registered head and multiscale targets")
    ckpt = out_dir / "checkpoint.pt"
    resume_state = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    if resume_state is not None and cfg.loader.multiscale_anchor:
        previous = resume_state.get("config", {})
        if previous.get("target_version") != cfg.target_version:
            raise ValueError("PSD target definition changed: use a NEW run directory, not resume")
    if resume_state is not None and (cfg.model.keep_readout or resume_state.get("config", {}).get("model", {}).get("keep_readout")):
        previous = resume_state["config"]
        if any(previous.get(k) != config_to_dict(cfg).get(k) for k in ("target_version", "readout_version", "keep_calibration_sha256", "model", "jepa")):
            raise ValueError("Step 1 resume identity changed: use a NEW run directory")
    calibration = None
    if cfg.model.keep_readout:
        if not cfg.keep_calibration or sha256_file(cfg.keep_calibration) != cfg.keep_calibration_sha256:
            raise ValueError("missing/changed keep calibration file")
        calibration = read_calibration(cfg.keep_calibration, calibration_signature(cfg, manifest, scales, window_s))
        if init_from is not None:
            raise ValueError("Step 1 warm-start not enabled: use paired NEW runs (no unchecked strict=False)")
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16 and str(device).startswith("cuda"))
    stop_file = out_dir / "STOP"
    if stop_file.exists():  # un allarme o uno stop numerico hanno fermato il run: i job successivi della catena non riprendono
        reason = stop_file.read_text().strip()
        log(f"run fermato in precedenza ({reason}): nessun passo")
        return {"stopped": f"fermato in precedenza: {reason}", "steps": None, "elapsed_s": 0.0}
    if cfg.loader.multiscale_anchor != cfg.model.multiscale_anchor:
        raise ValueError("ancora multi-scala a meta': servono insieme le teste (model) e i target (loader); vedi with_window1_rules")
    if bool(cfg.loader.rvq_codes_root) != bool(cfg.model.rvq_codes):
        raise ValueError("ancora RVQ a meta': servono insieme la testa (model.rvq_codes) e i codici (loader.rvq_codes_root); vedi with_rvq")
    torch.manual_seed(cfg.seed)
    index = load_index(manifest, roots, cfg.datasets)
    student = WearUsFM(cfg.model).to(device)
    if calibration is not None and resume_state is None:
        student.spectral_keep.fit_target_stats(calibration[0], calibration[1])
    teacher = make_teacher(student).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, monitor = 0, AlarmMonitor(cfg.alarm, cfg.model.dim)
    ckpt = out_dir / "checkpoint.pt"
    if ckpt.exists():
        state = resume_state  # load_state_dict porta pesi e momenti sul dispositivo dei parametri
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        opt.load_state_dict(state["optimizer"])
        step = int(state["step"])
        monitor.low_collapse, monitor.low_rank = state["alarm_counts"]
        torch.set_rng_state(state["torch_rng"].cpu())  # map_location porta anche lo stato casuale sulla GPU: set_rng_state vuole la CPU
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        log(f"ripresa dal passo {step}")
    elif init_from is not None:  # pesi di un altro run (studente e teacher), ottimizzatore e passi da zero: es. una diagnostica sul modello del sanity
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        log(f"pesi iniziali da {init_from} (passo {state['step']} di quel run); ottimizzatore e passi da zero")
    (out_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
    scales = dict(scales or {})
    val_loader = L.PretrainLoader(index, cfg.loader, dict(scales), window_s)
    val = _to_device(*L.to_model_inputs(val_loader.batch(min(cfg.batch_size, 16), np.random.default_rng([cfg.seed, 999]))), device)
    stream = torch.utils.data.DataLoader(BatchStream(index, cfg.loader, cfg.batch_size, cfg.seed * 1000 + step, scales, window_s), batch_size=None,
                                         num_workers=num_workers, persistent_workers=num_workers > 0)
    metrics = (out_dir / "metrics.jsonl").open("a")
    reason, it = "max_steps", iter(stream)
    numerical_failure = False

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
        codes_t = torch.from_numpy(RC.pack_codes(batch.rvq_codes)).to(device) if batch.rvq_codes is not None else None
        with amp:
            losses = jepa_losses(student, teacher, inp, visible, cfg.jepa, anchor_targets=batch.anchor_targets,
                                 kinds=torch.as_tensor(batch.kind).to(device), rvq_on=rvq_on if codes_t is not None else None,
                                 rvq_codes=(lambda c, w: codes_t[c, w]) if codes_t is not None else None)
        if not torch.isfinite(losses["total"]):
            reason = f"perdita non finita al passo {step}"
            numerical_failure = True
            break
        if cfg.jepa.keep_weight > 0 and cfg.keep_grad_every and step % cfg.keep_grad_every == 0:
            # Weighted gradient magnitudes on SHARED encoder params; loss sizes alone do not prove dominance.
            shared = [p for name, p in student.named_parameters() if p.requires_grad and
                      name.split(".")[0] in ("tokenizer", "identity", "local", "pool", "backbone")]
            components = {"jepa": losses["jepa"], "keep": cfg.jepa.keep_weight * losses["keep"]}
            components["anchors"] = losses["total"] - components["jepa"] - components["keep"]
            for name, component in components.items():
                grads = torch.autograd.grad(component, shared, retain_graph=True, allow_unused=True)
                squares = [g.detach().float().square().sum() for g in grads if g is not None]
                losses["shared_grad_" + name] = torch.stack(squares).sum().sqrt() if squares else losses["total"].new_zeros(())
        losses["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        # Loss finita non implica gradienti finiti. Gestione esplicita per registrare lo STOP anche senza un'eccezione del clipping.
        if not torch.isfinite(gnorm):
            reason = f"gradienti non finiti al passo {step} (norma: {float(gnorm)})"
            numerical_failure = True
            opt.zero_grad(set_to_none=True)
            break  # niente optimizer, EMA, incremento del passo o metriche di un aggiornamento non eseguito
        opt.step()
        ema_update(teacher, student, cfg.jepa.ema_momentum)
        step += 1
        rec = {"step": step, "lr": _lr_at(step - 1, cfg), "grad_norm": float(gnorm), "t_data_s": t_data, "t_step_s": time.time() - t0,
               "windows": len(batch.signals), "skipped_sessions": batch.skipped_sessions, **{k: float(v.detach()) for k, v in losses.items()}}
        # origine e topologia presentata (D9 decisione 1): finestre per classe di quota e quante presentate come montaggio virtuale (D6a)
        rec["classes"] = {c: sum(r.get("quota_class") == c for r in batch.rows) for c in ("A", "B", "C")}
        rec["virtual"] = sum(p != "full" for p in batch.presented or [])
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            with amp:
                d = diagnostics(student, teacher, val)
            # anche le varianze assolute (review del 03/10): rapporto e rango non vedono un'uscita quasi costante, le varianze si'
            rec.update({"student_collapse_ratio": d["student_collapse"]["ratio"], "teacher_collapse_ratio": d["teacher_collapse"]["ratio"],
                        "student_erank": d["student_erank"], "teacher_erank": d["teacher_erank"],
                        "student_var_samples": d["student_collapse"]["var_between_samples"],
                        "student_var_queries": d["student_collapse"]["var_between_queries"],
                        "teacher_var_samples": d["teacher_collapse"]["var_between_samples"]})
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
            mem = f", memoria GPU max {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if str(device).startswith("cuda") else ""
            log(f"passo {step}: totale {rec['total']:.4f}, jepa {rec['jepa']:.4f}, {rec['t_step_s']:.2f} s/passo (dati {t_data:.2f} s){mem}")
    if not numerical_failure:
        save()  # su errore numerico non sovrascrivere l'ultimo checkpoint valido, neppure con RNG avanzato nel passo fallito
    metrics.close()
    if reason.startswith("allarme") or numerical_failure:
        stop_file.write_text(reason + "\n")  # stop definitivo per i job successivi della catena
    summary = {"stopped": reason, "steps": step, "elapsed_s": time.time() - t_start, "parameters": sum(p.numel() for p in student.parameters()),
               "device": device, "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else None}
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
