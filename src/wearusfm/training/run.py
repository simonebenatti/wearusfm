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
import os
import shutil
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
class Schedule:
    """Forma del learning rate, in aggiornamenti dell'optimizer (decisione del 09/10/2026: mai in micro-batch). Il warmup lineare e' sempre
    `RunConfig.warmup_steps`.
    - "constant": warmup e poi il picco fino alla fine (il sanity e il pilota A/B, come sono stati collaudati);
    - "wsd" (D14, firmata il 04/10/2026): warmup, fase stabile al picco, **decay lineare a zero sul `decay_fraction` finale** di
      `total_steps`. Il decay comincia all'aggiornamento `decay_start` = total_steps - round(decay_fraction * total_steps); lì si salva il
      checkpoint pre-decay. `keep_checkpoints`: passi della fase stabile in cui tenere una copia del checkpoint, da cui partono i rami di
      decay del pilot sulle epoche e la continuazione a 2D (`train(..., branch_from=...)`)."""
    kind: str = "constant"
    total_steps: int | None = None
    decay_fraction: float = 0.2
    keep_checkpoints: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "wsd"):
            raise ValueError(f"schedule {self.kind!r}: 'constant' o 'wsd'")
        if self.kind == "wsd":
            if not self.total_steps or self.total_steps < 1 or not 0.0 < self.decay_fraction <= 1.0:
                raise ValueError("wsd: servono total_steps >= 1 e 0 < decay_fraction <= 1")
            if any(not 0 < s <= self.decay_start for s in self.keep_checkpoints):
                raise ValueError(f"wsd: i checkpoint da tenere stanno nella fase stabile (0, {self.decay_start}]: {self.keep_checkpoints}")
        elif self.total_steps is not None or self.keep_checkpoints:
            raise ValueError("constant: total_steps e keep_checkpoints valgono solo per 'wsd'")

    @property
    def decay_start(self) -> int | None:
        return None if self.kind != "wsd" else self.total_steps - int(round(self.decay_fraction * self.total_steps))


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
    pairing_audit: bool = False  # opt-in fresh-run audit; no data-stream resume guarantee
    # passi per segmento dello stream dei dati (`pretraining_loader.batch_stream`): la ripresa rigioca a secco al piu' un segmento per worker
    stream_segment: int = 1200
    schedule: Schedule = field(default_factory=Schedule)


def wsd_warmup_steps(steps_at_d: int) -> int:
    """D14: warmup lineare sul 2% dei passi a D, almeno 1.000. Nel pilot e nella continuazione a 2D il warmup resta quello di D per tutti i
    rami: si calcola una volta e si passa esplicito."""
    return max(1000, int(round(0.02 * steps_at_d)))


def with_wsd(cfg: RunConfig, total_steps: int, *, warmup_steps: int, keep_checkpoints: tuple[int, ...] = (),
             decay_fraction: float = 0.2) -> RunConfig:
    """Lo schedule WSD di D14 su `total_steps` aggiornamenti; il run finisce li' (`max_steps` = `total_steps`). Checkpoint con nome:
    `checkpoint_predecay.pt` (a `decay_start`), `checkpoint_postdecay.pt` (a `total_steps`), `checkpoint_step<N>.pt` (fase stabile)."""
    out = replace(cfg, warmup_steps=warmup_steps, max_steps=total_steps,
                  schedule=Schedule("wsd", total_steps, decay_fraction, tuple(sorted(set(keep_checkpoints)))))
    _check_wsd(out)
    return out


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
    """Batch del dataloader nei processi di torch: un `PretrainLoader` per worker e lo stream di `pretraining_loader.batch_stream`, che da'
    al batch di indice b (= il passo che lo consuma) lo stesso contenuto in un run continuo e in uno ripreso da `start` (a parita' di worker,
    processi e segmento). Con DDP ogni processo (`rank`) ha il suo stream."""

    def __init__(self, index: L.ManifestIndex, cfg: L.LoaderConfig, batch_size: int, seed: int, scales: dict, window_s: dict | None = None,
                 rank: int | None = None, start: int = 0, segment: int = 1200):
        self.index, self.cfg, self.batch_size, self.seed, self.scales, self.window_s = index, cfg, batch_size, seed, scales, window_s
        self.rank, self.start, self.segment = rank, start, segment

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, n = (info.id, info.num_workers) if info is not None else (0, 1)
        loader = L.PretrainLoader(self.index, self.cfg, dict(self.scales), self.window_s)
        yield from L.batch_stream(loader, self.batch_size, seed=self.seed, start=self.start, worker=wid, n_workers=n, segment=self.segment,
                                  rank=self.rank)


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


# --- piu' processi (DDP) -------------------------------------------------------------------------------------------------------------
# Decisione del 09/10/2026 (`docs/decisioni.md`, «Lavoro in parallelo»; `docs/contratto_loss_distribuite.md`): un processo per GPU, batch
# globale = `batch_size` finestre divise fra i processi, un solo aggiornamento per passo. Con un processo solo nulla cambia: nessuna
# operazione collettiva. **Provvisorio:** finche' il contratto delle loss (somme e conteggi per componente, `training/jepa.py`) non c'e', la
# perdita di ogni processo e' la sua media locale e DDP media i gradienti: NON e' la media del batch globale quando i denominatori differiscono
# fra processi. Per questo DDP si rifiuta, salvo che lo chieda esplicitamente una prova (`ddp_local_mean_for_tests`).


@dataclass(frozen=True)
class DistContext:
    """Il processo dentro un training DDP. `world_size` = 1: il percorso di sempre."""
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def active(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def dist_from_env() -> DistContext:
    """Dalle variabili di torchrun (RANK, WORLD_SIZE, LOCAL_RANK); senza, un processo solo."""
    return DistContext(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)))


class _Objective(torch.nn.Module):
    """Lo studente dentro un modulo il cui forward calcola le loss: DDP sincronizza i gradienti solo attraverso il forward del modulo che
    avvolge, mentre `jepa_losses` chiama `encode`, `decode` e le teste dello studente. Il teacher resta fuori: nessun gradiente, nessuna
    sincronizzazione dei buffer."""

    def __init__(self, student: WearUsFM):
        super().__init__()
        self.student = student

    def forward(self, compute):
        return compute(self.student)


def _any_flag(ctx: DistContext, device, *flags: bool) -> list[bool]:
    """Ogni decisione di arresto vale per tutti i processi: basta che un processo la prenda (MAX)."""
    if not ctx.active:
        return [bool(f) for f in flags]
    t = torch.tensor([1.0 if f else 0.0 for f in flags], device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return [bool(v > 0) for v in t.tolist()]


def _merge_records(parts: list[dict]) -> dict:
    """Il record di un passo dai record dei processi. Finestre, classi, montaggi virtuali e sessioni saltate si sommano; l'attesa sui dati e'
    la massima (il passo aspetta il processo piu' lento), con la media a parte; la durata del passo e' la massima. Le loss sono la media delle
    medie locali dei processi che le hanno (PROVVISORIO, vedi sopra: col contratto diventano somme / conteggi globali)."""
    out = dict(parts[0])
    out["windows"] = sum(p["windows"] for p in parts)
    out["skipped_sessions"] = sum(p["skipped_sessions"] for p in parts)
    out["virtual"] = sum(p["virtual"] for p in parts)
    out["classes"] = {c: sum(p["classes"][c] for p in parts) for c in parts[0]["classes"]}
    out["t_data_s"] = max(p["t_data_s"] for p in parts)
    out["t_data_s_mean"] = sum(p["t_data_s"] for p in parts) / len(parts)
    out["t_step_s"] = max(p["t_step_s"] for p in parts)
    for k in {k for p in parts for k in p.get("_losses", {})}:
        vals = [p["_losses"][k] for p in parts if k in p.get("_losses", {})]
        out[k] = sum(vals) / len(vals)
    out.pop("_losses", None)
    return out


# --- ciclo ---------------------------------------------------------------------------------------------------------------------------

def _check_wsd(cfg: RunConfig) -> None:
    """Un run WSD troppo corto per la forma firmata si rifiuta: il decay non comincia prima della fine del warmup e ha almeno un passo."""
    s = cfg.schedule
    if cfg.max_steps > s.total_steps:
        raise ValueError(f"wsd: max_steps {cfg.max_steps} oltre total_steps {s.total_steps} (dopo il decay il learning rate e' 0)")
    if s.total_steps - s.decay_start < 1:
        raise ValueError(f"wsd: total_steps {s.total_steps} troppo corto, nessun passo di decay con decay_fraction {s.decay_fraction}")
    if s.decay_start < cfg.warmup_steps:
        raise ValueError(f"wsd: il decay comincerebbe (passo {s.decay_start}) prima della fine del warmup ({cfg.warmup_steps} passi): "
                         "run troppo corto per lo schedule di D14")


def _check_branch(cfg: RunConfig, state: dict, source: Path) -> None:
    """Un ramo (pilot sulle epoche, continuazione a 2D; D14) e' identico a un run fatto da zero con lo schedule nuovo solo se: tutto il resto
    della configurazione e' uguale (stesso warmup e stesso picco compresi), il checkpoint e' della fase stabile del run d'origine e cade
    anche nella fase stabile del ramo, e lo stream dei dati del checkpoint e' ripetibile."""
    old, new = state.get("config", {}), config_to_dict(cfg)
    differ = sorted(k for k in set(old) | set(new) if k not in ("schedule", "max_steps") and old.get(k) != new.get(k))
    if differ:
        raise ValueError(f"ramo da {source}: configurazione diversa dall'origine in {differ}; cambiano solo schedule e max_steps")
    step = int(state["step"])
    src = Schedule(**{**old["schedule"], "keep_checkpoints": tuple(old["schedule"].get("keep_checkpoints", ()))}) if old.get("schedule") \
        else Schedule()
    if src.kind == "wsd" and step > src.decay_start:
        raise ValueError(f"ramo da {source}: il checkpoint (passo {step}) e' dentro il decay dell'origine (da {src.decay_start})")
    if cfg.schedule.kind != "wsd" or step > cfg.schedule.decay_start:
        raise ValueError(f"ramo da {source}: serve uno schedule wsd il cui decay cominci dopo il passo {step} del checkpoint")
    if state.get("stream") is None:
        raise ValueError(f"ramo da {source}: checkpoint senza lo stato dello stream, la sequenza dei batch non si ripete")


def _lr_at(step: int, cfg: RunConfig) -> float:
    """Learning rate dell'aggiornamento `step` (0 = il primo). Warmup lineare; con "wsd", dall'aggiornamento `decay_start` il fattore scende
    linearmente: 1 - (step - decay_start) / passi di decay, cioe' 1 al primo aggiornamento del decay e 1 / passi di decay all'ultimo (dopo,
    a `total_steps`, sarebbe 0: il run finisce li')."""
    lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
    s = cfg.schedule
    if s.kind == "wsd" and step >= s.decay_start:
        n_decay = s.total_steps - s.decay_start
        lr *= max(0.0, 1.0 - (step - s.decay_start) / max(1, n_decay))
    return lr


def _to_device(inp, visible, rvq_on, device):
    inp.signals = [x.to(device) for x in inp.signals]
    inp.codes = {k: v.to(device) for k, v in inp.codes.items()}
    inp.sets = {k: v.to(device) for k, v in inp.sets.items()}
    inp.qc_valid = inp.qc_valid.to(device)
    return inp, visible.to(device), rvq_on.to(device)


def train(cfg: RunConfig, manifest: Path, roots: list[Path], out_dir: Path, *, scales: dict | None = None, device: str = "cpu",
          num_workers: int = 0, time_limit_s: float | None = None, log=print, window_s: dict | None = None,
          init_from: Path | None = None, dist: DistContext | None = None, ddp_local_mean_for_tests: bool = False,
          branch_from: Path | None = None) -> dict:
    """Allena fino a `max_steps`, al limite di tempo o a un allarme; riprende da `out_dir/checkpoint.pt` se c'e'. Scrive `metrics.jsonl`,
    `checkpoint.pt` e `summary.json` in `out_dir` (fuori dal repo). Uno stop numerico preserva l'ultimo checkpoint gia' salvato;
    il passo difettoso non aggiorna optimizer/EMA e non viene contato.
    Con `dist` attivo (DDP, un processo per GPU, gruppo gia' inizializzato): `cfg.batch_size` e' il batch GLOBALE, diviso fra i processi;
    file, log e diagnostiche solo dal processo 0; ogni decisione di arresto vale per tutti; checkpoint col generatore casuale di ogni processo.
    Vedi la sezione «piu' processi» sopra per il limite provvisorio sulle loss."""
    t_start = time.time()
    ctx = dist or DistContext()
    say = log if ctx.is_main else (lambda m: None)
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
    if ctx.active:
        if not torch.distributed.is_initialized():
            raise ValueError("DDP: il gruppo dei processi va inizializzato prima (scripts/train_jepa.py --distributed)")
        if (torch.distributed.get_rank(), torch.distributed.get_world_size()) != (ctx.rank, ctx.world_size):
            raise ValueError("DDP: rank e world_size diversi da quelli del gruppo dei processi")
        if cfg.batch_size % ctx.world_size:
            raise ValueError(f"DDP: batch globale {cfg.batch_size} non divisibile per {ctx.world_size} processi")
        if not ddp_local_mean_for_tests:
            raise NotImplementedError("DDP: manca il contratto delle loss distribuite (somme e conteggi per componente in training/jepa.py, "
                                      "docs/contratto_loss_distribuite.md): la media locale non e' la media del batch globale")
        if cfg.pairing_audit or cfg.keep_grad_every:
            raise NotImplementedError("DDP: pairing_audit e keep_grad_every non ancora supportati (contratto, punto 4)")
        if str(device).startswith("cuda"):
            device = f"cuda:{ctx.local_rank}"
            torch.cuda.set_device(ctx.local_rank)
    per_process = cfg.batch_size // ctx.world_size
    ckpt = out_dir / "checkpoint.pt"
    if cfg.pairing_audit and (out_dir.exists() or init_from is not None):
        raise ValueError("pairing audit requires a NEW uninterrupted run (no resume/warm-start)")
    if cfg.schedule.kind == "wsd":
        _check_wsd(cfg)
    resume_state = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    default_schedule = json.loads(json.dumps(asdict(Schedule()), default=list))  # i checkpoint di prima del WSD: schedule costante
    if resume_state is not None and resume_state.get("config", {}).get("schedule", default_schedule) != config_to_dict(cfg)["schedule"]:
        raise ValueError("ripresa con uno schedule diverso da quello del checkpoint: per cambiarlo si fa un ramo in una cartella nuova "
                         "(branch_from)")
    branch = None
    if resume_state is None and branch_from is not None:  # un ramo: lo stato completo di un checkpoint della fase stabile, cartella nuova
        if init_from is not None or cfg.pairing_audit:
            raise ValueError("branch_from esclude init_from e pairing_audit")
        resume_state = torch.load(branch_from, map_location="cpu", weights_only=False)
        _check_branch(cfg, resume_state, branch_from)
        branch = {"from": str(branch_from), "sha256": sha256_file(branch_from), "step": int(resume_state["step"])}
    if resume_state is not None and cfg.loader.multiscale_anchor:
        previous = resume_state.get("config", {})
        if previous.get("target_version") != cfg.target_version:
            raise ValueError("PSD target definition changed: use a NEW run directory, not resume")
    if resume_state is not None and (cfg.model.keep_readout or resume_state.get("config", {}).get("model", {}).get("keep_readout")):
        previous = resume_state["config"]
        if any(previous.get(k) != config_to_dict(cfg).get(k) for k in ("target_version", "readout_version", "keep_calibration_sha256", "model", "jepa")):
            raise ValueError("Step 1 resume identity changed: use a NEW run directory")
    stream = {"num_workers": num_workers, "segment": cfg.stream_segment, "world_size": ctx.world_size}
    if resume_state is not None and resume_state.get("stream") is not None and resume_state["stream"] != stream:
        raise ValueError(f"ripresa con uno stream diverso ({stream}) da quello del checkpoint ({resume_state['stream']}): la sequenza dei batch "
                         "non sarebbe la stessa (contratto, punto 4). Stessi worker, processi e segmento, oppure un run nuovo")
    if resume_state is not None and int(resume_state.get("world_size", 1)) != ctx.world_size:
        raise ValueError(f"ripresa con {ctx.world_size} processi da un checkpoint di {resume_state.get('world_size', 1)}: non prevista "
                         "(nessuna ripresa elastica, contratto punto 4)")
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
        say(f"run fermato in precedenza ({reason}): nessun passo")
        return {"stopped": f"fermato in precedenza: {reason}", "steps": None, "elapsed_s": 0.0}
    if cfg.loader.multiscale_anchor != cfg.model.multiscale_anchor:
        raise ValueError("ancora multi-scala a meta': servono insieme le teste (model) e i target (loader); vedi with_window1_rules")
    if bool(cfg.loader.rvq_codes_root) != bool(cfg.model.rvq_codes):
        raise ValueError("ancora RVQ a meta': servono insieme la testa (model.rvq_codes) e i codici (loader.rvq_codes_root); vedi with_rvq")
    torch.manual_seed(cfg.seed)  # stessa inizializzazione su tutti i processi
    index = load_index(manifest, roots, cfg.datasets)
    student = WearUsFM(cfg.model).to(device)
    if calibration is not None and resume_state is None:
        student.spectral_keep.fit_target_stats(calibration[0], calibration[1])
    teacher = make_teacher(student).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step, monitor = 0, AlarmMonitor(cfg.alarm, cfg.model.dim)
    if resume_state is not None:
        state = resume_state  # load_state_dict porta pesi e momenti sul dispositivo dei parametri
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        opt.load_state_dict(state["optimizer"])
        step = int(state["step"])
        monitor.low_collapse, monitor.low_rank = state["alarm_counts"]
        if ctx.active:  # il generatore di questo processo
            mine = state["rank_rng"][ctx.rank]
            torch.set_rng_state(mine["torch_rng"].cpu())
            if mine.get("cuda_rng") is not None and str(device).startswith("cuda"):
                torch.cuda.set_rng_state(mine["cuda_rng"].cpu(), torch.device(device))
        else:
            torch.set_rng_state(state["torch_rng"].cpu())  # map_location porta anche lo stato casuale sulla GPU: set_rng_state vuole la CPU
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        say(f"ramo da {branch['from']} al passo {step}" if branch else f"ripresa dal passo {step}")
    else:
        if init_from is not None:  # pesi di un altro run (studente e teacher), ottimizzatore e passi da zero: es. una diagnostica sul sanity
            state = torch.load(init_from, map_location="cpu", weights_only=False)
            student.load_state_dict(state["student"])
            teacher.load_state_dict(state["teacher"])
            say(f"pesi iniziali da {init_from} (passo {state['step']} di quel run); ottimizzatore e passi da zero")
        if ctx.active:  # dropout del muscolo: un generatore diverso per processo (i dati sono gia' diversi)
            torch.manual_seed(cfg.seed * 1_000_003 + ctx.rank + 1)
    model = None
    if ctx.active:
        from torch.nn.parallel import DistributedDataParallel
        # find_unused_parameters: teste senza target in un passo, testa keep del braccio A, testa RVQ spenta (contratto, punto 4)
        model = DistributedDataParallel(_Objective(student), device_ids=[ctx.local_rank] if str(device).startswith("cuda") else None,
                                        find_unused_parameters=True)
        for t in teacher.state_dict().values():  # il teacher del processo 0 su tutti (lo studente lo allinea DDP)
            torch.distributed.broadcast(t, src=0)
    if cfg.pairing_audit:
        from wearusfm.training.pairing import state_sha256
        (out_dir / "pairing.json").write_text(json.dumps({
            "student_initial_sha256": state_sha256(student.state_dict()),
            "teacher_initial_sha256": state_sha256(teacher.state_dict()),
            "seed": cfg.seed, "num_workers": num_workers, "resume": False,
        }, indent=2))
    if ctx.is_main:
        (out_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
        if branch is not None:  # da dove parte il ramo (D14: pilot sulle epoche, continuazione a 2D)
            (out_dir / "branch.json").write_text(json.dumps(branch, indent=1))
    scales = dict(scales or {})
    val = None
    if ctx.is_main:  # diagnostiche solo sul processo 0
        val_loader = L.PretrainLoader(index, cfg.loader, dict(scales), window_s)
        val = _to_device(*L.to_model_inputs(val_loader.batch(min(cfg.batch_size, 16), np.random.default_rng([cfg.seed, 999]))), device)
    if resume_state is not None and resume_state.get("stream") is None:
        say("checkpoint senza lo stato dello stream (codice precedente al 09/10/2026): la ripresa non ripete la sequenza dei batch")
    data = torch.utils.data.DataLoader(BatchStream(index, cfg.loader, per_process, cfg.seed, scales, window_s,
                                                   rank=ctx.rank if ctx.active else None, start=step, segment=cfg.stream_segment),
                                       batch_size=None, num_workers=num_workers, persistent_workers=num_workers > 0,
                                       # un generatore proprio: creare l'iteratore estrae un seme, e dal generatore globale sposterebbe il
                                       # dropout del muscolo nei run ripresi (lo stream usa i suoi semi espliciti, non questo)
                                       generator=torch.Generator().manual_seed(cfg.seed))
    metrics = (out_dir / "metrics.jsonl").open("a") if ctx.is_main else None
    reason, it = "max_steps", iter(data)
    numerical_failure = False

    def save(names: tuple[str, ...] = ()):
        """checkpoint.pt (la ripresa) e, se `names`, copie con nome che restano (pre-decay e fase stabile dello schedule WSD)."""
        extra = {}
        if ctx.active:  # tutti i processi partecipano; scrive solo il processo 0
            mine = {"torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(torch.device(device)) if str(device).startswith("cuda") else None}
            rank_rng = [None] * ctx.world_size
            torch.distributed.all_gather_object(rank_rng, mine)
            extra = {"rank_rng": rank_rng, "world_size": ctx.world_size}
        if ctx.is_main:
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "alarm_counts": (monitor.low_collapse, monitor.low_rank), "config": config_to_dict(cfg),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "stream": stream, **extra},
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            for name in names:
                tmp = out_dir / (name + ".tmp")
                shutil.copyfile(ckpt, tmp)
                tmp.replace(out_dir / name)
        if ctx.active:
            torch.distributed.barrier()

    while step < cfg.max_steps:
        out_of_time, = _any_flag(ctx, device, time_limit_s is not None and time.time() - t_start > time_limit_s)
        if out_of_time:
            reason = "limite di tempo"
            break
        t0 = time.time()
        batch = next(it)
        t_data = time.time() - t0
        if batch.index != step:  # l'ordine dei worker del DataLoader deve ricomporre la sequenza degli indici
            raise RuntimeError(f"batch di indice {batch.index} al passo {step}: stream fuori sequenza")
        if cfg.pairing_audit:
            from wearusfm.training.pairing import batch_sha256
            batch_hash = batch_sha256(batch)  # CPU data BEFORE any conversion/device transfer
        inp, visible, rvq_on = _to_device(*L.to_model_inputs(batch), device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        student.train()
        opt.zero_grad(set_to_none=True)
        codes_t = torch.from_numpy(RC.pack_codes(batch.rvq_codes)).to(device) if batch.rvq_codes is not None else None

        def compute(m):
            return jepa_losses(m, teacher, inp, visible, cfg.jepa, anchor_targets=batch.anchor_targets,
                               kinds=torch.as_tensor(batch.kind).to(device), rvq_on=rvq_on if codes_t is not None else None,
                               rvq_codes=(lambda c, w: codes_t[c, w]) if codes_t is not None else None)

        with amp:
            losses = model(compute) if model is not None else compute(student)
        bad_loss, = _any_flag(ctx, device, not bool(torch.isfinite(losses["total"])))
        if bad_loss:  # anche se e' un altro processo: nessuno fa il backward
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
        losses["total"].backward()  # con DDP: i gradienti si mediano fra i processi qui
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)  # uguale su tutti i processi dopo la media
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
               "windows": len(batch.signals), "skipped_sessions": batch.skipped_sessions}
        loss_values = {k: float(v.detach()) for k, v in losses.items()}
        if not ctx.active:
            rec.update(loss_values)  # un processo solo: il record di sempre, stesso ordine dei campi
        # origine e topologia presentata (D9 decisione 1): finestre per classe di quota e quante presentate come montaggio virtuale (D6a)
        rec["classes"] = {c: sum(r.get("quota_class") == c for r in batch.rows) for c in ("A", "B", "C")}
        rec["virtual"] = sum(p != "full" for p in batch.presented or [])
        if ctx.active:  # il record del passo dai record di tutti i processi, sul processo 0
            parts = [None] * ctx.world_size if ctx.is_main else None
            torch.distributed.gather_object({**rec, "_losses": loss_values}, parts, dst=0)
            rec = _merge_records(parts) if ctx.is_main else {**rec, **loss_values}
        if cfg.pairing_audit:
            rec["batch_sha256"] = batch_hash
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            alarm = None
            if ctx.is_main:
                with amp:
                    d = diagnostics(student, teacher, val)
                # anche le varianze assolute (review del 03/10): rapporto e rango non vedono un'uscita quasi costante, le varianze si'
                rec.update({"student_collapse_ratio": d["student_collapse"]["ratio"], "teacher_collapse_ratio": d["teacher_collapse"]["ratio"],
                            "student_erank": d["student_erank"], "teacher_erank": d["teacher_erank"],
                            "student_var_samples": d["student_collapse"]["var_between_samples"],
                            "student_var_queries": d["student_collapse"]["var_between_queries"],
                            "teacher_var_samples": d["teacher_collapse"]["var_between_samples"]})
                alarm = monitor.update(min(d["student_collapse"]["ratio"], d["teacher_collapse"]["ratio"]), d["student_erank"])
            if ctx.active:  # la decisione del processo 0 vale per tutti
                box = [alarm]
                torch.distributed.broadcast_object_list(box, src=0)
                alarm = box[0]
            if alarm:
                if ctx.is_main:
                    rec["alarm"] = alarm
                    metrics.write(json.dumps(rec) + "\n")
                reason = "allarme: " + alarm
                break
        if ctx.is_main:
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
        names = ()
        if cfg.schedule.kind == "wsd":  # D14: il checkpoint pre-decay e quelli della fase stabile da cui partono i rami
            names = (("checkpoint_predecay.pt",) if step == cfg.schedule.decay_start else ()) + \
                    ((f"checkpoint_step{step:07d}.pt",) if step in cfg.schedule.keep_checkpoints else ()) + \
                    (("checkpoint_postdecay.pt",) if step == cfg.schedule.total_steps else ())
        if step % cfg.ckpt_every == 0 or names:
            save(names)
        if step % 50 == 0 or step == 1:
            mem = f", memoria GPU max {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if str(device).startswith("cuda") else ""
            say(f"passo {step}: totale {rec['total']:.4f}, jepa {rec['jepa']:.4f}, {rec['t_step_s']:.2f} s/passo (dati {rec['t_data_s']:.2f} s){mem}")
    if not numerical_failure:
        save()  # su errore numerico non sovrascrivere l'ultimo checkpoint valido, neppure con RNG avanzato nel passo fallito
    if metrics is not None:
        metrics.close()
    if ctx.is_main and (reason.startswith("allarme") or numerical_failure):
        stop_file.write_text(reason + "\n")  # stop definitivo per i job successivi della catena
    gpu_mem = torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else None
    if ctx.active and gpu_mem is not None:  # la memoria massima fra i processi
        t = torch.tensor([gpu_mem], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        gpu_mem = float(t.item())
    summary = {"stopped": reason, "steps": step, "elapsed_s": time.time() - t_start, "parameters": sum(p.numel() for p in student.parameters()),
               "device": device, "gpu_max_memory_gib": gpu_mem,
               # provenienza del codice, dai launcher (scripts/slurm/lib_code_dir.sh, 09/10/2026); None se il run non e' partito da un launcher
               "code": {"dir": os.environ.get("WEARUSFM_CODE_DIR"), "sha": os.environ.get("WEARUSFM_CODE_SHA")}}
    if ctx.active:
        summary.update({"world_size": ctx.world_size, "batch_per_process": per_process,
                        "ddp_loss": "media locale per processo, SOLO collaudo dell'impianto" if ddp_local_mean_for_tests else "contratto"})
    if ctx.is_main:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    say(f"fine: {reason}, passo {step}")
    return summary


def small_config(**overrides) -> RunConfig:
    """Una configurazione minuscola per le prove (stessi valori firmati per maschere, ancore, EMA e target)."""
    base = sanity_config(max_steps=4)
    model = replace(base.model, dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1)
    return replace(base, model=model, batch_size=3, eval_every=2, ckpt_every=2, warmup_steps=2, **overrides)


def count_parameters(cfg: FMConfig) -> int:
    return sum(p.numel() for p in WearUsFM(cfg).parameters())
