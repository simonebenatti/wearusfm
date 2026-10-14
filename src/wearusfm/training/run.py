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

import copy
import hashlib
import json
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
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
from wearusfm.training.jepa import JEPAConfig, effective_rank, ema_update, jepa_loss_terms, jepa_losses, loss_counts, make_teacher


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
    # micro-batch per processo e per aggiornamento (accumulo dei gradienti, contratto delle loss): il batch del processo si divide in
    # `accum_steps` parti contigue (`pretraining_loader.split_batch`); un solo clipping, optimizer ed EMA per aggiornamento
    accum_steps: int = 1
    # rami spettrali (`FMConfig.spectral_branches`): batch del training, con un generatore proprio, per le statistiche d'ingresso dei rami; si
    # calibrano una volta all'inizio di un run nuovo, poi stanno nel checkpoint
    branch_calibration_batches: int = 16
    # diagnostica dei rami (step 2C della proposta): encoder raw congelato dal teacher di `init_from`; si allenano solo rami e loro teste, col solo
    # termine dei rami sul readout (`jepa.branch_weight` > 0, `jepa.keep_weight` = 0)
    freeze_encoder: bool = False


# campi aggiunti dopo i primi checkpoint, coi valori che avevano implicitamente: un checkpoint di prima equivale a uno nuovo con questi valori
_LATER_DEFAULTS = {"accum_steps": 1, "branch_calibration_batches": 16, "freeze_encoder": False}
BRANCH_MODEL_FIELDS = {"spectral_branches": [], "branch_dim": 128, "branch_hidden": 48, "wavelet_learnable": False, "branch_alpha": 0.5}


def _config_with_defaults(d: dict) -> dict:
    out = {**_LATER_DEFAULTS, **d}
    if isinstance(out.get("model"), dict):
        out["model"] = {**BRANCH_MODEL_FIELDS, **out["model"]}
    if isinstance(out.get("jepa"), dict):
        out["jepa"] = {"branch_weight": 0.0, **out["jepa"]}
    return out


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


# Rung della ladder (v10 §10.4: «larghezza condivisa scalata ovunque; profondita' scalata soprattutto nel backbone; K fisso»). FIRMATI da
# Simone il 09/10/2026 («firmo i preset»): la larghezza e' quella della tabella di v10 (640, 896; teste da 64), la profondita' del backbone quella
# che centra il nominale con i nostri blocchi, che hanno due attenzioni (tempo e latenti) piu' l'MLP, ~16 d^2 parametri invece dei ~12 d^2
# di un blocco standard: 12 blocchi a 640 ~ 16 standard, 18 a 896 ~ 24 standard, cioe' il bordo basso dei layer di v10. Il 30M e' il preset
# del sanity. Conteggi (con l'ancora multi-scala): 30M 28,3 M; 100M 104,8 M; 300M 292,0 M. Ripartizione backbone / encoder locale /
# Perceiver: 67/13/6%, 75/9/5%, 79/10/3% (v10 prescrive 75-85 / 8-15 / 5-10: il Perceiver resta sotto a 300M perche' i suoi blocchi sono
# fissi). Il resto della configurazione (picco del learning rate, batch) e' quello del sanity finche' la calibrazione della finestra 1 non lo
# fissa (D14: «il batch dei rung maggiori lo fissa la calibrazione»).
LADDER = {
    "30m": dict(dim=384, n_heads=6, local_layers=2, backbone_layers=8, decoder_layers=2),
    "100m": dict(dim=640, n_heads=10, local_layers=2, backbone_layers=12, decoder_layers=2),
    "300m": dict(dim=896, n_heads=14, local_layers=3, backbone_layers=18, decoder_layers=2),
}


def ladder_config(rung: str, max_steps: int = 20000) -> RunConfig:
    """Il preset del sanity con l'architettura del rung `rung` (LADDER); stesse regole di dati, maschere, ancore e ottimizzatore."""
    base = sanity_config(max_steps)
    return replace(base, model=replace(base.model, **LADDER[rung]))


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


def with_spectral_branches(cfg: RunConfig, kinds: tuple[str, ...], *, weight: float = 0.0, wavelet_learnable: bool = False,
                           alpha: float = 0.5, freeze_encoder: bool = False, calibration_batches: int = 16) -> RunConfig:
    """Rami spettrali (step 2-3, Simone 13/10/2026) sopra una configurazione con la testa dello Step 1 (`with_spectral_keep`): stessi target e
    calibrazione. La keep dello Step 1 sulla testa raw resta com'e' (`keep_weight`); il termine dei rami ha il suo peso (`weight` ->
    `jepa.branch_weight`; parere GPT d5c53c1). Il controllo B e' il modello SENZA rami (con rami a peso 0 l'embedding conterrebbe rami non
    allenati: e' un altro controllo). `freeze_encoder` = la diagnostica a encoder congelato (solo il termine dei rami)."""
    return replace(cfg, model=replace(cfg.model, spectral_branches=tuple(kinds), wavelet_learnable=wavelet_learnable, branch_alpha=alpha),
                   jepa=replace(cfg.jepa, branch_weight=weight), freeze_encoder=freeze_encoder, branch_calibration_batches=calibration_batches)


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


class _Shared:
    """Un array grande in viaggio dal worker al processo principale come tensore torch, cioe' in memoria condivisa invece che copiato
    dentro il pickle (collaudo GPU del 12/10/2026: ~25 ms per passo di sola ricezione, due terzi per i target delle ancore)."""

    __slots__ = ("t",)

    def __init__(self, t: torch.Tensor):
        self.t = t


SHARED_MIN_BYTES = 64 * 1024


def _to_shared(x):
    """Nel worker (`collate_fn` del DataLoader): gli array numpy numerici da almeno `SHARED_MIN_BYTES` diventano `_Shared`, ovunque nel
    batch (segnali, target delle ancore, maschere...). Stessi valori, stesso dtype e forma; il resto resta com'e'."""
    if isinstance(x, np.ndarray):
        return _Shared(torch.from_numpy(np.ascontiguousarray(x))) if x.dtype.kind in "biuf" and x.nbytes >= SHARED_MIN_BYTES else x
    if isinstance(x, list):
        return [_to_shared(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_shared(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_shared(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        y = copy.copy(x)  # niente __init__/__post_init__ di nuovo: si sostituiscono solo i campi
        for f in fields(x):
            object.__setattr__(y, f.name, _to_shared(getattr(x, f.name)))
        return y
    return x


def _from_shared(x):
    """Nel processo principale: l'inverso esatto di `_to_shared` (array numpy che condividono la memoria dei tensori ricevuti)."""
    if isinstance(x, _Shared):
        return x.t.numpy()
    if isinstance(x, list):
        return [_from_shared(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_from_shared(v) for v in x)
    if isinstance(x, dict):
        return {k: _from_shared(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        y = copy.copy(x)
        for f in fields(x):
            object.__setattr__(y, f.name, _from_shared(getattr(x, f.name)))
        return y
    return x


def _ddp_wrap(student: WearUsFM, cfg: RunConfig, ctx: DistContext, device):
    """Lo studente in DDP. `find_unused_parameters` solo se il modello ha teste che in un passo possono restare fuori dal grafo: la testa
    RVQ (chiamata solo se ci sono finestre RVQ) e la testa keep (mai chiamata col peso 0, braccio A). Le teste delle ancore restano sempre
    nel grafo (senza target la perdita e' uno zero derivabile). Senza quelle teste il flag costa un attraversamento del grafo per passo per
    niente (avviso di DDP nel collaudo del 12/10/2026); un parametro inutilizzato con il flag spento fa fermare DDP con un errore."""
    from torch.nn.parallel import DistributedDataParallel
    unused = bool(cfg.model.rvq_codes) or bool(cfg.model.keep_readout)
    return DistributedDataParallel(_Objective(student), device_ids=[ctx.local_rank] if str(device).startswith("cuda") else None,
                                   find_unused_parameters=unused)


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
# globale = `batch_size` finestre divise fra i processi, un solo aggiornamento per passo. Con un processo solo e senza accumulo nulla cambia
# (`jepa_losses`, nessuna operazione collettiva). Con DDP o con l'accumulo le loss seguono il contratto (`_accumulate_terms`): numeratori
# locali e denominatori globali sommati su processi e micro-batch prima di ogni backward, metriche come somme / conteggi globali.


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
    avvolge, mentre `jepa_loss_terms` chiama `encode`, `decode` e le teste dello studente. Il teacher resta fuori: nessun gradiente, nessuna
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
    la massima (il passo aspetta il processo piu' lento), con la media a parte; la durata del passo e' la massima. Le loss non passano di qui:
    sono somme e conteggi globali (`_accumulate_terms`)."""
    out = dict(parts[0])
    out["windows"] = sum(p["windows"] for p in parts)
    out["skipped_sessions"] = sum(p["skipped_sessions"] for p in parts)
    out["virtual"] = sum(p["virtual"] for p in parts)
    out["classes"] = {c: sum(p["classes"][c] for p in parts) for c in parts[0]["classes"]}
    out["t_data_s"] = max(p["t_data_s"] for p in parts)
    out["t_data_s_mean"] = sum(p["t_data_s"] for p in parts) / len(parts)
    out["t_step_s"] = max(p["t_step_s"] for p in parts)
    return out


# --- loss col contratto (DDP e accumulo) -----------------------------------------------------------------------------------------------

@dataclass
class _Micro:
    """Un micro-batch sul dispositivo: ingressi del modello e argomenti delle loss."""

    inp: object
    visible: torch.Tensor
    kinds: torch.Tensor
    anchor_targets: list
    rvq_on: torch.Tensor | None
    rvq_codes: object  # lookup puro nei codici congelati (nessuna estrazione casuale), o None


def _prepare(batch: L.PretrainBatch, device, dtype: torch.dtype | None = None) -> _Micro:
    """Come il percorso a un processo; `dtype` solo per i test d'equivalenza sulla CPU (segnali in float64 per un modello in float64)."""
    inp, visible, rvq_on = _to_device(*L.to_model_inputs(batch), device)
    if dtype is not None:
        inp.signals = [x.to(dtype) for x in inp.signals]
    codes_t = torch.from_numpy(RC.pack_codes(batch.rvq_codes)).to(device) if batch.rvq_codes is not None else None
    return _Micro(inp, visible, torch.as_tensor(batch.kind).to(device), batch.anchor_targets, rvq_on if codes_t is not None else None,
                  (lambda c, w: codes_t[c, w]) if codes_t is not None else None)


def _keys_hash(*dicts) -> float:
    """Un numero per l'insieme delle chiavi (48 bit: esatto in float64), da confrontare fra processi con un solo all_reduce."""
    text = "|".join(",".join(sorted(d)) for d in dicts)
    return float(int.from_bytes(hashlib.sha256(text.encode()).digest()[:6], "little"))


def _all_max(ctx: DistContext, device, values: list[float]) -> list[float]:
    if not ctx.active:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return t.tolist()


def _keys_differ(ctx: DistContext, device, h: float) -> bool:
    """Vero su TUTTI i processi se almeno uno ha un insieme di chiavi diverso (max(h) != min(h))."""
    mx, mn = _all_max(ctx, device, [h, -h])
    return mx != h or -mn != h


def _accumulate_terms(model, student, teacher, micros: list[_Micro], cfg: RunConfig, ctx: DistContext, device, amp):
    """Forward e backward di un aggiornamento col contratto delle loss distribuite (DDP e/o accumulo; `docs/contratto_loss_distribuite.md`,
    consegna di GPT del 12/10/2026). Prima di ogni backward: conteggi `loss_counts` di tutti i micro-batch, sommati e ridotti fra i processi
    (denominatori globali D_c). Per micro-batch: forward (`no_sync` su tutti tranne l'ultimo), confronto dei conteggi del forward coi
    precalcolati, chiavi uguali fra micro-batch e processi, perdita finita; poi backward di `backward_loss(D, R)`: DDP media sui processi e il
    gradiente e' quello della media sul batch globale, senza dividere per i micro-batch. Ogni controllo vale per tutti i processi (un
    all_reduce). Ritorna (motivo d'arresto o None, metriche): le metriche sono somme e conteggi globali (numeratori, diagnostiche, contatori)."""
    keep_active = student.spectral_keep.target_active if cfg.jepa.keep_weight > 0 or cfg.jepa.branch_weight > 0 else None
    pre = [loss_counts(m.inp, m.visible, cfg.jepa, m.anchor_targets, m.rvq_on, m.rvq_codes, model_cfg=student.cfg,
                       keep_target_active=keep_active) for m in micros]
    keys = sorted(pre[0])
    if any(sorted(p) != keys for p in pre):
        raise RuntimeError(f"componenti delle loss diverse fra i micro-batch: {[sorted(p) for p in pre]}")
    if _keys_differ(ctx, device, _keys_hash(keys)):
        return "componenti delle loss diverse fra i processi", {}
    den = torch.stack([sum(p[k] for p in pre) for k in keys]).to(device=device, dtype=torch.int64)
    if ctx.active:
        torch.distributed.all_reduce(den)
    global_den = dict(zip(keys, den))
    if int(global_den["jepa"]) == 0:
        return "nessun target JEPA nel batch globale", {}
    sums: dict[tuple[str, str], torch.Tensor] = {}
    weight, first_hash = {}, None
    for i, m in enumerate(micros):
        last = i == len(micros) - 1
        with nullcontext() if model is None or last else model.no_sync():

            def compute(s, m=m):
                return jepa_loss_terms(s, teacher, m.inp, m.visible, cfg.jepa, anchor_targets=m.anchor_targets, rvq_on=m.rvq_on,
                                       rvq_codes=m.rvq_codes, kinds=m.kinds)

            with amp:
                terms = model(compute) if model is not None else compute(student)
            loss = terms.backward_loss(global_den, world_size=ctx.world_size)
            h = _keys_hash(terms.num, terms.log_num, terms.log_den, terms.log_sum)
            first_hash = h if first_hash is None else first_hash
            mismatch = sorted(terms.den) != keys or any(not torch.equal(terms.den[k], pre[i][k].to(terms.den[k].device)) for k in keys)
            flags = _all_max(ctx, device, [float(mismatch), float(not bool(torch.isfinite(loss))), float(h != first_hash), h, -h])
            if flags[0]:
                return "conteggi del forward diversi da quelli precalcolati", {}
            if flags[1]:
                return "perdita non finita", {}
            if flags[2] or flags[3] != h or -flags[4] != h:
                return "diagnostiche delle loss con chiavi diverse fra micro-batch o processi", {}
            loss.backward()
        weight = terms.weight
        for group, d in (("num", terms.num), ("log_num", terms.log_num), ("log_den", terms.log_den), ("log_sum", terms.log_sum)):
            for k, v in d.items():
                sums[group, k] = sums.get((group, k), 0) + v.detach().to(torch.float64)
    names = sorted(sums)
    vec = torch.stack([sums[n].reshape(()) for n in names]).to(device)
    if ctx.active:
        torch.distributed.all_reduce(vec)
    red = dict(zip(names, vec.tolist()))
    out = {k: red["num", k] / max(int(global_den[k]), 1) for k in keys}
    out["total"] = sum(weight[k] * out[k] for k in keys)
    diag = {k: red["log_num", k] / red["log_den", k] for (g, k) in names if g == "log_num" and red.get(("log_den", k), 0) > 0}
    diag.update({k: red["log_sum", k] for (g, k) in names if g == "log_sum"})
    if set(diag) & set(out):
        raise RuntimeError(f"diagnostiche con lo stesso nome di una componente: {sorted(set(diag) & set(out))}")
    out.update(diag)
    return None, out


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
    old, new = _config_with_defaults(state.get("config", {})), config_to_dict(cfg)  # checkpoint di prima dell'accumulo: un micro-batch
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


BRANCH_CALIBRATION_SEED = 4242


def _calibrate_branches(student: WearUsFM, cfg: RunConfig, index, scales: dict | None, window_s: dict | None, device) -> dict:
    """Statistiche d'ingresso dei rami spettrali da `branch_calibration_batches` batch del training (il corpus del pretraining), con un
    generatore proprio: lo stream dei batch del training non cambia. Uguali su tutti i processi (stesso seme). Ritorna la provenienza."""
    from wearusfm.training.pairing import state_sha256

    loader = L.PretrainLoader(index, cfg.loader, dict(scales or {}), window_s)
    rng = np.random.default_rng([cfg.seed, BRANCH_CALIBRATION_SEED])
    per_batch = min(cfg.batch_size, 32)

    def batches():
        for _ in range(cfg.branch_calibration_batches):
            yield _to_device(*L.to_model_inputs(loader.batch(per_batch, rng)), device)[0]

    frames = student.spectral_branches.calibrate(batches())
    return {"batches": cfg.branch_calibration_batches, "windows_per_batch": per_batch, "seed": [cfg.seed, BRANCH_CALIBRATION_SEED],
            "valid_frames": frames, "stats_sha256": state_sha256(dict(student.spectral_branches.named_buffers()))}


def _load_frozen_encoder(student: WearUsFM, cfg: RunConfig, init_from: Path | None) -> dict:
    """`freeze_encoder`: l'encoder raw dal TEACHER (EMA, quello delle sonde P1/P2) di `init_from`, un modello senza rami con la stessa
    configurazione (a parte rami, testa dello Step 1 e activation checkpointing). Rami e teste restano nuovi; la testa dello Step 1 si ricalibra.
    Ogni altra chiave deve caricarsi esattamente."""
    from wearusfm.model.spectral_branches import BRANCH_PREFIXES

    if init_from is None:
        raise ValueError("freeze_encoder: serve init_from (il checkpoint dell'encoder da congelare)")
    state = torch.load(init_from, map_location="cpu", weights_only=False)
    src = _config_with_defaults(state.get("config", {})).get("model", {})
    if src.get("spectral_branches"):
        raise ValueError("freeze_encoder: init_from ha gia' dei rami spettrali; si parte da un encoder senza rami")
    mine = config_to_dict(cfg)["model"]
    skip = set(BRANCH_MODEL_FIELDS) | {"keep_readout", "grad_checkpoint"}
    differ = sorted(k for k in set(src) | set(mine) if k not in skip and src.get(k) != mine.get(k))
    if differ:
        raise ValueError(f"freeze_encoder: configurazione del modello diversa da init_from in {differ}")
    missing, unexpected = student.load_state_dict(state["teacher"], strict=False)
    bad = [k for k in missing if not k.startswith(BRANCH_PREFIXES + ("spectral_keep.",))] + list(unexpected)
    if bad:
        raise ValueError(f"freeze_encoder: chiavi non caricate da init_from: {bad[:5]}")
    return {"init_from": str(init_from), "sha256": sha256_file(init_from), "step": int(state["step"]), "weights": "teacher"}


def train(cfg: RunConfig, manifest: Path, roots: list[Path], out_dir: Path, *, scales: dict | None = None, device: str = "cpu",
          num_workers: int = 0, time_limit_s: float | None = None, log=print, window_s: dict | None = None,
          init_from: Path | None = None, dist: DistContext | None = None, branch_from: Path | None = None) -> dict:
    """Allena fino a `max_steps`, al limite di tempo o a un allarme; riprende da `out_dir/checkpoint.pt` se c'e'. Scrive `metrics.jsonl`,
    `checkpoint.pt` e `summary.json` in `out_dir` (fuori dal repo). Uno stop numerico preserva l'ultimo checkpoint gia' salvato;
    il passo difettoso non aggiorna optimizer/EMA e non viene contato.
    Con `dist` attivo (DDP, un processo per GPU, gruppo gia' inizializzato): `cfg.batch_size` e' il batch GLOBALE, diviso fra i processi;
    file, log e diagnostiche solo dal processo 0; ogni decisione di arresto vale per tutti; checkpoint col generatore casuale di ogni processo.
    Con DDP o `cfg.accum_steps` > 1 le loss seguono il contratto delle loss distribuite (`_accumulate_terms`)."""
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
    if cfg.model.spectral_branches and not cfg.loader.multiscale_anchor:
        raise ValueError("rami spettrali: servono i target multiscala dello Step 1 (loader.multiscale_anchor)")
    if cfg.jepa.branch_weight > 0 and not cfg.model.spectral_branches:
        raise ValueError("branch_weight > 0 senza rami spettrali")
    if cfg.branch_calibration_batches < 1:
        raise ValueError("branch_calibration_batches almeno 1")
    if cfg.freeze_encoder:
        if not cfg.model.spectral_branches or cfg.jepa.branch_weight <= 0 or cfg.jepa.keep_weight != 0:
            raise ValueError("freeze_encoder: diagnostica dei rami spettrali, servono spectral_branches, branch_weight > 0 e keep_weight 0 "
                             "(la testa raw e' congelata)")
        if ctx.active or cfg.accum_steps > 1 or cfg.keep_grad_every or cfg.pairing_audit or branch_from is not None:
            raise NotImplementedError("freeze_encoder: un processo, senza accumulo, keep_grad_every, pairing_audit o branch_from")
    if ctx.active:
        if not torch.distributed.is_initialized():
            raise ValueError("DDP: il gruppo dei processi va inizializzato prima (scripts/train_jepa.py --distributed)")
        if (torch.distributed.get_rank(), torch.distributed.get_world_size()) != (ctx.rank, ctx.world_size):
            raise ValueError("DDP: rank e world_size diversi da quelli del gruppo dei processi")
        if cfg.batch_size % ctx.world_size:
            raise ValueError(f"DDP: batch globale {cfg.batch_size} non divisibile per {ctx.world_size} processi")
        if cfg.pairing_audit or cfg.keep_grad_every:
            raise NotImplementedError("DDP: pairing_audit e keep_grad_every non ancora supportati (contratto, punto 4)")
        if str(device).startswith("cuda"):
            device = f"cuda:{ctx.local_rank}"
            torch.cuda.set_device(ctx.local_rank)
    per_process = cfg.batch_size // ctx.world_size
    if cfg.accum_steps < 1 or per_process % cfg.accum_steps:
        raise ValueError(f"accumulo: {per_process} finestre per processo non divisibili in {cfg.accum_steps} micro-batch")
    if cfg.accum_steps > 1 and (cfg.pairing_audit or cfg.keep_grad_every):
        raise NotImplementedError("accumulo: pairing_audit e keep_grad_every non supportati")
    use_terms = ctx.active or cfg.accum_steps > 1  # il contratto delle loss; altrimenti il percorso di sempre, bit per bit
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
        previous = _config_with_defaults(resume_state["config"])
        if any(previous.get(k) != config_to_dict(cfg).get(k) for k in ("target_version", "readout_version", "keep_calibration_sha256", "model", "jepa",
                                                                       "branch_calibration_batches", "freeze_encoder")):
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
        if init_from is not None and not cfg.freeze_encoder:  # freeze_encoder: caricamento controllato chiave per chiave (_load_frozen_encoder)
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
    frozen_init = branch_calibration = None
    if cfg.freeze_encoder and resume_state is None:
        frozen_init = _load_frozen_encoder(student, cfg, init_from)
    if calibration is not None and resume_state is None:
        student.spectral_keep.fit_target_stats(calibration[0], calibration[1])
        for head in student.branch_heads.values() if student.branch_heads is not None else ():
            head.fit_target_stats(calibration[0], calibration[1])  # stessi target per tutte le teste
    if student.spectral_branches is not None and resume_state is None:
        branch_calibration = _calibrate_branches(student, cfg, index, scales, window_s, device)
    if cfg.freeze_encoder:
        from wearusfm.model.spectral_branches import freeze_raw
        freeze_raw(student)
    teacher = make_teacher(student).to(device)
    # i parametri che si allenano (senza rami, tutti: la lista di sempre)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        if init_from is not None and not cfg.freeze_encoder:  # pesi di un altro run (studente e teacher), ottimizzatore e passi da zero
            state = torch.load(init_from, map_location="cpu", weights_only=False)
            student.load_state_dict(state["student"])
            teacher.load_state_dict(state["teacher"])
            say(f"pesi iniziali da {init_from} (passo {state['step']} di quel run); ottimizzatore e passi da zero")
        if ctx.active:  # dropout del muscolo: un generatore diverso per processo (i dati sono gia' diversi)
            torch.manual_seed(cfg.seed * 1_000_003 + ctx.rank + 1)
    if student.branch_heads is not None:  # calibrazione dei target uguale per tutte le teste, anche in un checkpoint ripreso
        from wearusfm.model.spectral_branches import check_head_calibration
        check_head_calibration(student)
    model = None
    if ctx.active:
        model = _ddp_wrap(student, cfg, ctx, device)
        for t in teacher.state_dict().values():  # il teacher del processo 0 su tutti (lo studente lo allinea DDP)
            torch.distributed.broadcast(t, src=0)
    if cfg.pairing_audit:
        from wearusfm.training.pairing import state_sha256
        (out_dir / "pairing.json").write_text(json.dumps({
            "student_initial_sha256": state_sha256(student.state_dict()),
            "teacher_initial_sha256": state_sha256(teacher.state_dict()),
            "seed": cfg.seed, "num_workers": num_workers, "resume": False,
        }, indent=2))
    def payload(extra: dict) -> dict:
        return {"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "alarm_counts": (monitor.low_collapse, monitor.low_rank), "config": config_to_dict(cfg),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "stream": stream, **extra}

    init_sha = None
    if cfg.freeze_encoder and resume_state is None:  # un processo solo (controllato sopra)
        # step 2C: il controllo «rami non allenati» e' lo stesso modello al passo 0, rami calibrati e teste mai allenate
        init_ckpt = out_dir / "checkpoint_init.pt"
        torch.save(payload({}), init_ckpt.with_suffix(".tmp"))
        init_ckpt.with_suffix(".tmp").replace(init_ckpt)
        init_sha = sha256_file(init_ckpt)
    if ctx.is_main:
        (out_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
        if branch is not None:  # da dove parte il ramo (D14: pilot sulle epoche, continuazione a 2D)
            (out_dir / "branch.json").write_text(json.dumps(branch, indent=1))
        if frozen_init is not None or branch_calibration is not None:  # rami spettrali: encoder congelato e calibrazione (solo all'inizio)
            (out_dir / "spectral_branches.json").write_text(json.dumps({"frozen_init": frozen_init, "calibration": branch_calibration,
                                                                        "init_checkpoint_sha256": init_sha}, indent=1))
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
                                       generator=torch.Generator().manual_seed(cfg.seed),
                                       # con i worker, gli array grandi viaggiano in memoria condivisa (`_to_shared` / `_from_shared`)
                                       collate_fn=_to_shared if num_workers > 0 else None)
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
            torch.save(payload(extra), ckpt.with_suffix(".tmp"))
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
        batch = _from_shared(next(it)) if num_workers > 0 else next(it)
        t_data = time.time() - t0
        if batch.index != step:  # l'ordine dei worker del DataLoader deve ricomporre la sequenza degli indici
            raise RuntimeError(f"batch di indice {batch.index} al passo {step}: stream fuori sequenza")
        if cfg.pairing_audit:
            from wearusfm.training.pairing import batch_sha256
            batch_hash = batch_sha256(batch)  # CPU data BEFORE any conversion/device transfer
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        student.train()
        opt.zero_grad(set_to_none=True)
        if use_terms:  # DDP e/o accumulo: il contratto delle loss distribuite
            micros = [_prepare(mb, device) for mb in L.split_batch(batch, cfg.accum_steps)]
            failure, loss_values = _accumulate_terms(model, student, teacher, micros, cfg, ctx, device, amp)
            if failure is not None:  # deciso insieme da tutti i processi, prima del backward che l'avrebbe usato
                reason = f"{failure} al passo {step}"
                numerical_failure = True
                break
        else:  # un processo senza accumulo: il percorso di sempre
            inp, visible, rvq_on = _to_device(*L.to_model_inputs(batch), device)
            codes_t = torch.from_numpy(RC.pack_codes(batch.rvq_codes)).to(device) if batch.rvq_codes is not None else None
            with amp:
                if cfg.freeze_encoder:  # diagnostica dei rami: solo il termine dei rami sul readout, encoder congelato
                    from wearusfm.model.spectral_branches import readout_objectives
                    kb, kb_metrics = readout_objectives(student, inp, batch.anchor_targets, keep=False)["keep_branches"]
                    # col suo peso, come nel training congiunto (review GPT del 14/10: senza, branch_weight non controllava il backward)
                    losses = {"total": cfg.jepa.branch_weight * kb, "keep_branches": kb, **kb_metrics}
                else:
                    losses = jepa_losses(student, teacher, inp, visible, cfg.jepa, anchor_targets=batch.anchor_targets,
                                         kinds=torch.as_tensor(batch.kind).to(device), rvq_on=rvq_on if codes_t is not None else None,
                                         rvq_codes=(lambda c, w: codes_t[c, w]) if codes_t is not None else None)
            if not bool(torch.isfinite(losses["total"])):
                reason = f"perdita non finita al passo {step}"
                numerical_failure = True
                break
            if (cfg.jepa.keep_weight > 0 or cfg.jepa.branch_weight > 0) and cfg.keep_grad_every and step % cfg.keep_grad_every == 0:
                # Weighted gradient magnitudes on SHARED encoder params; loss sizes alone do not prove dominance. Con i rami il loro termine e'
                # una componente a se' (parere GPT d5c53c1: contributi dei rami separati, sui parametri raw condivisi)
                shared = [p for name, p in student.named_parameters() if p.requires_grad and
                          name.split(".")[0] in ("tokenizer", "identity", "local", "pool", "backbone")]
                components = {"jepa": losses["jepa"]}
                for part, weight in (("keep", cfg.jepa.keep_weight), ("keep_branches", cfg.jepa.branch_weight)):
                    if weight > 0:
                        components[part] = weight * losses[part]
                anchors = losses["total"]
                for value in list(components.values()):  # in ordine, come (totale - jepa) - keep dello Step 1
                    anchors = anchors - value
                components["anchors"] = anchors
                for name, component in components.items():
                    grads = torch.autograd.grad(component, shared, retain_graph=True, allow_unused=True)
                    squares = [g.detach().float().square().sum() for g in grads if g is not None]
                    losses["shared_grad_" + name] = torch.stack(squares).sum().sqrt() if squares else losses["total"].new_zeros(())
            losses["total"].backward()
            loss_values = {k: float(v.detach()) for k, v in losses.items()}
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)  # uguale su tutti i processi dopo la riduzione
        # Loss finita non implica gradienti finiti. Gestione esplicita per registrare lo STOP anche senza un'eccezione del clipping.
        if not torch.isfinite(gnorm):
            reason = f"gradienti non finiti al passo {step} (norma: {float(gnorm)})"
            numerical_failure = True
            opt.zero_grad(set_to_none=True)
            break  # niente optimizer, EMA, incremento del passo o metriche di un aggiornamento non eseguito
        opt.step()
        if cfg.freeze_encoder:  # EMA solo su rami e teste: il raw del teacher resta quello di init_from bit per bit (m a + (1 - m) a != a)
            for name in ("spectral_branches", "branch_heads"):
                ema_update(getattr(teacher, name), getattr(student, name), cfg.jepa.ema_momentum)
        else:
            ema_update(teacher, student, cfg.jepa.ema_momentum)
        step += 1
        rec = {"step": step, "lr": _lr_at(step - 1, cfg), "grad_norm": float(gnorm), "t_data_s": t_data, "t_step_s": time.time() - t0,
               "windows": len(batch.signals), "skipped_sessions": batch.skipped_sessions}
        if not ctx.active:
            rec.update(loss_values)  # un processo solo: il record di sempre, stesso ordine dei campi
        # origine e topologia presentata (D9 decisione 1): finestre per classe di quota e quante presentate come montaggio virtuale (D6a)
        rec["classes"] = {c: sum(r.get("quota_class") == c for r in batch.rows) for c in ("A", "B", "C")}
        rec["virtual"] = sum(p != "full" for p in batch.presented or [])
        if ctx.active:  # il record del passo dai record di tutti i processi, sul processo 0
            parts = [None] * ctx.world_size if ctx.is_main else None
            torch.distributed.gather_object(rec, parts, dst=0)
            rec = {**(_merge_records(parts) if ctx.is_main else rec), **loss_values}  # loss: gia' globali, uguali su ogni processo
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
            part = f"rami {rec['keep_branches']:.4f}" if cfg.freeze_encoder else f"jepa {rec['jepa']:.4f}"  # congelato: solo il termine dei rami
            say(f"passo {step}: totale {rec['total']:.4f}, {part}, {rec['t_step_s']:.2f} s/passo (dati {rec['t_data_s']:.2f} s){mem}")
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
        summary.update({"world_size": ctx.world_size, "batch_per_process": per_process})
    if use_terms:
        summary.update({"accum_steps": cfg.accum_steps, "micro_batch": per_process // cfg.accum_steps,
                        "loss_reduction": "contratto: numeratori locali / conteggi globali su processi e micro-batch"})
    if cfg.model.spectral_branches:
        prov = out_dir / "spectral_branches.json"
        summary["spectral_branches"] = {"kinds": list(cfg.model.spectral_branches), "freeze_encoder": cfg.freeze_encoder,
                                        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
                                        **(json.loads(prov.read_text()) if prov.exists() else {})}
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
