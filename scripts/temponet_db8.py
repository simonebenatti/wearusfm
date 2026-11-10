"""Replica di TEMPONet su NinaPro DB8 (Zanghieri et al., COINS 2021, [Z21]): training per soggetto e test sull'acquisizione 3.

Protocollo (`docs/temponet_db8_replication.md`, criteri in `docs/decisioni.md`): per ogni soggetto e seme tre modelli,
- `fold1`: training su A1, validazione su A2; `fold2`: training su A2, validazione su A1 (la «2-fold cross-validation» di [Z21] sez. II-D1 sulle
  sessioni 1 e 2 unite: i due fold sono le due sessioni, 10 ripetizioni per movimento ciascuna);
- `full`: training su A1 + A2, alpha dell'EMA dalle predizioni fuori fold dei due modelli precedenti (analisi secondaria).
Training: TEMPONet-64 (`wearusfm.model.temponet`), loss L1, Adam lr 1e-4 costante, batch 64, 19 epoche in float32 ([Z21] sez. II-D3; l'epoca
finale di quantization-aware training int8 non si replica), finestre di 256 campioni con passo 32 e bersaglio all'ultimo campione.
Post-processing: EMA causale con alpha per DoA scelto sulla griglia 0,00-0,99 con le sole predizioni di validazione; test su A3 con quell'alpha.
Nessuna selezione sul test: le epoche sono fisse, la curva di validazione per epoca e' solo descrittiva.
Proposta H5 (`docs/temponet_db8_h5_plan.md`): `--lr-schedule cosine` (W1) o `--final-epochs 1 --final-lr 1e-5` (W2). Solo con uno schedule il
learning rate e' un tensore, aggiornato prima di ogni passo e letto anche dal CUDA graph; col learning rate costante il percorso resta quello di
prima, a numero float.

Uso (GPU, `scripts/slurm/temponet_db8.sbatch`):
  python3 scripts/temponet_db8.py --cache <cache> --out <cartella NUOVA> --seeds 0 1 2 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from wearusfm.harness import db8_regression as db8

EMG_UNIT_SCALE = {"V": 1.0, "mV": 1e3, "uV": 1e6}
MODELS = ("fold1", "fold2", "full")
SPLITS = {"fold1": ((1,), 2), "fold2": ((2,), 1), "full": ((1, 2), None)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_cache(cache: Path, subject: int) -> dict:
    """sha256 dei file .npy del soggetto contro il manifest della cache; ritorna l'impronta del manifest."""
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "temponet_db8_cache_v1":
        raise ValueError(f"{manifest_path}: tipo di cache inatteso")
    if np.asarray(manifest["dof_to_doa"]).shape != db8.DOF_TO_DOA.shape or not np.allclose(manifest["dof_to_doa"], db8.DOF_TO_DOA):
        raise ValueError("matrice DoF->DoA della cache diversa da quella del codice")
    entries = [e for e in manifest["entries"] if e["subject"] == subject]
    if sorted(e["acquisition"] for e in entries) != list(db8.ACQUISITIONS):
        raise ValueError(f"cache incompleta per il soggetto {subject}")
    for e in entries:
        for name in ("emg", "doa"):
            out = e["outputs"][name]
            if sha256(cache / out["file"]) != out["sha256"]:
                raise ValueError(f"{out['file']}: sha256 diverso dal manifest")
    return {"manifest_sha256": sha256(manifest_path), "cache_code_sha": manifest.get("code_sha"),
            "n_samples": {str(e["acquisition"]): e["n_samples"] for e in entries}}


class Signal:
    """Registrazioni concatenate su dispositivo: emg (C, T) gia' scalato, DoA (T, 5); finestre per indice dell'ultimo campione."""

    def __init__(self, recordings, device, emg_scale: float, window: int, step: int):
        import torch

        self.window = window
        emg = np.concatenate([r.emg for r in recordings], axis=0)
        doa = np.concatenate([r.doa for r in recordings], axis=0)
        self.emg = torch.from_numpy(np.ascontiguousarray(emg.T)).to(device) * emg_scale
        self.doa = torch.from_numpy(doa).to(device)
        ends, offset = [], 0
        for r in recordings:  # una finestra finisce almeno window-1 campioni dopo l'inizio della sua registrazione: mai a cavallo di due
            ends.append(db8.window_ends(r.n_samples, window, step) + offset)
            offset += r.n_samples
        self.ends = torch.from_numpy(np.concatenate(ends)).to(device)
        self.offsets = torch.arange(-(window - 1), 1, device=device)

    def with_targets(self, doa):
        """Copia che condivide emg e finestre, con bersagli diversi (bersaglio standardizzato della variante V1)."""
        new = copy.copy(self)
        new.doa = doa
        return new

    def batch(self, ends):
        idx = ends[:, None] + self.offsets[None, :]
        return self.emg[:, idx].permute(1, 0, 2).contiguous(), self.doa[ends]


def predict(model, sig: Signal, batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(sig.ends), batch):
            x, y = sig.batch(sig.ends[i : i + batch])
            preds.append(model(x).double().cpu().numpy())
            targets.append(y.double().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


CUDNN_MODES = {"deterministic": (True, False), "default": (False, False), "benchmark": (False, True)}  # (deterministic, benchmark)
GRAPH_TOLERANCE = 1e-5  # differenza massima ammessa fra parametri eager e graph nel controllo
GRAPH_WARMUP = 3  # passi veri (contati nell'epoca) eseguiti in eager su uno stream laterale prima della cattura
LR_SCHEDULES = ("constant", "cosine")


class LRSchedule:
    """Learning rate del passo `step` (da 0) dell'epoca `epoch` (da 0); proposta H5, `docs/temponet_db8_h5_plan.md`.

    Le prime `epochs` epoche seguono `kind`: `constant` a `lr`, oppure `cosine` da `lr` a 0 aggiornato a ogni passo,
    lr * (1 + cos(pi * t / T)) / 2 con t = passi gia' fatti e T = epochs * steps_per_epoch (primo passo lr, ultimo ~0).
    Le `final_epochs` epoche successive sono a `final_lr` costante (W2: sostituto in fp32 dell'epoca finale di QAT).
    """

    def __init__(self, kind: str, lr: float, epochs: int, steps_per_epoch: int, final_epochs: int = 0, final_lr: float | None = None):
        if kind not in LR_SCHEDULES:
            raise ValueError(f"schedule sconosciuto: {kind}")
        if final_epochs and final_lr is None:
            raise ValueError("le epoche finali richiedono il loro learning rate")
        if epochs < 1 or steps_per_epoch < 1 or final_epochs < 0:
            raise ValueError("epoche e passi per epoca devono essere positivi")
        self.kind, self.lr, self.epochs, self.steps_per_epoch = kind, lr, epochs, steps_per_epoch
        self.final_epochs, self.final_lr = final_epochs, final_lr

    @property
    def total_epochs(self) -> int:
        return self.epochs + self.final_epochs

    def value(self, epoch: int, step: int) -> float:
        if epoch >= self.epochs:
            return self.final_lr
        if self.kind == "constant":
            return self.lr
        t = epoch * self.steps_per_epoch + step
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * t / (self.epochs * self.steps_per_epoch)))


def schedule_active(cfg: argparse.Namespace) -> bool:
    """Vero con uno schedule non costante o con epoche finali: solo allora il learning rate e' un tensore."""
    return getattr(cfg, "lr_schedule", "constant") != "constant" or getattr(cfg, "final_epochs", 0) > 0


def make_schedule(cfg: argparse.Namespace, steps_per_epoch: int) -> LRSchedule | None:
    """Schedule del training (None col learning rate costante: percorso a numero float, invariato)."""
    if not schedule_active(cfg):
        return None
    return LRSchedule(cfg.lr_schedule, cfg.lr, cfg.epochs, steps_per_epoch, cfg.final_epochs, cfg.final_lr)


def check_lrs(cfg: argparse.Namespace, steps: int) -> list[float] | None:
    """Learning rate dei passi del controllo eager/graph: lo schedule della variante compresso in `steps` passi (None se costante).

    Con le epoche finali, meta' dei passi segue lo schedule principale e meta' sta a `final_lr`: il cambio avviene dopo la cattura del graph.
    """
    if not schedule_active(cfg):
        return None
    final = 1 if cfg.final_epochs else 0
    spe = steps // (1 + final)
    sched = LRSchedule(cfg.lr_schedule, cfg.lr, 1, spe, final, cfg.final_lr)
    return [sched.value(i // spe, i % spe) for i in range(steps)]


class Stepper:
    """Un passo di training (finestre -> L1 -> backward -> Adam) in eager o catturato in un CUDA graph.

    Con il graph, i primi GRAPH_WARMUP passi della prima epoca sono passi veri in eager su uno stream laterale (ricetta PyTorch «whole network
    capture»); la cattura registra senza eseguire, poi ogni passo e' copia degli indici + replay. Gradienti messi a None prima della cattura: il
    backward catturato li scrive, non li accumula. Dropout: RNG compatibile con i graph (offset che avanza a ogni replay). Il numero di passi
    per epoca resta identico all'eager.
    Con uno schedule (H5) l'ottimizzatore ha un learning rate tensore: `step(ends, lr)` lo riscrive sul posto prima del passo, sullo stesso
    stream, e il replay lo legge dalla memoria (stesso schema degli indici `ends`).
    """

    def __init__(self, model, opt, loss_fn, sig: Signal, batch: int, use_graph: bool, device):
        import torch

        self.model, self.opt, self.loss_fn, self.sig, self.use_graph = model, opt, loss_fn, sig, use_graph
        self.ends = torch.zeros(batch, dtype=torch.long, device=device)
        self.total = torch.zeros((), dtype=torch.float64, device=device)
        lr = opt.param_groups[0]["lr"]
        self.lr_t = lr if torch.is_tensor(lr) else None
        self.graph = None
        self.side = torch.cuda.Stream(device) if use_graph else None
        self.done = 0

    def _eager(self):
        x, y = self.sig.batch(self.ends)
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()
        self.opt.step()
        self.total.add_(loss.detach())

    def step(self, ends, lr: float | None = None):
        import torch

        self.ends.copy_(ends)
        if lr is not None:
            if self.lr_t is None:
                raise ValueError("learning rate per passo richiesto con un ottimizzatore a learning rate float")
            self.lr_t.fill_(lr)
        if not self.use_graph:
            self._eager()
        elif self.graph is None and self.done < GRAPH_WARMUP:
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self._eager()
            torch.cuda.current_stream().wait_stream(self.side)
        else:
            if self.graph is None:
                self.opt.zero_grad(set_to_none=True)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    x, y = self.sig.batch(self.ends)
                    loss = self.loss_fn(self.model(x), y)
                    loss.backward()
                    self.opt.step()
                    self.total.add_(loss.detach())
            self.graph.replay()
        self.done += 1

    def take_total(self) -> float:
        value = float(self.total)
        self.total.zero_()
        return value


def build(cfg: argparse.Namespace, seed: int, device, dropout: float = 0.5):
    import torch
    from torch import nn

    from wearusfm.model.temponet import TEMPONet

    torch.manual_seed(seed)
    model = TEMPONet(in_channels=db8.N_EMG, window=cfg.window, out_dim=db8.N_DOA, causal=not cfg.symmetric_padding,
                     dropout=dropout).to(device)
    capturable = device.type == "cuda"  # stesso Adam in eager e nel graph
    lr = cfg.lr
    if schedule_active(cfg):  # H5: tensore riscritto prima di ogni passo; tipo dell'aritmetica che Adam usa con un numero float
        # (su GPU, ramo capturable foreach: il numero diventa float32; su CPU, ramo a tensore singolo: aritmetica Python in float64)
        lr = torch.tensor(cfg.lr, dtype=torch.float32 if capturable else torch.float64, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, capturable=capturable)
    return model, opt, nn.L1Loss()


def graph_check(sig: Signal, cfg: argparse.Namespace, device, steps: int = 60) -> dict:
    """Su GPU: stessi pesi iniziali, stessi batch, dropout 0; `steps` passi eager contro graph; differenza massima dei parametri e tempi.

    Con uno schedule (H5) il learning rate cambia durante i passi come nella variante (`check_lrs`): un graph che avesse fissato il valore
    alla cattura si allontanerebbe dall'eager ben oltre la tolleranza.
    """
    import torch

    gen = torch.Generator().manual_seed(12345)
    batches = sig.ends[torch.randperm(len(sig.ends), generator=gen)[: steps * cfg.batch].to(device)].view(steps, cfg.batch)
    lrs = check_lrs(cfg, steps) or [None] * steps
    saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # equivalenza esatta: algoritmi deterministici
    out = {}
    params = {}
    for mode in ("eager", "graph"):
        model, opt, loss_fn = build(cfg, 777, device, dropout=0.0)
        model.train()
        st = Stepper(model, opt, loss_fn, sig, cfg.batch, mode == "graph", device)
        for i in range(GRAPH_WARMUP + 1):
            st.step(batches[i], lrs[i])
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(GRAPH_WARMUP + 1, steps):
            st.step(batches[i], lrs[i])
        torch.cuda.synchronize()
        out[f"{mode}_ms_per_step"] = 1000 * (time.time() - t0) / (steps - GRAPH_WARMUP - 1)
        out[f"{mode}_loss_sum"] = st.take_total()
        params[mode] = [p.detach().clone() for p in model.parameters()] + [b.detach().clone() for b in model.buffers()]
        if lrs[0] is not None:
            out[f"{mode}_lr_after"] = float(opt.param_groups[0]["lr"])  # ultimo valore scritto nel tensore
    diffs = [float((a.double() - b.double()).abs().max()) for a, b in zip(params["eager"], params["graph"])]
    out["max_abs_param_diff"] = max(diffs)
    out["steps"] = steps
    if lrs[0] is not None:
        out["lr_schedule"] = {"first": lrs[0], "last": lrs[-1], "distinct": len(set(lrs))}
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved
    return out


def denormalizer(net, mu, sd):
    """Rete che restituisce gradi da una rete allenata sul bersaglio standardizzato: uscita * sd + mu (buffer salvati col modello)."""
    from torch import nn

    class Denorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mu", mu.clone())
            self.register_buffer("sd", sd.clone())

        def forward(self, x):
            return self.net(x) * self.sd + self.mu

    return Denorm()


def train(sig: Signal, val: Signal | None, cfg: argparse.Namespace, seed: int, device):
    import torch

    model, opt, loss_fn = build(cfg, seed, device)
    eval_model = model
    if getattr(cfg, "target_norm", False):  # V1: L1 sul bersaglio standardizzato con le statistiche delle finestre di training
        y = sig.doa[sig.ends].double()
        mu, sd = y.mean(dim=0).float(), y.std(dim=0).clamp_min(1e-6).float()
        sig = sig.with_targets((sig.doa - mu) / sd)
        eval_model = denormalizer(model, mu, sd)
    use_graph = device.type == "cuda" and not cfg.no_cuda_graph
    stepper = Stepper(model, opt, loss_fn, sig, cfg.batch, use_graph, device)
    gen = torch.Generator().manual_seed(seed)
    n = len(sig.ends)
    steps = n // cfg.batch
    if cfg.max_steps_per_epoch:
        steps = min(steps, cfg.max_steps_per_epoch)
    sched = make_schedule(cfg, steps)  # None col learning rate costante: percorso di prima
    history = []
    for epoch in range(cfg.epochs if sched is None else sched.total_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        batches = sig.ends[perm[: steps * cfg.batch]].view(steps, cfg.batch)
        if sched is None:
            for i in range(steps):
                stepper.step(batches[i])
        else:
            for i in range(steps):
                stepper.step(batches[i], sched.value(epoch, i))
        train_loss = stepper.take_total() / steps
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"loss non finita all'epoca {epoch + 1}")
        row = {"epoch": epoch + 1, "train_l1_deg": train_loss, "steps": steps, "seconds": round(time.time() - t0, 2)}
        if sched is not None:
            row["lr_first"], row["lr_last"] = sched.value(epoch, 0), sched.value(epoch, steps - 1)
        if val is not None:
            vp, vt = predict(eval_model, val)
            row["val_mae_raw_deg"] = float(db8.mae_per_doa(vp, vt).mean())
        history.append(row)
        if cfg.log_epochs:
            print(f"  seme {seed} epoca {epoch + 1}: train L1 {train_loss:.3f}"
                  + (f", val {row['val_mae_raw_deg']:.3f}" if "val_mae_raw_deg" in row else "") + f", {row['seconds']:.1f} s", flush=True)
    return eval_model, history


def run_unit(subject: int, seed: int, cfg: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = CUDNN_MODES[cfg.cudnn_mode]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    t0 = time.time()
    cache = Path(cfg.cache)
    provenance = check_cache(cache, subject)
    recs = {a: db8.load_cached(cache, subject, a) for a in db8.ACQUISITIONS}
    scale = EMG_UNIT_SCALE[cfg.emg_unit]
    sig = {a: Signal([recs[a]], device, scale, cfg.window, cfg.step) for a in db8.ACQUISITIONS}  # validazione e test: passo 16 ms
    train_step = cfg.train_step or cfg.step
    tsig = sig if train_step == cfg.step else {a: Signal([recs[a]], device, scale, cfg.window, train_step) for a in (1, 2)}
    out_dir = Path(cfg.out)
    result = {"subject": subject, "seed": seed, "provenance": provenance, "models": {}}
    if cfg.check_graph and device.type == "cuda" and subject == cfg.subjects[0] and seed == cfg.seeds[0]:
        check = graph_check(sig[1], cfg, device)
        result["graph_check"] = check
        sched = check.get("lr_schedule")
        print(f"controllo eager/graph: differenza massima {check['max_abs_param_diff']:.3g}, "
              f"{check['eager_ms_per_step']:.2f} ms contro {check['graph_ms_per_step']:.2f} ms per passo"
              + (f"; learning rate da {sched['first']:.3g} a {sched['last']:.3g} ({sched['distinct']} valori)" if sched else ""), flush=True)
        if check["max_abs_param_diff"] > GRAPH_TOLERANCE:
            raise RuntimeError(f"graph diverso dall'eager: {check['max_abs_param_diff']}")
    preds = {}
    for k, name in enumerate(MODELS):  # k fissa il seme del modello anche quando se ne allena solo una parte
        if name not in cfg.models or (name == "full" and cfg.full_seeds is not None and seed not in cfg.full_seeds):
            continue
        train_acqs, val_acq = SPLITS[name]
        tr = tsig[train_acqs[0]] if len(train_acqs) == 1 else Signal([recs[a] for a in train_acqs], device, scale, cfg.window, train_step)
        model_seed = 1_000_000 * seed + 1_000 * subject + k
        tm = time.time()
        model, history = train(tr, sig[val_acq] if val_acq else None, cfg, model_seed, device)
        test_pred, test_tgt = predict(model, sig[db8.TEST_ACQUISITION])
        entry = {"train_acquisitions": list(train_acqs), "val_acquisition": val_acq, "torch_seed": model_seed,
                 "n_train_windows": int(len(tr.ends)), "history": history, "train_seconds": round(time.time() - tm, 1)}
        if val_acq is not None:
            val_pred, val_tgt = predict(model, sig[val_acq])
            preds[name] = (val_pred, val_tgt)
            segments = [(val_pred, val_tgt)]
        else:  # full: alpha dalle predizioni fuori fold di fold1 (su A2) e fold2 (su A1)
            segments = [preds["fold1"], preds["fold2"]]
        alpha, _ = db8.tune_alpha(segments)
        entry["alpha"] = [float(a) for a in alpha]
        if val_acq is not None:
            entry["val_raw"] = db8.evaluate(val_pred, val_tgt)
            entry["val_ema"] = db8.evaluate(db8.ema(val_pred, alpha), val_tgt)
        entry["test_raw"] = db8.evaluate(test_pred, test_tgt)
        entry["test_ema"] = db8.evaluate(db8.ema(test_pred, alpha), test_tgt)
        result["models"][name] = entry
        (out_dir / "units" / f"S{subject}_seed{seed}.partial.json").write_text(json.dumps(result, indent=1))  # avanzamento
        torch.save(model.state_dict(), out_dir / "models" / f"S{subject}_seed{seed}_{name}.pt")
        np.save(out_dir / "predictions" / f"S{subject}_seed{seed}_{name}_test_raw.npy", test_pred.astype(np.float32))
        target_path = out_dir / "predictions" / f"S{subject}_test_target.npy"
        if not target_path.exists():
            np.save(target_path, test_tgt.astype(np.float32))
        if cfg.hide_test_in_log:  # screening del passo 2 della diagnosi: il test non si guarda prima del passo 3
            shown = f"validazione MAE {entry['val_ema']['mae_deg']:.3f}" if "val_ema" in entry else "test nascosto"
        else:
            shown = f"test MAE {entry['test_ema']['mae_deg']:.3f} (grezzo {entry['test_raw']['mae_deg']:.3f})"
        print(f"S{subject} seme {seed} {name}: {shown}, alpha medio {np.mean(alpha):.3f}, {entry['train_seconds']:.0f} s", flush=True)
    result["seconds"] = round(time.time() - t0, 1)
    result["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return result


def worker(job: tuple[int, int, dict]) -> tuple[int, int, str]:
    subject, seed, cfg_dict = job
    cfg = argparse.Namespace(**cfg_dict)
    dst = Path(cfg.out) / "units" / f"S{subject}_seed{seed}.json"
    if time.time() > cfg.deadline_epoch:
        dst.with_suffix(".skipped").write_text("tempo esaurito prima dell'avvio\n")
        return subject, seed, "saltato"
    try:
        result = run_unit(subject, seed, cfg)
        dst.write_text(json.dumps(result, indent=1))
        dst.with_suffix(".partial.json").unlink(missing_ok=True)
        return subject, seed, "ok"
    except Exception:  # registrato e riportato: il job esce con errore, nessun nuovo tentativo
        dst.with_suffix(".error").write_text(traceback.format_exc())
        return subject, seed, "errore"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="cartella nuova o vuota")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, db8.N_SUBJECTS + 1)))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=19)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=db8.WINDOW)
    ap.add_argument("--step", type=int, default=db8.STEP, help="passo delle finestre di validazione e test (16 ms)")
    ap.add_argument("--train-step", type=int, default=0, help="passo delle finestre di training (0 = come --step)")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS), help="modelli da allenare (full richiede fold1 e fold2)")
    ap.add_argument("--full-seeds", type=int, nargs="*", default=None, help="semi per cui allenare anche full (predefinito: tutti)")
    ap.add_argument("--emg-unit", choices=sorted(EMG_UNIT_SCALE), default="uV",
                    help="unita' dell'ingresso: cambio di scala costante, non una normalizzazione (vedi il documento)")
    ap.add_argument("--symmetric-padding", action="store_true", help="padding simmetrico come Q-PPG invece che causale")
    ap.add_argument("--target-norm", action="store_true", help="V1: L1 sul bersaglio standardizzato per DoA (statistiche di training)")
    ap.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                    help="H5: constant (come prima) o cosine (da --lr a 0 sulle --epochs epoche, aggiornato a ogni passo; W1)")
    ap.add_argument("--final-epochs", type=int, default=0, help="H5: epoche in piu' a --final-lr dopo le --epochs (W2: 1)")
    ap.add_argument("--final-lr", type=float, default=None, help="H5: learning rate delle epoche finali (W2: 1e-5)")
    ap.add_argument("--hide-test-in-log", action="store_true", help="nel log la validazione al posto del test (screening)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="solo collaudo: tetto ai passi per epoca")
    ap.add_argument("--workers", type=int, default=1, help="processi in parallelo; sulla stessa GPU senza MPS si alternano (time-slicing)")
    ap.add_argument("--no-cuda-graph", action="store_true", help="passo di training in eager anche su GPU")
    ap.add_argument("--cudnn-mode", choices=sorted(CUDNN_MODES), default="deterministic",
                    help="algoritmi cuDNN: deterministic (riproducibile), default o benchmark (scelta per misura, non deterministica)")
    ap.add_argument("--log-epochs", action="store_true", help="una riga per epoca (loss di training e MAE di validazione)")
    ap.add_argument("--check-graph", action="store_true", help="prima della prima unita': equivalenza eager/graph e tempi per passo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="nessuna unita' nuova dopo questo tempo (0 = nessun limite)")
    args = ap.parse_args(argv)
    if "full" in args.models and not {"fold1", "fold2"} <= set(args.models):
        ap.error("full richiede fold1 e fold2 (alpha dalle predizioni fuori fold)")
    if args.final_epochs < 0 or (args.final_epochs > 0) != (args.final_lr is not None):
        ap.error("--final-epochs (positivo) e --final-lr vanno insieme")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} non e' vuota: serve una cartella nuova", file=sys.stderr)
        return 2
    for sub in ("units", "models", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = vars(args) | {"deadline_epoch": (t0 + args.time_budget_s) if args.time_budget_s > 0 else float("inf")}
    import torch

    run = {"kind": "temponet_db8_run_v1", "args": vars(args), "code_dir": os.environ.get("WEARUSFM_CODE_DIR"),
           "code_sha": os.environ.get("WEARUSFM_CODE_SHA"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
           "cuda": torch.version.cuda, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    from wearusfm.model.temponet import TEMPONet, count_macs, count_parameters

    m = TEMPONet(in_channels=db8.N_EMG, window=args.window, out_dim=db8.N_DOA)
    run["model"] = {"parameters": count_parameters(m), "macs": count_macs(m), "out_length": m.out_length}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    jobs = [(s, seed, cfg) for seed in args.seeds for s in args.subjects]
    status = []
    if args.workers <= 1:
        status = [worker(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(worker, jobs):
                print(f"unita' S{res[0]} seme {res[1]}: {res[2]}", flush=True)
                status.append(res)
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run["seconds"] = round(time.time() - t0, 1)
    run["status"] = {f"S{s}_seed{seed}": st for s, seed, st in sorted(status)}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    bad = [k for k, v in run["status"].items() if v != "ok"]
    print(f"fine: {len(status) - len(bad)} unita' riuscite su {len(status)} in {run['seconds']:.0f} s" + (f"; non riuscite: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
