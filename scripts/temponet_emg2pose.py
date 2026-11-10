"""TEMPONet su emg2pose, regressione della posa (`docs/temponet_emg2pose_plan.md`; regole ed emendamento in `docs/decisioni.md`, 10/11/2026).

Due modi.
- `train` (passi 1 e 2): un solo modello sui 158 utenti dello split di training, entrambi i lati.
  - Rete: TEMPONet di [Z21] in V13, cioe' bersaglio standardizzato con `pose_stats.json` e padding simmetrico, a 20 uscite.
  - Ricetta: L1, Adam 1e-4 costante, batch 64, 19 epoche, seme 0; cuDNN benchmark; passo catturato in un CUDA graph, con controllo eager/graph
    prima di allenare.
  - Ingresso: EMG processato in unita' native per 10^k, con k dal manifest del passo 0.
  - Finestre: finiscono all'ultimo campione dei blocchi validi a 200 Hz, ogni 8 blocchi; si raccolgono dalla RAM in un thread, a blocchi di
    passi, in memoria bloccata.
  - Validazione grezza per epoca, solo descrittiva: istanti a 50 Hz sui blocchi validi. Si usa il modello finale.
- `eval` (passo 3): finestre e bersagli ufficiali del passo 0; alpha dell'EMA per giunto sulla validazione; cancello sulla validazione; poi, solo
  se passa, il test, una volta. Con `--val-only` (collaudo) il test non si calcola.

Uso: `scripts/slurm/temponet_emg2pose.sbatch` (GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temponet_db8 as T  # noqa: E402  (CUDNN_MODES, GRAPH_TOLERANCE, GRAPH_WARMUP, denormalizer)

from wearusfm.harness import emg2pose_regression as R  # noqa: E402

EVAL_BATCH = 8192


def max_rss_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1e6 if sys.platform.startswith("linux") else kb / 1e9  # Linux: kB; macOS: byte


def new_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"{p} non e' vuota: serve una cartella nuova")
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- dati in RAM ------------------------------------------------------------------------------------------------------------------------

class HostStore:
    """EMG di piu' registrazioni concatenato in RAM: X (T_tot, 16) int16 (torch); offset e fattore (10^k / int16_scale) per registrazione."""

    def __init__(self, rows: list[dict], processed_root: str | Path, k: int):
        import torch

        self.rows = rows
        metas = []
        for r in rows:
            u, s = R.recording_key(r)
            d = Path(processed_root) / u / s
            meta = json.loads((d / "metadata.json").read_text())
            metas.append((d, int(meta["shape"][0]), float(meta["int16_scale"])))
        self.dirs = [m[0] for m in metas]
        self.n = np.array([m[1] for m in metas], dtype=np.int64)
        self.offset = np.zeros(len(metas), dtype=np.int64)
        if len(metas) > 1:
            self.offset[1:] = np.cumsum(self.n)[:-1]
        self.factor = torch.tensor([10.0 ** k / m[2] for m in metas], dtype=torch.float32)
        t0 = time.time()
        self.X = torch.empty((int(self.n.sum()), R.N_CHANNELS), dtype=torch.int16)
        xn = self.X.numpy()
        for i, (d, n, _) in enumerate(metas):
            arr = np.load(d / "data_int16.npy", mmap_mode="r")
            if arr.shape != (n, R.N_CHANNELS) or arr.dtype != np.int16:
                raise ValueError(f"{d}: {arr.shape} {arr.dtype}, attesi ({n}, {R.N_CHANNELS}) int16")
            xn[self.offset[i] : self.offset[i] + n] = arr
        self.load_seconds = round(time.time() - t0, 1)
        self.win = torch.arange(-(R.WINDOW - 1), 1, dtype=torch.int64)

    @property
    def gib(self) -> float:
        return self.X.numel() * 2 / 2**30

    def gather(self, ends, out=None):
        """(B,) indici globali dell'ultimo campione -> (B, 256, 16) int16 (in `out`, se dato)."""
        import torch

        idx = (ends[:, None] + self.win[None, :]).reshape(-1)
        if out is None:
            return torch.index_select(self.X, 0, idx).view(len(ends), R.WINDOW, R.N_CHANNELS)
        torch.index_select(self.X, 0, idx, out=out.view(-1, R.N_CHANNELS))
        return out


def block_index(store: HostStore, pose_root: str | Path, stride_blocks: int):
    """Finestre sui blocchi validi a 200 Hz (regola di `R.train_blocks`): fine globale (N,), fattore (N,), bersaglio (N, 20) float16 in radianti."""
    import torch

    ends, rid, ys = [], [], []
    for i, d in enumerate(store.dirs):
        p = Path(pose_root) / d.parent.name / d.name
        blocks = np.load(p / "pose_200hz_f16.npy", mmap_mode="r")
        valid = np.load(p / "pose_valid.npy")
        if blocks.shape != (int(store.n[i]) // R.BLOCK, R.N_JOINTS) or valid.shape != (blocks.shape[0],):
            raise ValueError(f"{p}: blocchi {blocks.shape} per {store.n[i]} campioni")
        j = R.train_blocks(valid, int(store.n[i]), stride_blocks)
        ends.append(store.offset[i] + R.block_end(j))
        rid.append(np.full(j.size, i, dtype=np.int64))
        ys.append(np.asarray(blocks[j], dtype=np.float16))
    ends_t = torch.from_numpy(np.concatenate(ends))
    rid_t = torch.from_numpy(np.concatenate(rid))
    return ends_t, store.factor[rid_t], torch.from_numpy(np.concatenate(ys))


class Feeder:
    """Blocchi di `k` passi raccolti da un thread in due buffer (in memoria bloccata su GPU); il thread principale li copia sulla GPU e
    rilascia il buffer quando la copia e' finita."""

    def __init__(self, store: HostStore, ends, fw, ys, batch: int, k: int, pin: bool):
        import torch

        self.store, self.ends, self.fw, self.ys, self.batch, self.k = store, ends, fw, ys, batch, k
        n = batch * k

        def buf(shape, dtype):
            t = torch.empty(shape, dtype=dtype)
            return t.pin_memory() if pin else t

        self.bx = [buf((n, R.WINDOW, R.N_CHANNELS), torch.int16) for _ in range(2)]
        self.bf = [buf((n,), torch.float32) for _ in range(2)]
        self.by = [buf((n, R.N_JOINTS), torch.float16) for _ in range(2)]
        self.free: queue.Queue = queue.Queue()
        self.ready: queue.Queue = queue.Queue()
        for b in (0, 1):
            self.free.put(b)
        self.gather_seconds = 0.0

    def start(self, order, steps: int) -> None:
        """order: indici delle finestre dell'epoca nell'ordine dei passi (almeno steps * batch)."""
        def work():
            try:
                for c0 in range(0, steps, self.k):
                    c = min(self.k, steps - c0)
                    sel = order[c0 * self.batch : (c0 + c) * self.batch]
                    b = self.free.get()
                    t = time.time()
                    m = sel.numel()
                    self.store.gather(self.ends[sel], out=self.bx[b][:m])
                    self.bf[b][:m].copy_(self.fw[sel])
                    self.by[b][:m].copy_(self.ys[sel])
                    self.gather_seconds += time.time() - t
                    self.ready.put((b, c))
                self.ready.put(None)
            except BaseException as e:  # riportata al thread principale
                self.ready.put(e)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def chunks(self):
        while True:
            item = self.ready.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def release(self, b: int) -> None:
        self.free.put(b)


# --- modello e passo --------------------------------------------------------------------------------------------------------------------

def build(seed: int, device, lr: float, dropout: float = 0.5):
    import torch
    from torch import nn

    from wearusfm.model.temponet import TEMPONet

    torch.manual_seed(seed)
    model = TEMPONet(in_channels=R.N_CHANNELS, window=R.WINDOW, out_dim=R.N_JOINTS, causal=False, dropout=dropout).to(device)  # V13: simmetrico
    opt = torch.optim.Adam(model.parameters(), lr=lr, capturable=device.type == "cuda")
    return model, opt, nn.L1Loss()


class Stepper:
    """Passo di training su buffer statici (finestre int16, fattori, bersagli in radianti), in eager o in un CUDA graph (schema di
    `temponet_db8.Stepper`: GRAPH_WARMUP passi veri in eager su uno stream laterale, poi cattura e replay). Ingresso = int16 x fattore,
    bersaglio standardizzato con media e deviazione per giunto."""

    def __init__(self, model, opt, loss_fn, mu, sd, batch: int, use_graph: bool, device):
        import torch

        self.model, self.opt, self.loss_fn, self.use_graph = model, opt, loss_fn, use_graph
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(sd, dtype=torch.float32, device=device)
        self.xb = torch.zeros((batch, R.WINDOW, R.N_CHANNELS), dtype=torch.int16, device=device)
        self.fb = torch.ones(batch, dtype=torch.float32, device=device)
        self.yb = torch.zeros((batch, R.N_JOINTS), dtype=torch.float16, device=device)
        self.total = torch.zeros((), dtype=torch.float64, device=device)
        self.graph = None
        self.side = torch.cuda.Stream(device) if use_graph else None
        self.done = 0

    def _loss(self):
        x = self.xb.permute(0, 2, 1).contiguous().float() * self.fb[:, None, None]
        y = (self.yb.float() - self.mu) / self.sd
        return self.loss_fn(self.model(x), y)

    def _eager(self):
        self.opt.zero_grad(set_to_none=True)
        loss = self._loss()
        loss.backward()
        self.opt.step()
        self.total.add_(loss.detach())

    def step(self, x, f, y) -> None:
        import torch

        self.xb.copy_(x)
        self.fb.copy_(f)
        self.yb.copy_(y)
        if not self.use_graph:
            self._eager()
        elif self.graph is None and self.done < T.GRAPH_WARMUP:
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self._eager()
            torch.cuda.current_stream().wait_stream(self.side)
        else:
            if self.graph is None:
                self.opt.zero_grad(set_to_none=True)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    loss = self._loss()
                    loss.backward()
                    self.opt.step()
                    self.total.add_(loss.detach())
            self.graph.replay()
        self.done += 1

    def take_total(self) -> float:
        v = float(self.total)
        self.total.zero_()
        return v


def graph_check(xs, fs, ys, mu, sd, batch: int, device, lr: float) -> dict:
    """Stessi pesi iniziali (seme 777), stessi batch, dropout 0, cuDNN deterministico: len(xs) passi eager contro graph."""
    import torch

    saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    out, params = {"steps": len(xs)}, {}
    for mode in ("eager", "graph"):
        model, opt, loss_fn = build(777, device, lr, dropout=0.0)
        model.train()
        st = Stepper(model, opt, loss_fn, mu, sd, batch, mode == "graph", device)
        for i in range(T.GRAPH_WARMUP + 1):
            st.step(xs[i], fs[i], ys[i])
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(T.GRAPH_WARMUP + 1, len(xs)):
            st.step(xs[i], fs[i], ys[i])
        torch.cuda.synchronize()
        out[f"{mode}_ms_per_step"] = 1000 * (time.time() - t0) / (len(xs) - T.GRAPH_WARMUP - 1)
        out[f"{mode}_loss_sum"] = st.take_total()
        params[mode] = [p.detach().clone() for p in model.parameters()] + [b.detach().clone() for b in model.buffers()]
    out["max_abs_param_diff"] = max(float((a.double() - b.double()).abs().max()) for a, b in zip(params["eager"], params["graph"]))
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved
    return out


def predict(model, store: HostStore, ends, fw, device, batch: int = EVAL_BATCH) -> np.ndarray:
    """Uscite del modello (in radianti, se `model` e' il denormalizzatore) per le finestre che finiscono in `ends` (globali)."""
    import torch

    model.eval()
    out = np.empty((len(ends), R.N_JOINTS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            e = ends[i : i + batch]
            x = store.gather(e).to(device)
            x = x.permute(0, 2, 1).contiguous().float() * fw[i : i + batch].to(device)[:, None, None]
            out[i : i + len(e)] = model(x).float().cpu().numpy()
    return out


def setup_torch(cudnn_mode: str):
    import torch

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = T.CUDNN_MODES[cudnn_mode]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) - 1))
    return torch


def subset(rows: list[dict], n: int, seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["filename"])
    if n and n < len(rows):
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), size=n, replace=False))
        rows = [rows[i] for i in pick]
    return rows


# --- training ---------------------------------------------------------------------------------------------------------------------------

def run_train(args) -> int:
    torch = setup_torch(args.cudnn_mode)
    device = torch.device(args.device)
    out = new_dir(args.out)
    (out / "checkpoints").mkdir()
    t_start = time.time()
    manifest = json.loads(Path(args.targets_manifest).read_text())
    if not manifest.get("complete"):
        raise SystemExit("manifest del passo 0 incompleto")
    k = int(manifest["scale"]["exponent_k"])
    rows = R.read_rows(args.csv)
    train_rows = subset([r for r in rows if r["split"] == "train"], args.max_train_recordings, args.seed)
    val_rows = subset([r for r in rows if r["split"] == "val"], args.max_val_recordings, args.seed)
    mu, sd = R.pose_stats_all(args.pose_root)
    run = {"kind": "temponet_emg2pose_train_v1", "args": vars(args), "code_dir": os.environ.get("WEARUSFM_CODE_DIR"),
           "code_sha": os.environ.get("WEARUSFM_CODE_SHA"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scale_exponent_k": k,
           "targets_manifest": str(args.targets_manifest), "n_train_recordings": len(train_rows), "n_val_recordings": len(val_rows),
           "train_users": len({r["user"] for r in train_rows}), "status": "in corso"}
    from wearusfm.model.temponet import TEMPONet, count_parameters

    run["parameters"] = count_parameters(TEMPONet(in_channels=R.N_CHANNELS, window=R.WINDOW, out_dim=R.N_JOINTS, causal=False))
    (out / "run.json").write_text(json.dumps(run, indent=1))
    store = HostStore(train_rows, args.processed_root, k)
    ends, fw, ys = block_index(store, args.pose_root, args.train_stride_blocks)
    vstore = HostStore(val_rows, args.processed_root, k)
    vends, vfw, vys = block_index(vstore, args.pose_root, R.VAL_EPOCH_STRIDE_BLOCKS)
    n = len(ends)
    steps = n // args.batch
    if args.max_steps_per_epoch:
        steps = min(steps, args.max_steps_per_epoch)
    run.update({"train_gib": round(store.gib, 2), "train_load_seconds": store.load_seconds, "val_gib": round(vstore.gib, 2),
                "val_load_seconds": vstore.load_seconds, "n_train_windows": n, "steps_per_epoch": steps, "n_val_windows_epoch": len(vends),
                "max_rss_gb_after_load": round(max_rss_gb(), 2)})
    print(f"training: {len(train_rows)} registrazioni, {store.gib:.1f} GiB in {store.load_seconds:.0f} s, {n} finestre, {steps} passi per epoca; "
          f"validazione {len(vends)} finestre; k = {k}", flush=True)
    (out / "run.json").write_text(json.dumps(run, indent=1))
    mu_t, sd_t = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    use_graph = device.type == "cuda" and not args.no_cuda_graph
    if args.check_graph and device.type == "cuda":
        gen_c = torch.Generator().manual_seed(12345)
        sel = torch.randperm(n, generator=gen_c)[: 60 * args.batch]
        xs = store.gather(ends[sel]).to(device).view(60, args.batch, R.WINDOW, R.N_CHANNELS)
        fs = fw[sel].to(device).view(60, args.batch)
        yv = ys[sel].to(device).view(60, args.batch, R.N_JOINTS)
        check = graph_check(xs, fs, yv, mu, sd, args.batch, device, args.lr)
        (out / "graph_check.json").write_text(json.dumps(check, indent=1))
        print(f"controllo eager/graph: differenza massima {check['max_abs_param_diff']:.3g}, {check['eager_ms_per_step']:.2f} ms contro "
              f"{check['graph_ms_per_step']:.2f} ms per passo", flush=True)
        if check["max_abs_param_diff"] > T.GRAPH_TOLERANCE:
            run["status"] = "controllo eager/graph fallito: nessun training"
            (out / "run.json").write_text(json.dumps(run, indent=1))
            return 1
    model, opt, loss_fn = build(args.seed, device, args.lr)
    eval_model = T.denormalizer(model, mu_t.to(device), sd_t.to(device))
    stepper = Stepper(model, opt, loss_fn, mu, sd, args.batch, use_graph, device)
    feeder = Feeder(store, ends, fw, ys, args.batch, args.chunk_steps, pin=device.type == "cuda")
    chunk = args.batch * args.chunk_steps
    gx = torch.empty((chunk, R.WINDOW, R.N_CHANNELS), dtype=torch.int16, device=device)
    gf = torch.empty(chunk, dtype=torch.float32, device=device)
    gy = torch.empty((chunk, R.N_JOINTS), dtype=torch.float16, device=device)
    gen = torch.Generator().manual_seed(args.seed)
    vys_np = vys.float().numpy()
    deadline = t_start + args.time_budget_s if args.time_budget_s > 0 else float("inf")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(n, generator=gen)[: steps * args.batch]
        feeder.start(order, steps)
        for b, c in feeder.chunks():
            m = c * args.batch
            gx[:m].copy_(feeder.bx[b][:m], non_blocking=True)
            gf[:m].copy_(feeder.bf[b][:m], non_blocking=True)
            gy[:m].copy_(feeder.by[b][:m], non_blocking=True)
            if device.type == "cuda":
                ev = torch.cuda.Event()
                ev.record()
                ev.synchronize()  # il buffer bloccato torna libero solo a copia finita
            feeder.release(b)
            for i in range(c):
                s0, s1 = i * args.batch, (i + 1) * args.batch
                stepper.step(gx[s0:s1], gf[s0:s1], gy[s0:s1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_loss = stepper.take_total() / steps
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"loss non finita all'epoca {epoch + 1}")
        t_train = time.time() - t0
        vpred = predict(eval_model, vstore, vends, vfw, device)
        val_mae = float(np.abs(vpred.astype(np.float64) - vys_np).mean() * R.RAD2DEG)
        row = {"epoch": epoch + 1, "train_l1_std": train_loss, "steps": steps, "train_seconds": round(t_train, 1),
               "ms_per_step": round(1000 * t_train / steps, 4), "val_seconds": round(time.time() - t0 - t_train, 1), "val_mae_raw_deg": val_mae,
               "gather_seconds_total": round(feeder.gather_seconds, 1)}
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        torch.save(eval_model.state_dict(), out / "checkpoints" / f"epoch_{epoch + 1:02d}.pt")
        print(f"epoca {epoch + 1}: train L1 (std) {train_loss:.4f}, {row['ms_per_step']:.3f} ms per passo, validazione grezza {val_mae:.3f} gradi, "
              f"{row['train_seconds']:.0f} + {row['val_seconds']:.0f} s", flush=True)
        if time.time() > deadline and epoch + 1 < args.epochs:
            run["status"] = f"incompleto: tempo esaurito dopo l'epoca {epoch + 1}"
            break
    else:
        torch.save(eval_model.state_dict(), out / "model_final.pt")
        run["status"] = "ok"
    run.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t_start, 1),
                "max_rss_gb": round(max_rss_gb(), 2), "gather_seconds_total": round(feeder.gather_seconds, 1)})
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(f"fine: {run['status']} in {run['seconds']:.0f} s, memoria massima {run['max_rss_gb']} GB", flush=True)
    return 0 if run["status"] == "ok" else 1


# --- valutazione ------------------------------------------------------------------------------------------------------------------------

def collect(split: str, manifest: dict, targets_dir: Path, args, model, k: int, device):
    """Predizioni e bersagli di tutte le finestre ufficiali di uno split: (P, Y) torch (finestre, 1000, 20) su device, metadati per finestra."""
    import torch

    recs = sorted((stem, info) for stem, info in manifest["recordings"].items() if info["split"] == split and info["n_windows"] > 0)
    rows = [{"user": info["user"], "filename": stem} for stem, info in recs]
    store = HostStore(rows, args.processed_root, k)
    P, Y, meta = [], [], []
    for i, (stem, info) in enumerate(recs):
        z = np.load(targets_dir / info["file"])
        starts = z["starts"]
        inst = R.eval_instants(starts)
        if (inst + 0 >= store.n[i]).any():
            raise ValueError(f"{stem}: istanti oltre la registrazione")
        ends = torch.from_numpy(store.offset[i] + inst.reshape(-1))
        fwi = store.factor[i].repeat(len(ends))
        p = predict(model, store, ends, fwi, device).reshape(inst.shape + (R.N_JOINTS,))
        P.append(torch.from_numpy(p).to(device))
        Y.append(torch.from_numpy(np.asarray(z["targets"], dtype=np.float32)).to(device))
        meta.extend([(info["user"], info["generalization"], info["side"], R.is_moving_side(info))] * len(starts))
    return torch.cat(P), torch.cat(Y), meta, {"recordings": len(recs), "gib": round(store.gib, 2), "load_seconds": store.load_seconds}


def per_window_sums(P, Y):
    import torch

    return (P.to(torch.float64) - Y.to(torch.float64)).abs().sum(dim=(1, 2)).cpu().numpy()


def group_sums(sums: np.ndarray, meta: list, key) -> dict:
    """{gruppo: {utente: (somma, numero di valori)}} con key(meta) -> gruppo (o None per escludere)."""
    per_value = R.EVAL_PREDICTED // R.EVAL_STEP * R.N_JOINTS
    out: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for s, m in zip(sums, meta):
        g = key(m)
        if g is None:
            continue
        out[g][m[0]][0] += float(s)
        out[g][m[0]][1] += per_value
    return {g: {u: tuple(v) for u, v in d.items()} for g, d in out.items()}


def run_eval(args) -> int:
    torch = setup_torch(args.cudnn_mode)
    device = torch.device(args.device)
    out = new_dir(args.out)
    targets_dir = Path(args.targets_manifest).parent
    manifest = json.loads(Path(args.targets_manifest).read_text())
    proto = {"predicted": R.EVAL_PREDICTED, "context": R.CONTEXT, "stride": R.EVAL_STRIDE, "step": R.EVAL_STEP}
    if not manifest.get("complete") or manifest.get("protocol") != proto:
        raise SystemExit("manifest del passo 0 incompleto o con un protocollo diverso")
    train_run = json.loads((Path(args.model_dir) / "run.json").read_text())
    if train_run.get("status") != "ok" and not args.val_only:
        raise SystemExit(f"training non concluso: {train_run.get('status')}")
    k = int(manifest["scale"]["exponent_k"])
    if int(train_run["scale_exponent_k"]) != k:
        raise SystemExit("scala del training diversa da quella del manifest")
    mu, sd = R.pose_stats_all(args.pose_root)
    model, _, _ = build(0, device, 1e-4)
    eval_model = T.denormalizer(model, torch.tensor(mu, dtype=torch.float32).to(device), torch.tensor(sd, dtype=torch.float32).to(device))
    ckpt = Path(args.model_dir) / ("model_final.pt" if train_run.get("status") == "ok" else sorted((Path(args.model_dir) / "checkpoints").glob("*.pt"))[-1])
    eval_model.load_state_dict(torch.load(ckpt, map_location=device))
    t0 = time.time()
    report = {"kind": "temponet_emg2pose_eval_v1", "args": vars(args), "code_sha": os.environ.get("WEARUSFM_CODE_SHA"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "checkpoint": str(ckpt), "train_status": train_run.get("status"), "k": k}
    mu_dev = torch.tensor(mu, dtype=torch.float32, device=device)
    Pv, Yv, meta_v, info_v = collect("val", manifest, targets_dir, args, eval_model, k, device)
    alphas, mae_grid = R.tune_alpha_windows(Pv, Yv)
    Ev = R.ema_windows(Pv, torch.tensor(alphas, dtype=torch.float32, device=device))
    everyone = lambda m: "val"  # noqa: E731
    sums_v = {"ema": per_window_sums(Ev, Yv), "raw": per_window_sums(Pv, Yv), "constant": per_window_sums(mu_dev.expand_as(Yv), Yv)}
    val = {kind: R.summarize_users(group_sums(s, meta_v, everyone)["val"]) for kind, s in sums_v.items()}
    g = R.gate(group_sums(sums_v["ema"], meta_v, everyone)["val"], group_sums(sums_v["constant"], meta_v, everyone)["val"])
    report.update({"val": {"windows": len(meta_v), **info_v, **val}, "alpha": [float(a) for a in alphas],
                   "alpha_grid_mae_deg_min": float(mae_grid.min(axis=0).mean() * R.RAD2DEG), "gate": g})
    print(f"validazione: {len(meta_v)} finestre, MAE con EMA {val['ema']['mean_over_users_deg']:.3f}, grezza {val['raw']['mean_over_users_deg']:.3f}, "
          f"costante {val['constant']['mean_over_users_deg']:.3f}; cancello {g['ratio']:.3f} < {g['max_ratio']}: {g['passed']}", flush=True)
    del Pv, Yv, Ev
    if args.val_only or not g["passed"]:
        report["test"] = "non calcolato: " + ("collaudo (--val-only)" if args.val_only else "cancello non superato, si apre una diagnosi")
    else:
        Pt, Yt, meta_t, info_t = collect("test", manifest, targets_dir, args, eval_model, k, device)
        Et = R.ema_windows(Pt, torch.tensor(alphas, dtype=torch.float32, device=device))
        sums_t = {"ema": per_window_sums(Et, Yt), "raw": per_window_sums(Pt, Yt), "constant": per_window_sums(mu_dev.expand_as(Yt), Yt)}
        test = {"windows": len(meta_t), **info_t, "conditions": {}}
        for cond in R.CONDITIONS:
            by = {kind: R.summarize_users(group_sums(s, meta_t, lambda m: m[1]).get(cond, {})) for kind, s in sums_t.items()}
            main = by["ema"]["mean_over_users_deg"]
            test["conditions"][cond] = {**by, "reading": R.reading(main, cond) if main is not None else None,
                                        "tab4_regression_deg": R.TAB4_REGRESSION_DEG[cond]}
        test["by_side_ema"] = {s: R.summarize_users(d) for s, d in group_sums(sums_t["ema"], meta_t, lambda m: m[2]).items()}
        test["by_moving_ema"] = {("in movimento" if mv else "ferma"): R.summarize_users(d)
                                 for mv, d in group_sums(sums_t["ema"], meta_t, lambda m: m[3]).items()}
        report["test"] = test
    report["seconds"] = round(time.time() - t0, 1)
    (out / "eval.json").write_text(json.dumps(report, indent=1))
    (out / "eval.md").write_text(to_markdown(report))
    print(to_markdown(report), flush=True)
    return 0 if (args.val_only or g["passed"]) else 3


def to_markdown(r: dict) -> str:
    g, v = r["gate"], r["val"]
    lines = ["# TEMPONet su emg2pose: valutazione", "",
             f"Validazione ({v['n_users'] if 'n_users' in v else v['ema']['n_users']} utenti, {v['windows']} finestre ufficiali): MAE con EMA "
             f"{v['ema']['mean_over_users_deg']:.3f} gradi, grezza {v['raw']['mean_over_users_deg']:.3f}, predittore costante "
             f"{v['constant']['mean_over_users_deg']:.3f} (media fra utenti).",
             f"**Cancello**: {g['model_deg']:.3f} / {g['constant_deg']:.3f} = {g['ratio']:.3f}, soglia < {g['max_ratio']}: "
             f"**{'superato' if g['passed'] else 'NON superato'}**.", ""]
    t = r["test"]
    if isinstance(t, str):
        lines.append(f"Test {t}.")
        return "\n".join(lines) + "\n"
    lines += ["| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / "
              "NeuroPose / SensingDynamics |", "|---|---|---|---|---|---|---|---|"]
    for cond, c in t["conditions"].items():
        e = c["ema"]
        if e["mean_over_users_deg"] is None:
            lines.append(f"| {cond} | 0 | - | - | - | - | - | - |")
            continue
        tab = c["tab4_regression_deg"]
        lines.append(f"| {cond} | {e['n_users']} | {e['mean_over_users_deg']:.2f} ± {e['sd_over_users_deg'] or 0:.2f} | "
                     f"{c['raw']['mean_over_users_deg']:.2f} | {c['constant']['mean_over_users_deg']:.2f} | {e['pooled_deg']:.2f} | {c['reading']} | "
                     f"{tab['vemg2pose']} / {tab['NeuroPose']} / {tab['SensingDynamics']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("train", "eval"))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--pose-root", required=True)
    ap.add_argument("--targets-manifest", required=True, help="manifest.json del passo 0")
    ap.add_argument("--out", required=True, help="cartella nuova o vuota")
    ap.add_argument("--model-dir", default=None, help="eval: cartella del training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=19)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-stride-blocks", type=int, default=R.TRAIN_STRIDE_BLOCKS)
    ap.add_argument("--chunk-steps", type=int, default=512, help="passi per blocco raccolto dal thread")
    ap.add_argument("--max-train-recordings", type=int, default=0, help="solo collaudo")
    ap.add_argument("--max-val-recordings", type=int, default=0, help="solo collaudo")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="solo collaudo")
    ap.add_argument("--val-only", action="store_true", help="eval: solo validazione e cancello, nessun test (collaudo)")
    ap.add_argument("--check-graph", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--cudnn-mode", choices=sorted(T.CUDNN_MODES), default="benchmark")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="train: nessuna epoca nuova dopo questo tempo (run incompleto)")
    args = ap.parse_args(argv)
    if args.mode == "eval" and not args.model_dir:
        ap.error("eval richiede --model-dir")
    return run_train(args) if args.mode == "train" else run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
