"""Replica di TEMPONet su NinaPro DB8 (Zanghieri et al., COINS 2021, [Z21]): training per soggetto e test sull'acquisizione 3.

Protocollo (`docs/temponet_db8_replication.md`, criteri in `docs/decisioni.md`): per ogni soggetto e seme tre modelli,
- `fold1`: training su A1, validazione su A2; `fold2`: training su A2, validazione su A1 (la «2-fold cross-validation» di [Z21] sez. II-D1 sulle
  sessioni 1 e 2 unite: i due fold sono le due sessioni, 10 ripetizioni per movimento ciascuna);
- `full`: training su A1 + A2, alpha dell'EMA dalle predizioni fuori fold dei due modelli precedenti (analisi secondaria).
Training: TEMPONet-64 (`wearusfm.model.temponet`), loss L1, Adam lr 1e-4 costante, batch 64, 19 epoche in float32 ([Z21] sez. II-D3; l'epoca
finale di quantization-aware training int8 non si replica), finestre di 256 campioni con passo 32 e bersaglio all'ultimo campione.
Post-processing: EMA causale con alpha per DoA scelto sulla griglia 0,00-0,99 con le sole predizioni di validazione; test su A3 con quell'alpha.
Nessuna selezione sul test: le epoche sono fisse, la curva di validazione per epoca e' solo descrittiva.

Uso (GPU, `scripts/slurm/temponet_db8.sbatch`):
  python3 scripts/temponet_db8.py --cache <cache> --out <cartella NUOVA> --seeds 0 1 2 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def train(sig: Signal, val: Signal | None, cfg: argparse.Namespace, seed: int, device):
    import torch
    from torch import nn

    from wearusfm.model.temponet import TEMPONet

    torch.manual_seed(seed)
    model = TEMPONet(in_channels=db8.N_EMG, window=cfg.window, out_dim=db8.N_DOA, causal=not cfg.symmetric_padding).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()
    gen = torch.Generator().manual_seed(seed)
    n = len(sig.ends)
    steps = n // cfg.batch
    if cfg.max_steps_per_epoch:
        steps = min(steps, cfg.max_steps_per_epoch)
    history = []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = torch.zeros((), device=device, dtype=torch.float64)
        for i in range(steps):
            x, y = sig.batch(sig.ends[perm[i * cfg.batch : (i + 1) * cfg.batch]])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.detach()
        train_loss = float(total / steps)
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"loss non finita all'epoca {epoch + 1}")
        row = {"epoch": epoch + 1, "train_l1_deg": train_loss, "steps": steps, "seconds": round(time.time() - t0, 2)}
        if val is not None:
            vp, vt = predict(model, val)
            row["val_mae_raw_deg"] = float(db8.mae_per_doa(vp, vt).mean())
        history.append(row)
    return model, history


def run_unit(subject: int, seed: int, cfg: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    t0 = time.time()
    cache = Path(cfg.cache)
    provenance = check_cache(cache, subject)
    recs = {a: db8.load_cached(cache, subject, a) for a in db8.ACQUISITIONS}
    scale = EMG_UNIT_SCALE[cfg.emg_unit]
    sig = {a: Signal([recs[a]], device, scale, cfg.window, cfg.step) for a in db8.ACQUISITIONS}
    out_dir = Path(cfg.out)
    result = {"subject": subject, "seed": seed, "provenance": provenance, "models": {}}
    preds = {}
    for k, name in enumerate(MODELS):
        train_acqs, val_acq = SPLITS[name]
        tr = sig[train_acqs[0]] if len(train_acqs) == 1 else Signal([recs[a] for a in train_acqs], device, scale, cfg.window, cfg.step)
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
        torch.save(model.state_dict(), out_dir / "models" / f"S{subject}_seed{seed}_{name}.pt")
        np.save(out_dir / "predictions" / f"S{subject}_seed{seed}_{name}_test_raw.npy", test_pred.astype(np.float32))
        if seed == cfg.seeds[0] and k == 0:
            np.save(out_dir / "predictions" / f"S{subject}_test_target.npy", test_tgt.astype(np.float32))
        print(f"S{subject} seme {seed} {name}: test MAE {entry['test_ema']['mae_deg']:.3f} (grezzo {entry['test_raw']['mae_deg']:.3f}), "
              f"alpha medio {np.mean(alpha):.3f}, {entry['train_seconds']:.0f} s", flush=True)
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
    ap.add_argument("--step", type=int, default=db8.STEP)
    ap.add_argument("--emg-unit", choices=sorted(EMG_UNIT_SCALE), default="uV",
                    help="unita' dell'ingresso: cambio di scala costante, non una normalizzazione (vedi il documento)")
    ap.add_argument("--symmetric-padding", action="store_true", help="padding simmetrico come Q-PPG invece che causale")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="solo collaudo: tetto ai passi per epoca")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="nessuna unita' nuova dopo questo tempo (0 = nessun limite)")
    args = ap.parse_args(argv)
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
