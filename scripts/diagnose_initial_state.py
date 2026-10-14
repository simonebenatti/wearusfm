#!/usr/bin/env python3
"""Diagnostica dello stato iniziale dei teacher dell'A/B dello Step 1, per la baseline casuale (punto 1 della lista comune, Simone 14/10/2026:
«parti con lo script per la baseline casuale»). Sola lettura dei run; nessun training, nessun forward, nessun CUDA.

La baseline casuale di GPT ricostruisce il modello al passo 0 e ne controlla lo SHA (`training.pairing.state_sha256`) contro quello registrato
dall'A/B in `pairing.json`; sul cluster non torna (r1 e r2). Lo stato iniziale non e' salvato: si confrontano ricostruzioni fatte in ambienti
diversi, ciascuna in un PROCESSO NUOVO (l'ambiente conta dall'avvio: thread di OpenMP/MKL/OpenBLAS, ordine degli import, libreria OpenMP
caricata):
- `ab_replica`: come il processo dell'A/B (launcher `step1_ab_train.sbatch`: OMP/MKL/OpenBLAS a 1 dall'avvio; import di `scripts/step1_ab.py`,
  invariato da 69fa2da; `torch.manual_seed(seed)` e poi `WearUsFM(cfg.model)`, come `training.run.train`);
- `torch_only_env1`: thread a 1 dall'avvio, solo torch e il modello;
- `gpt_like_env8`: thread a 8 dall'avvio (launcher della baseline), poi come `harness.random_encoder.initial_teacher` (GPT, eb80059/4fb4208):
  `torch.set_num_threads(1)` e `default_generator.manual_seed(seed)` dentro `fork_rng`;
- `gpt_like_env8_sklearn_first`: come il precedente, ma con scipy e sklearn importati prima di torch;
- `env8_plain`: thread a 8, `torch.manual_seed(seed)`, nessun cambio dei thread.
In ogni ricostruzione i buffer di calibrazione della testa dello Step 1 si copiano dal teacher finale (come fa GPT: stimati sulla GPU al passo
0 e mai cambiati dal training). Il rapporto dice, per seme: quale ricostruzione riproduce lo SHA di `pairing.json`; quali tensori differiscono
fra le ricostruzioni (hash per tensore, somma e massimo in float64); quali buffer (che il training non cambia: l'EMA li copia) differiscono dal
teacher finale; l'ambiente di ogni processo (host, CPU, thread, librerie OpenMP caricate).

  python scripts/diagnose_initial_state.py --runs-root RUNS --seeds 0 1 --out REPORT.json
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_BUFFERS = ("target_mean", "target_std", "target_active", "stats_fitted")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
VARIANTS = {
    "ab_replica": {"threads": "1", "imports": "step1_ab", "seeding": "manual_seed", "set_threads_1": False},
    "torch_only_env1": {"threads": "1", "imports": "torch", "seeding": "manual_seed", "set_threads_1": False},
    "gpt_like_env8": {"threads": "8", "imports": "torch", "seeding": "default_generator", "set_threads_1": True},
    "gpt_like_env8_sklearn_first": {"threads": "8", "imports": "sklearn_first", "seeding": "default_generator", "set_threads_1": True},
    "env8_plain": {"threads": "8", "imports": "torch", "seeding": "manual_seed", "set_threads_1": False},
}


def _environment() -> dict:
    """Host, CPU, thread e librerie caricate nel processo (le librerie OpenMP/BLAS da /proc/self/maps, dove c'e')."""
    import torch

    cpu = None
    try:
        cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), None)
    except OSError:
        pass
    libs = set()
    try:
        for line in open("/proc/self/maps"):
            name = line.split()[-1] if "/" in line else ""
            base = os.path.basename(name)
            if any(k in base for k in ("gomp", "iomp", "omp", "mkl", "openblas", "blas")):
                libs.add(base)
    except OSError:
        pass
    return {"host": socket.gethostname(), "cpu": cpu, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "env": {k: os.environ.get(k) for k in THREAD_VARS + ("MKL_THREADING_LAYER",)},
            "parallel_info": torch.__config__.parallel_info(), "libraries": sorted(libs)}


def child(variant: str, reference: Path, seed: int, out: Path) -> None:
    """Una ricostruzione nel processo corrente, gia' avviato con l'ambiente della variante."""
    spec = VARIANTS[variant]
    if spec["imports"] == "step1_ab":
        sys.path.insert(0, str(ROOT / "scripts"))
        import step1_ab  # noqa: F401  gli import del processo dell'A/B, nello stesso ordine
    elif spec["imports"] == "sklearn_first":
        import scipy.signal  # noqa: F401
        import sklearn.linear_model  # noqa: F401
    import torch
    from dataclasses import fields

    sys.path.insert(0, str(ROOT / "src"))
    from wearusfm.model.fm import FMConfig, WearUsFM
    from wearusfm.training.pairing import state_sha256

    state = torch.load(reference, map_location="cpu", weights_only=False)
    mcfg = state["config"]["model"]
    cfg = FMConfig(**{f.name: mcfg[f.name] for f in fields(FMConfig) if f.name in mcfg})
    if spec["seeding"] == "manual_seed":
        threads = torch.get_num_threads()
        if spec["set_threads_1"]:
            torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = WearUsFM(cfg)
        torch.set_num_threads(threads)
    else:
        threads = torch.get_num_threads()
        try:
            if spec["set_threads_1"]:
                torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                model = WearUsFM(cfg)
        finally:
            torch.set_num_threads(threads)
    if model.spectral_keep is not None:
        with torch.no_grad():
            for name in CALIBRATION_BUFFERS:
                getattr(model.spectral_keep, name).copy_(state["teacher"]["spectral_keep." + name])
    sd = model.state_dict()
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu().contiguous()
        d = t.double()
        tensors[k] = {"sha256": hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                      "dtype": str(t.dtype), "shape": list(t.shape), "sum": float(d.sum()), "absmax": float(d.abs().max()) if t.numel() else 0.0,
                      "values": t.double().reshape(-1).tolist() if t.numel() <= 8 else None}
    out.write_text(json.dumps({"variant": variant, "seed": seed, "sha256": state_sha256(sd), "tensors": tensors,
                               "buffers": [k for k, _ in model.named_buffers()], "environment": _environment()}, allow_nan=False))


def _tensor_hashes(state: dict) -> dict:
    import torch

    return {k: hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for k, v in state.items() if isinstance(v, torch.Tensor)}


def diagnose(runs_root: Path, seeds, workdir: Path, variants=tuple(VARIANTS)) -> dict:
    import torch

    report = {"protocol": "diagnose_initial_state_v1", "variants": {v: VARIANTS[v] for v in variants}, "seeds": {}}
    for seed in seeds:
        refs = {arm: runs_root / f"{arm}_seed{seed}" for arm in ("A", "B")}
        pairing = {arm: json.loads((p / "pairing.json").read_text()) for arm, p in refs.items() if (p / "pairing.json").exists()}
        reference = refs["B"] / "checkpoint.pt"
        teacher = torch.load(reference, map_location="cpu", weights_only=False)["teacher"]
        teacher_hashes = _tensor_hashes(teacher)
        results = {}
        for v in variants:
            out = workdir / f"seed{seed}_{v}.json"
            env = {**os.environ, **{k: VARIANTS[v]["threads"] for k in THREAD_VARS}, "PYTHONDONTWRITEBYTECODE": "1"}
            env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", v, "--reference", str(reference),
                                   "--seed", str(seed), "--child-out", str(out)], env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                results[v] = {"error": proc.stderr[-2000:]}
                continue
            results[v] = json.loads(out.read_text())
        expected = {f"{arm}_{who}": p.get(f"{who}_initial_sha256") for arm, p in pairing.items() for who in ("student", "teacher")}
        row = {"reference": str(reference), "pairing_initial_sha256": expected, "variants": {}}
        base = results.get("ab_replica")
        for v, r in results.items():
            if "error" in r:
                row["variants"][v] = r
                continue
            buffers_vs_teacher = sorted(k for k in r["buffers"] if k in teacher_hashes and r["tensors"][k]["sha256"] != teacher_hashes[k])
            diff = None
            if base is not None and "error" not in base and v != "ab_replica":
                diff = {k: {"sum": [base["tensors"][k]["sum"], t["sum"]], "absmax": [base["tensors"][k]["absmax"], t["absmax"]],
                            "values": [base["tensors"][k]["values"], t["values"]]}
                        for k, t in r["tensors"].items() if k in base["tensors"] and t["sha256"] != base["tensors"][k]["sha256"]}
            row["variants"][v] = {"sha256": r["sha256"], "matches": sorted(k for k, s in expected.items() if s == r["sha256"]),
                                  "buffers_differing_from_final_teacher": {k: {"variant": r["tensors"][k]["values"],
                                                                               "variant_sum": r["tensors"][k]["sum"]}
                                                                           for k in buffers_vs_teacher},
                                  "tensors_differing_from_ab_replica": diff, "environment": r["environment"]}
        report["seeds"][str(seed)] = row
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, help="runs dell'A/B (A_seed<k>, B_seed<k> con checkpoint.pt e pairing.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--out", type=Path, help="rapporto JSON (cartella nuova, fuori dal repo)")
    ap.add_argument("--child", choices=list(VARIANTS), help=argparse.SUPPRESS)
    ap.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        child(args.child, args.reference, args.seed, args.child_out)
        return 0
    if args.runs_root is None or args.out is None:
        ap.error("servono --runs-root e --out")
    if args.out.exists():
        ap.error(f"{args.out} esiste: rapporto nuovo")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = args.out.parent / (args.out.stem + "_processi")
    work.mkdir()
    report = diagnose(args.runs_root, args.seeds, work, tuple(args.variants))
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False))
    for seed, row in report["seeds"].items():
        print(f"seme {seed}: SHA attesi {row['pairing_initial_sha256']}")
        for v, r in row["variants"].items():
            if "error" in r:
                print(f"  {v}: ERRORE {r['error'][-300:]}")
                continue
            ndiff = len(r["tensors_differing_from_ab_replica"] or {})
            print(f"  {v}: sha {r['sha256'][:12]} uguale a {r['matches'] or 'nessuno'}; buffer diversi dal teacher finale "
                  f"{sorted(r['buffers_differing_from_final_teacher'])}; tensori diversi da ab_replica {ndiff}; "
                  f"thread torch {r['environment']['torch_threads']}, host {r['environment']['host']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
