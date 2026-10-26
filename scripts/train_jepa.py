#!/usr/bin/env python3
"""Training JEPA con ancore (passo 6): oggi il preset `sanity` (valori firmati da Simone il 03/10/2026, `docs/fogli_firma_d9_d10.md` 15-20; le
altre scelte sono in `src/wearusfm/training/run.py`). Scrive in `--out-dir`, FUORI dal repo; riprende da `checkpoint.pt` se c'e' (job concatenati).

  python3 scripts/train_jepa.py --preset sanity --manifest manifest_59253155.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed \\
      --scales session_scales.json --out-dir $WORK/wearusfm_runs/runs/sanity_<id> --time-limit-s 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

from wearusfm.training import run as R  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Il parser delle opzioni (usato anche da scripts/wsd_chain.py per controllare gli argomenti liberi dei piani)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["sanity", "small", "30m", "100m", "300m"], required=True,
                    help="sanity (firmato), small (prove), o un rung della ladder (training.run.LADDER, firmati il 09/10/2026)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--scales", type=Path, default=None, help="session_scales.json di scripts/measure_loader.py (senza: calcolate al volo)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--time-limit-s", type=float, default=None, help="si ferma e salva il checkpoint prima del limite del job")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", default=None, help="elenco separato da virgole, o 'all'; default: quello del preset (sanity: emg2qwerty)")
    ap.add_argument("--nested", choices=list(R.PM.DERIVED_FRACTIONS), default=None,
                    help="asse dei soggetti (D16): la frazione dichiarata dal manifest derivato passato con --manifest "
                         "(scripts/build_nested_manifests.py; '1' = derivato al 100%%); default manifest non derivato")
    ap.add_argument("--rvq-codes", type=Path, default=None,
                    help="radice dei codici RVQ precalcolati (scripts/precompute_rvq_codes.py): accende l'ancora RVQ (D5b); senza, l'ancora e' spenta")
    ap.add_argument("--window1-rules", action="store_true",
                    help="decisioni del 04/10/2026: quote nel tempo, perdita per finestra, ancore lontane dai bordi (serve --window-seconds)")
    ap.add_argument("--window-seconds", type=Path, default=None, help="JSON di scripts/window_seconds.py (lunghezza media delle finestre per sessione)")
    ap.add_argument("--manifest-sha256", default=None,
                    help="SHA-256 atteso del file del manifest (dal piano): un file diverso ferma il run prima di partire")
    ap.add_argument("--init-from", type=Path, default=None, help="checkpoint.pt di un altro run: pesi iniziali (ottimizzatore e passi da zero)")
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="learning rate di picco (calibrazione C1-C4); default quello del preset (3e-4)")
    ap.add_argument("--erank-min", type=float, default=None,
                    help="soglia assoluta dell'allarme sul rango effettivo (firmata il 23/10/2026, R.W1_ERANK_MIN); default la regola (16), 10%% della dimensione")
    ap.add_argument("--anchor-weight", type=float, default=None,
                    help="peso complessivo delle ancore; default quello del preset (0,2, decisione 18); 0 = JEPA puro (braccio W1-OBJ)")
    ap.add_argument("--jepa-weight", type=float, default=None,
                    help="peso della perdita JEPA; default 1; 0 = ancore attuali da sole (braccio W1-OBJ)")
    ap.add_argument("--jepa-target", choices=["a", "b"], default=None,
                    help="D11: dove si leggono i target del teacher; default quello del preset (b); a = encoder locale (R1 del pilot degli obiettivi)")
    ap.add_argument("--window-ms", type=float, default=None,
                    help="finestra del kernel del front-end piu' lunga del passo di 25 ms (R5 del pilot degli obiettivi, p.es. 150): la griglia "
                         "resta la patch, la perdita conta solo i token interamente nascosti; default nessuna (il banco di sempre)")
    ap.add_argument("--jepa-region-target", action="store_true",
                    help="target JEPA per regione nascosta (media del teacher sulla maschera estratta) invece che per token: R2 del pilot degli "
                         "obiettivi; default per token")
    ap.add_argument("--pose-root", default=None,
                    help="R3 (seconda ondata del pilot degli obiettivi, firmata il 26/10/2026): albero della posa di emg2pose scritto da "
                         "scripts/ingest_emg2pose_pose.py (con pose_stats.json); accende la testa di posa e il suo termine; default nessuno")
    ap.add_argument("--pose-weight", type=float, default=None,
                    help="peso del termine di posa; default 1,0 con --pose-root (firma del 26/10), non ammesso senza")
    ap.add_argument("--pose-unprompted", choices=["on", "off"], default="on",
                    help="registrazioni di emg2pose a mano non sollecitata nel termine di posa: on (firma del 26/10) o off (mascherate)")
    ap.add_argument("--mask-shares", default=None,
                    help="quote delle maschere corta:media:lunga:spaziale (somma 1; D10 firmata: 0.15:0.25:0.20:0.40); per R5 "
                         "0:0.294:0.235:0.471, scostamento dichiarato da D10; default le quote firmate")
    ap.add_argument("--batch-size", type=int, default=None, help="batch GLOBALE in finestre (con --distributed diviso fra i processi)")
    ap.add_argument("--keep-calibration", type=Path, default=None, help="TRAIN-only calibration NPZ from scripts/spectral_step1.py")
    ap.add_argument("--keep-weight", type=float, default=0., help="Step 1: 0 for paired arm A; >0 for B (experimental)")
    ap.add_argument("--spectral-branches", default=None,
                    help="rami spettrali in parallelo al raw (step 2-3): 'psd', 'wavelet' o 'psd,wavelet'; servono --keep-calibration e le regole "
                         "della finestra 1 (target multiscala)")
    ap.add_argument("--branch-weight", type=float, default=0.0,
                    help="peso del termine dei rami sul readout (la keep raw dello Step 1 resta --keep-weight); da fissare prima dai gradienti")
    ap.add_argument("--wavelet-learnable", action="store_true", help="centri e larghezze del banco wavelet apprendibili (entro limiti)")
    ap.add_argument("--branch-alpha", type=float, default=0.5, help="loss sul readout: (1 - alpha) testa fusa + alpha media delle teste dei rami")
    ap.add_argument("--branch-calibration-batches", type=int, default=16, help="batch del training per le statistiche d'ingresso dei rami")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="diagnostica dei rami: encoder raw congelato dal teacher di --init-from, si allenano solo rami e loro teste "
                         "(--branch-weight > 0, --keep-weight 0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--keep-grad-every", type=int, default=0, help="diagnostic weighted gradient norms on shared encoder parameters")
    ap.add_argument("--fixed-eval-every", type=int, default=0,
                    help="ogni N aggiornamenti le loss su un insieme fisso di batch del training (fixed_eval.jsonl); 0 = spenta")
    ap.add_argument("--fixed-eval-batches", type=int, default=8, help="batch dell'insieme fisso (fino a 32 finestre ciascuno)")
    ap.add_argument("--log-batch-origin", action="store_true",
                    help="in metrics.jsonl l'origine di ogni finestra (sessione, trial, inizio); sessioni in origin_sessions.json")
    ap.add_argument("--prefetch", action="store_true",
                    help="il batch successivo preso, preparato e copiato sul dispositivo in un thread mentre gira il passo (stessi batch)")
    ap.add_argument("--prefetch-audit", action="store_true",
                    help="verifica: a fine passo gli ingressi sul dispositivo uguali bit per bit a quelli preparati dal batch sulla CPU")
    ap.add_argument("--rvq-targets", choices=["slab", "channel"], default=None,
                    help="finestre dell'ancora RVQ; 'channel' e' SOLO DIAGNOSTICA (v10 §6.3: nel training mai sul masking di canale)")
    ap.add_argument("--schedule", choices=["constant", "wsd"], default="constant",
                    help="learning rate: 'constant' (warmup poi costante, il sanity) o 'wsd' (D14: warmup, stabile, decay lineare a zero)")
    ap.add_argument("--total-steps", type=int, default=None, help="wsd: aggiornamenti totali del run (fine del decay); max_steps = questo")
    ap.add_argument("--steps-at-d", type=int, default=None,
                    help="wsd: aggiornamenti a D, per il warmup di D14 (2%%, almeno 1.000); uguale in tutti i rami. In alternativa --warmup-steps")
    ap.add_argument("--decay-fraction", type=float, default=0.2, help="wsd: frazione finale di decay (D14: 0,2)")
    ap.add_argument("--keep-checkpoints", default="",
                    help="wsd: passi della fase stabile di cui tenere il checkpoint, separati da virgole o da ':' (nelle catene: sbatch --export "
                         "usa le virgole come separatore)")
    ap.add_argument("--branch-from", type=Path, default=None,
                    help="un ramo (pilot sulle epoche, continuazione a 2D): stato completo da un checkpoint della fase stabile, --out-dir nuova")
    ap.add_argument("--distributed", action="store_true",
                    help="DDP sotto torchrun, un processo per GPU (RANK/WORLD_SIZE/LOCAL_RANK dall'ambiente); il batch del preset e' GLOBALE")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="micro-batch per processo e per aggiornamento (accumulo dei gradienti; loss col contratto delle loss distribuite)")
    return ap


def build_config(args, error=None) -> "R.RunConfig":
    """La configurazione del run dagli argomenti di `build_parser()`: la stessa strada per il training (`main`) e per la verifica
    dell'identita' dei candidati della calibrazione (`scripts/calibration_select.py`, critico del 16/10/2026). Non legge i dati; legge solo
    la calibrazione della keep, se data (il suo SHA entra nella configurazione). `error`: chiamato con il messaggio (es. `ap.error`); in ogni
    caso un argomento non valido e' un ValueError."""
    def fail(msg):
        if error is not None:
            error(msg)
        raise ValueError(msg)

    cfg = {"sanity": R.sanity_config, "small": R.small_config}.get(args.preset, lambda: R.ladder_config(args.preset))()
    if args.rvq_codes is not None:
        cfg = R.with_rvq(cfg, args.rvq_codes)
    if args.window1_rules:
        cfg = R.with_window1_rules(cfg)
    if args.warmup_steps is not None:
        cfg = replace(cfg, warmup_steps=args.warmup_steps)
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.lr is not None:
        if not args.lr > 0:
            fail("--lr deve essere positivo")
        cfg = replace(cfg, lr=args.lr)
    if getattr(args, "erank_min", None) is not None:
        if not args.erank_min > 0:
            fail("--erank-min deve essere positivo")
        cfg = replace(cfg, alarm=replace(cfg.alarm, erank_min=float(args.erank_min)))
    if getattr(args, "jepa_target", None) is not None:
        cfg = replace(cfg, jepa=replace(cfg.jepa, target=args.jepa_target))
    if getattr(args, "window_ms", None) is not None:
        try:
            cfg = replace(cfg, model=replace(cfg.model, window_ms=float(args.window_ms)))
        except ValueError as e:
            fail(f"--window-ms: {e}")
    if getattr(args, "jepa_region_target", False):
        cfg = replace(cfg, jepa=replace(cfg.jepa, region_target=True))
    if getattr(args, "mask_shares", None) is not None:
        try:
            shares = [float(x) for x in args.mask_shares.split(":")]
            if len(shares) != 4:
                raise ValueError("servono quattro quote corta:media:lunga:spaziale")
            cfg = replace(cfg, loader=replace(cfg.loader, mask=replace(cfg.loader.mask, share_short=shares[0], share_medium=shares[1],
                                                                       share_long=shares[2], share_spatial=shares[3])))
        except ValueError as e:
            fail(f"--mask-shares: {e}")
    weights = {k: v for k, v in (("anchor_weight", args.anchor_weight), ("jepa_weight", args.jepa_weight)) if v is not None}
    if weights:
        try:
            cfg = replace(cfg, jepa=replace(cfg.jepa, **weights))
        except ValueError as e:
            fail(f"--anchor-weight / --jepa-weight: {e}")
    pose_root, pose_weight = getattr(args, "pose_root", None), getattr(args, "pose_weight", None)
    if pose_weight is not None and pose_root is None:
        fail("--pose-weight senza --pose-root")
    if pose_root is not None:  # R3: target di posa nel loader, testa nel modello, termine nella loss (peso firmato 1,0)
        w = 1.0 if pose_weight is None else float(pose_weight)
        if not w > 0:
            fail("--pose-weight deve essere positivo")
        try:
            cfg = replace(cfg, loader=replace(cfg.loader, pose_root=str(pose_root), pose_unprompted=getattr(args, "pose_unprompted", "on") == "on"),
                          model=replace(cfg.model, pose_head=True), jepa=replace(cfg.jepa, pose_weight=w))
        except ValueError as e:
            fail(f"--pose-root / --pose-weight: {e}")
    if args.accum_steps is not None:
        cfg = replace(cfg, accum_steps=args.accum_steps)
    if args.schedule == "wsd":
        if args.total_steps is None:
            fail("--schedule wsd richiede --total-steps")
        if (args.steps_at_d is None) == (args.warmup_steps is None):
            fail("--schedule wsd: il warmup da --steps-at-d (D14) oppure da --warmup-steps, uno dei due")
        warmup = R.wsd_warmup_steps(args.steps_at_d) if args.steps_at_d is not None else args.warmup_steps
        keep = tuple(int(x) for x in args.keep_checkpoints.replace(":", ",").split(",") if x.strip())
        cfg = R.with_wsd(cfg, args.total_steps, warmup_steps=warmup, keep_checkpoints=keep, decay_fraction=args.decay_fraction)
    elif args.total_steps is not None or args.steps_at_d is not None or args.keep_checkpoints:
        fail("--total-steps, --steps-at-d e --keep-checkpoints valgono con --schedule wsd")
    if args.rvq_targets is not None:
        cfg = replace(cfg, jepa=replace(cfg.jepa, rvq_targets=args.rvq_targets))
    if args.max_steps is not None:
        cfg = replace(cfg, max_steps=args.max_steps)
    if args.datasets is not None:
        cfg = replace(cfg, datasets=None if args.datasets == "all" else tuple(args.datasets.split(",")))
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    cfg = replace(cfg, keep_grad_every=args.keep_grad_every, fixed_eval_every=args.fixed_eval_every,
                  fixed_eval_batches=args.fixed_eval_batches, log_batch_origin=args.log_batch_origin, prefetch=args.prefetch,
                  prefetch_audit=args.prefetch_audit, nested=args.nested)
    if args.keep_calibration is not None:
        cfg = R.with_spectral_keep(cfg, args.keep_calibration, args.keep_weight)
    elif args.keep_weight != 0:
        fail("--keep-weight requires --keep-calibration")
    if args.spectral_branches:
        if args.keep_calibration is None:
            fail("--spectral-branches richiede --keep-calibration (target e teste dello Step 1)")
        cfg = R.with_spectral_branches(cfg, tuple(k.strip() for k in args.spectral_branches.split(",") if k.strip()),
                                       weight=args.branch_weight, wavelet_learnable=args.wavelet_learnable, alpha=args.branch_alpha,
                                       freeze_encoder=args.freeze_encoder, calibration_batches=args.branch_calibration_batches)
    elif args.wavelet_learnable or args.freeze_encoder or args.branch_weight != 0:
        fail("--branch-weight, --wavelet-learnable e --freeze-encoder valgono con --spectral-branches")
    return cfg


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    ctx = None
    if args.distributed:
        ctx = R.dist_from_env()
        torch.distributed.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")  # env:// da torchrun
    main_process = ctx is None or ctx.is_main
    try:  # identita' dei file d'ingresso: manifest del piano e, per un derivato, le durate con cui e' stato costruito (critico, 16/10)
        R.check_input_files(args.manifest, args.window_seconds, args.nested, args.manifest_sha256)
    except ValueError as e:
        ap.error(str(e))
    window_s = json.loads(args.window_seconds.read_text())["window_s"] if args.window_seconds else None
    cfg = build_config(args, ap.error)  # dopo i file, come prima dell'estrazione di build_config (stesso ordine degli errori)
    scales = json.loads(args.scales.read_text())["scales"] if args.scales else {}
    if main_process:
        print(f"preset {args.preset}: {R.count_parameters(cfg.model) / 1e6:.1f} M parametri, {cfg.max_steps} passi, batch {cfg.batch_size}"
              + (f" globale su {ctx.world_size} processi" if ctx else "") + f", dispositivo {args.device}, {len(scales)} scale di sessione, "
              f"ancora RVQ {'accesa: ' + str(args.rvq_codes) if args.rvq_codes else 'spenta'}", flush=True)
    try:
        summary = R.train(cfg, args.manifest, args.root, args.out_dir, scales=scales, device=args.device, num_workers=args.num_workers,
                          time_limit_s=args.time_limit_s, log=lambda m: print(m, flush=True), window_s=window_s, init_from=args.init_from,
                          dist=ctx, branch_from=args.branch_from)
        if main_process:
            print(json.dumps(summary), flush=True)
    finally:
        if ctx is not None:
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
