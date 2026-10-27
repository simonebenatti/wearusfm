"""Ingest della POSA di emg2pose (angoli articolari) per la seconda ondata del pilot degli obiettivi: vedi src/wearusfm/ingest/emg2pose_pose.py
e docs/pilot_posa_v1.md §2 e §6 (punti 2, 5, 6). Un file alla volta nell'ordine del tar, un file rotto non ferma gli altri, RIPRENDIBILE
(`--skip-existing`), con un budget di tempo (`--time-budget-s`) per fermarsi in ordine prima del limite di SLURM e scrivere il report.

Collaudo (40 registrazioni, 20 per lato, quote uguali per `moving_hand`):
  python3 scripts/ingest_emg2pose_pose.py --tar <...tar> --csv <...metadata.csv> --out-root <...> --report <r.json> --limit 40 --both-sides
Ingest intero (dopo che il margine di validita' e' stato congelato in decisioni.md):
  python3 scripts/ingest_emg2pose_pose.py --tar ... --csv ... --out-root ... --report ... --skip-existing --zero-margin-samples <M> --stats
Solo statistiche (`pose_stats.json` da un albero gia' scritto):  ... --stats-only
Il report va FUORI dal repo ($WORK/wearusfm_runs/results/passo2/); il riepilogo si copia sul Mac e si committa in results/passo2/.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest import emg2pose as E2P  # noqa: E402
from wearusfm.ingest import emg2pose_pose as PP  # noqa: E402


def _aggregate_profiles(done: list, metas: list[dict]) -> dict:
    """Riepilogo del collaudo: frazione di righe a zero, massimo assoluto per giunto, profilo dei bordi (medie pesate per bin) e minimi
    simultanei, su tutte le registrazioni fatte."""
    out = {"n_recordings": len(done), "per_side": {}, "per_moving_hand": {}}
    rows = sum(r["n_samples"] for r in done)
    out["frac_zero_rows"] = sum(r["n_zero_rows"] for r in done) / rows if rows else None
    out["n_nonfinite_rows"] = sum(r["n_nonfinite_rows"] for r in done)
    out["max_abs_per_joint"] = [max(r["max_abs_per_joint"][j] for r in done) for j in range(PP.N_JOINTS)] if done else None
    out["max_abs_overall"] = max(out["max_abs_per_joint"]) if done else None
    for key in ("side", "moving_hand"):
        bucket = out["per_side"] if key == "side" else out["per_moving_hand"]
        for r in done:
            bucket.setdefault(str(r.get(key)), 0)
            bucket[str(r.get(key))] += 1
    # statistiche per lato sulle registrazioni fatte (media e deviazione per giunto sui blocchi validi; varianza media per registrazione)
    out["per_side_stats"] = {}
    for side in sorted({str(m.get("side")) for m in metas}):
        ms = [m for m in metas if str(m.get("side")) == side]
        acc = {"n": 0, "sum": [0.0] * PP.N_JOINTS, "sumsq": [0.0] * PP.N_JOINTS, "min": None, "max": None}
        for m in ms:
            PP._accumulate(acc, m["joint_stats_valid_blocks"])
        fin = PP._finish(acc)
        var = [m["per_joint_var_valid_blocks"] for m in ms if m.get("per_joint_var_valid_blocks") and m["per_joint_var_valid_blocks"][0] is not None]
        out["per_side_stats"][side] = {"n_recordings": len(ms), "hours": sum(m["n_samples"] for m in ms) / PP.NATIVE_FS_HZ / 3600,
                                       "mean": fin.get("mean"), "std": fin.get("std"), "n_valid_blocks": fin["n"],
                                       "mean_per_recording_var": [float(sum(v[j] for v in var) / len(var)) for j in range(PP.N_JOINTS)] if var else None,
                                       "per_recording_var_mean_over_joints": [float(sum(v) / len(v)) for v in var]}
    prof = [m["edge_profile"] for m in metas if "edge_profile" in m and m["edge_profile"]["bins"]]
    out["frac_zero_rows_per_recording"] = sorted(r["frac_zero_rows"] for r in done)
    if prof:
        bins = []
        for i in range(len(prof[0]["bins"])):
            n = sum(p["bins"][i]["n"] for p in prof)
            s = sum(p["bins"][i]["n"] * p["bins"][i]["mean_abs_diff"] for p in prof if p["bins"][i]["mean_abs_diff"] is not None)
            bins.append({"lag_from": prof[0]["bins"][i]["lag_from"], "lag_to": prof[0]["bins"][i]["lag_to"], "n": n,
                         "mean_abs_diff": s / n if n else None})
        far_n = sum(p["far"].get("n", 0) for p in prof)
        far_mean = sum(p["far"]["n"] * p["far"]["mean"] for p in prof if p["far"].get("n")) / far_n if far_n else None
        out["edge_profile"] = {"bins": bins, "far_mean_abs_diff": far_mean, "far_n": far_n,
                               "note": "contaminazione dei bordi = bin con mean_abs_diff lontano da far_mean_abs_diff; il margine e' il primo lag "
                                       "oltre il quale il profilo torna al valore lontano (da congelare in decisioni.md)"}
    dips = [m["simultaneous_dips"] for m in metas if "simultaneous_dips" in m]
    if dips:
        out["simultaneous_dips"] = {"n_dip_runs_total": sum(d["n_dip_runs"] for d in dips), "n_dip_rows_total": sum(d["n_dip_rows"] for d in dips),
                                    "recordings_with_dips": sum(1 for d in dips if d["n_dip_runs"])}
    return out


def _write_report(path: Path, done: list, failed: dict, skipped: int, t0: float, stopped: str | None, args, extra: dict | None) -> dict:
    report = {
        "dataset": "emg2pose_pose", "params": {"zero_margin_samples": args.zero_margin_samples, "limit": args.limit, "both_sides": args.both_sides,
                                               "profile_max_lag": args.profile_max_lag,
                                               "measure": not args.no_measure, "pose_fs_hz": PP.POSE_FS_HZ, "block_samples": PP.BLOCK},
        "n_recordings": len(done), "n_failed": len(failed), "n_skipped_existing": skipped, "n_users": len({r["user"] for r in done}),
        "hours_total": sum(r["n_samples"] for r in done) / PP.NATIVE_FS_HZ / 3600, "n_valid_blocks_total": sum(r["n_valid_blocks"] for r in done),
        "n_blocks_total": sum(r["n_blocks"] for r in done), "failed": failed, "stopped": stopped, "elapsed_s": time.time() - t0,
        "collaudo": extra, "per_recording": done,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tar", type=Path, help="emg2pose_dataset.tar (non compresso)")
    p.add_argument("--csv", type=Path, required=True, help="CSV dei metadati di emg2pose (utente, lato, moving_hand, split)")
    p.add_argument("--out-root", type=Path, required=True, help="albero della posa, FUORI da quello processato dell'EMG")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="collaudo: al massimo N registrazioni (nell'ordine del tar)")
    p.add_argument("--both-sides", action="store_true", help="col --limit: meta' per lato e quote uguali per moving_hand")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=None)
    p.add_argument("--zero-margin-samples", type=int, default=0, help="campioni non validi attorno a ogni riga a zero (congelato in decisioni.md "
                                                                      "dopo il collaudo; 0 = solo le righe a zero)")
    p.add_argument("--no-measure", action="store_true", help="senza profilo dei bordi e minimi simultanei (ingest intero, piu' veloce)")
    p.add_argument("--profile-max-lag", type=int, default=PP.PROFILE_MAX_LAG, help="collaudo: lunghezza del profilo dei bordi in campioni a 2 kHz "
                                                                                   "(default 400 = 200 ms; 4000 = 2 s per il profilo lungo)")
    p.add_argument("--stats", action="store_true", help="alla fine scrive pose_stats.json sullo split di pretraining")
    p.add_argument("--stats-only", action="store_true", help="solo pose_stats.json da un albero gia' scritto")
    args = p.parse_args()
    if args.zero_margin_samples < 0:
        p.error("--zero-margin-samples >= 0")
    if args.profile_max_lag < PP.PROFILE_BIN:
        p.error("--profile-max-lag >= un bin")
    if args.stats_only:
        stats = PP.aggregate_stats(args.out_root)
        print(json.dumps({k: stats[k] for k in ("n_recordings", "n_recordings_per_side", "side_criterion")}, indent=1))
        return 0
    if args.tar is None:
        p.error("--tar e' obbligatorio (salvo --stats-only)")

    t0 = time.time()
    done, metas, failed, skipped, stopped = [], [], {}, 0, None
    picker = PP.CollaudoPicker(args.limit, args.both_sides) if args.limit is not None else None
    meta = E2P.load_metadata_csv(args.csv)
    fields = None

    def out_of_budget() -> bool:
        return args.time_budget_s is not None and time.time() - t0 > args.time_budget_s

    with tarfile.open(args.tar, "r:") as tf:
        print(f"{len(meta)} righe nel CSV; si scorre il tar e si elabora man mano (margine {args.zero_margin_samples} campioni)", flush=True)
        for ref in E2P.iter_recordings(tf):
            if picker is not None and picker.full:
                stopped = "limit"
                break
            if out_of_budget():
                stopped = "time-budget"
                break
            row = meta.get(ref.stem)
            if picker is not None and not picker.accept(row):
                continue
            user = (row or {}).get("user", "sconosciuto")
            if args.skip_existing and PP.already_done(PP.out_dir_for(args.out_root, user, ref.stem), args.zero_margin_samples):
                skipped += 1  # saltata solo se scritta con la STESSA regola di validita' (margine e tolleranza): altrimenti si riscrive
                continue
            try:
                fobj = tf.extractfile(ref.info if ref.info is not None else ref.member)  # col membro, non col nome (vedi E2P.RecordingRef)
                if fields is None:
                    fields = PP.fields_in_file(fobj)
                    fobj.seek(0)
                done.append(PP.ingest_pose_recording(fobj, ref, args.out_root, row, zero_margin_samples=args.zero_margin_samples,
                                                     measure=not args.no_measure, fields=fields, profile_max_lag=args.profile_max_lag))
                metas.append(json.loads((PP.out_dir_for(args.out_root, user, ref.stem) / PP.META_FILE).read_text()))
            except Exception as e:  # un file rotto non ferma gli altri
                failed[ref.stem] = f"{type(e).__name__}: {e}"
                print(f"  FALLITO {ref.stem}: {failed[ref.stem]}", flush=True)
                continue
            if len(done) % 200 == 0:
                print(f"  {len(done)} fatte", flush=True)
                _write_report(args.report, done, failed, skipped, t0, "in corso", args, None)
    extra = _aggregate_profiles(done, metas) if done else None
    if extra is not None:
        extra["fields_in_first_file"] = fields
    report = _write_report(args.report, done, failed, skipped, t0, stopped, args, extra)
    if args.stats and stopped != "time-budget":
        stats = PP.aggregate_stats(args.out_root)
        report["stats"] = {k: stats[k] for k in ("n_recordings", "n_recordings_per_side", "side_criterion")}
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_recording", "failed")}, indent=2, default=str))
    if failed:
        print(f"FALLITI: {len(failed)} (primi 5: {dict(list(failed.items())[:5])})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
