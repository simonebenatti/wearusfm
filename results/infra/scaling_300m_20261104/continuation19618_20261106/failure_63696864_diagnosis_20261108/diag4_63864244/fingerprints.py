"""Diagnosi 4: sha256, dimensione, mtime/ctime e inode dei file delle sessioni dell'insieme fisso e delle registrazioni del pannello."""
import hashlib, importlib.util, json, os, time
from pathlib import Path
import numpy as np  # noqa: F401
from wearusfm.training import run as R

OUT = Path(os.environ["DIAG_OUT"])
code = Path(os.environ["CODE_DIR"]).resolve()
S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
B = Path(os.environ["DIAG_B"])
argv = json.loads((OUT / "argv_plain_cli.json").read_text())
spec = importlib.util.spec_from_file_location("train_jepa_cli_fp", code / "scripts/train_jepa.py")
CLI = importlib.util.module_from_spec(spec); spec.loader.exec_module(CLI)
ap = CLI.build_parser(); args = ap.parse_args(argv); cfg = CLI.build_config(args, ap.error)
index = R.load_index(args.manifest, args.root, cfg.datasets, cfg.nested)
by_key = {f"{r['dataset']}/{r['subject']}/{r['session']}": r for r in index.rows}
A_fixed = json.loads((PREV / "fixed_eval_set.json").read_text())
A_panel = json.loads((PREV / "pose_panel_set.json").read_text())
B_panel = json.loads((B / "pose_panel_ident_now.json").read_text())
unchanged = {rec["recording"]: (a == b) for rec, a, b in zip(A_panel["recordings"], A_panel["batch_sha256"], B_panel["batch_sha256"])}
fixed_keys = {w[0] for w in A_fixed["windows"]}
keys = sorted(fixed_keys | set(unchanged))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
def describe(p):
    st = p.stat()
    return {"bytes": st.st_size, "mtime": iso(st.st_mtime), "ctime": iso(st.st_ctime), "inode": st.st_ino, "sha256": sha(p)}

pose_root = Path(cfg.loader.pose_root) if getattr(cfg.loader, "pose_root", None) else None
t0 = time.time(); out = {}; n_files = 0; n_bytes = 0
for k in keys:
    row = by_key.get(k)
    e = {"in_index": row is not None, "in_fixed": k in fixed_keys, "in_panel": k in unchanged, "panel_unchanged_A_vs_B": unchanged.get(k)}
    if row is not None:
        try:
            d = index.path_of(row)
            e["dir"] = str(d); e["files"] = {}
            for name in ("metadata.json", "data_int16.npy"):
                p = d / name
                if p.exists():
                    e["files"][name] = describe(p); n_files += 1; n_bytes += e["files"][name]["bytes"]
            if pose_root is not None and row["dataset"] == "emg2pose":
                for cand in (pose_root / row["subject"] / row["session"], pose_root / ("u" + row["subject"]) / row["session"]):
                    if cand.exists():
                        e["pose_dir"] = str(cand)
                        for name in ("pose_200hz_f16.npy", "pose_valid.npy", "pose_meta.json"):
                            p = cand / name
                            if p.exists():
                                e["files"][name] = describe(p); n_files += 1; n_bytes += e["files"][name]["bytes"]
                        break
        except Exception as ex:  # noqa: BLE001
            e["error"] = f"{type(ex).__name__}: {ex}"
    out[k] = e
summary = {"protocol": "scaling-diag-importorder-20261108-v4", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "n_sessions": len(keys),
           "n_fixed_sessions": len(fixed_keys), "n_panel_recordings": len(unchanged), "n_files": n_files, "n_bytes": n_bytes,
           "elapsed_s": round(time.time() - t0, 1), "pose_root": str(pose_root), "roots": [str(r) for r in index.roots]}
(OUT / "fingerprints.json").write_text(json.dumps({"summary": summary, "sessions": out}, indent=1))
print("IMPRONTE:", json.dumps(summary), flush=True)
