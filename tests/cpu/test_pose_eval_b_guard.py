"""Operational guards around frozen E1/E2, without running Slurm or changing scientific code."""
import csv
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/slurm/pose_eval_b_guarded.sbatch"


@pytest.fixture
def guard(tmp_path, monkeypatch):
    text = SCRIPT.read_text()
    code = text.split("<<'PY_B_GUARD'\n", 1)[1].split("\nPY_B_GUARD\n", 1)[0]
    ns = {"__name__": "guard_test"}
    exec(compile(code, str(SCRIPT), "exec"), ns)
    for k, v in {"WORK": str(tmp_path / "work"), "SCRATCH": str(tmp_path / "scratch"),
                 "CODE_DIR": str(tmp_path / "code"), "WEARUSFM_CODE_SHA": ns["CODE_SHA"],
                 "WRAPPER_SHA256": "ac" * 32}.items():
        monkeypatch.setenv(k, v)
    loc = ns["paths"]()
    for p in loc.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    loc["pose"].mkdir()
    (loc["pose"] / ns["PT"].STATS_FILE).write_text('{"fixture": true}')
    ns["STATS_SHA"] = ns["sha"](loc["pose"] / ns["PT"].STATS_FILE)
    rows = [{"split": "val", "filename": f"r{i}.h5", "user": u, "session": f"s{i}", "stage": "moving",
             "side": "left", "moving_hand": "both"} for i, u in enumerate(("a", "b"))]
    with loc["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for i, u in enumerate(("a", "b")):
        processed = loc["processed"] / ("u" + u) / f"r{i}"
        processed.mkdir(parents=True)
        (processed / "metadata.json").write_text("{}")
        (processed / "data_int16.npy").write_bytes(b"fixture")
        pose = loc["pose"] / ("u" + u) / f"r{i}"
        pose.mkdir(parents=True)
        for name in (ns["PT"].POSE_FILE, ns["PT"].VALID_FILE):
            (pose / name).write_bytes(b"fixture")
    doc = {"columns": ["dataset", "subject", "split"], "rows": [["emg2pose", "uc", "pretraining"]]}
    loc["manifest"].write_bytes(gzip.compress(json.dumps(doc).encode()))
    ns["MANIFEST_SHA"] = ns["sha"](loc["manifest"])
    run = tmp_path / "B"
    run.mkdir()
    config = {"model": {"pose_head": True, "dim": 4}}
    (run / "config.json").write_text(json.dumps(config))
    ns["CONFIG_SHA"] = ns["sha"](run / "config.json")
    (run / "summary.json").write_text(json.dumps({"steps": 5000, "stopped": "max_steps", "code": {"sha": ns["CODE_SHA"]},
                                                "pose_stats_sha256": ns["STATS_SHA"]}))
    state = {"step": 5000, "config": config, "pose_stats_sha256": ns["STATS_SHA"],
             "student": {"pose_head.norm.weight": torch.ones(4), "pose_head.norm.bias": torch.zeros(4),
                         "pose_head.out.weight": torch.ones(20, 4), "pose_head.out.bias": torch.zeros(20)}}
    checkpoint = run / "checkpoint.pt"
    torch.save(state, checkpoint)
    ns.update(CHECKPOINT_BYTES=checkpoint.stat().st_size, VAL_RECORDINGS=2, VAL_USERS=2)
    return ns, loc, checkpoint, tmp_path / "outputs" / "val.npz", state


def make_cache(guard):
    ns, loc, ck, out, _ = guard
    ns["main"]("extract", str(ck), str(out))
    prov = json.loads(Path(str(out) + ".extract_provenance.json").read_text())
    meta = [{**{k: r[k] for k in ("user", "session", "stage", "side", "stem")}, "recording": i, "start": 0, "fs": 2000., "n_patches": 2}
            for i, r in enumerate(prov["records"])]
    data = {"lat": np.ones((2, 2, 4), dtype=np.float16), "pred": np.ones((2, 2, 20), dtype=np.float32),
            "target": np.ones((2, 2, 20), dtype=np.float32), "mask": np.ones((2, 2, 20), dtype=bool),
            "std": np.ones((2, 20), dtype=np.float32), "pose_var": np.ones(2), "has_head": True, "dim": 4,
            "meta": meta, "n_recordings": 2,
            "config": {"which": "student", "split": "val", "code_sha": ns["CODE_SHA"], "pose_stats_sha256": ns["STATS_SHA"],
                       "checkpoint_pose_stats_sha256": ns["STATS_SHA"], "checkpoint": str(ck), "pose_root": str(loc["pose"]),
                       "n_windows": 10, "window_s": 4.0, "seed": ns["PE"].SEED_WINDOWS}}
    ns["PE"].save_npz(out, data)
    return data


def test_guard_roundtrip_records_hashes_and_does_not_compute_scientific_report(guard):
    ns, _, ck, out, _ = guard
    make_cache(guard)
    report = out.parent / "report.json"
    ns["main"]("report", str(out), str(report))
    prov = json.loads(Path(str(report) + ".report_provenance.json").read_text())
    assert prov["cache_sha256"] == ns["sha"](out) and prov["checkpoint_sha256"] == ns["sha"](ck)
    assert prov["n_recordings_with_windows"] == 2 and prov["n_users_with_valid_pose"] == 2
    assert not report.exists()  # frozen launcher alone computes E1/E2
    with pytest.raises(ValueError, match="provenance already exists"):
        ns["main"]("report", str(out), str(report))


@pytest.mark.parametrize("bad", ["size", "stats", "config", "step", "head", "overlap", "csv", "processed", "pose"])
def test_extract_rejects_wrong_checkpoint_statistics_data_or_split(guard, bad):
    ns, loc, ck, out, state = guard
    if bad == "size":
        ns["CHECKPOINT_BYTES"] += 1
    elif bad == "stats":
        (loc["pose"] / ns["PT"].STATS_FILE).write_text("changed")
    elif bad == "config":
        (ck.parent / "config.json").write_text("{}")
    elif bad in ("step", "head"):
        if bad == "step":
            state["step"] = 4999
        else:
            state["student"]["pose_head.out.weight"].zero_()
        torch.save(state, ck)
        ns["CHECKPOINT_BYTES"] = ck.stat().st_size
    elif bad == "overlap":
        doc = {"columns": ["dataset", "subject", "split"], "rows": [["emg2pose", "ua", "pretraining"]]}
        loc["manifest"].write_bytes(gzip.compress(json.dumps(doc).encode()))
        ns["MANIFEST_SHA"] = ns["sha"](loc["manifest"])
    elif bad == "csv":
        ns["VAL_RECORDINGS"] = 3
    elif bad == "processed":
        (loc["processed"] / "ua/r0/metadata.json").unlink()
    elif bad == "pose":
        (loc["pose"] / "ua/r0" / ns["PT"].POSE_FILE).unlink()
    with pytest.raises(ValueError):
        ns["main"]("extract", str(ck), str(out))
    assert not Path(str(out) + ".extract_provenance.json").exists()


@pytest.mark.parametrize("bad", ["teacher", "stats", "checkpoint", "metadata", "support", "no_head", "duplicate"])
def test_report_rejects_cache_or_checkpoint_disagreement(guard, bad):
    ns, _, ck, out, _ = guard
    data = make_cache(guard)
    if bad == "teacher":
        data["config"]["which"] = "teacher"
    elif bad == "stats":
        data["config"]["checkpoint_pose_stats_sha256"] = "00" * 32
    elif bad == "checkpoint":
        with ck.open("ab") as stream:
            stream.write(b"changed")
    elif bad == "metadata":
        data["meta"][0]["side"] = "right"
    elif bad == "support":
        data["mask"][0] = False
    elif bad == "no_head":
        data["has_head"] = False
    elif bad == "duplicate":
        data["meta"][1] = dict(data["meta"][0])
    ns["PE"].save_npz(out, data)
    with pytest.raises(ValueError):
        ns["main"]("report", str(out), str(out.parent / "report.json"))


def test_output_guard_and_shell_contract(guard):
    ns, _, ck, out, _ = guard
    out.parent.mkdir()
    out.write_bytes(b"existing")
    with pytest.raises(ValueError, match="output already exists"):
        ns["main"]("extract", str(ck), str(out))
    code_out = Path(ns["os"].environ["CODE_DIR"]) / "out.npz"
    with pytest.raises(ValueError, match="inside code"):
        ns["main"]("extract", str(ck), str(code_out))
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert '${B_EVAL_MODE:?}' in text and '--which student --split val' in text
    assert 'exec bash "$CODE_DIR/scripts/slurm/pose_eval_report.sbatch" "$INPUT" "$OUTPUT"' in text
