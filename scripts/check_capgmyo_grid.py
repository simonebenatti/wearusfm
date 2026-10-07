#!/usr/bin/env python3
"""Local, descriptive CapgMyo channel-layout diagnostic; never modifies input data.

Not an anatomical calibration or a preregistered acceptance gate. Report distinct
adjacency graphs, rather than counting transposes/reflections as independent evidence.
Requires NumPy, SciPy and (for figures) Matplotlib. Raw MAT files stay outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[1]
AUTHOR_SHA = "1726e9ed09ffdf70b4c2f33f2baee01efe9394d6"
FILE_RE = re.compile(r"^(\d{3})-(\d{3})-(\d{3})\.mat$")


def candidate_maps() -> dict[str, np.ndarray]:
    """Grid cells contain zero-based input-column indices, NOT reordered signals."""
    ids = np.arange(128)
    row = ids.reshape(16, 8)
    col = ids.reshape(16, 8, order="F")
    snake_row = row.copy()
    snake_row[1::2] = snake_row[1::2, ::-1]
    snake_col = col.copy()
    snake_col[:, 1::2] = snake_col[::-1, 1::2]
    out = {
        "row16x8_C": row,
        "column16x8_F": col,
        "snake_rows": snake_row,
        "snake_columns": snake_col,
    }
    for order in ("C", "F"):
        physical = np.concatenate([
            np.arange(m * 16, (m + 1) * 16).reshape(8, 2, order=order)
            for m in range(8)
        ], axis=1)
        out[f"modules8x2_{order}_transposed"] = physical.T.copy()
    out["row16x8_C_mirrored"] = row[:, ::-1].copy()
    out["legacy8x16_C"] = ids.reshape(8, 16)
    for name, grid in out.items():
        if sorted(grid.ravel().tolist()) != list(range(128)):
            raise ValueError(f"{name}: not a channel bijection")
    return out


def edge_axes(grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "row_axis": np.column_stack((grid[:-1].ravel(), grid[1:].ravel())),
        "column_axis": np.column_stack((grid[:, :-1].ravel(), grid[:, 1:].ravel())),
    }


def graph_key(grid: np.ndarray) -> tuple:
    edges = np.concatenate(list(edge_axes(grid).values()))
    return tuple(sorted(map(tuple, np.sort(edges, axis=1))))


def equivalent_graphs(maps: dict[str, np.ndarray]) -> list[list[str]]:
    groups: dict[tuple, list[str]] = {}
    for name, grid in maps.items():
        groups.setdefault(graph_key(grid), []).append(name)
    return list(groups.values())


def correlation(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    centered = data - data.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    good = valid & (norms > 0)
    z = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    corr = np.clip(z.T @ z, -1.0, 1.0)
    corr[~good, :] = np.nan
    corr[:, ~good] = np.nan
    return corr


def rms_windows(data: np.ndarray, window: int = 50, hop: int = 25) -> np.ndarray:
    if window < 2 or hop < 1 or len(data) < window:
        raise ValueError("invalid RMS window/hop or insufficient samples")
    return np.stack([np.sqrt(np.mean(data[t:t + window] ** 2, axis=0))
                     for t in range(0, len(data) - window + 1, hop)])


def lagged_correlations(data: np.ndarray, edges: np.ndarray,
                        max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson at each lag, independently centered/normalized on overlap.

    Positive lag compares channel a[t] with b[t+lag]. Peak is selected by absolute
    correlation; this selection is biased upwards and is NOT propagation evidence.
    """
    if max_lag < 0 or max_lag >= len(data) - 2:
        raise ValueError("invalid maximum lag")
    a, b = edges.T
    values = []
    lags = np.arange(-max_lag, max_lag + 1)
    for lag in lags:
        if lag >= 0:
            x, y = data[:len(data) - lag], data[lag:]
        else:
            x, y = data[-lag:], data[:len(data) + lag]
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        numerator = np.einsum("ij,ij->j", x[:, a], y[:, b])
        denominator = np.linalg.norm(x, axis=0)[a] * np.linalg.norm(y, axis=0)[b]
        values.append(np.divide(numerator, denominator,
                                out=np.full(len(edges), np.nan), where=denominator > 0))
    values = np.array(values)
    finite = np.isfinite(values).any(axis=0)
    best = np.argmax(np.where(np.isfinite(values), np.abs(values), -np.inf), axis=0)
    peaks = values[best, np.arange(len(edges))]
    peaks[~finite] = np.nan
    best_lags = lags[best].astype(float)
    best_lags[~finite] = np.nan
    return peaks, best_lags


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def analyze_trial(data: np.ndarray, maps: dict[str, np.ndarray], *,
                  fs: float = 1000, rms_ms: float = 50, hop_ms: float = 25,
                  max_lag_ms: float = 3, min_std_ratio: float = 1e-3) -> tuple[dict, dict]:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 128 or data.shape[0] < 50:
        raise ValueError("expected a time x 128 array with at least 50 samples")
    if not np.isfinite(data).all():
        raise ValueError("non-finite input: refused, not silently imputed")
    if fs <= 0 or rms_ms <= 0 or hop_ms <= 0 or min_std_ratio < 0:
        raise ValueError("invalid diagnostic parameters")
    std = data.std(axis=0)
    median = np.median(std)
    valid = (std > 0) & (std >= min_std_ratio * median)
    if valid.sum() < 2:
        raise ValueError("fewer than two non-flat channels")
    rms = rms_windows(data, round(fs * rms_ms / 1000), round(fs * hop_ms / 1000))
    if len(rms) < 3:
        raise ValueError("fewer than three RMS windows")
    raw = correlation(data, valid)
    # Sensitivity analysis only: this is not the signed preprocessing pipeline.
    cmr = data - np.median(data[:, valid], axis=1, keepdims=True)
    removed = correlation(cmr, valid)
    env = correlation(rms, valid)
    log_rms = np.log(np.maximum(rms, np.finfo(float).tiny))
    all_edges = np.array(sorted(set(e for grid in maps.values() for e in graph_key(grid))))
    lookup = {tuple(e): i for i, e in enumerate(all_edges)}
    peaks, peak_lags = lagged_correlations(data, all_edges, round(fs * max_lag_ms / 1000))
    result = {}
    for name, grid in maps.items():
        axes = edge_axes(grid)
        axes["all"] = np.concatenate(list(axes.values()))
        scored = {}
        for axis, edges in axes.items():
            edges = edges[valid[edges[:, 0]] & valid[edges[:, 1]]]
            a, b = edges.T
            idx = np.array([lookup[tuple(sorted(e))] for e in edges], dtype=int)
            scored[axis] = {
                "edges": len(edges),
                "raw_signed": finite_mean(raw[a, b]),
                "raw_abs": finite_mean(np.abs(raw[a, b])),
                "cmr_signed": finite_mean(removed[a, b]),
                "cmr_abs": finite_mean(np.abs(removed[a, b])),
                "envelope_signed": finite_mean(env[a, b]),
                "log_rms_tv": finite_mean(np.abs(log_rms[:, a] - log_rms[:, b])),
                "lag_peak_abs": finite_mean(np.abs(peaks[idx])),
                "lag_peak_abs_ms": finite_mean(np.abs(peak_lags[idx]) * 1000 / fs),
            }
        result[name] = scored
    upper = np.triu_indices(128, k=1)
    baseline = {
        "raw_abs": finite_mean(np.abs(raw[upper])),
        "cmr_abs": finite_mean(np.abs(removed[upper])),
        "envelope_signed": finite_mean(env[upper]),
    }
    return {"invalid_channels": np.flatnonzero(~valid).tolist(), "rms_windows": len(rms),
            "candidates": result, "all_pairs_baseline": baseline}, {
                "raw_corr": raw, "cmr_corr": removed, "envelope_corr": env,
                "rms_profile": np.median(rms, axis=0),
            }


def load_trial(path: Path) -> tuple[np.ndarray, dict]:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"invalid filename {path.name}")
    subject, gesture, trial = map(int, match.groups())
    mat = loadmat(path)
    for key, expected in (("subject", subject), ("gesture", gesture), ("trial", trial)):
        if key not in mat or int(np.asarray(mat[key]).item()) != expected:
            raise ValueError(f"{path.name}: missing/inconsistent {key}")
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.shape != (1000, 128):
        raise ValueError(f"{path.name}: expected (1000, 128), got {data.shape}")
    return data, {"file": path.name, "subject": subject, "gesture": gesture,
                  "trial": trial, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def aggregate(trials: list[dict], maps: dict[str, np.ndarray]) -> dict:
    """Equal trial weighting within subject, then equal subject weighting."""
    subjects = sorted({t["subject"] for t in trials})
    result = {}
    metrics = list(trials[0]["candidates"][next(iter(maps))]["all"])
    for name in maps:
        result[name] = {}
        for axis in ("row_axis", "column_axis", "all"):
            result[name][axis] = {}
            for metric in metrics:
                per_subject = {}
                for subject in subjects:
                    vals = [t["candidates"][name][axis][metric] for t in trials if t["subject"] == subject]
                    per_subject[str(subject)] = finite_mean(np.array([np.nan if v is None else v for v in vals]))
                vals = np.array([np.nan if v is None else v for v in per_subject.values()])
                result[name][axis][metric] = {"mean": finite_mean(vals), "by_subject": per_subject}
    return result


def source_audit(source_dir: Path | None) -> dict:
    result = {"repository": "https://github.com/Answeror/srep", "commit": AUTHOR_SHA, "files": {}}
    if source_dir is None:
        result["status"] = "not supplied; no source verification claimed"
        return result
    for name in ("capgmyo_init.py", "data_init.py", "preprocess.py"):
        path = source_dir / name
        body = path.read_text()
        matches = [{"line": i, "text": line.strip()} for i, line in enumerate(body.splitlines(), 1)
                   if any(s in line for s in ("NUM_SEMG_", "loadmat(path)", "np.vstack(data).reshape",
                                               "data = data.reshape", "data = get_trial"))]
        result["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "matches": matches}
    result["status"] = "local source snapshots; review SHA/URLs and processing path before interpreting"
    return result


def make_figures(out: Path, maps: dict[str, np.ndarray], trials: list[dict],
                 visual: dict, summary: dict, groups: list[list[str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    representatives = [g[0] for g in groups]
    subjects = sorted(visual)
    for subject in subjects:
        # One raw-file example: identical amplitude scale, offsets for readability.
        from matplotlib.patches import Rectangle
        data = visual[subject]["example_data"]
        channels = np.arange(32, 40)  # row 4 in the proposed 16x8 C mapping
        centered = data[:, channels] - data[:, channels].mean(axis=0)
        scale = float(np.quantile(np.abs(centered), 0.995))
        spacing = 2.5 * scale
        offsets = np.arange(len(channels))[::-1] * spacing
        time_ms = np.arange(len(data))  # 1000 Hz, one sample per ms
        colors = plt.get_cmap("tab10")(np.arange(8))
        fig = plt.figure(figsize=(14, 9), layout="constrained")
        layout = fig.add_gridspec(2, 2, width_ratios=[1.8, 1])
        ax = fig.add_subplot(layout[:, 0])
        for i, channel in enumerate(channels):
            ax.plot(time_ms, centered[:, i] + offsets[i], color=colors[i], lw=0.6)
        ax.set_yticks(offsets, [f"ch {i + 1}" for i in channels])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("raw MAT amplitude; mean removed, vertical offsets only")
        ax.set_title("8 consecutive input channels | same amplitude scale")
        ax.grid(axis="x", alpha=0.2)
        ax.plot([1030, 1030], [0, scale], color="black", lw=2)
        ax.text(1040, scale / 2, f"{scale:.3g}\nMAT units", fontsize=9, va="center")
        ax.set_xlim(0, 1140)
        ax = fig.add_subplot(layout[0, 1])
        profile = visual[subject]["rms_profile"]
        im = ax.imshow(profile.reshape(16, 8), cmap="viridis", aspect="equal")
        for col in range(8):
            ax.add_patch(Rectangle((col - 0.5, 3.5), 1, 1, fill=False, edgecolor=colors[col], lw=2))
        ax.set_title("16x8 C | median RMS; outlined channels 33-40")
        ax.set_xlabel("column index (not anatomical)")
        ax.set_ylabel("row index")
        fig.colorbar(im, ax=ax, label="RMS (original MAT units)", shrink=0.8)
        ax = fig.add_subplot(layout[1, 1])
        rms = rms_windows(data[:, channels])
        centers_ms = np.arange(len(rms)) * 25 + 24.5
        for i, channel in enumerate(channels):
            ax.plot(centers_ms, rms[:, i], color=colors[i], label=f"ch {channel + 1}", lw=1.2)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("RMS (original MAT units)")
        ax.set_title("RMS: 50 ms window, 25 ms hop")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        fig.suptitle(f"Real CapgMyo | subject {subject:03d}, gesture 1, trial 1 | no channel-wise normalization")
        fig.savefig(out / f"signals_s{subject:03d}.png", dpi=150)
        plt.close(fig)
        profile = visual[subject]["rms_profile"]
        relative = np.log10(np.maximum(profile, np.finfo(float).tiny) / np.median(profile[profile > 0]))
        lo, hi = np.quantile(relative[np.isfinite(relative)], [0.02, 0.98])
        fig, axs = plt.subplots(2, 3, figsize=(12, 10), layout="constrained")
        for ax, name in zip(axs.ravel(), representatives):
            im = ax.imshow(relative[maps[name]], cmap="viridis", vmin=lo, vmax=hi, aspect="equal")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("column index (not anatomical)")
            ax.set_ylabel("row index")
        for ax in axs.ravel()[len(representatives):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axs.ravel()), label="log10(RMS / trial median RMS)", shrink=0.7)
        fig.suptitle(f"Subject {subject:03d}, gesture 1, trial 1 | same values, different adjacency", fontsize=13)
        fig.savefig(out / f"rms_s{subject:03d}.png", dpi=150)
        plt.close(fig)
    fig, axs = plt.subplots(len(subjects), 3, figsize=(12, 4 * len(subjects)), layout="constrained", squeeze=False)
    for row, subject in enumerate(subjects):
        for col, key in enumerate(("raw_corr", "cmr_corr", "envelope_corr")):
            matrix = np.mean([t[key] for t in visual[subject]["matrices"]], axis=0)
            im = axs[row, col].imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
            axs[row, col].set_title(f"s{subject:03d} | {key}")
            axs[row, col].set_xlabel("input column, zero-based")
            axs[row, col].set_ylabel("input column, zero-based")
    fig.colorbar(im, ax=list(axs.ravel()), label="Pearson r", shrink=0.7)
    fig.savefig(out / "correlations.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for ax, metric in zip(axs, ("raw_abs", "cmr_abs", "log_rms_tv")):
        vals = [summary[n]["all"][metric]["mean"] for n in representatives]
        ax.bar(np.arange(len(vals)), vals, color="#447799")
        for subject in subjects:
            points = [summary[n]["all"][metric]["by_subject"][str(subject)] for n in representatives]
            ax.plot(np.arange(len(points)), points, "o", label=f"s{subject:03d}", markersize=4)
        ax.set_xticks(np.arange(len(vals)), representatives, rotation=70, ha="right", fontsize=8)
        ax.set_title(metric + (" (lower = smoother)" if metric == "log_rms_tv" else " (higher = more similar)"))
        ax.legend(fontsize=8)
    fig.suptitle("Descriptive scores: subject means, no anatomical certification")
    fig.savefig(out / "scores.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 7, 13])
    ap.add_argument("--gestures", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--trials", type=int, nargs="+", default=[1, 5, 9])
    ap.add_argument("--source-dir", type=Path)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    out = args.out_dir.resolve()
    if out == REPO or REPO in out.parents:
        ap.error("output must be outside the repository: no data-derived maps in Git")
    if out.exists() and any(out.iterdir()):
        ap.error("output directory is nonempty; use a fresh directory")
    if not args.no_plots and (1 not in args.gestures or 1 not in args.trials):
        ap.error("plots require gesture 1 and trial 1 in the selection; use --no-plots otherwise")
    maps = candidate_maps()
    groups = equivalent_graphs(maps)
    records, visual = [], {}
    for subject in sorted(set(args.subjects)):
        for gesture in sorted(set(args.gestures)):
            for trial in sorted(set(args.trials)):
                path = args.raw_root / f"{subject:03d}-{gesture:03d}-{trial:03d}.mat"
                data, record = load_trial(path)
                scores, arrays = analyze_trial(data, maps)
                record.update(scores)
                records.append(record)
                visual.setdefault(subject, {"matrices": []})["matrices"].append(arrays)
                if gesture == 1 and trial == 1:
                    visual[subject]["rms_profile"] = arrays["rms_profile"]
                    visual[subject]["example_data"] = data
        print(f"subject {subject:03d}: analyzed", flush=True)
    if not records:
        ap.error("empty selection")
    summary = aggregate(records, maps)
    def order(metric, ascending=False):
        def score(name):
            val = summary[name]["all"][metric]["mean"]
            return (val if ascending else -val) if val is not None else float("inf")
        return sorted([g[0] for g in groups], key=score)
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {
        "kind": "exploratory channel-layout diagnostic, NOT an acceptance gate",
        "code_base_commit": git_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": {"subjects": args.subjects, "gestures": args.gestures, "trials": args.trials},
        "parameters": {"fs_hz": 1000, "rms_ms": 50, "hop_ms": 25, "max_lag_ms": 3, "min_std_ratio": 1e-3},
        "preprocessing": "raw as supplied; channel centering for correlations; median common-mode removal is a separate sensitivity analysis",
        "n_files": len(records), "equivalent_adjacency_graphs": groups,
        "channel_maps": {n: g.tolist() for n, g in maps.items()},
        "rankings": {"raw_abs": order("raw_abs"), "cmr_abs": order("cmr_abs"), "log_rms_tv": order("log_rms_tv", True)},
        "summary": summary, "trials": records, "source_audit": source_audit(args.source_dir),
        "limitations": [
            "Three subjects are not the entire dataset; all tests are descriptive.",
            "No inferential p-value or anatomical acceptance threshold is defined.",
            "Common reference, crosstalk and phase inversion can affect correlations.",
            "Absolute lag peaks have selection bias; they do not prove fibre propagation.",
            "Reflections/transposes preserve adjacency and cannot fix proximal/distal orientation.",
            "Graphs are open rectangles; no undocumented circumferential wrap is assumed.",
            "RMS maps show the same values under permutations; smoothness alone is not ground truth.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        make_figures(out, maps, records, visual, summary, groups)
    print(json.dumps({"files": len(records), "rankings": report["rankings"], "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
