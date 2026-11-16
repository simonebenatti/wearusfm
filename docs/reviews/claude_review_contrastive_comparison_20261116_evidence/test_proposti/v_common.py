"""verifica-rnc: oracolo numpy (float64, nessun autograd) scritto dal paragrafo 3 del protocollo, e imbragatura del punto di
chiamata vero (ContrastiveRuntime.attach) con oggetti finti al posto di encoder e micro-batch. Solo tensori sintetici."""
import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

PATCH_MS = 25.0


def np_oracle(windows, *, temperature=0.1, stride=8, patch_ms=PATCH_MS, exclude_same_instant=True, details=False):
    """windows: lista di dict con zbar (P,D) float64 numpy (media sui K slot), y (P,J) o None, m (P,J) bool, tv (P,) bool,
    ident = (dataset, subject, session, trial, start, fs). Ritorna (loss float, finestre contate, info)."""
    pts = []
    for w, win in enumerate(windows):
        if win["y"] is None:
            continue
        P = win["zbar"].shape[0]
        for p in range(4 * stride // 8 if stride == 8 else stride // 2, P, stride):
            if win["tv"][p] and win["m"][p].any():
                pts.append((w, p))
    n = len(pts)
    U = np.zeros((n, windows[0]["zbar"].shape[1]))
    inst = []
    for i, (w, p) in enumerate(pts):
        h = windows[w]["zbar"][p].astype(np.float64)
        h = h - h.mean()
        U[i] = h / np.sqrt((h * h).sum())
        ds, sub, ses, trial, start, fs = windows[w]["ident"]
        inst.append((ds, sub, ses, trial, fs, start + int(math.ceil((p + 0.5) * patch_ms * fs / 1000.0 - 1e-9))))
    S = U @ U.T / temperature
    per_window, anchors, pairs, excluded = {}, 0, 0, 0
    for i, (wi, pi) in enumerate(pts):
        mi, yi = windows[wi]["m"][pi], windows[wi]["y"][pi].astype(np.float64)
        cand = []
        for k, (wk, pk) in enumerate(pts):
            if k == i:
                continue
            common = mi & windows[wk]["m"][pk]
            if not common.any():
                continue
            if exclude_same_instant and inst[i] == inst[k]:
                excluded += 1
                continue
            yk = windows[wk]["y"][pk].astype(np.float64)
            d = math.sqrt(float(((yi - yk)[common] ** 2).sum()) / int(common.sum()))
            cand.append((d, k))
        if len(cand) < 2:
            continue
        num, ref = 0.0, 0.0
        for dj, j in cand:
            tail = [S[i, k] for dk, k in cand if dk >= dj]
            num += float(np.logaddexp.reduce(np.array(tail))) - S[i, j]
            ref += math.log(len(tail))
        per_window.setdefault(wi, []).append(num / ref)
        anchors += 1
        pairs += len(cand)
    if not per_window:
        return 0.0, 0, {"anchors": 0, "pairs": 0, "points": n, "excluded_pairs": excluded, "instants": inst}
    loss = float(np.mean([np.mean(v) for v in per_window.values()]))
    return loss, len(per_window), {"anchors": anchors, "pairs": pairs, "points": n, "excluded_pairs": excluded, "instants": inst,
                                   "per_window": {k: float(np.mean(v)) for k, v in per_window.items()}}


def make_windows(seed, specs, *, K=3, D=24, J=20, per_joint_masks=False, invalid_block_prob=0.0, pose_fn=None):
    """specs: lista di (subject, session, start_campioni, n_patch, has_pose). Posa = funzione liscia dell'istante ASSOLUTO della
    registrazione: finestre sovrapposte della stessa registrazione hanno target identici agli stessi istanti, come nel flusso vero."""
    g = np.random.default_rng(seed)
    fs = 2000.0
    freqs = g.uniform(0.2, 3.0, size=(J,))
    phases = {}
    out = []
    for subject, session, start, P, has_pose in specs:
        key = (subject, session)
        if key not in phases:
            phases[key] = (g.uniform(0, 6.28, size=(J,)), g.uniform(0.3, 1.5, size=(J,)))
        ph, amp = phases[key]
        t = (start / 50.0 + np.arange(P))[:, None] * 0.025           # secondi assoluti dell'inizio della patch
        if pose_fn is not None:
            y = pose_fn(key, t, J).astype(np.float32)
        else:
            y = (amp[None] * np.sin(2 * np.pi * freqs[None] * t + ph[None]) + 0.2 * np.sin(1.7 * t + ph[None])).astype(np.float32)
        m = np.ones((P, J), dtype=bool)
        if invalid_block_prob > 0:  # validita' per patch, uguale sui giunti (come i blocchi veri)
            bad = g.random(P) < invalid_block_prob
            m[bad] = False
        if per_joint_masks:
            m &= g.random((P, J)) > 0.3
        z = g.standard_normal((P, K, D))
        out.append({"z": z, "zbar": z.mean(1), "y": y if has_pose else None, "m": m if has_pose else np.zeros((P, J), bool),
                    "tv": np.ones(P, dtype=bool), "ident": ("emg2pose", subject, session, None, int(start), fs),
                    "stop": int(start) + int(P * 50)})
    return out


def fake_inputs(windows, *, dtype=torch.float64, channels=2, pad_to=None, y_dtype=torch.float32):
    """Oggetti finti con i soli attributi letti da ContrastiveRuntime.attach. Le finestre piu' corte sono riempite a destra;
    il riempimento ha patch_valid False, readout casuali NON nulli (per smascherare un uso spurio) e target NaN."""
    S = len(windows)
    P = max(w["z"].shape[0] for w in windows) if pad_to is None else pad_to
    K, D = windows[0]["z"].shape[1:]
    J = windows[0]["m"].shape[1]
    g = torch.Generator().manual_seed(1234)
    z = torch.randn(S, P, K, D, generator=g, dtype=torch.float64)
    patch_valid = torch.zeros(S * channels, P, dtype=torch.bool)
    y = torch.full((S, P, J), float("nan"), dtype=torch.float32)
    valid = torch.zeros(S, P, J, dtype=torch.bool)
    any_pose = False
    for s, w in enumerate(windows):
        p = w["z"].shape[0]
        z[s, :p] = torch.from_numpy(w["z"])
        patch_valid[s * channels:(s + 1) * channels, :p] = torch.from_numpy(w["tv"])[None]
        if w["y"] is not None:
            any_pose = True
            y[s, :p] = torch.from_numpy(w["y"])
            valid[s, :p] = torch.from_numpy(w["m"])
    y = torch.where(valid, y, torch.zeros_like(y)) if False else y   # i NaN mascherati restano: devono essere ignorati
    z = z.to(dtype).requires_grad_(True)
    zt = torch.randn(S, P, K, D, generator=g, dtype=torch.float64).to(dtype)
    counts = [channels] * S
    inp = SimpleNamespace(counts=counts, qc_valid=torch.ones(S * channels, dtype=torch.bool))
    pose = (y.to(y_dtype), valid, torch.ones(S, J)) if any_pose else None
    micro = SimpleNamespace(inp=inp, pose=pose)
    tv = torch.stack([patch_valid[s * channels:(s + 1) * channels].any(0) for s in range(S)])
    enc = SimpleNamespace(z=z, patch_valid=patch_valid, key_time_valid=tv.clone())
    enc_t = SimpleNamespace(z=zt, patch_valid=patch_valid)
    rows = [{"dataset": w["ident"][0], "subject": w["ident"][1], "session": w["ident"][2]} for w in windows]
    wins = [SimpleNamespace(trial=w["ident"][3], start=w["ident"][4], stop=w["stop"]) for w in windows]
    batch = SimpleNamespace(rows=rows, windows=wins, fs=[w["ident"][5] for w in windows])
    return micro, enc, enc_t, batch


def heads(D, dtype=torch.float64, patch_ms=PATCH_MS):
    from wearusfm.model.contrastive import ProjectionHead
    torch.manual_seed(7)
    student = SimpleNamespace(contrastive_head=ProjectionHead(D, D, 16).to(dtype), cfg=SimpleNamespace(patch_ms=patch_ms))
    teacher = SimpleNamespace(contrastive_head=ProjectionHead(D, D, 16).to(dtype))
    return student, teacher


def run_attach(windows, *, arm="CR", dtype=torch.float64, step=0, cfg_over=None, pad_to=None, y_dtype=torch.float32, engine=None):
    from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime
    from wearusfm.training.jepa import LossTerms
    cfg = dict(arm=arm, fill_steps=0, ramp_steps=1, min_negatives=1, gap_seconds=0.001)
    cfg.update(cfg_over or {})
    if engine is None:
        engine = ContrastiveRuntime(asdict(ContrastiveConfig(**cfg)))
    micro, enc, enc_t, batch = fake_inputs(windows, dtype=dtype, pad_to=pad_to, y_dtype=y_dtype)
    student, teacher = heads(windows[0]["z"].shape[2], dtype)
    engine.prepare(batch, step)
    terms = LossTerms()
    terms = engine.attach(terms, student, teacher, micro, enc, enc_t)
    return terms, enc, engine
