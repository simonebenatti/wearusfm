# Independent oracle adapted from Claude review 0ab9d98, rnc_oracle_common.py.
"""Helper comuni dell'agente rnc (review Claude di db5ac41). Solo tensori sintetici."""
import math
import torch

from wearusfm.training.contrastive_memory import WindowIdentity

PATCH_MS = 25.0
FS = 2000.0


def instant(ident, p, patch_ms=PATCH_MS):
    """Indice nativo del protocollo par. 3: start + ceil((p+0,5) * patch_ms * fs / 1000 - 1e-9)."""
    return (ident.dataset, ident.subject, ident.session, ident.trial, ident.fs,
            ident.start + int(math.ceil((p + 0.5) * patch_ms * ident.fs / 1000.0 - 1e-9)))


def oracle(windows, *, temperature=0.1, stride=8, phase=None, patch_ms=PATCH_MS, per_window=True, dedup_pairs=True):
    """Definizione coppia per coppia, scritta dal protocollo (non dal codice di GPT).

    windows: lista di dict con z (P,D) float64 con grad, y (P,J), m (P,J) bool, tv (P,) bool, ident (WindowIdentity) oppure
    y None per una finestra senza posa. Ritorna (loss, n_finestre_contate, dettagli)."""
    phase = stride // 2 if phase is None else phase
    pts = []
    for w, win in enumerate(windows):
        if win.get("y") is None:
            continue
        for p in range(phase, win["z"].shape[0], stride):
            if bool(win["tv"][p]) and bool(win["m"][p].any()):
                pts.append((w, p))
    feats = []
    for w, p in pts:
        h = windows[w]["z"][p]
        h = h - h.mean()
        feats.append(h / h.norm())
    per_win, n_anchor, n_pairs = {}, 0, 0
    for i, (wi, pi) in enumerate(pts):
        mi, yi = windows[wi]["m"][pi], windows[wi]["y"][pi]
        ds, sc = {}, {}
        for k, (wk, pk) in enumerate(pts):
            if k == i:
                continue
            common = mi & windows[wk]["m"][pk]
            if not bool(common.any()):
                continue
            if dedup_pairs and instant(windows[wi]["ident"], pi, patch_ms) == instant(windows[wk]["ident"], pk, patch_ms):
                continue
            yk = windows[wk]["y"][pk]
            ds[k] = math.sqrt(sum(float(yi[j].double() - yk[j].double()) ** 2 for j in range(yi.shape[0]) if bool(common[j]))
                              / int(common.sum()))
            sc[k] = (feats[i] * feats[k]).sum() / temperature
        if len(ds) < 2:
            continue
        num, chance = 0.0, 0.0
        for j in ds:
            keys = [k for k in ds if ds[k] >= ds[j]]
            num = num + torch.logsumexp(torch.stack([sc[k] for k in keys]), 0) - sc[j]
            chance += math.log(len(keys))
        per_win.setdefault(wi, []).append(num / chance)
        n_anchor += 1
        n_pairs += len(ds)
    if not per_win:
        return None, 0, {"anchors": 0, "pairs": 0, "points": len(pts)}
    if per_window:
        loss = torch.stack([torch.stack(v).mean() for v in per_win.values()]).mean()
        count = len(per_win)
    else:
        flat = [v for g in per_win.values() for v in g]
        loss, count = torch.stack(flat).mean(), len(flat)
    return loss, count, {"anchors": n_anchor, "pairs": n_pairs, "points": len(pts)}


def ident(subject, session, start, n_patches, dataset="emg2pose", fs=FS, trial=None, patch_ms=PATCH_MS):
    stop = start + int(math.ceil(n_patches * patch_ms * fs / 1000.0 - 1e-9))
    return WindowIdentity(dataset, subject, session, trial, start, stop, fs)
